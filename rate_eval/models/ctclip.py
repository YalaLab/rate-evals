"""CT-CLIP implementation."""

import os
import copy
from functools import wraps
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
from transformers import BertModel
from torch.utils.checkpoint import checkpoint
from vector_quantize_pytorch import VectorQuantize

from ..common import get_logger, setup_device
from ..config import load_model_config, get_config_value, merge_configs
from .common import batch_apply_ct_windowing, batch_apply_normalization


logger = get_logger(__name__)


def l2norm(t):
    return F.normalize(t, dim=-1)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def leaky_relu(p=0.1):
    return nn.LeakyReLU(p)


def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret


def make_checkpointable(fn):
    @wraps(fn)
    def inner(*args):
        input_needs_grad = any([isinstance(el, torch.Tensor) and el.requires_grad for el in args])
        if not input_needs_grad:
            return fn(*args)
        return checkpoint(fn, *args)

    return inner


def identity(t, *args, **kwargs):
    return t


def max_neg_value(dtype):
    return -torch.finfo(dtype).max


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class CLIPLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4, dropout=0.0):
    inner_dim = int(mult * (2 / 3) * dim)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
    )


class CLIPFeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2, bias=False),
            GEGLU(),
            CLIPLayerNorm(inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context=None,
        dim_head=64,
        heads=8,
        causal=False,
        num_null_kv=0,
        norm_context=True,
        dropout=0.0,
        scale=8,
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = scale
        inner_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        self.attn_dropout = nn.Dropout(dropout)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(heads, 2 * num_null_kv, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias=False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, mask=None, context=None, attn_bias=None):
        batch, device, dtype = x.shape[0], x.device, x.dtype

        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        x = self.norm(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))

        nk, nv = repeat(self.null_kv, "h (n r) d -> b h n r d", b=batch, r=2).unbind(dim=-2)

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        i, j = sim.shape[-2:]

        if exists(attn_bias):
            attn_bias = F.pad(attn_bias, (self.num_null_kv, 0), value=0.0)
            sim = sim + attn_bias

        if exists(mask):
            mask = F.pad(mask, (self.num_null_kv, 0), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            causal_mask = torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class PEG(nn.Module):
    def __init__(self, dim, causal=False):
        super().__init__()
        self.causal = causal
        self.dsconv = nn.Conv3d(dim, dim, 3, groups=dim)

    def forward(self, x, shape: Tuple[int, int, int, int] = None):
        needs_shape = x.ndim == 3
        assert not (needs_shape and not exists(shape))

        orig_shape = x.shape

        if needs_shape:
            x = x.reshape(*shape, -1)

        x = rearrange(x, "b ... d -> b d ...")

        frame_padding = (2, 0) if self.causal else (1, 1)

        x = F.pad(x, (1, 1, 1, 1, *frame_padding), value=0.0)
        x = self.dsconv(x)

        x = rearrange(x, "b d ... -> b ... d")

        if needs_shape:
            x = rearrange(x, "b ... d -> b (...) d")

        return x.reshape(orig_shape)


class ContinuousPositionBias(nn.Module):
    def __init__(self, *, dim, heads, num_dims=2, layers=2, log_dist=True, cache_rel_pos=False):
        super().__init__()
        self.num_dims = num_dims
        self.log_dist = log_dist

        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(self.num_dims, dim), leaky_relu()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), leaky_relu()))

        self.net.append(nn.Linear(dim, heads))

        self.cache_rel_pos = cache_rel_pos
        self.register_buffer("rel_pos", None, persistent=False)

    def forward(self, *dimensions, device=torch.device("cpu")):
        if not exists(self.rel_pos) or not self.cache_rel_pos:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            positions = [torch.arange(d, device=device) for d in dimensions]
            grid = torch.stack(torch.meshgrid(*positions, indexing="ij"))
            grid = rearrange(grid, "c ... -> (...) c")
            rel_pos = rearrange(grid, "i c -> i 1 c") - rearrange(grid, "j c -> 1 j c")

            if self.log_dist:
                rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)

            self.register_buffer("rel_pos", rel_pos, persistent=False)

        rel_pos = self.rel_pos.to(torch.float32)

        for layer in self.net:
            rel_pos = layer(rel_pos.float())

        return rearrange(rel_pos, "i j h -> h i j")


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_context=None,
        causal=False,
        dim_head=64,
        heads=8,
        ff_mult=4,
        peg=False,
        peg_causal=False,
        attn_num_null_kv=2,
        has_cross_attn=False,
        attn_dropout=0.0,
        ff_dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PEG(dim=dim, causal=peg_causal) if peg else None,
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            causal=causal,
                            dropout=attn_dropout,
                        ),
                        (
                            Attention(
                                dim=dim,
                                dim_head=dim_head,
                                dim_context=dim_context,
                                heads=heads,
                                causal=False,
                                num_null_kv=attn_num_null_kv,
                                dropout=attn_dropout,
                            )
                            if has_cross_attn
                            else None
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.norm_out = LayerNorm(dim)

    def forward(
        self,
        x,
        video_shape: Tuple[int, int, int, int] = None,
        attn_bias=None,
        context=None,
        self_attn_mask=None,
        cross_attn_context_mask=None,
    ):
        for peg, self_attn, cross_attn, ff in self.layers:
            if exists(peg):
                x = peg(x, shape=video_shape) + x

            x = self_attn(x, attn_bias=attn_bias, mask=self_attn_mask) + x

            if exists(cross_attn) and exists(context):
                x = cross_attn(x, context=context, mask=cross_attn_context_mask) + x

            x = ff(x) + x

        return self.norm_out(x)


class CTViT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        codebook_size,
        image_size,
        patch_size,
        temporal_patch_size,
        spatial_depth,
        temporal_depth,
        dim_head=64,
        heads=8,
        channels=1,
        attn_dropout=0.0,
        ff_dropout=0.0,
    ):
        super().__init__()

        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size

        self.temporal_patch_size = temporal_patch_size

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads)

        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0

        self.to_patch_emb_first_frame = nn.Sequential(
            Rearrange("b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(channels * patch_width * patch_height),
            nn.Linear(channels * patch_width * patch_height, dim),
            nn.LayerNorm(dim),
        )

        self.to_patch_emb = nn.Sequential(
            Rearrange(
                "b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)",
                p1=patch_height,
                p2=patch_width,
                pt=temporal_patch_size,
            ),
            nn.LayerNorm(channels * patch_width * patch_height * temporal_patch_size),
            nn.Linear(channels * patch_width * patch_height * temporal_patch_size, dim),
            nn.LayerNorm(dim),
        )

        transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,
            peg_causal=True,
        )
        self.enc_spatial_transformer = Transformer(depth=spatial_depth, **transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth=temporal_depth, **transformer_kwargs)
        self.vq = VectorQuantize(dim=dim, codebook_size=codebook_size, use_cosine_sim=True)

        self.to_pixels_first_frame = nn.Sequential(
            nn.Linear(dim, channels * patch_width * patch_height),
            Rearrange("b 1 h w (c p1 p2) -> b c 1 (h p1) (w p2)", p1=patch_height, p2=patch_width),
        )

        self.to_pixels = nn.Sequential(
            nn.Linear(dim, channels * patch_width * patch_height * temporal_patch_size),
            Rearrange(
                "b t h w (c pt p1 p2) -> b c (t pt) (h p1) (w p2)",
                p1=patch_height,
                p2=patch_width,
                pt=temporal_patch_size,
            ),
        )

    @property
    def image_num_tokens(self):
        return int(self.image_size[0] / self.patch_size[0]) * int(
            self.image_size[1] / self.patch_size[1]
        )

    @property
    def patch_height_width(self):
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    def encode(self, tokens):
        b = tokens.shape[0]
        h, w = self.patch_height_width

        video_shape = tuple(tokens.shape[:-1])

        tokens = rearrange(tokens, "b t h w d -> (b t) (h w) d")
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        attn_bias = self.spatial_rel_pos_bias(h, w, device=device)

        tokens = self.enc_spatial_transformer(tokens, attn_bias=attn_bias, video_shape=video_shape)

        tokens = rearrange(tokens, "(b t) (h w) d -> b t h w d", b=b, h=h, w=w)

        tokens = rearrange(tokens, "b t h w d -> (b h w) t d")
        tokens = self.enc_temporal_transformer(tokens, video_shape=video_shape)
        tokens = rearrange(tokens, "(b h w) t d -> b t h w d", b=b, h=h, w=w)

        return tokens

    def forward(
        self, video, mask=None, return_only_codebook_ids=False, return_encoded_tokens=False
    ):
        assert video.ndim in {4, 5}

        is_image = video.ndim == 4

        if is_image:
            video = rearrange(video, "b c h w -> b c 1 h w")
            assert not exists(mask)

        b, c, f, *image_dims, device = *video.shape, video.device
        assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f

        tokens = self.to_patch_emb(video)

        shape = tokens.shape
        *_, h, w, _ = shape

        tokens = self.encode(tokens)

        tokens, packed_fhw_shape = pack([tokens], "b * d")

        vq_mask = None

        tokens, indices, commit_loss = self.vq(tokens, mask=vq_mask)

        if return_only_codebook_ids:
            (indices,) = unpack(indices, packed_fhw_shape, "b *")
            return indices

        tokens = rearrange(tokens, "b (t h w) d -> b t h w d", h=h, w=w)

        if return_encoded_tokens:
            return tokens

        return tokens


class RearrangeImage(nn.Module):
    def forward(self, x):
        global h_r, w_r, z_r
        return rearrange(x, "b (h w z) c -> b c h w z", h=h_r, w=w_r)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = CLIPLayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class CLIPAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, causal=False, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias=False), CLIPLayerNorm(dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, rotary_pos_emb=None):
        h, device, scale = self.heads, x.device, self.scale

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = q * self.scale

        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k)

        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.type(sim.dtype)

        attn = self.dropout(attn)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CLIPTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head=64,
        heads=8,
        causal=False,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        checkpoint_during_training=False,
    ):
        super().__init__()
        self.checkpoint_during_training = checkpoint_during_training

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            CLIPAttention(
                                dim=dim,
                                dim_head=dim_head,
                                heads=heads,
                                causal=causal,
                                dropout=attn_dropout,
                            ),
                        ),
                        PreNorm(dim, CLIPFeedForward(dim=dim, mult=ff_mult)),
                    ]
                )
            )

        self.norm_in = CLIPLayerNorm(dim)
        self.norm_out = CLIPLayerNorm(dim)

    def forward(self, x, rotary_pos_emb=None, mask=None):
        can_checkpoint = self.training and self.checkpoint_during_training
        checkpoint_fn = make_checkpointable if can_checkpoint else identity

        x = self.norm_in(x)

        for attn, ff in self.layers:
            attn, ff = map(checkpoint_fn, (attn, ff))

            x = attn(x, mask, rotary_pos_emb) + x
            x = ff(x) + x

        return self.norm_out(x)


class CTCLIPModel(nn.Module):
    """CT-CLIP model architecture."""

    def __init__(
        self,
        *,
        image_encoder=None,
        text_encoder=None,
        dim_text=768,
        dim_image=294912,
        dim_latent=512,
        visual_enc_depth=6,
        visual_heads=8,
        visual_dim_head=64,
        visual_image_size=480,
        visual_patch_size=20,
        visual_temporal_patch_size=10,
        visual_spatial_depth=4,
        visual_temporal_depth=4,
        channels=1,
        codebook_size=8192,
        extra_latent_projection=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False,
        **kwargs,
    ):
        super().__init__()
        self.dtype = torch.float32

        self.dim_text = dim_text
        self.dim_image = dim_image
        self.dim_latent = dim_latent

        self.image_channels = channels
        self.image_size = visual_image_size

        self.use_all_token_embeds = use_all_token_embeds
        self.extra_latent_projection = extra_latent_projection

        # Text encoder
        if exists(text_encoder):
            self.text_transformer = text_encoder
        else:
            self.text_transformer = BertModel.from_pretrained(
                "microsoft/BiomedVLP-CXR-BERT-specialized"
            )

        # Visual encoder
        if exists(image_encoder):
            self.visual_transformer = image_encoder
        else:
            self.visual_transformer = CTViT(
                dim=512,
                codebook_size=codebook_size,
                image_size=visual_image_size,
                patch_size=visual_patch_size,
                temporal_patch_size=visual_temporal_patch_size,
                spatial_depth=visual_spatial_depth,
                temporal_depth=visual_temporal_depth,
                dim_head=visual_dim_head,
                heads=visual_heads,
                channels=channels,
            )

        # Projection layers
        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias=False)

        if downsample_image_embeds:
            dim_conv = 512
            self.to_visual_latent = nn.Sequential(
                RearrangeImage(),
                nn.Conv3d(dim_conv, dim_conv, 4, stride=2, padding=1, bias=False, groups=dim_conv),
                nn.Conv3d(dim_conv, dim_latent, 1),
                Rearrange("b c h w z -> b (h w z c)"),
                nn.Linear(dim_image, dim_latent, bias=False),
            )
        else:
            self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias=False)

        # Temperature parameter
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Extra projection layers
        self.to_text_latent_extra = copy.deepcopy(self.to_text_latent)
        self.to_visual_latent_extra = copy.deepcopy(self.to_visual_latent)

    def forward(
        self,
        text,
        image,
        device,
        return_loss=False,
        return_encodings=False,
        return_latents=False,
        text_to_image=True,
    ):

        def _print_tensor_stats(label: str, tensor: torch.Tensor) -> None:
            return

            try:
                tensor_cpu = tensor.detach().cpu()
            except AttributeError:
                tensor_cpu = torch.as_tensor(tensor).detach().cpu()

            flat = tensor_cpu.flatten()
            preview = flat[: min(10, flat.numel())].tolist() if flat.numel() else []
            mean_val = float(tensor_cpu.mean()) if tensor_cpu.numel() else float("nan")

            print(f"[CTCLIP] {label} shape: {tuple(tensor.shape)}")
            print(f"[CTCLIP] {label} preview: {preview}")
            print(f"[CTCLIP] {label} mean: {mean_val}")

        _print_tensor_stats("image input", image)

        # Encode text
        text_embeddings = self.text_transformer(text.input_ids, attention_mask=text.attention_mask)
        enc_text = text_embeddings[0]

        # Encode image
        enc_image = self.visual_transformer(image, return_encoded_tokens=True)

        _print_tensor_stats("encoded vision tokens", enc_image)

        global h_r, w_r, z_r
        h_r, w_r, z_r = enc_image.shape[1], enc_image.shape[2], enc_image.shape[3]

        enc_image_send = enc_image
        enc_image = torch.mean(enc_image, dim=1)
        enc_image = enc_image.view(enc_image.shape[0], -1)

        if return_encodings:
            return enc_text, enc_image

        # Get embeddings
        if self.use_all_token_embeds:
            text_embeds = (
                enc_text[:, 1:]
                if hasattr(self, "text_has_cls_token") and self.text_has_cls_token
                else enc_text
            )
            image_embeds = (
                enc_image[:, 1:]
                if hasattr(self, "visual_has_cls_token") and self.visual_has_cls_token
                else enc_image
            )
        else:
            text_embeds = enc_text[:, 0, :] if enc_text.ndim == 3 else enc_text
            image_embeds = enc_image[:, :] if enc_image.ndim == 3 else enc_image

        # Project to latents
        text_latents = self.to_text_latent(text_embeds)
        image_latents = self.to_visual_latent(image_embeds)

        text_latents = l2norm(text_latents)
        image_latents = l2norm(image_latents)

        _print_tensor_stats("image latents", image_latents)

        # Extra projections
        text_latents_extra, image_latents_extra = text_latents, image_latents
        if self.extra_latent_projection:
            text_latents_extra = self.to_text_latent_extra(text_embeds)
            image_latents_extra = self.to_visual_latent_extra(image_embeds)
            text_latents_extra = l2norm(text_latents_extra)
            image_latents_extra = l2norm(image_latents_extra)

        if return_latents:
            if self.extra_latent_projection:
                return text_latents, image_latents, text_latents_extra, image_latents_extra
            return text_latents, image_latents, enc_image_send

        # Calculate similarity
        temp = self.temperature.exp()

        if not return_loss and self.use_all_token_embeds:
            einsum_args = (
                (text_latents_extra, image_latents_extra)
                if self.extra_latent_projection and not text_to_image
                else (text_latents, image_latents)
            )
            return torch.einsum("b d, b i d -> b t i", *einsum_args) * temp

        if not return_loss and not self.use_all_token_embeds:
            einsum_args = (
                (text_latents_extra, image_latents_extra)
                if self.extra_latent_projection and not text_to_image
                else (text_latents, image_latents)
            )
            return torch.einsum("b d, b d -> b", *einsum_args) * temp

        return torch.tensor(0.0, device=device, dtype=torch.float32)


h_r, w_r, z_r = None, None, None  # Global variables for RearrangeImage


class CTCLIP:
    """CT-CLIP medical image analysis."""

    def __init__(self, config: dict):
        self.config = config

        # Load model-specific config and merge with CLI overrides
        base_model_config = load_model_config("ctclip")
        self.model_config = merge_configs(base_model_config, config)
        self.device = setup_device(get_config_value(config, "device"))

        # Config is already merged with CLI overrides taking precedence
        self.model_checkpoint = get_config_value(self.model_config, "model_checkpoint")

        # Setup model
        self.setup_model()

        logger.info(f"Initialized CT-CLIP on device: {self.device}")

    def setup_model(self) -> None:
        """Initialize the model architecture and load pretrained weights."""
        logger.info("Loading CT-CLIP model")

        # Initialize text encoder
        text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

        # Initialize image encoder
        image_encoder = CTViT(
            dim=512,
            codebook_size=8192,
            image_size=480,
            patch_size=20,
            temporal_patch_size=10,
            spatial_depth=4,
            temporal_depth=4,
            dim_head=32,
            heads=8,
            channels=1,
        )

        # Initialize CLIP model
        self.model = CTCLIPModel(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            dim_image=294912,
            dim_text=768,
            dim_latent=512,
            visual_image_size=480,
            visual_patch_size=20,
            visual_temporal_patch_size=10,
            visual_spatial_depth=4,
            visual_temporal_depth=4,
            visual_dim_head=32,
            visual_heads=8,
            channels=1,
            codebook_size=8192,
            extra_latent_projection=False,
            downsample_image_embeds=False,
            use_all_token_embeds=False,
        )

        # Load checkpoint if provided
        if self.model_checkpoint and os.path.exists(self.model_checkpoint):
            logger.info(f"Loading CT-CLIP checkpoint from {self.model_checkpoint}")
            checkpoint_data = torch.load(self.model_checkpoint, map_location=self.device)

            # Load state dict with strict=False to handle missing keys
            mismatched_keys = self.model.load_state_dict(checkpoint_data, strict=False)
            if mismatched_keys.missing_keys:
                logger.warning(f"Missing keys: {mismatched_keys.missing_keys[:5]}...")
            if mismatched_keys.unexpected_keys:
                logger.warning(f"Unexpected keys: {mismatched_keys.unexpected_keys[:5]}...")
            logger.info("Checkpoint loaded successfully")
        else:
            logger.warning(
                f"No checkpoint found at {self.model_checkpoint}, using random initialization"
            )

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded successfully on device: {self.device}")

    @staticmethod
    def preprocess_single(image, model_config, metadata=None, modality=None):
        """
        Preprocess a single exam for CT-CLIP (for use in dataset __getitem__).

        Args:
            image: Tensor from dataset (normalized to [0,1] range)
            model_config: Model configuration dictionary

        Returns:
            Preprocessed tensor ready for CT-CLIP
        """
        # CT-CLIP expects input in [-1, 1] range after HU windowing
        # The windowing and normalization will be handled in extract_features
        return image

    def preprocess_ct(self, volumes: torch.Tensor) -> torch.Tensor:
        """
        Preprocess CT volumes for CT-CLIP.
        CT-CLIP expects input clipped to [-1000, 1000] HU and normalized to [-1, 1].
        """
        # Detect whether the input is already normalized to [-1, 1]
        with torch.no_grad():
            max_abs = volumes.detach().abs().max().item() if volumes.numel() > 0 else 0.0

        if max_abs <= 1.5:
            # Volume already normalized; clamp lightly to remove minor numerical drift
            return torch.clamp(volumes, -1.0, 1.0)

        # Otherwise assume we are in raw HU and perform standard normalization
        hu_min, hu_max = -1000, 1000
        volumes = torch.clamp(volumes, hu_min, hu_max)

        return volumes / 1000.0

    def preprocess(self, volumes: torch.Tensor, modality: str) -> torch.Tensor:
        """
        Preprocess input volumes before extraction.
        """
        if modality in ["chest_ct", "abdomen_ct"]:
            # Apply CT-specific preprocessing
            volumes = self.preprocess_ct(volumes)

            # Apply windowing if configured
            ct_window_type = get_config_value(self.model_config, "ct_window_type")
            if ct_window_type:
                assert volumes.dim() == 5, f"Volumes should be 5D, got {volumes.dim()}"

                # Get normalization params from merged config
                normalize_mean = get_config_value(self.model_config, "ct_normalize_mean")
                normalize_std = get_config_value(self.model_config, "ct_normalize_std")

                if ct_window_type and ct_window_type != "none":
                    per_sample = get_config_value(self.model_config, "per_sample_windowing")
                    volumes = batch_apply_ct_windowing(
                        volumes,
                        ct_window_type=ct_window_type,
                        modality="CT",
                        per_sample=per_sample,
                    )

                if normalize_mean is not None and normalize_std is not None:
                    volumes = batch_apply_normalization(volumes, normalize_mean, normalize_std)
        else:
            # For non-CT modalities, apply standard normalization
            normalize_mean = get_config_value(self.model_config, "xray_normalize_mean")
            normalize_std = get_config_value(self.model_config, "xray_normalize_std")
            volumes = batch_apply_normalization(volumes, normalize_mean, normalize_std)

        return volumes

    def resize_volume(self, volume: torch.Tensor, target_shape=(480, 480, 240)) -> torch.Tensor:
        """
        Resize volume to target shape with center crop/pad.
        Matches standalone implementation: input (C, D, H, W), target_shape (H, W, D)

        Args:
            volume: Input volume tensor of shape (C, D, H, W)
            target_shape: Target shape (H, W, D) = (480, 480, 240)

        Returns:
            Resized volume tensor of shape (C, D, H, W)
        """
        if volume.dim() != 4:
            raise ValueError(f"Expected 4D tensor (C, D, H, W), got {volume.dim()}D")

        C, D, H, W = volume.shape
        target_h, target_w, target_d = target_shape

        # Permute to (C, H, W, D) for processing
        volume = volume.permute(0, 2, 3, 1)  # (C, D, H, W) -> (C, H, W, D)

        # Center crop/pad in (H, W, D) space
        h_start = max((H - target_h) // 2, 0)
        h_end = min(h_start + target_h, H)
        w_start = max((W - target_w) // 2, 0)
        w_end = min(w_start + target_w, W)
        d_start = max((D - target_d) // 2, 0)
        d_end = min(d_start + target_d, D)

        cropped = volume[:, h_start:h_end, w_start:w_end, d_start:d_end]

        # Pad if necessary
        pad_h_before = (target_h - cropped.size(1)) // 2
        pad_h_after = target_h - cropped.size(1) - pad_h_before
        pad_w_before = (target_w - cropped.size(2)) // 2
        pad_w_after = target_w - cropped.size(2) - pad_w_before
        pad_d_before = (target_d - cropped.size(3)) // 2
        pad_d_after = target_d - cropped.size(3) - pad_d_before

        # Pad format: (left, right, top, bottom, front, back) for last 3 dims
        padded = F.pad(
            cropped,
            (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after),
        )

        # Convert back to (C, D, H, W) format to match expected model input
        padded = padded.permute(0, 3, 1, 2)  # (C, H, W, D) -> (C, D, H, W)

        return padded

    @torch.no_grad()
    def extract_image_features(self, inputs: torch.Tensor, modality="chest_ct") -> np.ndarray:
        """
        Extract features from input images/volumes using image encoder only.

        Args:
            inputs: Input tensor of shape (B, C, H, W) for 2D or (B, C, D, H, W) for 3D
            modality: Type of medical imaging modality

        Returns:
            Feature embeddings as numpy array of shape (B, feature_dim)
        """
        inputs = inputs.to(self.device)
        inputs = self.preprocess(inputs, modality)

        def _print_tensor_stats(label: str, tensor: torch.Tensor) -> None:
            return
            tensor_cpu = tensor.detach().cpu()
            flat = tensor_cpu.flatten()
            preview = flat[: min(10, flat.numel())].tolist() if flat.numel() else []
            mean_val = float(tensor_cpu.mean()) if tensor_cpu.numel() else float("nan")
            print(f"[CTCLIP] {label} shape: {tuple(tensor.shape)}")
            print(f"[CTCLIP] {label} preview: {preview}")
            print(f"[CTCLIP] {label} mean: {mean_val}")

        # Handle different input shapes
        if modality in ["chest_ct", "abdomen_ct"]:
            # For CT volumes, resize to expected shape
            if inputs.dim() == 5:  # (B, C, D, H, W)
                B, C, D, H, W = inputs.shape
                # Resize each volume in the batch
                resized = []
                for i in range(B):
                    volume = self.resize_volume(inputs[i], target_shape=(480, 480, 240))
                    resized.append(volume)
                inputs = torch.stack(resized)

        _print_tensor_stats("image input", inputs)

        # Extract features using only the visual encoder
        with torch.no_grad():
            # Direct visual encoding without text
            enc_image_features = self.model.visual_transformer(inputs, return_encoded_tokens=True)

            # Process the features
            global h_r, w_r, z_r
            h_r, w_r, z_r = (
                enc_image_features.shape[1],
                enc_image_features.shape[2],
                enc_image_features.shape[3],
            )

            print(f"enc_image_features shape: {enc_image_features.shape}")
            # Pool features (mean pooling according to CT-CLIP)
            features = torch.mean(enc_image_features, dim=1)

            features = features.view(features.shape[0], -1)

            # Project to latent space
            latent_features = self.model.to_visual_latent(features)
            latent_features = l2norm(latent_features)

            _print_tensor_stats("image latents", latent_features)

            features = latent_features.cpu().numpy().astype(np.float32)

        return features

    @torch.no_grad()
    def extract_features(self, inputs: torch.Tensor, modality="chest_ct") -> np.ndarray:
        """
        Extract features from input images/volumes.

        Args:
            inputs: Input tensor of shape (B, C, H, W) for 2D or (B, C, D, H, W) for 3D
            modality: Type of medical imaging modality

        Returns:
            Feature embeddings as numpy array of shape (B, feature_dim)
        """
        # Use the image-only feature extraction method
        return self.extract_image_features(inputs, modality)

    def eval(self):
        """Set model to evaluation mode."""
        if hasattr(self, "model"):
            self.model.eval()
        return self
