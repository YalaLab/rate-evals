"""MedImageInsight model for medical image analysis."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from collections import OrderedDict
from einops import rearrange
import torchvision.transforms as T
from safetensors.torch import load_file
from timm.layers import DropPath, trunc_normal_
from huggingface_hub import hf_hub_download
from .common import batch_apply_ct_windowing, batch_apply_normalization

from ..common import get_logger, setup_device, ModelError, ModelDownloadLock
from ..config import load_model_config, get_config_value, merge_configs

logger = get_logger(__name__)


# Configuration constants
DEFAULT_CONFIG = {
    "IMAGE_ENCODER": {
        "NAME": "davit_v1",
        "NUM_CLASSES": 0,
        "IMAGE_SIZE": [480, 480],
        "LOAD_PRETRAINED": True,
        "PRETRAINED": "",
        "PRETRAINED_LAYERS": ["*"],
        "SPEC": {
            "DROP_RATE": 0.1,
            "DROP_PATH_RATE": 0.2,
            "PATCH_SIZE": [7, 3, 3, 3],
            "PATCH_STRIDE": [4, 2, 2, 2],
            "PATCH_PADDING": [3, 1, 1, 1],
            "PATCH_PRENORM": [False, True, True, True],
            "DIM_EMBED": [256, 512, 1024, 2048],
            "NUM_HEADS": [8, 16, 32, 64],
            "NUM_GROUPS": [8, 16, 32, 64],
            "DEPTHS": [1, 1, 9, 1],
            "WINDOW_SIZE": 12,
            "ENABLE_CHECKPOINT": True,
            "STANDPARAM": True,
            "CONV_AT_ATTN": True,
            "CONV_AT_FFN": True,
            "DYNAMIC_SCALE": True,
        },
    },
    "UNICL_MODEL": {
        "DIM_PROJECTION": 1024,
        "GATHER_TENSORS": True,
        "LOAD_PRETRAINED": True,
        "PRETRAINED": "",
        "PRETRAINED_LAYERS": ["*"],
    },
    "AUG": {
        "INTERPOLATION": "bicubic",
    },
    "TEST": {
        "CENTER_CROP": False,
    },
    "VERBOSE": True,
}

# Hugging Face model repository defaults
DEFAULT_HF_REPO_ID = "lion-ai/MedImageInsights"
DEFAULT_HF_SUBFOLDER = "2024.09.27"
DEFAULT_VISION_MODEL_NAME = "medimageinsigt-v1.0.0.pt"
DEFAULT_LANGUAGE_MODEL_NAME = "language_model.pth"


# DaViT Components
class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class PreNorm(nn.Module):
    def __init__(self, norm, fn, drop_path=None):
        super().__init__()
        self.norm = norm
        self.fn = fn
        self.drop_path = drop_path

    def forward(self, x, *args, **kwargs):
        shortcut = x
        if self.norm != None:
            x, size = self.fn(self.norm(x), *args, **kwargs)
        else:
            x, size = self.fn(x, *args, **kwargs)

        if self.drop_path:
            x = self.drop_path(x)

        x = shortcut + x

        return x, size


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(in_features, hidden_features)),
                    ("act", act_layer()),
                    ("fc2", nn.Linear(hidden_features, out_features)),
                ]
            )
        )

    def forward(self, x, size):
        return self.net(x), size


class DepthWiseConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        kernel_size,
        padding,
        stride,
        bias=True,
    ):
        super().__init__()
        self.dw = nn.Conv2d(
            dim_in,
            dim_in,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim_in,
            stride=stride,
            bias=bias,
        )

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        x = self.dw(x.transpose(1, 2).view(B, C, H, W))
        size = (x.size(-2), x.size(-1))
        x = x.flatten(2).transpose(1, 2)
        return x, size


class ConvEmbed(nn.Module):
    def __init__(
        self,
        patch_size=7,
        in_chans=3,
        embed_dim=64,
        stride=4,
        padding=2,
        norm_layer=None,
        pre_norm=True,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding
        )

        dim_norm = in_chans if pre_norm else embed_dim
        self.norm = norm_layer(dim_norm) if norm_layer else None

        self.pre_norm = pre_norm

    def forward(self, x, size):
        H, W = size
        if len(x.size()) == 3:
            if self.norm and self.pre_norm:
                x = self.norm(x)
            x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        x = self.proj(x)

        _, _, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.norm and not self.pre_norm:
            x = self.norm(x)

        return x, (H, W)


class ChannelAttention(nn.Module):
    def __init__(
        self,
        dim,
        base_dim,
        groups=8,
        base_groups=8,
        qkv_bias=True,
        dynamic_scale=True,
        standparam=True,
    ):
        super().__init__()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dynamic_scale = dynamic_scale

        self.dim = dim
        self.groups = groups
        self.group_dim = dim // groups

        self.base_dim = base_dim
        self.base_groups = base_groups
        self.base_group_dim = base_dim // base_groups

        self.group_wm = self.group_dim / self.base_group_dim
        self.standparam = standparam

    def forward(self, x, size):
        B, N, C = x.shape
        assert C == self.dim

        qkv = self.qkv(x).reshape(B, N, 3, self.groups, C // self.groups).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = N**-0.5 if self.dynamic_scale else self.dim**-0.5

        if self.standparam:
            scale = N**-0.5 if self.dynamic_scale else self.dim**-0.5
        else:
            assert self.dynamic_scale
            scale = N**-0.5

        q = q * scale
        attention = q.transpose(-1, -2) @ k
        attention = attention.softmax(dim=-1)

        if not self.standparam:
            attention = attention / self.group_wm

        x = (attention @ v.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, size


class ChannelBlock(nn.Module):
    def __init__(
        self,
        dim,
        base_dim,
        groups,
        base_groups,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        conv_at_attn=True,
        conv_at_ffn=True,
        dynamic_scale=True,
        standparam=True,
    ):
        super().__init__()

        drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.conv1 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_attn else None
        self.channel_attn = PreNorm(
            norm_layer(dim),
            ChannelAttention(
                dim,
                base_dim,
                groups=groups,
                base_groups=base_groups,
                qkv_bias=qkv_bias,
                dynamic_scale=dynamic_scale,
                standparam=standparam,
            ),
            drop_path,
        )
        self.conv2 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_ffn else None
        self.ffn = PreNorm(
            norm_layer(dim),
            Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer),
            drop_path,
        )

    def forward(self, x, size):
        if self.conv1:
            x, size = self.conv1(x, size)
        x, size = self.channel_attn(x, size)

        if self.conv2:
            x, size = self.conv2(x, size)
        x, size = self.ffn(x, size)

        return x, size


def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    B = windows.shape[0] // (H * W // window_size // window_size)
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(
        self, dim, base_dim, num_heads, base_num_heads, window_size, qkv_bias=True, standparam=True
    ):
        super().__init__()

        self.window_size = window_size

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.base_dim = base_dim
        self.base_num_heads = base_num_heads
        base_head_dim = base_dim // base_num_heads

        if standparam:
            scale = float(head_dim) ** -0.5
        else:
            base_scale = float(base_head_dim) ** -0.5
            head_wm = head_dim / base_head_dim
            scale = base_scale / head_wm
        self.scale = scale

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, size):
        H, W = size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)

        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, Hp, Wp)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        return x, size


class SpatialBlock(nn.Module):
    def __init__(
        self,
        dim,
        base_dim,
        num_heads,
        base_num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        conv_at_attn=True,
        conv_at_ffn=True,
        standparam=True,
    ):
        super().__init__()

        drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.conv1 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_attn else None
        self.window_attn = PreNorm(
            norm_layer(dim),
            WindowAttention(
                dim,
                base_dim,
                num_heads,
                base_num_heads,
                window_size,
                qkv_bias=qkv_bias,
                standparam=standparam,
            ),
            drop_path,
        )
        self.conv2 = PreNorm(None, DepthWiseConv2d(dim, 3, 1, 1)) if conv_at_ffn else None
        self.ffn = PreNorm(
            norm_layer(dim),
            Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer),
            drop_path,
        )

    def forward(self, x, size):
        if self.conv1:
            x, size = self.conv1(x, size)
        x, size = self.window_attn(x, size)

        if self.conv2:
            x, size = self.conv2(x, size)
        x, size = self.ffn(x, size)
        return x, size


class DaViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        depths=(1, 1, 3, 1),
        patch_size=(7, 2, 2, 2),
        patch_stride=(4, 2, 2, 2),
        patch_padding=(3, 0, 0, 0),
        patch_prenorm=(False, False, False, False),
        embed_dims=(64, 128, 192, 256),
        base_embed_dims=(64, 128, 192, 256),
        num_heads=(3, 6, 12, 24),
        base_num_heads=(3, 6, 12, 24),
        num_groups=(3, 6, 12, 24),
        base_num_groups=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        enable_checkpoint=False,
        conv_at_attn=True,
        conv_at_ffn=True,
        dynamic_scale=True,
        standparam=True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_stages = len(self.embed_dims)
        self.enable_checkpoint = enable_checkpoint
        assert self.num_stages == len(self.num_heads) == len(self.num_groups)

        num_stages = len(embed_dims)
        self.img_size = img_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) * 2)]

        depth_offset = 0
        convs = []
        blocks = []
        for i in range(num_stages):
            conv_embed = ConvEmbed(
                patch_size=patch_size[i],
                stride=patch_stride[i],
                padding=patch_padding[i],
                in_chans=in_chans if i == 0 else self.embed_dims[i - 1],
                embed_dim=self.embed_dims[i],
                norm_layer=norm_layer,
                pre_norm=patch_prenorm[i],
            )
            convs.append(conv_embed)

            block = MySequential(
                *[
                    MySequential(
                        OrderedDict(
                            [
                                (
                                    "spatial_block",
                                    SpatialBlock(
                                        embed_dims[i],
                                        base_embed_dims[i],
                                        num_heads[i],
                                        base_num_heads[i],
                                        window_size,
                                        drop_path_rate=dpr[depth_offset + j * 2],
                                        qkv_bias=qkv_bias,
                                        mlp_ratio=mlp_ratio,
                                        conv_at_attn=conv_at_attn,
                                        conv_at_ffn=conv_at_ffn,
                                        standparam=standparam,
                                    ),
                                ),
                                (
                                    "channel_block",
                                    ChannelBlock(
                                        embed_dims[i],
                                        base_embed_dims[i],
                                        num_groups[i],
                                        base_num_groups[i],
                                        drop_path_rate=dpr[depth_offset + j * 2 + 1],
                                        qkv_bias=qkv_bias,
                                        mlp_ratio=mlp_ratio,
                                        conv_at_attn=conv_at_attn,
                                        conv_at_ffn=conv_at_ffn,
                                        dynamic_scale=dynamic_scale,
                                        standparam=standparam,
                                    ),
                                ),
                            ]
                        )
                    )
                    for j in range(depths[i])
                ]
            )
            blocks.append(block)
            depth_offset += depths[i] * 2

        self.convs = nn.ModuleList(convs)
        self.blocks = nn.ModuleList(blocks)

        self.norms = norm_layer(self.embed_dims[-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        if standparam:
            self.head = (
                nn.Linear(self.embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
            )
        else:
            self.head = (
                nn.Linear(self.embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
            )

        self.apply(self._custom_init_weights)

    @property
    def dim_out(self):
        return self.embed_dims[-1]

    def _custom_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def _try_remap_keys(self, pretrained_dict):
        remap_keys = {
            "conv_embeds": "convs",
            "main_blocks": "blocks",
            "0.cpe.0.proj": "spatial_block.conv1.fn.dw",
            "0.attn": "spatial_block.window_attn.fn",
            "0.cpe.1.proj": "spatial_block.conv2.fn.dw",
            "0.mlp": "spatial_block.ffn.fn.net",
            "1.cpe.0.proj": "channel_block.conv1.fn.dw",
            "1.attn": "channel_block.channel_attn.fn",
            "1.cpe.1.proj": "channel_block.conv2.fn.dw",
            "1.mlp": "channel_block.ffn.fn.net",
            "0.norm1": "spatial_block.window_attn.norm",
            "0.norm2": "spatial_block.ffn.norm",
            "1.norm1": "channel_block.channel_attn.norm",
            "1.norm2": "channel_block.ffn.norm",
        }

        full_key_mappings = {}
        for k in pretrained_dict.keys():
            old_k = k
            for remap_key in remap_keys.keys():
                if remap_key in k:
                    k = k.replace(remap_key, remap_keys[remap_key])
            full_key_mappings[old_k] = k

        return full_key_mappings

    def from_state_dict(self, pretrained_dict, pretrained_layers=["*"], verbose=True):
        model_dict = self.state_dict()
        stripped_key = lambda x: x[14:] if x.startswith("image_encoder.") else x
        full_key_mappings = self._try_remap_keys(pretrained_dict)

        pretrained_dict = {
            stripped_key(full_key_mappings[k]): v.to(self.device)
            for k, v in pretrained_dict.items()
            if stripped_key(full_key_mappings[k]) in model_dict.keys()
        }
        need_init_state_dict = {}
        for k, v in pretrained_dict.items():
            need_init = k.split(".")[0] in pretrained_layers or pretrained_layers[0] == "*"
            if need_init:
                if verbose:
                    logger.debug(f"=> init {k} from pretrained state dict")
                need_init_state_dict[k] = v.to(self.device)
        self.load_state_dict(need_init_state_dict, strict=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward_features(self, x):
        input_size = (x.size(2), x.size(3))
        for conv, block in zip(self.convs, self.blocks):
            x, input_size = conv(x, input_size)
            if self.enable_checkpoint:
                x, input_size = checkpoint.checkpoint(block, x, input_size)
            else:
                x, input_size = block(x, input_size)

        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.norms(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class UniCLModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.conf_image_encoder = config["IMAGE_ENCODER"]

        # Build image encoder (DaViT)
        spec = self.conf_image_encoder["SPEC"]

        self.image_encoder = DaViT(
            num_classes=self.conf_image_encoder["NUM_CLASSES"],
            depths=spec["DEPTHS"],
            embed_dims=spec["DIM_EMBED"],
            base_embed_dims=spec["DIM_EMBED"],
            num_heads=spec["NUM_HEADS"],
            base_num_heads=spec["NUM_HEADS"],
            num_groups=spec["NUM_GROUPS"],
            base_num_groups=spec["NUM_GROUPS"],
            patch_size=spec["PATCH_SIZE"],
            patch_stride=spec["PATCH_STRIDE"],
            patch_padding=spec["PATCH_PADDING"],
            patch_prenorm=spec["PATCH_PRENORM"],
            drop_path_rate=spec["DROP_PATH_RATE"],
            img_size=self.conf_image_encoder["IMAGE_SIZE"],
            window_size=spec.get("WINDOW_SIZE", 7),
            enable_checkpoint=spec.get("ENABLE_CHECKPOINT", False),
            conv_at_attn=spec.get("CONV_AT_ATTN", True),
            conv_at_ffn=spec.get("CONV_AT_FFN", True),
            dynamic_scale=spec.get("DYNAMIC_SCALE", True),
            standparam=spec.get("STANDPARAM", True),
        )

        # Projection layer for image embeddings
        dim_projection = config["UNICL_MODEL"]["DIM_PROJECTION"]
        self.image_projection = nn.Parameter(
            torch.empty(self.image_encoder.dim_out, dim_projection)
        )

        # Initialize
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.image_projection, std=0.02)

    def _convert_old_weights(self, model_dict):
        model_dict_updated = {}
        for k, v in model_dict.items():
            if k.startswith("visual."):
                model_dict_updated["image_encoder." + k[7:]] = v
            elif k == "vision_projection":
                model_dict_updated["image_projection"] = v
            elif not k.startswith("text.") and k != "text_projection":
                model_dict_updated[k] = v

        return model_dict_updated

    def from_pretrained(self, pretrained="", pretrained_layers=["*"], verbose=True):
        if not os.path.isfile(pretrained):
            logger.warning(f"=> Pretrained model ({pretrained}) is not a file, skip init weight")
            return

        pretrained_dict = load_file(pretrained)
        logger.info(f"=> Loading pretrained model {pretrained}")
        model_dict = self.state_dict()
        pretrained_dict = self._convert_old_weights(pretrained_dict)

        pretrained_dict = {k: v.to(self.device) for k, v in pretrained_dict.items()}
        need_init_state_dict = {}
        image_encoder_state_dict = {}
        for k, v in pretrained_dict.items():
            need_init = k.split(".")[0] in pretrained_layers or pretrained_layers[0] == "*"

            if need_init:
                if k.startswith("image_encoder."):
                    image_encoder_state_dict[k] = v.to(self.device)
                else:
                    if verbose:
                        logger.debug(f"=> init {k} from {pretrained}")
                    need_init_state_dict[k] = v.to(self.device)
        self.image_encoder.from_state_dict(image_encoder_state_dict, pretrained_layers, verbose)
        self.load_state_dict(need_init_state_dict, strict=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode_image(self, image, norm=True):
        x = self.image_encoder.forward_features(image)
        x = x @ self.image_projection
        if norm:
            x = x / x.norm(dim=-1, keepdim=True)
        return x


class MedImageInsight:
    """
    MedImageInsight model for medical image analysis.

    This model processes medical images using a dual-attention transformer architecture
    with both spatial and channel attention mechanisms for comprehensive feature extraction.
    """

    def __init__(self, config: dict):
        self.config = config

        # Unified config structure - CLI overrides are already in config.model
        self.model_config = config.model
        self.device = setup_device(get_config_value(config, "device"))

        # Setup model
        self.setup_model()

        logger.info(f"Initialized MedImageInsight on device: {self.device}")

    def setup_model(self) -> None:
        """Initialize the model architecture and load pretrained weights."""
        logger.info("Loading MedImageInsight model")

        try:
            # Get model configuration from config file or use defaults
            try:
                hf_repo_id = get_config_value(self.model_config, "repo_id")
            except ValueError:
                hf_repo_id = DEFAULT_HF_REPO_ID

            try:
                hf_subfolder = get_config_value(self.model_config, "hf_subfolder")
            except ValueError:
                hf_subfolder = DEFAULT_HF_SUBFOLDER

            try:
                vision_model_name = get_config_value(self.model_config, "vision_model_name")
            except ValueError:
                vision_model_name = DEFAULT_VISION_MODEL_NAME

            try:
                hf_revision = get_config_value(self.model_config, "revision")
            except ValueError:
                hf_revision = "main"

            # Get model configuration from config file or use defaults
            try:
                model_config = get_config_value(self.model_config, "model_config")
            except ValueError:
                model_config = DEFAULT_CONFIG

            # Build model
            self.model = UniCLModel(model_config)
            self.model.to(self.device)
            self.model.eval()

            # Load pretrained weights from Hugging Face
            try:
                # Create download lock for vision model
                download_lock = ModelDownloadLock(model_repo_id=hf_repo_id, revision=hf_revision)

                with download_lock.acquire_download_lock(timeout=600):
                    vision_model_path = hf_hub_download(
                        repo_id=hf_repo_id,
                        subfolder=f"{hf_subfolder}/vision_model",
                        filename=vision_model_name,
                    )
                    logger.info(
                        f"Loading pretrained weights from Hugging Face: {vision_model_path}"
                    )
                    self.model.from_pretrained(
                        vision_model_path,
                        model_config["UNICL_MODEL"]["PRETRAINED_LAYERS"],
                        model_config["VERBOSE"],
                    )
            except Exception as e:
                logger.warning(f"Could not load pretrained weights from Hugging Face: {e}")
                logger.info("Using randomly initialized weights")

            logger.info(f"MedImageInsight loaded successfully on device: {self.device}")

        except Exception as e:
            logger.error(f"Failed to load MedImageInsight model: {e}")
            raise ModelError(f"Could not load MedImageInsight model: {e}")

    @staticmethod
    def preprocess_single(image, model_config, metadata=None, modality=None):
        """
        Preprocess a single image for MedImageInsight (for use in dataset __getitem__).

        Args:
            image: Tensor from dataset (normalized to [0,1] range)
            model_config: Model configuration dictionary

        Returns:
            Preprocessed tensor ready for MedImageInsight (with ImageNet normalization applied)
        """
        # MedImageInsight expects 480x480 images
        try:
            config_dict = get_config_value(model_config, "model_config")
        except ValueError:
            config_dict = DEFAULT_CONFIG
        target_size = config_dict["IMAGE_ENCODER"]["IMAGE_SIZE"][0]

        # Handle different input shapes - resize each slice
        if len(image.shape) == 4:  # (C, D, H, W) - 3D volume
            C, D, H, W = image.shape
            # Process all slices, not just middle slice
            # Resize each slice to target size
            resized_slices = []
            for d in range(D):
                slice_2d = image[:, d, :, :]  # (C, H, W)
                resized_slice = T.functional.resize(
                    slice_2d, (target_size, target_size), interpolation=T.InterpolationMode.BICUBIC
                )
                resized_slices.append(resized_slice)
            image = torch.stack(resized_slices, dim=1)  # (C, D, target_size, target_size)
        elif len(image.shape) == 3:  # (C, H, W) - 2D image
            # Resize to target size
            image = T.functional.resize(
                image, (target_size, target_size), interpolation=T.InterpolationMode.BICUBIC
            )
        else:
            raise ValueError(f"Expected 3D (C,H,W) or 4D (C,D,H,W) input, got shape {image.shape}")

        assert not torch.isnan(image).any(), f"NaN detected in image: {image.shape}"
        return image

    def forward(self, images: torch.Tensor) -> np.ndarray:
        """
        Forward pass through the model.

        Args:
            images: Input images tensor with shape (B, C, H, W) for 2D or (B, C, D, H, W) for 3D

        Returns:
            Model embeddings as numpy array with shape (B, feature_dim)
        """
        images = images.to(self.device)

        # Handle both 2D and 3D inputs
        if len(images.shape) == 4:  # 2D images (B, C, H, W)
            N, C, H_in, W_in = images.shape
            D_in = 1
            # Add depth dimension: (B, C, H, W) -> (B, C, 1, H, W)
            images = images.unsqueeze(2)
        elif len(images.shape) == 5:  # 3D volumes (B, C, D, H, W)
            N, C, D_in, H_in, W_in = images.shape
        else:
            raise ValueError(f"Expected 4D or 5D input tensor, got shape {images.shape}")

        # Rearrange to process all slices/views: (B, C, D, H, W) -> (B*D, C, H, W)
        images_rearranged = images.permute(0, 2, 1, 3, 4).reshape(-1, C, H_in, W_in)

        # Get slice_batch_size config
        slice_batch_size = get_config_value(self.model_config, "extraction.slice_batch_size")

        # Extract features using the model with optional micro-batching
        with torch.no_grad():
            if slice_batch_size is None or D_in * N <= slice_batch_size:
                # Process all slices at once (original behavior)
                features = self.model.encode_image(images_rearranged, norm=True)
            else:
                # Micro-batch processing to avoid OOM
                feature_list = []
                total_slices = images_rearranged.shape[0]

                for start_idx in range(0, total_slices, slice_batch_size):
                    end_idx = min(start_idx + slice_batch_size, total_slices)
                    batch_slices = images_rearranged[start_idx:end_idx]
                    batch_features = self.model.encode_image(batch_slices, norm=True)
                    feature_list.append(batch_features)

                features = torch.cat(feature_list, dim=0)

        # Reshape features back to (B, D, feature_dim)
        feature_dim = features.shape[-1]
        features_reshaped = features.view(N, D_in, feature_dim)

        # Apply pooling operation across depth dimension
        try:
            pool_op = get_config_value(self.model_config, "extraction.pool_op")
        except ValueError:
            pool_op = get_config_value(self.model_config, "pool_op")

        if pool_op == "max":
            aggregated_features = features_reshaped.max(1).values.float().cpu().numpy()
        elif pool_op == "mean":
            aggregated_features = features_reshaped.mean(1).float().cpu().numpy()
        elif pool_op == "median":
            aggregated_features = features_reshaped.median(1).values.float().cpu().numpy()
        elif pool_op == "middle":
            # Select the middle frame only
            middle_idx = D_in // 2
            aggregated_features = features_reshaped[:, middle_idx, :].float().cpu().numpy()
        else:
            raise ValueError(f"Unsupported pooling operation: {pool_op}")

        return aggregated_features

    def preprocess(self, volumes: torch.Tensor, modality: str) -> torch.Tensor:
        """
        Preprocess input volumes before extraction.
        """
        # Apply CT windowing if modality is chest_ct
        if modality in ["chest_ct", "abdomen_ct", "brain_ct", "breast_mr"]:
            # The model has merged config with CLI overrides
            ct_window_type = get_config_value(self.model_config, "preprocessing.ct.window_type")

            assert ct_window_type is not None, "CT window type is not set"
            assert volumes.dim() == 5, f"Volumes should be 5D, got {volumes.dim()}"

            # Get normalization params from merged config
            try:
                normalize_mean = get_config_value(
                    self.model_config, "preprocessing.ct.normalize_mean"
                )
            except ValueError:
                normalize_mean = get_config_value(self.model_config, "ct_normalize_mean")

            try:
                normalize_std = get_config_value(
                    self.model_config, "preprocessing.ct.normalize_std"
                )
            except ValueError:
                normalize_std = get_config_value(self.model_config, "ct_normalize_std")

            per_sample = get_config_value(self.model_config, "per_sample_windowing")
            volumes = batch_apply_ct_windowing(
                volumes,
                ct_window_type=ct_window_type,
                modality="CT",
                per_sample=per_sample,
            )
        else:
            assert "xray" in modality.lower(), f"Modality {modality} is not supported"
            try:
                normalize_mean = get_config_value(
                    self.model_config, "preprocessing.xray.normalize_mean"
                )
            except ValueError:
                normalize_mean = get_config_value(self.model_config, "xray_normalize_mean")

            try:
                normalize_std = get_config_value(
                    self.model_config, "preprocessing.xray.normalize_std"
                )
            except ValueError:
                normalize_std = get_config_value(self.model_config, "xray_normalize_std")

        volumes = batch_apply_normalization(volumes, normalize_mean, normalize_std)
        return volumes

    @torch.no_grad()
    def extract_features(self, inputs: torch.Tensor, modality="chest_xray") -> np.ndarray:
        """
        Extract features from input images.

        Args:
            inputs: Input tensor of shape (B, C, H, W) or (B, C, D, H, W)
            modality: Imaging modality (for consistency with other models)

        Returns:
            Feature embeddings as numpy array
        """
        inputs = self.preprocess(inputs, modality)

        # inputs should be (B, C, D, H, W) or (B, C, H, W)
        return self.forward(inputs)
