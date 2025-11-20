"""Merlin multimodal medical image analysis implementation."""

import os
import warnings
import torch
import numpy as np
import copy
import math
from pathlib import Path
from typing import List, Optional
import torchvision
from torch import nn
from torch.nn import Parameter
import torch.utils.checkpoint as checkpoint
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import wordpunct_tokenize
from huggingface_hub import hf_hub_download
import monai
from monai.transforms import (
    ScaleIntensityRange,
    SpatialPad,
    CenterSpatialCrop,
    Spacingd,
    Resize,
)

from ..common import get_logger, setup_device, ModelError, ModelDownloadLock
from ..config import load_model_config, get_config_value, merge_configs

torch.serialization.add_safe_globals([monai.data.meta_tensor.MetaTensor])

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


def sanitize_report(report):
    """Sanitize radiology report text"""
    report = report.lower()
    return " ".join(wordpunct_tokenize(report))


def inflate_conv(conv2d, time_dim=3, time_padding=0, time_stride=1, time_dilation=1, center=False):
    """Inflate 2D convolution to 3D"""
    if conv2d.kernel_size[0] == 7:
        kernel_dim = (3, 7, 7)
        padding = (1, 3, 3)
        stride = (1, 2, 2)
        dilation = (1, 1, 1)
        conv3d = torch.nn.Conv3d(
            conv2d.in_channels,
            conv2d.out_channels,
            kernel_dim,
            padding=padding,
            dilation=dilation,
            stride=stride,
        )
        weight_2d = conv2d.weight.data
        if center:
            weight_3d = torch.zeros(*weight_2d.shape)
            weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            middle_idx = time_dim // 2
            weight_3d[:, :, middle_idx, :, :] = weight_2d
        else:
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            weight_3d = weight_3d / time_dim

        conv3d.weight = Parameter(weight_3d)
        conv3d.bias = conv2d.bias
    else:
        kernel_dim = (time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1])
        padding = (time_padding, conv2d.padding[0], conv2d.padding[1])
        stride = (time_stride, conv2d.stride[0], conv2d.stride[0])
        dilation = (time_dilation, conv2d.dilation[0], conv2d.dilation[1])
        conv3d = torch.nn.Conv3d(
            conv2d.in_channels,
            conv2d.out_channels,
            kernel_dim,
            padding=padding,
            dilation=dilation,
            stride=stride,
        )
        weight_2d = conv2d.weight.data
        if center:
            weight_3d = torch.zeros(*weight_2d.shape)
            weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            middle_idx = time_dim // 2
            weight_3d[:, :, middle_idx, :, :] = weight_2d
        else:
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
            weight_3d = weight_3d / time_dim

        conv3d.weight = Parameter(weight_3d)
        conv3d.bias = conv2d.bias
    return conv3d


def inflate_linear(linear2d, time_dim):
    """Inflate 2D linear layer to handle 3D features"""
    linear3d = torch.nn.Linear(linear2d.in_features * time_dim, linear2d.out_features)
    weight3d = linear2d.weight.data.repeat(1, time_dim)
    weight3d = weight3d / time_dim

    linear3d.weight = Parameter(weight3d)
    linear3d.bias = linear2d.bias
    return linear3d


def inflate_batch_norm(batch2d):
    """Inflate 2D batch norm to 3D"""
    batch3d = torch.nn.BatchNorm3d(batch2d.num_features)
    batch2d._check_input_dim = batch3d._check_input_dim
    return batch2d


def inflate_pool(pool2d, time_dim=1, time_padding=0, time_stride=None, time_dilation=1):
    """Inflate 2D pooling layer to 3D"""
    if isinstance(pool2d, torch.nn.AdaptiveAvgPool2d):
        pool3d = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
    else:
        kernel_dim = (time_dim, pool2d.kernel_size, pool2d.kernel_size)
        padding = (time_padding, pool2d.padding, pool2d.padding)
        if time_stride is None:
            time_stride = time_dim
        stride = (time_stride, pool2d.stride, pool2d.stride)
        if isinstance(pool2d, torch.nn.MaxPool2d):
            dilation = (time_dilation, pool2d.dilation, pool2d.dilation)
            pool3d = torch.nn.MaxPool3d(
                kernel_dim,
                padding=padding,
                dilation=dilation,
                stride=stride,
                ceil_mode=pool2d.ceil_mode,
            )
        elif isinstance(pool2d, torch.nn.AvgPool2d):
            pool3d = torch.nn.AvgPool3d(kernel_dim, stride=stride)
        else:
            raise ValueError("{} is not among known pooling classes".format(type(pool2d)))
    return pool3d


def inflate_downsample(downsample2d, time_stride=1):
    """Inflate 2D downsample layer to 3D"""
    downsample3d = torch.nn.Sequential(
        inflate_conv(downsample2d[0], time_dim=1, time_stride=time_stride, center=True),
        inflate_batch_norm(downsample2d[1]),
    )
    return downsample3d


class Bottleneck3d(torch.nn.Module):
    """3D Bottleneck block for ResNet"""

    def __init__(self, bottleneck2d):
        super(Bottleneck3d, self).__init__()

        spatial_stride = bottleneck2d.conv2.stride[0]

        self.conv1 = inflate_conv(bottleneck2d.conv1, time_dim=1, center=True)
        self.bn1 = inflate_batch_norm(bottleneck2d.bn1)

        self.conv2 = inflate_conv(
            bottleneck2d.conv2,
            time_dim=3,
            time_padding=1,
            time_stride=spatial_stride,
            center=True,
        )
        self.bn2 = inflate_batch_norm(bottleneck2d.bn2)

        self.conv3 = inflate_conv(bottleneck2d.conv3, time_dim=1, center=True)
        self.bn3 = inflate_batch_norm(bottleneck2d.bn3)

        self.relu = torch.nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = inflate_downsample(
                bottleneck2d.downsample, time_stride=spatial_stride
            )
        else:
            self.downsample = None

        self.stride = bottleneck2d.stride

    def forward(self, x):
        def run_function(input_x):
            out = self.conv1(input_x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)
            return out

        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        if x.requires_grad:
            out = checkpoint.checkpoint(run_function, x)
        else:
            out = run_function(x)

        out = out + residual
        out = self.relu(out)
        return out


def inflate_reslayer(reslayer2d):
    """Inflate 2D ResNet layer to 3D"""
    reslayers3d = []
    for layer2d in reslayer2d:
        layer3d = Bottleneck3d(layer2d)
        reslayers3d.append(layer3d)
    return torch.nn.Sequential(*reslayers3d)


class I3ResNet(torch.nn.Module):
    """I3D ResNet for 3D medical imaging"""

    def __init__(
        self,
        resnet2d,
        frame_nb=16,
        class_nb=1000,
        conv_class=False,
        return_skips=False,
        ImageEmbedding=False,
    ):
        super(I3ResNet, self).__init__()
        self.return_skips = return_skips
        self.conv_class = conv_class
        self.ImageEmbedding = ImageEmbedding

        self.conv1 = inflate_conv(resnet2d.conv1, time_dim=3, time_padding=1, center=True)
        self.bn1 = inflate_batch_norm(resnet2d.bn1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = inflate_pool(resnet2d.maxpool, time_dim=3, time_padding=1, time_stride=2)

        self.layer1 = inflate_reslayer(resnet2d.layer1)
        self.layer2 = inflate_reslayer(resnet2d.layer2)
        self.layer3 = inflate_reslayer(resnet2d.layer3)
        self.layer4 = inflate_reslayer(resnet2d.layer4)

        if conv_class:
            self.avgpool = inflate_pool(resnet2d.avgpool, time_dim=1)
            self.classifier = torch.nn.Conv3d(
                in_channels=2048,
                out_channels=class_nb,
                kernel_size=(1, 1, 1),
                bias=True,
            )
            self.contrastive_head = torch.nn.Conv3d(
                in_channels=2048, out_channels=512, kernel_size=(1, 1, 1), bias=True
            )
        else:
            final_time_dim = int(math.ceil(frame_nb / 16))
            self.avgpool = inflate_pool(resnet2d.avgpool, time_dim=final_time_dim)
            self.fc = inflate_linear(resnet2d.fc, 1)

    def forward(self, x):
        skips = []
        x = x.permute(0, 1, 4, 2, 3)
        x = torch.cat((x, x, x), dim=1)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = self.maxpool(x)

        x = checkpoint.checkpoint(self.layer1, x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = checkpoint.checkpoint(self.layer2, x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = checkpoint.checkpoint(self.layer3, x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))
        x = checkpoint.checkpoint(self.layer4, x)
        if self.return_skips:
            skips.append(x.permute(0, 1, 3, 4, 2))

        if self.conv_class:
            logger.debug(f"I3ResNet - Before avgpool: x.shape={x.shape}")
            x_features = self.avgpool(x)
            logger.debug(f"I3ResNet - After avgpool: x_features.shape={x_features.shape}")

            # Check for empty features before operations
            if x_features.numel() == 0:
                logger.error(f"I3ResNet - Empty features after avgpool")
                raise ValueError("Empty features after avgpool")

            if self.ImageEmbedding:
                logger.debug(f"I3ResNet - ImageEmbedding mode: applying squeeze operations")
                logger.debug(
                    f"I3ResNet - Before squeeze operations: x_features.shape={x_features.shape}"
                )

                # Check dimensions before squeezing
                if x_features.shape[2] == 0 or x_features.shape[3] == 0 or x_features.shape[4] == 0:
                    logger.error(
                        f"I3ResNet - Cannot squeeze zero-sized dimensions: shape={x_features.shape}"
                    )
                    raise ValueError(
                        f"Cannot squeeze zero-sized dimensions: shape={x_features.shape}"
                    )

                # Squeeze the spatial dimensions (2, 3, 4) to get (batch_size, embedding_dim)
                # Instead of adding unsqueeze(0), return the proper batch format
                result = x_features.squeeze(2).squeeze(2).squeeze(2)  # Remove .unsqueeze(0)
                logger.debug(f"I3ResNet - After squeeze operations: result.shape={result.shape}")
                return result

            logger.debug(
                f"I3ResNet - Non-ImageEmbedding mode: processing classifier and contrastive head"
            )
            x_ehr = self.classifier(x_features)
            logger.debug(f"I3ResNet - After classifier: x_ehr.shape={x_ehr.shape}")

            if x_ehr.shape[3] == 0 or x_ehr.shape[2] == 0:
                logger.error(
                    f"I3ResNet - Cannot squeeze/mean zero-sized dimensions in x_ehr: shape={x_ehr.shape}"
                )
                raise ValueError(
                    f"Cannot process zero-sized dimensions in x_ehr: shape={x_ehr.shape}"
                )

            x_ehr = x_ehr.squeeze(3)
            x_ehr = x_ehr.squeeze(3)
            x_ehr = x_ehr.mean(2)
            logger.debug(f"I3ResNet - After x_ehr processing: x_ehr.shape={x_ehr.shape}")

            x_contrastive = self.contrastive_head(x_features)
            logger.debug(
                f"I3ResNet - After contrastive_head: x_contrastive.shape={x_contrastive.shape}"
            )

            if x_contrastive.shape[3] == 0 or x_contrastive.shape[2] == 0:
                logger.error(
                    f"I3ResNet - Cannot squeeze/mean zero-sized dimensions in x_contrastive: shape={x_contrastive.shape}"
                )
                raise ValueError(
                    f"Cannot process zero-sized dimensions in x_contrastive: shape={x_contrastive.shape}"
                )

            x_contrastive = x_contrastive.squeeze(3)
            x_contrastive = x_contrastive.squeeze(3)
            x_contrastive = x_contrastive.mean(2)
            logger.debug(
                f"I3ResNet - After x_contrastive processing: x_contrastive.shape={x_contrastive.shape}"
            )
            if self.return_skips:
                return x_contrastive, x_ehr, skips
            else:
                return x_contrastive, x_ehr
        else:
            x = self.avgpool(x)
            x_reshape = x.view(x.size(0), -1)
            x = self.fc(x_reshape)
        return x


class ImageEncoder(nn.Module):
    """Image encoder using I3D ResNet"""

    def __init__(self, ImageEmbedding: bool = False):
        super().__init__()
        self.ImageEmbedding = ImageEmbedding
        resnet = torchvision.models.resnet152(weights=None)
        self.i3_resnet = I3ResNet(
            copy.deepcopy(resnet),
            class_nb=1692,
            conv_class=True,
            ImageEmbedding=self.ImageEmbedding,
        )

    def forward(self, image):
        if self.ImageEmbedding:
            contrastive_features = self.i3_resnet(image)
            return contrastive_features
        else:
            contrastive_features, ehr_features = self.i3_resnet(image)
            return contrastive_features, ehr_features


class TextEncoder(nn.Module):
    """Text encoder using Clinical-Longformer"""

    def __init__(self):
        super().__init__()

        # Suppress expected Longformer warnings about pooler weights
        import transformers
        import logging

        transformers_logger = logging.getLogger("transformers.modeling_utils")
        original_level = transformers_logger.level
        transformers_logger.setLevel(logging.ERROR)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
            self.text_encoder = AutoModel.from_pretrained("yikuan8/Clinical-Longformer")
            self.text_encoder.gradient_checkpointing_enable()
            self.linear_layer = nn.Linear(768, 512)
        finally:
            # Restore original logging level
            transformers_logger.setLevel(original_level)

    def forward(self, text_labels):
        text_labels = [sanitize_report(text) for text in text_labels]
        inputs = self.tokenizer(
            text_labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(self.text_encoder.device) for k, v in inputs.items()}
        text_embeddings = self.text_encoder(**inputs).last_hidden_state[:, 0, :]
        text_embeddings = self.linear_layer(text_embeddings)
        return text_embeddings


class MerlinArchitecture(nn.Module):
    """Complete Merlin model architecture"""

    def __init__(self, init_logit_scale: float = 1.0, ImageEmbedding: bool = False):
        super().__init__()
        self.ImageEmbedding = ImageEmbedding
        self.encode_image = ImageEncoder(ImageEmbedding=self.ImageEmbedding)
        self.encode_text = TextEncoder() if not self.ImageEmbedding else None
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

    def forward(self, image, text=None):
        # logger.info(f"Images shape: {image.shape}, Images max: {image.max()}, Images min: {image.min()}, Images mean: {image.mean()}, Images std: {image.std()}, Images dtype: {image.dtype}, Images device: {image.device} (start of forward)")

        # def summarize_tensor(tensor, label):
        #     tensor_cpu = tensor.detach().float().cpu()
        #     flat = tensor_cpu.reshape(-1)
        #     total = flat.numel()
        #     first_vals = flat[:10].tolist()
        #     mid_start = max(0, total // 2 - 5)
        #     mid_vals = flat[mid_start:mid_start + 10].tolist()
        #     last_vals = flat[-10:].tolist()
        #     percentiles = np.percentile(flat.numpy(), [0, 25, 50, 75, 100]) if total > 0 else [0] * 5
        #     summary = {
        #         "shape": tuple(tensor.shape),
        #         "dtype": str(tensor.dtype),
        #         "min": float(tensor_cpu.min()) if total > 0 else None,
        #         "max": float(tensor_cpu.max()) if total > 0 else None,
        #         "mean": float(tensor_cpu.mean()) if total > 0 else None,
        #         "std": float(tensor_cpu.std(unbiased=False)) if total > 1 else None,
        #         "percentiles": {
        #             "p0": float(percentiles[0]),
        #             "p25": float(percentiles[1]),
        #             "p50": float(percentiles[2]),
        #             "p75": float(percentiles[3]),
        #             "p100": float(percentiles[4]),
        #         },
        #         "first10": [float(x) for x in first_vals],
        #         "mid10": [float(x) for x in mid_vals],
        #         "last10": [float(x) for x in last_vals],
        #     }
        #     logger.info(f"[RATE] {label}:", summary)

        # summarize_tensor(image, "Input image stats")
        if self.ImageEmbedding and text is None:
            image_features = self.encode_image(image)
            # summarize_tensor(image_features, "Output image features stats")
            return image_features
        elif self.ImageEmbedding and text is not None:
            raise ValueError("Text input not required for image embedding")
        elif text is None:
            raise ValueError("Text input required for Image and Text embedding")

        image_features, ehr_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # summarize_tensor(image_features, "Output image features stats")
        # summarize_tensor(ehr_features, "Output ehr features stats")
        # summarize_tensor(text_features, "Output text features stats")

        if len(image_features.shape) == 1:
            image_features = image_features.unsqueeze(0)
        if len(text_features.shape) == 1:
            text_features = text_features.unsqueeze(0)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return (
            image_features,
            ehr_features,
            text_features,
        )


class Merlin:
    """Merlin multimodal medical image analysis."""

    def __init__(self, config: dict):
        self.config = config

        # Unified config structure - CLI overrides are already in config.model
        self.model_config = config.model
        self.device = setup_device(get_config_value(config, "device"))

        self.model_repo_id = get_config_value(self.model_config, "repo_id")
        self.model_revision = get_config_value(self.model_config, "revision")
        self.checkpoint_name = get_config_value(self.model_config, "checkpoint_name")
        self.image_embedding_mode = get_config_value(self.model_config, "image_embedding_mode")

        # Setup model
        self.setup_model()

        logger.info(
            f"Initialized Merlin {self.model_repo_id}@{self.model_revision} on device: {self.device}"
        )

    def setup_model(self) -> None:
        """Initialize the model architecture and load pretrained weights."""
        # Create download lock for this model
        download_lock = ModelDownloadLock(
            model_repo_id=self.model_repo_id, revision=self.model_revision
        )

        # Use download lock to prevent concurrent downloads
        with download_lock.acquire_download_lock(timeout=600):  # 10 minute timeout
            logger.info(f"Loading Merlin model {self.model_repo_id}@{self.model_revision}")

            # Download checkpoint
            checkpoint_path = self._download_checkpoint()

            # Initialize model architecture
            self.model = MerlinArchitecture(ImageEmbedding=self.image_embedding_mode)

            # Load pretrained weights
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("encode_text.")}
            load_strict = not self.image_embedding_mode
            missing, unexpected = self.model.load_state_dict(state_dict, strict=load_strict)
            if missing:
                logger.warning("Merlin: Missing keys during checkpoint load: %s", missing)
            if unexpected:
                logger.warning("Merlin: Unexpected keys during checkpoint load: %s", unexpected)

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded successfully on device: {self.device}")

    def _download_checkpoint(self):
        """Download the Merlin weights from the Hugging Face Hub to cache"""
        return hf_hub_download(
            repo_id=self.model_repo_id, filename=self.checkpoint_name, revision=self.model_revision
        )

    @staticmethod
    def preprocess_single(image, model_config, metadata=None, modality=None):
        """
        Prepare a single volume for Merlin.

        The preferred path assumes the dataset already applied the MONAI
        preprocessing pipeline from ``merlin_standalone.py`` and therefore
        only needs lightweight tensor hygiene here. A legacy path is kept for
        cached tensors that still rely on metadata-driven resampling.
        """

        if isinstance(image, dict) and "image" in image:
            image = image["image"]

        # logger.info(f"Image shape: {image.shape}, Image max: {image.max()}, Image min: {image.min()}, Image mean: {image.float().mean()}, Image std: {image.float().std()}, Image device: {image.device} (start of preprocess_single)")

        if metadata and isinstance(metadata, dict) and "processing_metadata" in metadata:
            # Legacy preprocessing path for cached tensors that provide spacing via metadata
            if image.dim() == 3:
                image = image.unsqueeze(0)

            image = image.permute(0, 3, 2, 1)
            image = torch.flip(image, dims=[1])

            transform_intensity = ScaleIntensityRange(
                a_min=-1000,
                a_max=1000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
            image = transform_intensity(image)

            spacing = metadata["processing_metadata"].get("target_spacing")
            if spacing is None:
                raise ValueError("Legacy preprocessing requires 'target_spacing' in metadata")

            affine = np.eye(4)
            affine[0, 0] = spacing[0]
            affine[1, 1] = spacing[1]
            affine[2, 2] = spacing[2]

            data_dict = {"image": image, "image_meta_dict": {"affine": affine}}
            transform_spacing = Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear"))
            data_dict = transform_spacing(data_dict)
            image = data_dict["image"]

            target_size = [224, 224, 160]
            if image.shape[1] < 224 or image.shape[2] < 224 or image.shape[3] < 160:
                pad = SpatialPad(spatial_size=target_size, mode="constant")
                image = pad(image)

            if image.shape[1] > 224 or image.shape[2] > 224 or image.shape[3] > 160:
                crop = CenterSpatialCrop(roi_size=target_size)
                image = crop(image)

            # logger.info(f"Image shape: {image.shape}, Image max: {image.max()}, Image min: {image.min()}, Image mean: {image.mean()}, Image std: {image.std()}, Image dtype: {image.dtype}, Image device: {image.device} (end of preprocess_single)")

            return image.float()

        # Preferred path: dataset already returned a MONAI-processed tensor in (C, H, W, D)
        if torch.is_tensor(image) is False:
            image = torch.as_tensor(image)

        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Ensure channel-first ordering and float dtype
        if image.dim() == 4 and image.shape[0] != 1:
            # Unexpected extra channels: keep original ordering but warn for diagnostics
            logger.warning(
                "Merlin.preprocess_single received image with %d channels; expected 1",
                image.shape[0],
            )

        image = image.float().contiguous()

        # logger.info(f"Image shape: {image.shape}, Image max: {image.max()}, Image min: {image.min()}, Image mean: {image.mean()}, Image std: {image.std()}, Image dtype: {image.dtype}, Image device: {image.device} (end of preprocess_single)")

        return image

    def preprocess(self, volumes: torch.Tensor, modality: str) -> torch.Tensor:
        """
        Preprocess input volumes before extraction.
        Note: Intensity scaling is now handled in preprocess_single via MONAI transforms.
        """
        # No additional preprocessing needed here since preprocess_single handles it
        return volumes

    @torch.no_grad()
    def extract_features(
        self, inputs: torch.Tensor, modality="abdomen_ct", text=None
    ) -> np.ndarray:
        """
        Extract features from input images/volumes.

        Args:
            inputs: Input tensor of shape (B, C, D, H, W) for 3D
            modality: Modality type
            text: Optional text input for multimodal mode

        Returns:
            Feature embeddings as numpy array
        """

        logger.debug(
            f"Merlin.extract_features - Input shape: {inputs.shape}, dtype: {inputs.dtype}, device: {inputs.device}"
        )

        # Check for empty input
        if inputs.numel() == 0:
            logger.error(f"Merlin.extract_features - Empty input tensor received")
            raise ValueError("Empty input tensor")

        inputs = inputs.to(self.device)
        logger.debug(
            f"Merlin.extract_features - After moving to device {self.device}: shape={inputs.shape}"
        )

        inputs = self.preprocess(inputs, modality)
        logger.debug(f"Merlin.extract_features - After preprocessing: shape={inputs.shape}")

        # Check if preprocessing resulted in empty tensor
        if inputs.numel() == 0:
            logger.error(f"Merlin.extract_features - Empty tensor after preprocessing")
            raise ValueError("Empty tensor after preprocessing")

        if self.image_embedding_mode:
            # Image-only mode
            logger.debug(f"Merlin.extract_features - Running in image-only mode")
            with torch.no_grad():
                logger.debug(f"Merlin.extract_features - Calling model forward pass")
                # torch.save(inputs, "inputs_merlin.pt")
                features = self.model(inputs)
                logger.debug(
                    f"Merlin.extract_features - Model output shape: {features.shape}, dtype: {features.dtype}"
                )

                # Check for empty output
                if features.numel() == 0:
                    logger.error(f"Merlin.extract_features - Model returned empty tensor")
                    raise ValueError("Model returned empty tensor")

                features = features.cpu().numpy().astype(np.float32)
                logger.debug(
                    f"Merlin.extract_features - Final features shape: {features.shape}, dtype: {features.dtype}"
                )
        else:
            # Multimodal mode - requires text input
            if text is None:
                raise ValueError("Text input required for multimodal mode")

            logger.debug(f"Merlin.extract_features - Running in multimodal mode")
            with torch.no_grad():
                image_features, ehr_features, text_features = self.model(inputs, text)
                logger.debug(
                    f"Merlin.extract_features - Image features shape: {image_features.shape}"
                )

                # Check for empty output
                if image_features.numel() == 0:
                    logger.error(f"Merlin.extract_features - Model returned empty image features")
                    raise ValueError("Model returned empty image features")

                # Return image features for feature extraction
                features = image_features.cpu().numpy().astype(np.float32)
                logger.debug(
                    f"Merlin.extract_features - Final features shape: {features.shape}, dtype: {features.dtype}"
                )
                logger.debug(
                    f"Merlin.extract_features - First 10 values: {features.flatten()[:10]}"
                )

        return features

    def eval(self):
        """Set model to evaluation mode."""
        if hasattr(self, "model"):
            self.model.eval()
        return self
