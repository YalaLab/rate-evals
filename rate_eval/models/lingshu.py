"""Lingshu model using the new architecture."""

import torch
import numpy as np
import math
import os
from einops import rearrange
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    Qwen2VLImageProcessorFast,
)
import torchvision.transforms as transforms
import torch.nn.functional as F

from ..common import get_logger, setup_device, ModelError, ModelDownloadLock
from ..config import load_model_config, get_config_value, merge_configs
from .common import batch_apply_ct_windowing, batch_apply_normalization

logger = get_logger(__name__)

image_processor_config = {
    "crop_size": None,
    "data_format": "channels_first",
    "default_to_square": True,
    "device": None,
    "disable_grouping": None,
    "do_center_crop": None,
    "do_convert_rgb": True,
    "do_normalize": True,  # we need this as it will repeat the dimensions in the image processor
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    # "image_mean": [0., 0., 0.],
    "image_processor_type": "Qwen2VLImageProcessorFast",
    "image_std": [0.26862954, 0.26130258, 0.27577711],
    # "image_std": [1., 1., 1.],
    "input_data_format": None,
    "max_pixels": 12845056,
    "merge_size": 2,
    "min_pixels": 3136,
    "patch_size": 14,
    "processor_class": "Qwen2_5_VLProcessor",
    "resample": 3,
    # "rescale_factor": 0.00392156862745098,
    "rescale_factor": 1.0,  # disable rescale
    "return_tensors": None,
    "size": {"longest_edge": 12845056, "shortest_edge": 3136},
    "temporal_patch_size": 2,
}


class Lingshu:
    """
    Lingshu model for medical image analysis.

    This model processes both 2D images and 3D volumes. For 3D volumes, it treats
    each slice as a separate image and aggregates features across the depth dimension.
    """

    def __init__(self, config: dict):
        self.config = config

        # Load model-specific config and merge with CLI overrides
        base_model_config = load_model_config("lingshu")
        self.model_config = merge_configs(base_model_config, config)
        self.device = setup_device(get_config_value(config, "device"))

        # Setup model
        self.setup_model()

        logger.info(f"Initialized Lingshu on device: {self.device}")

    def setup_model(self) -> None:
        """Initialize the model architecture and load pretrained weights."""
        model_name = get_config_value(self.model_config, "model_name")
        logger.info(f"Loading Lingshu model: {model_name}")

        # Get dtype from config, default to bfloat16 with warning
        try:
            dtype = get_config_value(self.model_config, "dtype")
        except ValueError:
            dtype = None
        if dtype is None:
            dtype = torch.bfloat16
            logger.warning(
                "No dtype specified in config, defaulting to bfloat16. "
                "To suppress this warning, add 'dtype: bfloat16' to your model config."
            )
        else:
            # Convert string dtype to torch dtype if needed
            if isinstance(dtype, str):
                if dtype == "float16":
                    dtype = torch.float16
                elif dtype == "float32":
                    dtype = torch.float32
                elif dtype == "bfloat16":
                    dtype = torch.bfloat16
                else:
                    logger.warning(f"Unknown dtype '{dtype}', defaulting to bfloat16")
                    dtype = torch.bfloat16

        # Create download lock for this model
        download_lock = ModelDownloadLock(model_repo_id=model_name, revision="main")

        # Use download lock to prevent concurrent downloads
        with download_lock.acquire_download_lock(timeout=600):  # 10 minute timeout
            logger.info(f"Loading Lingshu model {model_name}")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=dtype, attn_implementation="flash_attention_2"
            )

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded successfully with dtype: {dtype}")

    @staticmethod
    def preprocess_single(image, model_config, metadata=None, modality=None):
        """
        Preprocess a single image for Lingshu (for use in dataset __getitem__).

        Args:
            image: Tensor from dataset (normalized to [0,1] range)
            model_config: Model configuration dictionary

        Returns:
            Preprocessed tensor ready for Lingshu (resized using processor)
        """
        try:
            target_size = get_config_value(model_config, "target_slice_h")
        except ValueError:
            target_size = get_config_value(model_config, "preprocessing.target_slice_h")

        if image.shape[-2] != target_size or image.shape[-1] != target_size:
            # Handle different input shapes - resize first
            if len(image.shape) == 4:  # (C, D, H, W) - 3D volume
                C, D, H, W = image.shape
                # Resize each slice
                resized_slices = []
                for d in range(D):
                    slice_2d = image[:, d, :, :]  # (C, H, W)
                    slice_resized = transforms.functional.resize(
                        slice_2d,
                        (target_size, target_size),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    )
                    resized_slices.append(slice_resized)
                image = torch.stack(resized_slices, dim=1)  # (C, D, H, W)
            elif len(image.shape) == 3:  # (C, H, W) - 2D image
                image = transforms.functional.resize(
                    image,
                    (target_size, target_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )
            else:
                raise ValueError(
                    f"Expected 3D (C,H,W) or 4D (C,D,H,W) input, got shape {image.shape}"
                )

        # apply windowing if it's CT
        if modality == "chest_ct" or modality == "abdomen_ct":
            assert len(image.shape) == 4, f"Expected 4D input, got shape {image.shape}"
            ct_window_type = get_config_value(model_config, "ct_window_type")
            # The tensor in preprocess_single is not batched
            image = batch_apply_ct_windowing(
                image[None, ...],
                ct_window_type=ct_window_type,
                modality="CT",
                per_sample=False,
            )[0]

        image_processor = Qwen2VLImageProcessorFast(**image_processor_config)

        if len(image.shape) == 4:
            image_list = []
            image_grid_thw_list = []
            for i in range(image.shape[1]):
                processed_slice = image_processor(image[:, i, ...], return_tensors="pt")
                image_list.append(processed_slice["pixel_values"])
                image_grid_thw_list.append(processed_slice["image_grid_thw"])
            stacked_views = torch.stack(image_list, dim=0)
            stacked_views = stacked_views[None, ...]
            image_grid_thw = torch.stack(image_grid_thw_list, dim=0)

            extra_infos = {"image_grid_thw": image_grid_thw}
            return stacked_views, extra_infos
        else:
            inputs = image_processor(image, return_tensors="pt")
            image, image_grid_thw = inputs["pixel_values"], inputs["image_grid_thw"]

            assert not torch.isnan(image).any(), f"NaN detected in image: {image.shape}"
            return {"image": image, "image_grid_thw": image_grid_thw}

    def preprocess(self, volumes: torch.Tensor, modality: str) -> torch.Tensor:
        """
        Preprocess input volumes before extraction.

        Args:
            volumes: Input tensor of shape (B, 1, D, num_tokens, feature_dim)
            modality: Modality string for applying appropriate preprocessing

        Returns:
            Preprocessed tensor (windowed but not normalized - normalization is separate)
        """
        # # Apply CT windowing if modality is chest_ct
        # if modality == "chest_ct" or modality == "abdomen_ct":
        #     # The model has merged config with CLI overrides
        #     ct_window_type = get_config_value(self.model_config, 'ct_window_type')
        #     assert ct_window_type is not None, "CT window type is not set"
        #     assert volumes.dim() == 5, f"Volumes should be 5D, got {volumes.dim()}"

        #     # Get normalization params from merged config
        #     normalize_mean = get_config_value(self.model_config, 'ct_normalize_mean')
        #     normalize_std = get_config_value(self.model_config, 'ct_normalize_std')

        #     # We might need to do windowing in preprocess_single, since normalization is done in preprocess_single.
        #     # volumes = batch_apply_ct_windowing(
        #     #     volumes,
        #     #     ct_window_type=ct_window_type,
        #     #     modality='CT',
        #     # )
        # else:
        #     assert "xray" in modality.lower(), f"Modality {modality} is not supported"
        #     normalize_mean = get_config_value(self.model_config, 'xray_normalize_mean')
        #     normalize_std = get_config_value(self.model_config, 'xray_normalize_std')

        # volumes = batch_apply_normalization(volumes, normalize_mean, normalize_std)

        return volumes

    def forward(self, images: torch.Tensor, extra_infos: dict = None) -> np.ndarray:
        """
        Forward pass through the model.

        Args:
            images: Input images tensor with shape (B, C, H, W) for 2D or (B, C, D, H, W) for 3D

        Returns:
            Model embeddings as numpy array with shape (B, feature_dim)
        """
        images = images.to(self.device)

        # Handle both 2D and 3D inputs (same as MedGemma)
        if len(images.shape) == 4:  # 2D images (B, C, H, W)
            N, C, H_in, W_in = images.shape
            D_in = 1
            # Add depth dimension: (B, C, H, W) -> (B, C, 1, H, W)
            images = images.unsqueeze(2)
        elif len(images.shape) == 5:  # 3D volumes (B, C, D, H, W)
            N, C, D_in, H_in, W_in = images.shape
        else:
            raise ValueError(f"Expected 4D or 5D input tensor, got shape {images.shape}")

        images_for_processing = images.permute(0, 2, 1, 3, 4)  # NDCHW
        inputs = rearrange(images_for_processing, "n d c h w -> (n d) c h w")

        # Remove the extra dimension

        inputs = inputs.squeeze(1)
        image_grid_thw = extra_infos["image_grid_thw"].flatten(0, 2).cuda()

        # Get slice_batch_size config
        slice_batch_size = get_config_value(self.model_config, "slice_batch_size")

        # Extract features with optional micro-batching
        total_slices = inputs.shape[0]
        if slice_batch_size is None or total_slices <= slice_batch_size:
            # Process all slices at once (original behavior)
            model_embed = torch.stack(
                self.model.get_image_features(inputs, image_grid_thw=image_grid_thw), dim=0
            )
        else:
            # Micro-batch processing to avoid OOM
            embed_list = []
            for start_idx in range(0, total_slices, slice_batch_size):
                end_idx = min(start_idx + slice_batch_size, total_slices)
                batch_inputs = inputs[start_idx:end_idx]
                batch_grid = image_grid_thw[start_idx:end_idx]
                batch_embed = torch.stack(
                    self.model.get_image_features(batch_inputs, image_grid_thw=batch_grid), dim=0
                )
                embed_list.append(batch_embed)
            model_embed = torch.cat(embed_list, dim=0)

        # Rearrange for depth pooling (same as MedGemma)
        rearranged_for_pool = rearrange(model_embed, "(n d) k c -> n c (d k)", d=D_in)

        # Apply depth pooling operation (same as MedGemma)
        pool_op = get_config_value(self.model_config, "pool_op")
        if pool_op == "max":
            model_embed = rearranged_for_pool.max(-1).values.float().cpu().numpy()
        elif pool_op == "mean":
            model_embed = rearranged_for_pool.mean(-1).float().cpu().numpy()
        elif pool_op == "median":
            model_embed = rearranged_for_pool.median(-1).values.float().cpu().numpy()
        elif pool_op == "middle":
            # Select the middle frame only
            # rearranged_for_pool shape: (n, c, d*k) where d is depth and k is features per slice
            # We need to select the middle depth slice from the d*k dimension
            total_features = rearranged_for_pool.shape[-1]  # d*k
            features_per_slice = total_features // D_in  # k
            middle_idx = D_in // 2
            start_idx = middle_idx * features_per_slice
            end_idx = (middle_idx + 1) * features_per_slice
            model_embed = rearranged_for_pool[:, :, start_idx:end_idx].float().cpu().numpy()
        else:
            raise ValueError(f"Unsupported pooling operation: {pool_op}")

        return model_embed

    @torch.no_grad()
    def extract_features(
        self, inputs: torch.Tensor, modality="chest_ct", extra_infos: dict = None
    ) -> np.ndarray:
        """
        Extract features from input volumes.

        Args:
            inputs: Input tensor of shape (B, C, D, H, W) or (B, C, H, W)
            modality: Modality string for applying appropriate preprocessing

        Returns:
            Feature embeddings as numpy array
        """
        preprocessed_inputs = self.preprocess(inputs, modality)
        return self.forward(preprocessed_inputs, extra_infos=extra_infos)
