"""Refactored MedGemma model using the new architecture."""

import torch
import numpy as np
from einops import rearrange
from transformers import AutoProcessor, AutoModelForImageTextToText
import torchvision.transforms as transforms
import time

from ..common import get_logger, setup_device, ModelError, ModelDownloadLock
from ..config import (
    load_model_config,
    get_config_value,
    merge_configs,
)
from .common import batch_apply_ct_windowing, batch_apply_normalization

logger = get_logger(__name__)


class MedGemma:
    """
    MedGemma model for medical image analysis.

    This model processes both 2D images and 3D volumes. For 3D volumes, it treats
    each slice as a separate image and aggregates features across the depth dimension.
    """

    def __init__(self, config: dict):
        self.config = config

        # Load model-specific config and merge with CLI overrides
        base_model_config = load_model_config("medgemma")
        self.model_config = merge_configs(base_model_config, config)
        self.device = setup_device(get_config_value(config, "device"))

        # Setup model
        self.setup_model()

        logger.info(f"Initialized MedGemma on device: {self.device}")

    def setup_model(self) -> None:
        """Initialize the model architecture and load pretrained weights."""
        model_name = get_config_value(self.model_config, "repo_id")

        logger.info(f"Loading MedGemma model {model_name}")

        # Get dtype from config, default to float16 with warning
        try:
            dtype = get_config_value(self.model_config, "dtype")
        except ValueError:
            dtype = None
        if dtype is None:
            dtype = torch.float16
            logger.warning(
                "No dtype specified in config, defaulting to float16. "
                "To suppress this warning, add 'dtype: float16' to your model config."
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
                    logger.warning(f"Unknown dtype '{dtype}', defaulting to float16")
                    dtype = torch.float16

        try:
            # Create download lock for this model
            download_lock = ModelDownloadLock(
                model_repo_id=model_name, revision="main"  # MedGemma typically uses main branch
            )

            # Use download lock to prevent concurrent downloads
            with download_lock.acquire_download_lock(timeout=600):  # 10 minute timeout
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                )

            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded successfully on device: {self.device} with dtype: {dtype}")

        except Exception as e:
            logger.error(f"Failed to load MedGemma model: {e}")
            logger.error("Make sure you are authenticated with HuggingFace for gated models")
            logger.error("Run: huggingface-cli login")
            raise ModelError(f"Could not load MedGemma model: {e}")

    @staticmethod
    def preprocess_single(image, model_config, metadata=None, modality=None):
        """
        Preprocess a single image for MedGemma (for use in dataset __getitem__).

        Args:
            image: Tensor from dataset (normalized to [0,1] range)
            model_config: Model configuration dictionary

        Returns:
            Preprocessed tensor ready for MedGemma (normalized to [-1,1] range for X-rays)
        """
        # MedGemma vision tower expects 896x896 images
        target_size = get_config_value(model_config, "preprocessing.target_slice_h")

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
            raise ValueError(f"Expected 3D (C,H,W) or 4D (C,D,H,W) input, got shape {image.shape}")

        # Note: No normalization applied here as it will be handled differently based on modality:
        # - CT data: Gets windowing + normalization applied in extract.py based on ct_normalize_mean/std
        # - X-ray data: Gets normalization applied in extract_features based on xray_normalize_mean/std

        assert not torch.isnan(image).any(), f"NaN detected in image: {image.shape}"
        return image

    def preprocess(self, volumes: torch.Tensor, modality: str) -> torch.Tensor:
        """
        Preprocess input volumes before extraction.
        """
        # Apply CT windowing if modality is chest_ct
        if modality in ["chest_ct", "abdomen_ct", "brain_ct", "breast_mr"]:
            # The model has merged config with CLI overrides
            ct_window_type = get_config_value(
                self.model_config,
                "preprocessing.ct.window_type",
            )
            assert ct_window_type is not None, "CT window type is not set"
            assert volumes.dim() == 5, f"Volumes should be 5D, got {volumes.dim()}"

            # Get normalization params from merged config
            normalize_mean = get_config_value(
                self.model_config,
                "preprocessing.ct.normalize_mean",
            )

            normalize_std = get_config_value(
                self.model_config,
                "preprocessing.ct.normalize_std",
            )

            per_sample = get_config_value(self.model_config, "per_sample_windowing")
            volumes = batch_apply_ct_windowing(
                volumes,
                ct_window_type=ct_window_type,
                modality="CT",
                per_sample=per_sample,
            )
        else:
            assert "xray" in modality.lower(), f"Modality {modality} is not supported"
            normalize_mean = get_config_value(
                self.model_config,
                "preprocessing.xray.normalize_mean",
            )

            normalize_std = get_config_value(
                self.model_config,
                "preprocessing.xray.normalize_std",
            )

        volumes = batch_apply_normalization(volumes, normalize_mean, normalize_std)
        return volumes

    def forward(self, images: torch.Tensor) -> np.ndarray:
        """
        Forward pass through the model.

        Args:
            images: Input images tensor with shape (B, C, H, W) for 2D or (B, C, D, H, W) for 3D

        Returns:
            Model embeddings as numpy array with shape (B, feature_dim)
        """
        t_start = time.time()

        images = images.to(self.device)
        t_gpu_transfer = time.time()
        print(f"[MEDGEMMA] GPU transfer: {t_gpu_transfer - t_start:.3f}s", flush=True)

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

        t_shape_handling = time.time()
        print(f"[MEDGEMMA] Shape handling: {t_shape_handling - t_gpu_transfer:.3f}s", flush=True)

        # Convert to 3-channel format expected by vision transformer
        # Handle different channel counts
        if C == 1:  # Grayscale - repeat to 3 channels
            images_3channel = images.repeat(1, 3, 1, 1, 1)
        elif C == 3:  # Already RGB
            images_3channel = images
        else:
            raise ValueError(f"Expected 1 or 3 channels, got {C}")

        images_for_vit = images_3channel.permute(0, 2, 1, 3, 4)  # NDCHW
        images_rearranged = rearrange(images_for_vit, "n d c h w -> (n d) c h w")

        t_rearrange = time.time()
        print(
            f"[MEDGEMMA] Channel conversion & rearrange: {t_rearrange - t_shape_handling:.3f}s",
            flush=True,
        )
        print(
            f"[MEDGEMMA] Input shape to vision tower: {images_rearranged.shape}, dtype: {images_rearranged.dtype}",
            flush=True,
        )

        # Get slice_batch_size config
        try:
            slice_batch_size = get_config_value(self.model_config, "extraction.slice_batch_size")
        except ValueError:
            slice_batch_size = None

        # Extract features using vision tower with optional micro-batching
        torch.cuda.synchronize()
        t_before_vit = time.time()

        total_slices = images_rearranged.shape[0]
        if slice_batch_size is None or total_slices <= slice_batch_size:
            # Process all slices at once (original behavior)
            model_embed = self.model.vision_tower(images_rearranged).last_hidden_state
        else:
            # Micro-batch processing to avoid OOM
            embed_list = []
            for start_idx in range(0, total_slices, slice_batch_size):
                end_idx = min(start_idx + slice_batch_size, total_slices)
                batch_slices = images_rearranged[start_idx:end_idx]
                batch_embed = self.model.vision_tower(batch_slices).last_hidden_state
                embed_list.append(batch_embed)
            model_embed = torch.cat(embed_list, dim=0)

        torch.cuda.synchronize()
        t_after_vit = time.time()
        print(f"[MEDGEMMA] Vision tower inference: {t_after_vit - t_before_vit:.3f}s", flush=True)
        print(f"[MEDGEMMA] Vision tower output shape: {model_embed.shape}", flush=True)

        # Apply multi-modal projector if specified
        extractor = get_config_value(self.model_config, "architecture.extractor")
        if extractor == "multi_modal_projector":
            torch.cuda.synchronize()
            t_before_proj = time.time()
            model_embed = self.model.multi_modal_projector(model_embed)
            torch.cuda.synchronize()
            t_after_proj = time.time()
            print(
                f"[MEDGEMMA] Multi-modal projector: {t_after_proj - t_before_proj:.3f}s", flush=True
            )
            print(f"[MEDGEMMA] Projector output shape: {model_embed.shape}", flush=True)

        # Rearrange for pooling across depth dimension
        t_before_pool_rearrange = time.time()
        rearranged_for_pool = rearrange(model_embed, "(n d) k c -> n c (d k)", d=D_in)
        t_after_pool_rearrange = time.time()
        print(
            f"[MEDGEMMA] Rearrange for pooling: {t_after_pool_rearrange - t_before_pool_rearrange:.3f}s",
            flush=True,
        )
        print(f"[MEDGEMMA] Shape for pooling: {rearranged_for_pool.shape}", flush=True)

        # Apply pooling operation
        pool_op = get_config_value(self.model_config, "extraction.pool_op")
        t_before_pool = time.time()
        # We need to convert to float as numpy doesn't support bfloat16
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

        t_after_pool = time.time()
        print(
            f"[MEDGEMMA] Pooling ({pool_op}) & CPU transfer: {t_after_pool - t_before_pool:.3f}s",
            flush=True,
        )
        print(f"[MEDGEMMA] Final output shape: {model_embed.shape}", flush=True)

        t_end = time.time()
        print(f"[MEDGEMMA] Total forward time: {t_end - t_start:.3f}s", flush=True)

        return model_embed

    @torch.no_grad()
    def extract_features(self, inputs: torch.Tensor, modality="chest_ct") -> np.ndarray:
        """
        Extract features from input volumes.

        Args:
            inputs: Input tensor of shape (B, C, D, H, W)
            modality: Modality string for applying appropriate preprocessing

        Returns:
            Feature embeddings as numpy array
        """
        t_start = time.time()
        inputs = self.preprocess(inputs, modality)
        t_preprocess = time.time()
        print(f"[MEDGEMMA] Preprocessing time: {t_preprocess - t_start:.3f}s", flush=True)

        result = self.forward(inputs)
        return result
