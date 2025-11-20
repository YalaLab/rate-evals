"""Pillar0 multimodal medical image model implementation."""

import torch
import numpy as np
from einops import rearrange

from ..common import get_logger, setup_device, ModelError, ModelDownloadLock
from ..config import load_model_config, get_config_value, merge_configs
from .common import batch_apply_ct_windowing, batch_apply_mr_windowing, batch_apply_normalization
from transformers import AutoModel


logger = get_logger(__name__)


class Pillar0:
    """Pillar0 multimodal medical image analysis."""

    def __init__(self, config: dict):
        self.config = config

        # Unified config structure - CLI overrides are already in config.model
        self.model_config = config.model
        self.device = setup_device(get_config_value(config, "device"))

        # Get repo_id and revision from unified config
        self.model_repo_id = get_config_value(self.model_config, "repo_id")
        self.model_revision = get_config_value(self.model_config, "revision")

        # Setup model
        self.setup_model()

        logger.info(
            "Initialized Pillar0 %s@%s on device: %s",
            self.model_repo_id,
            self.model_revision,
            self.device,
        )

    def setup_model(self) -> None:
        """Initialize the model architecture and load pretrained weights."""
        # Create download lock for this model
        download_lock = ModelDownloadLock(
            model_repo_id=self.model_repo_id, revision=self.model_revision
        )

        # Use download lock to prevent concurrent downloads
        with download_lock.acquire_download_lock(timeout=600):  # 10 minute timeout
            logger.info(
                "Loading Pillar0 model %s@%s",
                self.model_repo_id,
                self.model_revision,
            )
            self.model = AutoModel.from_pretrained(
                self.model_repo_id, revision=self.model_revision, trust_remote_code=True
            )

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded successfully on device: {self.device}")

    @staticmethod
    def preprocess_single(image, model_config, metadata=None, modality=None):
        """
        Preprocess a single exam for Pillar0 (for use in dataset __getitem__).

        Args:
            image: Tensor from dataset (normalized to [0,1] range)
            model_config: Model configuration dictionary

        Returns:
            Preprocessed tensor ready for Pillar0 (normalized for X-rays)
        """
        # Note: No normalization applied here as it will be handled differently based on modality:
        # - CT data: Gets windowing + normalization applied in extract.py based on ct_normalize_mean/std (model-agnostic)
        # - X-ray data: Gets normalization applied in extract_features based on xray_normalize_mean/std (model-specific)

        return image

    def preprocess(self, volumes: torch.Tensor, modality: str) -> torch.Tensor:
        """
        Preprocess input volumes before extraction.
        """
        apply_normalization = get_config_value(self.model_config, "apply_normalization")
        # Apply CT windowing if modality is chest_ct
        if modality in ["chest_ct", "abdomen_ct", "brain_ct"]:
            # The model has merged config with CLI overrides
            ct_window_type = get_config_value(self.model_config, "ct_window_type")
            assert ct_window_type is not None, "CT window type is not set"
            assert volumes.dim() == 5, f"Volumes should be 5D, got {volumes.dim()}"

            # Get normalization params from merged config
            normalize_mean = get_config_value(self.model_config, "ct_normalize_mean")
            normalize_std = get_config_value(self.model_config, "ct_normalize_std")

            per_sample = get_config_value(self.model_config, "per_sample_windowing")
            volumes = batch_apply_ct_windowing(
                volumes,
                ct_window_type=ct_window_type,
                modality="CT",
                per_sample=per_sample,
            )
        elif modality in ["breast_mr"]:
            mr_window_type = get_config_value(self.model_config, "mr_window_type")
            assert mr_window_type is not None, "MR window type is not set"
            assert volumes.dim() == 5, f"Volumes should be 5D, got {volumes.dim()}"
            per_sample = get_config_value(self.model_config, "per_sample_windowing")
            volumes = batch_apply_mr_windowing(
                volumes, mr_window_type, modality="MR", per_sample=per_sample
            )
            normalize_mean = get_config_value(self.model_config, "mr_normalize_mean")
            normalize_std = get_config_value(self.model_config, "mr_normalize_std")
        else:
            assert "xray" in modality.lower(), f"Modality {modality} is not supported"
            normalize_mean = get_config_value(self.model_config, "xray_normalize_mean")
            normalize_std = get_config_value(self.model_config, "xray_normalize_std")

        if apply_normalization:
            volumes = batch_apply_normalization(volumes, normalize_mean, normalize_std)
        return volumes

    @torch.no_grad()
    def extract_features(self, inputs: torch.Tensor, modality="chest_ct") -> np.ndarray:
        """
        Extract features from input images/volumes.

        Args:
            inputs: Input tensor of shape (B, C, H, W) for 2D or (B, C, D, H, W) for 3D

        Returns:
            Feature embeddings as numpy array of shape (B, feature_dim)
        """
        inputs = inputs.to(self.device)
        inputs = self.preprocess(inputs, modality)

        # Handle different inference patterns based on modality
        if modality == "chest_xray_two_view":
            # Original Pillar0 two-view inference: process two views together
            logger.debug(
                "Using Pillar0 two-view inference for modality: %s",
                modality,
            )

            # Process two views together (original Pillar0 approach)
            inputs_as_dict = {modality: inputs}

            with torch.no_grad():
                features = self.model.extract_vision_feats(inputs_as_dict)
                features = features.cpu().numpy().astype(np.float32)

        elif modality == "chest_xray_single_view":
            # MedGemma-style single-view inference: reshape and process each view separately
            logger.debug(f"Using MedGemma-style single-view inference for modality: {modality}")

            # Handle different input shapes
            if len(inputs.shape) == 4:  # (B, C, H, W) - Single view
                N, C, H, W = inputs.shape
                D_in = 1
                # Add depth dimension: (B, C, H, W) -> (B, C, 1, H, W)
                inputs = inputs.unsqueeze(2)
            elif len(inputs.shape) == 5:  # (B, C, D, H, W) - Two view or 3D volume
                N, C, D_in, H, W = inputs.shape
            else:
                raise ValueError(f"Expected 4D or 5D input tensor, got shape {inputs.shape}")

            # MedGemma-style processing: reshape and process each view separately
            # Rearrange to process each view: (B, C, D, H, W) -> (B*D, C, H, W)
            inputs_rearranged = rearrange(inputs, "n c d h w -> (n d) c 1 h w")

            # Process through Pillar0 model
            inputs_as_dict = {modality: inputs_rearranged}

            with torch.no_grad():
                # Extract features for each view
                features = self.model.extract_vision_feats(inputs_as_dict)

                # Rearrange back for pooling: (B*D, feature_dim) -> (B, D, feature_dim)
                features_rearranged = rearrange(features, "(n d) f -> n f d", d=D_in)

                # Apply pooling operation across views (MedGemma style)
                pool_op = get_config_value(self.model_config, "pool_op")
                if pool_op == "max":
                    features = features_rearranged.max(-1).values.cpu().numpy()
                elif pool_op == "mean":
                    features = features_rearranged.mean(-1).cpu().numpy()
                elif pool_op == "median":
                    features = features_rearranged.median(-1).values.cpu().numpy()
                elif pool_op == "middle":
                    # Select the middle frame only
                    middle_idx = D_in // 2
                    features = features_rearranged[:, :, middle_idx].cpu().numpy()
                else:
                    raise ValueError(f"Unsupported pooling operation: {pool_op}")

                # Convert to numpy for multiprocessing compatibility
                features = features.astype(np.float32)

        else:
            # Default behavior for other modalities
            logger.debug(f"Using default inference for modality: {modality}")
            inputs_as_dict = {modality: inputs}

            with torch.no_grad():
                features = self.model.extract_vision_feats(inputs_as_dict)
                features = features.cpu().numpy().astype(np.float32)

        return features

    def eval(self):
        """Set model to evaluation mode."""
        if hasattr(self, "model"):
            self.model.eval()
        return self
