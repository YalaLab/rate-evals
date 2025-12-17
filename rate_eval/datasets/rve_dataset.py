"""Merlin Abdominal CT dataset using the new architecture."""

import os
import math
import joblib
import torch
import cv2
import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image
from io import StringIO
from typing import Any, Union, Optional, List
import rve
import json
from ..common import get_logger, DatasetError, load_metadata_from_directory
from ..config import get_config_value

logger = get_logger(__name__)


def load_cached_volume(
    filepath: str,
    transforms: callable,
    transforms_3d: callable,
    load_metadata: bool = False,
    target_slices: Optional[int] = None,
    target_h: Optional[int] = 256,
    target_w: Optional[int] = 256,
    pad_value: float = -1024.0,
) -> torch.Tensor:
    """
    Load a 3D volume from an RVE tar file with optional cropping/padding to target dimensions.

    Args:
        filepath: Path to the .tar file from RVE
        transforms: 2D transforms to apply to the volume
        transforms_3d: 3D transforms to apply to the volume
        use_opencv: Whether to use OpenCV to load the volume
        image_mode: Mode to use for the image
        target_slices: Target number of slices (D dimension). If provided, will pad or crop.
        target_h: Target height (H dimension). If provided, will pad or crop.
        target_w: Target width (W dimension). If provided, will pad or crop.
        pad_value: Value to use for padding (default: -1024 for CT)

    Returns:
        Volume tensor with shape (1, D, H, W)
    """
    try:
        logger.debug(f"load_cached_volume - Loading volume from: {filepath}")

        # This is an optimization fix to avoid loading libraries multiple times.
        if "LD_LIBRARY_PATH" in os.environ:
            LD_LIBRARY_PATH = os.environ["LD_LIBRARY_PATH"]
            del os.environ["LD_LIBRARY_PATH"]
        else:
            LD_LIBRARY_PATH = None
        # Load metadata first if requested (before RVE to avoid LD_LIBRARY_PATH issues)
        metadata = None
        if load_metadata:
            metadata = load_metadata_from_directory(filepath)

        # load volume from rve
        # start_time = time.time()
        volume = rve.load_sample(filepath, use_hardware_acceleration=False)
        # end_time = time.time()
        # load_time = end_time - start_time
        # logger.debug(f"RVE load_sample took {load_time:.4f} seconds")
        if LD_LIBRARY_PATH is not None:
            os.environ["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH

        # Load cached tensor [deprecate in favor of rve]
        # volume = torch.load(filepath, map_location='cpu')

        logger.debug(
            f"load_cached_volume - Loaded volume shape: {volume.shape}, dtype: {volume.dtype}"
        )

        # Check for empty volume
        if volume.numel() == 0:
            logger.error(f"load_cached_volume - Loaded empty volume from {filepath}")
            raise ValueError(f"Empty volume loaded from {filepath}")

        # Ensure it's a 4D tensor (C, D, H, W)
        if volume.dim() == 3:
            volume = volume.unsqueeze(0)  # Add channel dimension
            logger.debug(f"load_cached_volume - Added channel dimension, new shape: {volume.shape}")
        elif volume.dim() == 4:
            pass  # Already 4D
        else:
            raise ValueError(f"Unexpected tensor shape: {volume.shape}")

        # Apply 2D transforms if provided
        if hasattr(transforms, "transforms"):
            for transform in transforms.transforms[1:]:  # Skip ToTensor
                volume = transform(volume)

        if transforms_3d:
            volume = transforms_3d(volume)

        # Pad Z dimension if needed
        if target_slices is not None and volume.shape[1] < target_slices:
            z_diff = target_slices - volume.shape[1]
            z_pad_before = z_diff // 2
            z_pad_after = z_diff - z_pad_before
            # Pad only the Z dimension (dimension 1 after channel)
            padding = (0, 0, 0, 0, z_pad_before, z_pad_after)  # (W, H, D) padding in reverse order
            volume = torch.nn.functional.pad(volume, padding, mode="constant", value=pad_value)
        elif target_slices is not None and volume.shape[1] > target_slices:
            # Center crop for validation/inference
            start_d = (volume.shape[1] - target_slices) // 2
            volume = volume[:, start_d : start_d + target_slices]

        # Handle H dimension (height)
        if target_h is not None:
            current_h = volume.shape[2]
            if current_h < target_h:
                # Pad H dimension
                h_diff = target_h - current_h
                h_pad_before = h_diff // 2
                h_pad_after = h_diff - h_pad_before
                padding = (
                    0,
                    0,
                    h_pad_before,
                    h_pad_after,
                    0,
                    0,
                )  # (W, H, D) padding in reverse order
                volume = torch.nn.functional.pad(volume, padding, mode="constant", value=pad_value)
            elif current_h > target_h:
                start_h = (current_h - target_h) // 2
                volume = volume[:, :, start_h : start_h + target_h, :]

        # Handle W dimension (width)
        if target_w is not None:
            current_w = volume.shape[3]
            if current_w < target_w:
                # Pad W dimension
                w_diff = target_w - current_w
                w_pad_before = w_diff // 2
                w_pad_after = w_diff - w_pad_before
                padding = (
                    w_pad_before,
                    w_pad_after,
                    0,
                    0,
                    0,
                    0,
                )  # (W, H, D) padding in reverse order
                volume = torch.nn.functional.pad(volume, padding, mode="constant", value=pad_value)
            elif current_w > target_w:
                start_w = (current_w - target_w) // 2
                volume = volume[:, :, :, start_w : start_w + target_w]

        if load_metadata:
            return volume, metadata
        else:
            return volume

    except Exception as e:
        logger.error(f"Failed to load cached volume {filepath}: {e}")
        raise DatasetError(f"Failed to load cached volume {filepath}: {e}")


def apply_3d_transforms(
    volume: torch.Tensor, target_d: int, pad_value: float, transform_option: str
) -> torch.Tensor:
    """
    Apply 3D transforms to adjust volume depth.
    """
    current_d = volume.shape[0]
    option_insufficient, option_excess = transform_option.split(",")

    if current_d < target_d:
        if option_insufficient == "pad":
            pad_size = math.ceil((target_d - current_d) / 2)
            if volume.dim() == 3:  # (D, H, W)
                volume = torch.nn.functional.pad(
                    volume, (0, 0, 0, 0, pad_size, target_d - current_d - pad_size), value=pad_value
                )
            elif volume.dim() == 4:  # (C, D, H, W)
                volume = torch.nn.functional.pad(
                    volume, (0, 0, 0, 0, pad_size, target_d - current_d - pad_size), value=pad_value
                )
        else:
            # Repeat slices to reach target depth
            repeat_factor = math.ceil(target_d / current_d)
            volume = (
                volume.repeat(repeat_factor, 1, 1)
                if volume.dim() == 3
                else volume.repeat(1, repeat_factor, 1, 1)
            )
            volume = volume[:target_d] if volume.dim() == 3 else volume[:, :target_d]

    elif current_d > target_d:
        if option_excess == "center":
            # Take center slices
            start_idx = (current_d - target_d) // 2
            volume = volume[start_idx : start_idx + target_d]
        elif option_excess == "nearest":
            # Downsample using nearest neighbor
            indices = torch.linspace(0, current_d - 1, target_d).long()
            volume = volume[indices]
        else:
            # Take first target_d slices
            volume = volume[:target_d]

    return volume


class RVEDataset:
    """
    RVE dataset for medical image analysis.

    This dataset loads 3D CT volumes from cached .pt files or NII.gz files.
    Labels are loaded separately during evaluation from JSON files.
    """

    def __init__(
        self,
        config: dict,
        split: str = "train",
        transforms=None,
        model_preprocess=None,
        name_override=None,
    ):
        self.config = config
        self.split = split

        self.transforms = transforms
        self.model_preprocess = model_preprocess  # Model's preprocess_single method

        # Initialize dataset
        self.setup_dataset()

        logger.info(f"Initialized MerlinAbdCT for split '{split}' with {len(self.df)} samples")

    def setup_dataset(self) -> None:
        """Initialize the dataset by loading metadata."""
        # Get JSON path based on split
        json_path = get_config_value(self.config, f"data.{self.split}_json")

        if json_path is None:
            raise ValueError(f"json path not found for split '{self.split}' in config")

        logger.info(f"Loading {self.split} dataset from {json_path}")

        # Load JSON file line by line (each line is a JSON object)
        samples = []
        with open(json_path, "r") as f:
            for line in f:
                if line.strip():
                    samples.append(pd.read_json(StringIO(line.strip()), typ="series"))

        self.df = pd.DataFrame(samples)

        # Validate required columns
        if "sample_name" not in self.df.columns:
            raise ValueError("json must contain 'sample_name' column")
        if "nii_path" not in self.df.columns:
            raise ValueError("json must contain 'nii_path' column")

        # Load cache manifest
        cache_manifest_path = get_config_value(self.config, "data.cache_manifest")
        if cache_manifest_path and os.path.exists(cache_manifest_path):
            logger.info(f"Loading cache manifest from {cache_manifest_path}")
            self.cache_manifest = pd.read_csv(cache_manifest_path)
            self.use_cache = True
        else:
            logger.warning(
                f"Cache manifest not found for {cache_manifest_path}, falling back to direct NII loading"
            )
            self.cache_manifest = None
            self.use_cache = False

        self.root_dir = get_config_value(self.config, "data.root_dir")
        self.modality = get_config_value(self.config, "modality")

        # Dataset configuration
        self.target_d = get_config_value(self.config, "target_d")
        self.target_h = get_config_value(self.config, "target_h")
        self.target_w = get_config_value(self.config, "target_w")
        self.pad_value = get_config_value(self.config, "pad_value")
        self.transform_option = get_config_value(self.config, "transform_option")
        self.use_opencv = get_config_value(self.config, "use_opencv")
        self.image_mode = get_config_value(self.config, "image_mode")
        self.transforms_3d_type = get_config_value(self.config, "transforms_3d")

        logger.info(f"Loaded {len(self.df)} samples for {self.split} split")
        logger.info(f"Using cached volumes: {self.use_cache}")
        logger.info(f"Sample accessions: {self.df['sample_name'].head().tolist()}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.df.iloc[idx]
        sample_name = row["sample_name"]

        if self.use_cache and self.cache_manifest is not None:
            # Use cached volume
            cache_row = self.cache_manifest[self.cache_manifest["sample_name"] == sample_name]
            if not cache_row.empty:
                cache_path = cache_row.iloc[0]["image_cache_path"]
                if os.path.exists(cache_path):
                    volume, metadata = load_cached_volume(
                        cache_path,
                        self.transforms,
                        ## do permute from CHWD to CDHW if needed (not needed in RVE)
                        ## NOTE: we dont use permute for the RVE generated caches
                        # lambda x: x.permute(0, 3, 1, 2),
                        None,
                        load_metadata=True,
                        target_slices=self.target_d,
                        target_h=self.target_h,
                        target_w=self.target_w,
                        pad_value=self.pad_value,
                    )
                    # logger.debug(f"**Volume min: {volume.min()}, max: {volume.max()}, transforms: {self.transforms}, volume shape: {volume.shape}**")
                else:
                    logger.warning(f"Cached file not found: {cache_path}, falling back to NII")
                    volume, metadata = self._load_nii_fallback(row)
            else:
                logger.warning(
                    f"Sample {sample_name} not found in cache manifest, falling back to NII"
                )
                volume, metadata = self._load_nii_fallback(row)
        else:
            # Fallback to direct NII loading
            volume, metadata = self._load_nii_fallback(row)

        # flip the volume along the first dimension to match UCSF orientation
        # in practice, this means that tensor is viewed from top to bottom (depth dimension)
        volume = torch.flip(volume, dims=[1])

        # Model-specific preprocessing
        ## NOTE: windowing is applied at batch level for efficiency, but resizing is needed per-model
        if self.model_preprocess:
            volume = self.model_preprocess(volume, metadata=metadata, modality=self.modality)

        return volume

    def _load_nii_fallback(self, row) -> tuple[torch.Tensor, dict]:
        """Fallback method to load NII file directly."""
        raise RuntimeError(f"Loading data failed for {row['sample_name']}")

    def get_sample_info(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        return {
            "sample_name": row["sample_name"],
            "split": self.split,
            "dataset_class": self.__class__.__name__,
            "accession": row["sample_name"],
            "sample_id": f'{self.split}_{row["sample_name"]}',
        }

    def get_accession(self, idx: int) -> str:
        """Get the accession ID for a specific sample."""
        return self.df.iloc[idx]["sample_name"]

    def get_all_accessions(self) -> List[str]:
        """Get all accessions without loading any data - much faster for filtering."""
        return self.df["sample_name"].tolist()

    def get_accessions_batch(self, indices: List[int]) -> List[str]:
        """Get accessions for a batch of indices without loading data."""
        return self.df.iloc[indices]["sample_name"].tolist()


MerlinAbdCT = RVEDataset