"""Dummy dataset for testing and development."""

import torch
import numpy as np
import os
from PIL import Image

# Removed Tuple import since datasets now return only volumes

from ..common import get_logger
from ..config import get_config_value

logger = get_logger(__name__)


class DummyDataset:
    """
    Dummy dataset that reads a sample image and repeats it for testing.

    Similar to MIMIC-CXR processing but using a single image repeated.
    """

    def __init__(self, config: dict, split: str = "train", transforms=None, model_preprocess=None):
        self.config = config
        self.split = split
        self.transforms = transforms
        self.model_preprocess = model_preprocess  # Model's preprocess_single method

        # Initialize dataset
        self.setup_dataset()

        logger.info(f"Initialized DummyDataset for split '{split}' with {self.num_samples} samples")

    def setup_dataset(self) -> None:
        """Initialize the dummy dataset."""
        # Get configuration similar to MIMIC-CXR
        try:
            target_size = get_config_value(self.config, "target_size")
        except ValueError:
            target_size = get_config_value(self.config, "processing.target_size")
        self.target_size = target_size

        try:
            image_mode = get_config_value(self.config, "image_mode")
        except ValueError:
            image_mode = get_config_value(self.config, "processing.image_mode")
        self.image_mode = image_mode

        # Set number of samples (default 1000)
        try:
            num_samples = get_config_value(self.config, "num_samples")
        except ValueError:
            num_samples = get_config_value(self.config, "generation.num_samples")
        self.num_samples = num_samples

        # Set modality similar to MIMIC-CXR
        self.modality = get_config_value(self.config, "modality")

        # Find sample image path
        self.sample_image_path = self._find_sample_image()

        logger.info(f"Created dummy {self.split} dataset with {self.num_samples} samples")
        logger.info(f"Using sample image: {self.sample_image_path}")
        logger.info(f"Target size: {self.target_size}")

    def _find_sample_image(self) -> str:
        """Find a sample image to use for the dummy dataset."""
        # Look for the chest X-ray image in the project directory
        possible_paths = [
            "assets/CXR145_IM-0290-1001.png",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # If no sample image found, raise an error
        raise FileNotFoundError(f"No sample image found in any of the paths: {possible_paths}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a sample from the dataset (reads and processes image similar to MIMIC-CXR).

        Args:
            idx: Sample index

        Returns:
            Processed image tensor with two views: (C, 2, H, W)
            Same format as MIMIC-CXR dataset
        """
        # Load the same image twice to create two views (like MIMIC-CXR)
        images = self._load_image_pair()

        # Process images similar to MIMIC-CXR
        processed_images = []
        for img in images:
            # First apply dataset transforms or default preprocessing
            if self.transforms:
                img = self.transforms(img)
            else:
                # Default preprocessing (same as MIMIC-CXR)
                img = img.resize(self.target_size, Image.LANCZOS)
                img = np.array(img)
                if len(img.shape) == 3:
                    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                else:
                    img = torch.from_numpy(img).unsqueeze(0).float() / 255.0

            # Then apply model-specific preprocessing if available
            if self.model_preprocess:
                img = self.model_preprocess(img)

            processed_images.append(img)

        # Handle model preprocessing that returns dict (same as MIMIC-CXR)
        if isinstance(processed_images[0], dict):
            extra_infos = {
                k: torch.stack(
                    [processed_images[i][k] for i in range(len(processed_images))], dim=0
                )
                for k in processed_images[0].keys()
                if k != "image"
            }
            processed_images = [processed_image["image"] for processed_image in processed_images]
        else:
            extra_infos = None

        # Always return exactly two views (same as MIMIC-CXR)
        if len(processed_images) == 1:
            # Duplicate the single view to create two views
            view1 = processed_images[0]
            view2 = processed_images[0].clone()  # Clone to avoid sharing memory
            processed_images = [view1, view2]

            # also repeat items in extra_infos
            if extra_infos is not None:
                extra_infos = {
                    k: torch.stack([extra_infos[k][0], extra_infos[k][0]], dim=0)
                    for k in extra_infos.keys()
                }

        if extra_infos is not None:
            stacked_views = torch.stack(processed_images, dim=0)
            stacked_views = stacked_views[None, ...]
            return stacked_views, extra_infos
        else:
            # Stack the two views along a new depth dimension: (C, H, W) -> (C, 2, H, W)
            stacked_views = torch.stack(processed_images, dim=1)  # Stack along depth dimension
            return stacked_views

    def _load_image_pair(self) -> list:
        """Load the same image twice to simulate a pair (like MIMIC-CXR)."""
        try:
            image = Image.open(self.sample_image_path)
            if self.image_mode and image.mode != self.image_mode:
                image = image.convert(self.image_mode)
            # Return the same image twice to simulate a pair
            return [image, image.copy()]
        except Exception as e:
            raise RuntimeError(f"Failed to load sample image {self.sample_image_path}: {e}")

    def get_sample_info(self, idx: int) -> dict:
        """Get metadata for a sample (similar to MIMIC-CXR format)."""
        return {
            "index": idx,
            "split": self.split,
            "dataset_class": self.__class__.__name__,
            "study_id": f"dummy_study_{idx:04d}",
            "subject_id": f"dummy_subject_{idx:04d}",
            "sample_id": f"{self.split}_{idx:04d}",
            "num_images": 2,
            "view_positions": ["PA", "PA"],  # Same image repeated as PA view
            "output_format": "Two views stacked as (C, 2, H, W)",
            "views_returned": 2,  # Always returns exactly 2 views
            "synthetic": True,
        }

    def get_accession(self, idx: int) -> str:
        """Get the study ID as accession for evaluation (similar to MIMIC-CXR)."""
        return f"dummy_study_{idx:04d}"

    def get_all_accessions(self) -> list:
        """Get all accessions without loading any data - much faster for filtering."""
        return [f"dummy_study_{idx:04d}" for idx in range(self.num_samples)]

    def get_accessions_batch(self, indices: list) -> list:
        """Get accessions for a batch of indices without loading data."""
        return [f"dummy_study_{idx:04d}" for idx in indices]
