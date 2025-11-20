"""Merlin Abdominal CT dataset that mirrors the standalone preprocessing pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import pandas as pd
from monai.data import Dataset as MonaiDataset
from monai.data import PersistentDataset
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    SpatialPadd,
    Spacingd,
    ToTensord,
)

from ..common import DatasetError, get_logger
from ..config import get_config_value

logger = get_logger(__name__)


def _resolve_path(path: str, roots: Iterable[Optional[str]]) -> str:
    """Resolve image path against a list of candidate roots."""
    if os.path.isabs(path) and os.path.exists(path):
        return path

    for root in roots:
        if not root:
            continue
        candidate = os.path.join(root, path)
        if os.path.exists(candidate):
            return candidate

    if os.path.exists(path):
        return path

    raise FileNotFoundError(f"Unable to resolve path '{path}' using roots {list(roots)}")


def _build_image_transforms() -> Compose:
    """Create the MONAI transform pipeline used in merlin_standalone.py."""
    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1000,
                a_max=1000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            SpatialPadd(keys=["image"], spatial_size=[224, 224, 160]),
            CenterSpatialCropd(keys=["image"], roi_size=[224, 224, 160]),
            ToTensord(keys=["image"]),
        ]
    )


class MerlinAbdCT:
    """Dataset that loads abdominal CT volumes from NIfTI files using MONAI transforms."""

    def __init__(
        self,
        config: Dict[str, Any],
        split: str = "train",
        transforms: Optional[Any] = None,
        model_preprocess: Optional[Any] = None,
        name_override: Optional[str] = None,
    ) -> None:
        self.config = config
        self.split = split

        self.modality = get_config_value(self.config, "modality")
        self.image_key = get_config_value(self.config, "img_paths_key")
        self.text_key = get_config_value(self.config, "csv_caption_key")
        self.text_separator = get_config_value(self.config, "separator")
        self.root_dir = get_config_value(self.config, "data.root_dir")
        self.additional_roots = get_config_value(self.config, "data.additional_roots")
        if not isinstance(self.additional_roots, list):
            self.additional_roots = [self.additional_roots]
        self.nii_root_dir = get_config_value(self.config, "data.nii_root_dir")
        self.search_roots: List[Optional[str]] = [
            self.root_dir,
            self.nii_root_dir,
            *self.additional_roots,
        ]
        self.cache_dir = get_config_value(self.config, "data.cache_dir")

        self.additional_transforms = transforms
        self.model_preprocess = model_preprocess

        self.image_transforms = _build_image_transforms()

        self._load_metadata()
        self._build_monai_dataset()

        logger.info(
            "Initialized MerlinAbdCT (split=%s) with %d samples",
            split,
            len(self.df),
        )

    # ---------------------------------------------------------------------
    # Dataset construction helpers
    # ---------------------------------------------------------------------
    def _load_metadata(self) -> None:
        json_path = get_config_value(self.config, f"data.{self.split}_json")
        if json_path is None:
            raise DatasetError(f"json path not found for split '{self.split}' in config")

        if not os.path.exists(json_path):
            raise DatasetError(f"JSON file not found for split '{self.split}': {json_path}")

        logger.info("Loading %s metadata from %s", self.split, json_path)

        records: List[Dict[str, Any]] = []
        with open(json_path, "r") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise DatasetError(f"Failed to parse JSON line: {exc}") from exc

        if not records:
            raise DatasetError(f"No samples found in {json_path}")

        self.df = pd.DataFrame(records)

        if "sample_name" not in self.df.columns:
            raise DatasetError("JSON manifest missing 'sample_name' column")
        if self.image_key not in self.df.columns:
            raise DatasetError(f"JSON manifest missing '{self.image_key}' column")

        logger.debug("Sample columns: %s", list(self.df.columns))

        self.samples: List[Dict[str, Any]] = []
        for row in self.df.to_dict(orient="records"):
            try:
                image_path = _resolve_path(str(row[self.image_key]), self.search_roots)
            except FileNotFoundError as exc:
                raise DatasetError(str(exc)) from exc

            sample: Dict[str, Any] = {
                "sample_name": row["sample_name"],
                "image_path": image_path,
            }

            if self.text_key and self.text_key in row and row[self.text_key] is not None:
                text_value = row[self.text_key]
                if isinstance(text_value, list) and self.text_separator:
                    text_value = self.text_separator.join(map(str, text_value))
                sample["text"] = text_value

            self.samples.append(sample)

    def _build_monai_dataset(self) -> None:
        data_list = [{"image": sample["image_path"]} for sample in self.samples]

        if self.cache_dir:
            cache_dir = Path(self.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._monai_dataset = PersistentDataset(
                data=data_list,
                transform=self.image_transforms,
                cache_dir=cache_dir,
            )
            logger.info("Using PersistentDataset cache at %s", cache_dir)
        else:
            self._monai_dataset = MonaiDataset(
                data=data_list,
                transform=self.image_transforms,
            )

    # ---------------------------------------------------------------------
    # PyTorch dataset interface
    # ---------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        try:
            sample_dict = self._monai_dataset[idx]
        except FileNotFoundError as exc:
            logger.error("Failed to load sample at index %d: %s", idx, exc)
            raise DatasetError(f"Failed to load sample at index {idx}: {exc}") from exc

        image = sample_dict.get("image")
        if image is None:
            raise DatasetError("MONAI transform pipeline did not return an 'image' tensor")

        if not torch.is_tensor(image):
            image = torch.as_tensor(image)

        if self.additional_transforms is not None:
            image = self.additional_transforms(image)

        if self.model_preprocess is not None:
            image = self.model_preprocess(image, modality=self.modality)

        return image

    # ---------------------------------------------------------------------
    # Metadata helpers
    # ---------------------------------------------------------------------
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        return {
            "sample_name": sample["sample_name"],
            "split": self.split,
            "dataset_class": self.__class__.__name__,
            "accession": sample["sample_name"],
            "sample_id": f"{self.split}_{sample['sample_name']}",
        }

    def get_accession(self, idx: int) -> str:
        return self.samples[idx]["sample_name"]

    def get_all_accessions(self) -> List[str]:
        return [sample["sample_name"] for sample in self.samples]

    def get_accessions_batch(self, indices: List[int]) -> List[str]:
        return [self.samples[i]["sample_name"] for i in indices]
