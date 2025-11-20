"""Merlin Abdominal CT dataset that reproduces the CT-CLIP preprocessing pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..common import DatasetError, get_logger
from ..config import get_config_value


logger = get_logger(__name__)

TARGET_SPACING = (1.5, 0.75, 0.75)
TARGET_SHAPE = (480, 480, 240)
CLIP_RANGE = (-1000.0, 1000.0)


def _resolve_path(path: str, roots: Iterable[Optional[str]]) -> str:
    if Path(path).is_absolute() and Path(path).exists():
        return path

    for root in roots:
        if not root:
            continue
        candidate = Path(root) / path
        if candidate.exists():
            return str(candidate)

    if Path(path).exists():
        return path

    raise FileNotFoundError(f"Unable to resolve path '{path}' using roots {list(roots)}")


def _resize_array(
    array: torch.Tensor,
    current_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
) -> np.ndarray:
    original_shape = array.shape[2:]
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [
        int(max(1, round(original_shape[i] * scaling_factors[i])))
        for i in range(len(original_shape))
    ]
    resized = F.interpolate(array, size=new_shape, mode="trilinear", align_corners=False)
    return resized.cpu().numpy()


def _safe_float(value: Any, default: float) -> float:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.item()

    try:
        value = float(value)
    except (TypeError, ValueError):
        return default

    if not math.isfinite(value):
        return default

    return value


def _parse_xy_spacing(value: Any) -> Optional[float]:
    if value is None:
        return None

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped[1:-1]
        parts = [part.strip() for part in stripped.split(",") if part.strip()]
        if not parts:
            return None
        value = parts[0]

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class MerlinAbdCTForCTCLIPModel:
    """Dataset that loads abdominal CT volumes using the standalone CT-CLIP preprocessing."""

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
        if isinstance(self.additional_roots, tuple):
            self.additional_roots = list(self.additional_roots)
        elif hasattr(self.additional_roots, "__iter__") and not isinstance(
            self.additional_roots, (list, str)
        ):
            self.additional_roots = list(self.additional_roots)
        if not isinstance(self.additional_roots, list):
            self.additional_roots = [self.additional_roots]
        self.nii_root_dir = get_config_value(self.config, "data.nii_root_dir")
        self.search_roots = [
            str(root) if root is not None else None
            for root in [self.root_dir, self.nii_root_dir, *self.additional_roots]
        ]
        self.metadata_csv = get_config_value(self.config, "data.metadata_csv")

        self.additional_transforms = transforms
        self.model_preprocess = model_preprocess

        self._csv_metadata: Optional[Dict[str, Dict[str, float]]] = None
        self.samples: List[Dict[str, Any]] = []

        self._load_metadata()

        logger.info(
            "Initialized MerlinAbdCTForCTCLIPModel (split=%s) with %d samples",
            split,
            len(self.samples),
        )

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------
    def _load_metadata(self) -> None:
        json_path = get_config_value(self.config, f"data.{self.split}_json")
        if json_path is None:
            raise DatasetError(f"json path not found for split '{self.split}' in config")

        json_path = Path(json_path)
        if not json_path.exists():
            raise DatasetError(f"JSON file not found for split '{self.split}': {json_path}")

        logger.info("Loading %s metadata from %s", self.split, json_path)

        records: List[Dict[str, Any]] = []
        with json_path.open("r") as fh:
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

        df = pd.DataFrame(records)

        if "sample_name" not in df.columns:
            raise DatasetError("JSON manifest missing 'sample_name' column")
        if self.image_key not in df.columns:
            raise DatasetError(f"JSON manifest missing '{self.image_key}' column")

        for row in df.to_dict(orient="records"):
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

    def _ensure_csv_metadata_loaded(self) -> None:
        if self._csv_metadata is not None or not self.metadata_csv:
            return

        csv_path = Path(self.metadata_csv)
        if not csv_path.exists():
            logger.warning(
                "Metadata CSV not found at %s; falling back to NIfTI headers only", csv_path
            )
            self._csv_metadata = {}
            return

        df = pd.read_csv(csv_path)
        required_cols = {"VolumeName", "RescaleSlope", "RescaleIntercept", "XYSpacing", "ZSpacing"}
        if not required_cols.issubset(df.columns):
            raise DatasetError(f"Metadata CSV missing required columns: {required_cols}")

        metadata_map: Dict[str, Dict[str, float]] = {}
        for record in df.to_dict(orient="records"):
            volume_name = str(record["VolumeName"])
            xy_spacing = _parse_xy_spacing(record["XYSpacing"])
            if xy_spacing is None:
                xy_spacing = float("nan")
            metadata_map[volume_name] = {
                "slope": _safe_float(record.get("RescaleSlope"), 1.0),
                "intercept": _safe_float(record.get("RescaleIntercept"), 0.0),
                "xy_spacing": _safe_float(xy_spacing, float("nan")),
                "z_spacing": _safe_float(record.get("ZSpacing"), float("nan")),
            }

        self._csv_metadata = metadata_map

    def _lookup_csv_metadata(self, volume_name: str) -> Optional[Dict[str, float]]:
        self._ensure_csv_metadata_loaded()
        if not self._csv_metadata:
            return None
        return self._csv_metadata.get(volume_name)

    def _extract_volume_metadata(
        self, nii_img: nib.Nifti1Image, volume_path: Path
    ) -> Tuple[float, float, float, float]:
        header = nii_img.header

        slope = _safe_float(header.get("scl_slope", 1.0), 1.0)
        intercept = _safe_float(header.get("scl_inter", 0.0), 0.0)

        zooms = header.get_zooms()[:3]
        x_spacing = _safe_float(zooms[0] if len(zooms) > 0 else float("nan"), float("nan"))
        y_spacing = _safe_float(zooms[1] if len(zooms) > 1 else x_spacing, float("nan"))
        z_spacing = _safe_float(zooms[2] if len(zooms) > 2 else float("nan"), float("nan"))

        if (
            (not math.isfinite(x_spacing))
            or (not math.isfinite(y_spacing))
            or (not math.isfinite(z_spacing))
        ):
            csv_meta = self._lookup_csv_metadata(volume_path.name)
            if csv_meta is not None:
                slope = csv_meta.get("slope", slope)
                intercept = csv_meta.get("intercept", intercept)
                x_spacing = csv_meta.get("xy_spacing", x_spacing)
                y_spacing = csv_meta.get("xy_spacing", y_spacing)
                z_spacing = csv_meta.get("z_spacing", z_spacing)

        if not math.isfinite(x_spacing):
            x_spacing = 1.0
        if not math.isfinite(y_spacing):
            y_spacing = x_spacing
        if not math.isfinite(z_spacing):
            z_spacing = 1.0

        xy_spacing = (x_spacing + y_spacing) / 2.0 if math.isfinite(y_spacing) else x_spacing

        return slope, intercept, xy_spacing, z_spacing

    def _preprocess_volume(self, image_path: str) -> torch.Tensor:
        path = Path(image_path)
        if not path.exists():
            raise DatasetError(f"Volume not found at {path}")

        nii_img = nib.load(str(path))
        slope, intercept, xy_spacing, z_spacing = self._extract_volume_metadata(nii_img, path)

        raw_data = np.asarray(nii_img.dataobj.get_unscaled(), dtype=np.float32)
        img_data = slope * raw_data + intercept
        img_data = np.clip(img_data, CLIP_RANGE[0], CLIP_RANGE[1])

        img_data = img_data.transpose(2, 0, 1)
        tensor = torch.from_numpy(img_data).unsqueeze(0).unsqueeze(0)

        resized = _resize_array(tensor, (z_spacing, xy_spacing, xy_spacing), TARGET_SPACING)
        img_data = resized[0, 0]
        img_data = np.transpose(img_data, (1, 2, 0))
        img_data = (img_data / CLIP_RANGE[1]).astype(np.float32)

        volume = torch.from_numpy(img_data)

        target_h, target_w, target_d = TARGET_SHAPE
        h, w, d = volume.shape

        h_start = max((h - target_h) // 2, 0)
        w_start = max((w - target_w) // 2, 0)
        d_start = max((d - target_d) // 2, 0)

        volume = volume[
            h_start : min(h_start + target_h, h),
            w_start : min(w_start + target_w, w),
            d_start : min(d_start + target_d, d),
        ]

        pad_h_before = max((target_h - volume.shape[0]) // 2, 0)
        pad_h_after = target_h - volume.shape[0] - pad_h_before
        pad_w_before = max((target_w - volume.shape[1]) // 2, 0)
        pad_w_after = target_w - volume.shape[1] - pad_w_before
        pad_d_before = max((target_d - volume.shape[2]) // 2, 0)
        pad_d_after = target_d - volume.shape[2] - pad_d_before

        volume = F.pad(
            volume,
            (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after),
            value=-1.0,
        )

        volume = volume.permute(2, 0, 1)
        volume = volume.unsqueeze(0)
        return volume.contiguous()

    # ------------------------------------------------------------------
    # PyTorch dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        try:
            sample = self.samples[idx]
        except IndexError as exc:
            raise DatasetError(f"Sample index {idx} out of range") from exc

        image = self._preprocess_volume(sample["image_path"])

        if self.additional_transforms is not None:
            image = self.additional_transforms(image)

        if self.model_preprocess is not None:
            image = self.model_preprocess(image, modality=self.modality)

        return image

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
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
