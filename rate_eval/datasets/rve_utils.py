"""Shared utilities for datasets that rely on rad-vision-engine (RVE)."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from ..common import get_logger, load_metadata_from_directory

logger = get_logger(__name__)

try:
    import rve  # type: ignore

    HAS_RVE = True
except ImportError:
    HAS_RVE = False


@contextmanager
def _without_env_var(key: str):
    """Temporarily remove an environment variable if it exists."""
    previous = os.environ.pop(key, None)
    try:
        yield
    finally:
        if previous is not None:
            os.environ[key] = previous


def _pad_volume(
    volume: torch.Tensor,
    pad: Tuple[int, int, int, int, int, int],
    pad_value: float,
) -> torch.Tensor:
    """Apply symmetric padding to a 3D or 4D volume."""
    if not any(pad):
        return volume

    if volume.dim() == 3:
        padded = F.pad(volume.unsqueeze(0).unsqueeze(0), pad, value=pad_value)
        return padded.squeeze(0).squeeze(0)
    if volume.dim() == 4:
        padded = F.pad(volume.unsqueeze(0), pad, value=pad_value)
        return padded.squeeze(0)

    raise ValueError(f"Expected 3D or 4D tensor for padding, got shape {tuple(volume.shape)}")


def _crop_indices(length: int, target: int, crop_mode: str) -> slice:
    """Compute the slice that crops a dimension to the requested target size."""
    if length == target:
        return slice(None)

    if crop_mode == "random" and length > target:
        start = torch.randint(0, length - target + 1, (1,)).item()
    else:
        start = max((length - target) // 2, 0)
    end = start + target
    return slice(start, end)


def load_rve_volume(
    filepath: str,
    *,
    target_d: Optional[int] = None,
    target_h: Optional[int] = None,
    target_w: Optional[int] = None,
    pad_value: float = 0.0,
    crop_mode: str = "center",
    add_channel_dim: bool = False,
    dtype: Optional[torch.dtype] = torch.float32,
    load_metadata: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[dict]]]:
    """
    Load a 3D volume from an RVE sample and optionally pad/crop to target dimensions.

    Args:
        filepath: Directory containing the RVE-exported sample.
        target_d: Desired depth (Z) dimension.
        target_h: Desired height (Y) dimension.
        target_w: Desired width (X) dimension.
        pad_value: Value used when padding is required.
        crop_mode: How to crop when the volume exceeds target size ('center' or 'random').
        add_channel_dim: If True, ensure an explicit channel dimension is present.
        dtype: Optional dtype to cast the result to (default: float32).
        load_metadata: Whether to return `metadata.json` contents alongside the volume.

    Returns:
        The processed volume tensor, or a tuple of (tensor, metadata) when requested.
    """
    if not HAS_RVE:
        raise RuntimeError(
            "rad-vision-engine (rve) is not installed. Install it to enable RVE-backed datasets."
        )

    metadata = load_metadata_from_directory(filepath) if load_metadata else None

    with _without_env_var("LD_LIBRARY_PATH"):
        volume = rve.load_sample(filepath, use_hardware_acceleration=False)

    if dtype is not None:
        volume = volume.to(dtype)

    if add_channel_dim and volume.dim() == 3:
        volume = volume.unsqueeze(0)
    elif not add_channel_dim and volume.dim() == 4 and volume.shape[0] == 1:
        volume = volume.squeeze(0)

    pad_args = [0, 0, 0, 0, 0, 0]  # (w_left, w_right, h_left, h_right, d_left, d_right)

    dim_targets = (
        (target_d, -3),
        (target_h, -2),
        (target_w, -1),
    )

    # Ensure volume has at least 3 dimensions corresponding to D, H, W.
    if volume.dim() not in (3, 4):
        raise ValueError(
            f"Expected RVE volume to have 3 or 4 dimensions, got {tuple(volume.shape)}"
        )

    for idx, (target, axis) in enumerate(dim_targets):
        if target is None:
            continue

        current = volume.shape[axis]

        if current < target:
            diff = target - current
            before = diff // 2
            after = diff - before

            if axis == -3:  # depth
                pad_args[4], pad_args[5] = before, after
            elif axis == -2:  # height
                pad_args[2], pad_args[3] = before, after
            else:  # width
                pad_args[0], pad_args[1] = before, after

    if any(pad_args):
        volume = _pad_volume(volume, tuple(pad_args), pad_value)

    slices = []
    for target, axis in dim_targets:
        if target is None:
            slices.append(slice(None))
            continue

        current = volume.shape[axis]
        if current > target:
            slices.append(_crop_indices(current, target, crop_mode))
        else:
            slices.append(slice(None))

    # slices currently ordered in iteration; map back to D,H,W
    d_slice, h_slice, w_slice = slices

    if volume.dim() == 4:
        volume = volume[:, d_slice, h_slice, w_slice]
    else:
        volume = volume[d_slice, h_slice, w_slice]

    if load_metadata:
        return volume, metadata
    return volume
