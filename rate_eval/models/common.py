import torch
import rve
from typing import Optional, Union, List


def batch_apply_mr_windowing(
    batch: torch.Tensor,
    mr_window_type: str = "all",
    modality: str = "MR",
    normalize_mean: Optional[float] = None,
    normalize_std: Optional[float] = None,
    per_sample: bool = True,
) -> torch.Tensor:
    """Apply windowing and normalization transforms to MR volumes."""

    B, C, D, H, W = batch.shape
    device = batch.device
    if per_sample:
        # Per-sample windowing implementation
        if mr_window_type in ["high_contrast", "znorm"]:
            # Single window
            num_output_channels = C
            windowed = torch.zeros(
                (B, num_output_channels, D, H, W), device=device, dtype=batch.dtype
            )
            for b in range(B):
                for cidx in range(C):
                    sample = batch[b : b + 1, cidx : cidx + 1]  # Keep batch dimension
                    windowed[b, cidx] = rve.apply_windowing(sample, mr_window_type, modality)
            batch = windowed
        else:
            raise ValueError(f"Invalid window type: {mr_window_type}")
    else:
        if mr_window_type in ["high_contrast", "znorm"]:
            # Single window
            windowed = torch.zeros((B, C, D, H, W), device=device, dtype=batch.dtype)

            # Apply each window type to the entire batch
            for cidx in range(C):
                windowed[:, cidx] = rve.apply_windowing(
                    batch[:, cidx : cidx + 1], mr_window_type, modality
                )
            batch = windowed
        else:
            raise ValueError(f"Invalid window type: {mr_window_type}")

    # Apply normalization after windowing
    assert (normalize_mean is None) == (
        normalize_std is None
    ), "Either both or none of normalize_mean and normalize_std must be provided"
    if normalize_mean is not None or normalize_std is not None:
        normalize_mean = torch.tensor(normalize_mean, device=device)
        normalize_std = torch.tensor(normalize_std, device=device)
        batch = (batch - normalize_mean) / normalize_std

    return batch


def batch_apply_ct_windowing(
    batch: torch.Tensor,
    ct_window_type=None,
    modality: str = "CT",
    normalize_mean: Optional[float] = None,
    normalize_std: Optional[float] = None,
    per_sample: bool = True,
) -> torch.Tensor:
    """Apply windowing and normalization transforms to CT volumes."""
    B, C, D, H, W = batch.shape
    device = batch.device

    if per_sample:
        # Per-sample windowing implementation
        if ct_window_type == "all":
            # Get all available windows
            windows = rve.get_available_windows(modality)
            # Allocate output tensor
            windowed = torch.zeros((B, len(windows), D, H, W), device=device, dtype=batch.dtype)

            # Apply each window type per sample
            for b in range(B):
                sample = batch[b : b + 1]  # Keep batch dimension
                for i, window in enumerate(windows):
                    if C == 1:
                        windowed[b, i] = rve.apply_windowing(
                            sample.squeeze(0).squeeze(0), window, modality
                        )
                    else:
                        windowed[b, i] = rve.apply_windowing(sample[0, 0], window, modality)
            batch = windowed
        elif isinstance(ct_window_type, list):
            # Multiple specific windows
            windowed = torch.zeros(
                (B, len(ct_window_type), D, H, W), device=device, dtype=batch.dtype
            )
            for b in range(B):
                sample = batch[b : b + 1]  # Keep batch dimension
                for i, window in enumerate(ct_window_type):
                    if C == 1:
                        windowed[b, i] = rve.apply_windowing(
                            sample.squeeze(0).squeeze(0), window, modality
                        )
                    else:
                        windowed[b, i] = rve.apply_windowing(sample[0, 0], window, modality)
            batch = windowed
        else:
            # Single window
            windowed = torch.zeros((B, 1, D, H, W), device=device, dtype=batch.dtype)
            for b in range(B):
                sample = batch[b : b + 1]  # Keep batch dimension
                if C == 1:
                    windowed[b, 0] = rve.apply_windowing(
                        sample.squeeze(0).squeeze(0), ct_window_type, modality
                    )
                else:
                    windowed[b, 0] = rve.apply_windowing(sample[0, 0], ct_window_type, modality)
            batch = windowed
    else:
        # Original batch-wise windowing (normalization across entire batch)
        if ct_window_type == "all":
            # Get all available windows
            windows = rve.get_available_windows(modality)
            # Allocate output tensor
            windowed = torch.zeros((B, len(windows), D, H, W), device=device, dtype=batch.dtype)

            # Apply each window type to the entire batch
            for i, window in enumerate(windows):
                if C == 1:
                    windowed[:, i] = rve.apply_windowing(batch.squeeze(1), window, modality)
                else:
                    windowed[:, i] = rve.apply_windowing(batch[:, 0], window, modality)
            batch = windowed
        elif isinstance(ct_window_type, list):
            # Multiple specific windows
            windowed = torch.zeros(
                (B, len(ct_window_type), D, H, W), device=device, dtype=batch.dtype
            )
            for i, window in enumerate(ct_window_type):
                if C == 1:
                    windowed[:, i] = rve.apply_windowing(batch.squeeze(1), window, modality)
                else:
                    windowed[:, i] = rve.apply_windowing(batch[:, 0], window, modality)
            batch = windowed
        else:
            # Single window
            if C == 1:
                batch = rve.apply_windowing(batch.squeeze(1), ct_window_type, modality).unsqueeze(1)
            else:
                batch = rve.apply_windowing(batch[:, 0], ct_window_type, modality).unsqueeze(1)

    # Apply normalization after windowing
    assert (normalize_mean is None) == (
        normalize_std is None
    ), "Either both or none of normalize_mean and normalize_std must be provided"
    if normalize_mean is not None or normalize_std is not None:
        normalize_mean = torch.tensor(normalize_mean, device=device)
        normalize_std = torch.tensor(normalize_std, device=device)
        batch = (batch - normalize_mean) / normalize_std

    return batch


def batch_apply_normalization(
    batch: torch.Tensor,
    normalize_mean: Union[float, List[float]],
    normalize_std: Union[float, List[float]],
) -> torch.Tensor:
    """Apply normalization to a batch of volumes."""
    assert isinstance(normalize_mean, (float, list)) and isinstance(
        normalize_std, (float, list)
    ), f"normalize_mean and normalize_std must be either float or list, got {type(normalize_mean)} and {type(normalize_std)}"

    if isinstance(normalize_mean, float):
        # If we apply windowing, C might not be 1. In this case, we only support one mean and std for all channels.
        if len(batch.shape) == 5:  # (B, C, D, H, W)
            mean = torch.tensor(normalize_mean, device=batch.device).view(1, -1, 1, 1, 1)
            std = torch.tensor(normalize_std, device=batch.device).view(1, -1, 1, 1, 1)
        elif len(batch.shape) == 4:  # (B, C, H, W)
            mean = torch.tensor(normalize_mean, device=batch.device).view(1, -1, 1, 1)
            std = torch.tensor(normalize_std, device=batch.device).view(1, -1, 1, 1)
        else:
            raise ValueError(f"Expected 4D or 5D input tensor, got shape {batch.shape}")
    else:
        assert (
            len(normalize_mean) == len(normalize_std) == 3
        ), f"Expected 3 channels, got {len(normalize_mean)} and {len(normalize_std)}"
        if len(batch.shape) == 5:  # (B, C, D, H, W)
            assert batch.shape[1] == 1, f"Expected 1 channel, got {batch.shape[1]}"
            mean = torch.tensor(normalize_mean, device=batch.device).view(1, -1, 1, 1, 1)
            std = torch.tensor(normalize_std, device=batch.device).view(1, -1, 1, 1, 1)
            batch = batch.repeat(1, 3, 1, 1, 1)
        elif len(batch.shape) == 4:  # (B, C, H, W)
            assert batch.shape[1] == 1, f"Expected 1 channel, got {batch.shape[1]}"
            mean = torch.tensor(normalize_mean, device=batch.device).view(1, -1, 1, 1)
            std = torch.tensor(normalize_std, device=batch.device).view(1, -1, 1, 1)
            batch = batch.repeat(1, 3, 1, 1)
        else:
            raise ValueError(f"Expected 4D or 5D input tensor, got shape {batch.shape}")

    batch = (batch - mean) / std

    return batch
