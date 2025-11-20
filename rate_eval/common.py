"""Common utilities for the RATE evaluation pipeline.

This module consolidates all shared functionality to eliminate code duplication.
"""

import os
import logging
import sys
import warnings
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Any, Dict, Union, Optional, List, Tuple, Set

try:
    import colorlog

    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False

# Suppress common warnings early
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="runpy")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


# =============================================================================
# LOGGING UTILITIES
# =============================================================================


def setup_logging(
    level: str = "INFO",
    format_str: Optional[str] = None,
    colored: bool = True,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    debug: bool = False,
) -> None:
    """
    Set up logging configuration for the entire pipeline with optional coloring.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_str: Custom format string for log messages
        colored: Whether to use colored output (requires colorlog package)
        rank: GPU rank for multi-GPU settings (0-based, None for single GPU)
        world_size: Total number of GPUs (None for single GPU)
        debug: Whether debug mode is enabled
    """
    # Adjust log level based on multi-GPU rank and debug settings
    if world_size is not None and world_size > 1 and not debug:
        # Multi-GPU setting with debug disabled
        if rank == 0:
            # Rank 0: use INFO level unless explicitly set to something else
            actual_level = level if level != "INFO" else "INFO"
        else:
            # Other ranks: use WARN level
            actual_level = "WARNING"
    else:
        # Single GPU or debug mode: use the specified level
        actual_level = level
    # Convert string level to logging constant
    numeric_level = getattr(logging, actual_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {actual_level}")

    # Setup colored logging if available and requested
    if colored and HAS_COLORLOG:
        if format_str is None:
            format_str = "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        formatter = colorlog.ColoredFormatter(
            format_str,
            datefmt="%H:%M:%S",
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)

        # Configure root logger
        logging.basicConfig(level=numeric_level, handlers=[handler])
    else:
        # Fallback to regular logging
        if format_str is None:
            format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        logging.basicConfig(
            level=numeric_level, format=format_str, handlers=[logging.StreamHandler(sys.stdout)]
        )

    # Set specific loggers to reduce noise
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Name of the module (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_colored(logger: logging.Logger, level: str, message: str, color: str = None) -> None:
    """
    Log a message with specific color (when colorlog is available).

    Args:
        logger: Logger instance
        level: Log level (info, warning, error, debug, critical)
        message: Message to log
        color: Color name (red, green, blue, yellow, magenta, cyan, white, black)
               or color with background (e.g., 'red,bg_white')
    """
    if color and HAS_COLORLOG:
        # Temporarily modify the formatter to use custom color
        original_colors = None
        for handler in logger.handlers:
            if isinstance(handler.formatter, colorlog.ColoredFormatter):
                original_colors = handler.formatter.log_colors.copy()
                handler.formatter.log_colors[level.upper()] = color
                break

        # Log the message
        getattr(logger, level.lower())(message)

        # Restore original colors
        if original_colors:
            for handler in logger.handlers:
                if isinstance(handler.formatter, colorlog.ColoredFormatter):
                    handler.formatter.log_colors = original_colors
                    break
    else:
        # Fallback to regular logging
        getattr(logger, level.lower())(message)


# =============================================================================
# DEVICE MANAGEMENT
# =============================================================================


def setup_device(device_spec: Optional[str] = None) -> torch.device:
    """
    Set up and validate device for model/data operations.

    Args:
        device_spec: Device specification ('cuda', 'cpu', 'cuda:0', etc.)

    Returns:
        Torch device object
    """
    if device_spec is None:
        device_spec = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_spec)

    if device.type == "cuda" and not torch.cuda.is_available():
        logger = get_logger(__name__)
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = torch.device("cpu")

    return device


# =============================================================================
# FILE I/O UTILITIES
# =============================================================================


def ensure_dir_exists(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_metadata_from_directory(
    dirpath: Union[str, Path], filename: str = "metadata.json"
) -> Optional[Dict[str, Any]]:
    """
    Load metadata JSON file from a directory.

    Args:
        dirpath: Path to the directory containing the metadata file
        filename: Name of the metadata file (default: "metadata.json")

    Returns:
        Dictionary containing metadata, or None if file doesn't exist

    Raises:
        ValueError: If the file exists but cannot be parsed as JSON
    """
    import json

    dirpath = Path(dirpath)
    metadata_file = dirpath / filename

    if not metadata_file.exists():
        return None

    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        return metadata
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse metadata file {metadata_file}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load metadata file {metadata_file}: {e}")


# =============================================================================
# ERROR HANDLING
# =============================================================================


class RATEEvalError(Exception):
    """Base exception for the RATE evaluation pipeline."""

    pass


class ModelError(RATEEvalError):
    """Raised when there's a model-related error."""

    pass


class DatasetError(RATEEvalError):
    """Raised when there's a dataset-related error."""

    pass


# =============================================================================
# CHECKPOINT MANAGEMENT - REFACTORED
# =============================================================================

import pandas as pd
from datetime import datetime
import fcntl  # For file locking on Unix systems
import contextlib
import time
from transformers.utils import cached_file


# =============================================================================
# MODEL DOWNLOAD MANAGEMENT
# =============================================================================


class ModelDownloadLock:
    """Thread-safe model download lock to prevent concurrent downloads of the same model."""

    def __init__(self, model_repo_id: str, revision: str = "main", cache_dir: Optional[str] = None):
        """
        Initialize download lock for a specific model.

        Args:
            model_repo_id: HuggingFace model repository ID
            revision: Model revision/branch to download
            cache_dir: Custom cache directory (defaults to HF cache)
        """
        self.model_repo_id = model_repo_id
        self.revision = revision
        self.cache_dir = cache_dir

        # Create a lock file path based on model repo and revision
        cache_root = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "huggingface"
        lock_dir = cache_root / "download_locks"
        lock_dir.mkdir(parents=True, exist_ok=True)

        # Create a safe filename from repo_id and revision
        safe_repo_id = model_repo_id.replace("/", "_").replace(":", "_")
        safe_revision = revision.replace("/", "_").replace(":", "_")
        self.lock_file = lock_dir / f"{safe_repo_id}_{safe_revision}.lock"

        logger = get_logger(__name__)
        logger.debug(f"ModelDownloadLock initialized for {model_repo_id}@{revision}")
        logger.debug(f"Lock file: {self.lock_file}")

    def is_model_downloaded(self) -> bool:
        """
        Check if the model is already downloaded and available.

        Returns:
            True if model is available locally, False otherwise
        """
        try:
            # Try to access a core model file to check if download is complete
            # We'll check for config.json which should always exist
            cached_file(
                path_or_repo_id=self.model_repo_id,
                filename="config.json",
                revision=self.revision,
                cache_dir=self.cache_dir,
                local_files_only=True,  # Only check local cache, don't download
            )
            return True
        except Exception:
            # If any error occurs (file not found, cache miss, etc.), model isn't fully downloaded
            return False

    @contextlib.contextmanager
    def acquire_download_lock(self, timeout: int = 300):
        """
        Context manager to acquire exclusive download lock.

        Args:
            timeout: Maximum time to wait for lock in seconds

        Yields:
            None when lock is acquired

        Raises:
            TimeoutError: If lock cannot be acquired within timeout
        """
        logger = get_logger(__name__)

        if os.environ.get("RATE_SKIP_DOWNLOAD_LOCK", "0") == "1":
            logger.debug(
                "Skipping download lock for %s@%s due to RATE_SKIP_DOWNLOAD_LOCK",
                self.model_repo_id,
                self.revision,
            )
            yield
            return

        # If model is already downloaded, no need to lock
        if self.is_model_downloaded():
            logger.debug(
                f"Model {self.model_repo_id}@{self.revision} already downloaded, skipping lock"
            )
            yield
            return

        logger.info(f"Acquiring download lock for {self.model_repo_id}@{self.revision}")

        start_time = time.time()
        lock_acquired = False
        lock_file_handle = None

        try:
            while time.time() - start_time < timeout:
                try:
                    # Try to acquire exclusive lock
                    lock_file_handle = open(self.lock_file, "w")
                    fcntl.flock(lock_file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    lock_acquired = True

                    # Write process info to lock file for debugging
                    lock_file_handle.write(f"pid:{os.getpid()}\n")
                    lock_file_handle.write(f"model:{self.model_repo_id}@{self.revision}\n")
                    lock_file_handle.write(f"timestamp:{time.time()}\n")
                    lock_file_handle.flush()

                    logger.info(f"Download lock acquired for {self.model_repo_id}@{self.revision}")
                    break

                except (OSError, IOError):
                    # Lock is held by another process, wait and retry
                    if lock_file_handle:
                        lock_file_handle.close()
                        lock_file_handle = None

                    time.sleep(1)  # Wait 1 second before retry

                    # Check if model became available while waiting
                    if self.is_model_downloaded():
                        logger.debug(
                            f"Model {self.model_repo_id}@{self.revision} downloaded by another process"
                        )
                        yield
                        return

            if not lock_acquired:
                raise TimeoutError(
                    f"Could not acquire download lock for {self.model_repo_id}@{self.revision} within {timeout} seconds"
                )

            # Double-check if model was downloaded while we were acquiring the lock
            if self.is_model_downloaded():
                logger.debug(
                    f"Model {self.model_repo_id}@{self.revision} already downloaded, releasing lock"
                )
                yield
                return

            # Lock acquired and model not yet downloaded - proceed with download
            logger.debug(f"Proceeding with download for {self.model_repo_id}@{self.revision}")
            yield

        finally:
            # Release the lock
            if lock_acquired and lock_file_handle:
                try:
                    fcntl.flock(lock_file_handle.fileno(), fcntl.LOCK_UN)
                    lock_file_handle.close()
                    logger.debug(f"Download lock released for {self.model_repo_id}@{self.revision}")
                except Exception as e:
                    logger.warning(f"Error releasing download lock: {e}")

            # Clean up lock file
            try:
                if self.lock_file.exists():
                    self.lock_file.unlink()
            except Exception as e:
                logger.warning(f"Error removing lock file: {e}")


class SimpleCheckpointManager:
    """Simple checkpoint manager that stores individual .npz files per sample."""

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        split: str,
        cache_dir: str = "cache",
        skip_existing_cache: bool = False,
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.split = split
        self.skip_existing_cache = skip_existing_cache

        # Setup directory structure
        self.cache_root = Path(cache_dir)
        self.embeddings_dir = self.cache_root / "embeddings" / split
        self.processed_csv = self.cache_root / "processed.csv"

        # Create directories
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        logger = get_logger(__name__)

        # Track previously processed samples so they can be skipped if requested
        self._initial_processed_samples: Set[str] = set()
        if self.skip_existing_cache:
            self._initial_processed_samples = self._load_processed_samples()
            if self._initial_processed_samples:
                logger.info(
                    "Ignoring %d cached samples for %s/%s %s",
                    len(self._initial_processed_samples),
                    self.model_name,
                    self.dataset_name,
                    self.split,
                )
            self.processed_samples: Set[str] = set()
        else:
            # Load processed samples
            self.processed_samples = self._load_processed_samples()

        logger.info(
            "Checkpoint manager initialized: %d samples already processed",
            len(self.processed_samples),
        )

    def _load_processed_samples(self) -> set:
        """Load processed samples from CSV into a set for fast lookup."""
        if not self.processed_csv.exists():
            return set()

        try:
            df = pd.read_csv(self.processed_csv)

            # Filter for this specific model/dataset/split combination
            filtered_df = df[
                (df["model_name"] == self.model_name)
                & (df["dataset_name"] == self.dataset_name)
                & (df["split"] == self.split)
            ]

            # Convert all accessions to strings for consistent lookup
            processed_set = set(str(acc) for acc in filtered_df["accession"].tolist())
            logger = get_logger(__name__)
            logger.info(f"Found {len(processed_set)} previously processed samples")
            return processed_set
        except Exception as e:
            logger = get_logger(__name__)
            logger.warning(f"Failed to load processed.csv: {e}")
            return set()

    def is_sample_processed(self, accession: str) -> bool:
        """Check if a sample has already been processed."""
        return str(accession) in self.processed_samples

    def save_sample_embedding(
        self, accession: str, embedding: np.ndarray, continue_on_error: bool = False
    ) -> None:
        """Save a single sample's embedding to .npz file and update processed.csv."""
        logger = get_logger(__name__)

        # Ensure accession is string
        accession_str = str(accession)

        logger.debug(
            f"Checkpoint: Attempting to save embedding for {accession_str}, shape: {embedding.shape}, dtype: {embedding.dtype}"
        )
        logger.debug(
            f"Checkpoint: Embedding stats for {accession_str}: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}"
        )

        # Check for NaN values before saving
        nan_count = np.isnan(embedding).sum()
        logger.debug(
            f"Checkpoint: NaN check for {accession_str}: {nan_count} out of {embedding.size} elements are NaN"
        )

        if nan_count > 0:
            logger.error(f"NaN detected in embedding for sample {accession_str}")
            logger.error(f"Embedding shape: {embedding.shape}")
            logger.error(f"NaN locations: {nan_count} out of {embedding.size} elements")

            # Additional debug info for NaN troubleshooting
            logger.debug(f"Checkpoint: NaN pattern analysis for {accession_str}:")
            nan_mask = np.isnan(embedding)
            for batch_idx in range(embedding.shape[0]):
                batch_nan_count = nan_mask[batch_idx].sum()
                if batch_nan_count > 0:
                    logger.debug(f"  Batch {batch_idx}: {batch_nan_count} NaN values")

            if continue_on_error:
                logger.warning(
                    f"⚠️  SAVING NaN embedding for {accession_str} (continue-on-error enabled)"
                )
                logger.warning(f"This embedding contains corrupted data!")
            else:
                raise RuntimeError(
                    f"NaN values detected in embedding for sample {accession_str}. Refusing to save corrupted data."
                )

        # Save embedding to .npz file
        embedding_file = self.embeddings_dir / f"{accession_str}.npz"
        logger.debug(f"Checkpoint: Saving embedding to {embedding_file}")
        np.savez_compressed(embedding_file, embedding=embedding)
        logger.debug(f"Checkpoint: Successfully saved embedding for {accession_str}")

        if self.skip_existing_cache and accession_str in self._initial_processed_samples:
            self._initial_processed_samples.remove(accession_str)

        # Add to processed samples set
        self.processed_samples.add(accession_str)

        # Append to processed.csv
        self._append_to_processed_csv(accession_str)
        logger.debug(f"Checkpoint: Added {accession_str} to processed samples")

    def _append_to_processed_csv(self, accession: str) -> None:
        """Append a processed sample to the CSV file with file locking for multi-GPU safety."""
        new_row = {
            "accession": str(accession),  # Ensure string format
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "timestamp": datetime.now().isoformat(),
        }

        # Use file locking to prevent race conditions from multiple GPUs
        try:
            # Check if file exists before opening
            file_exists = self.processed_csv.exists()

            # Open file in append mode with exclusive lock
            with open(self.processed_csv, "a", newline="") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)

                # Write header if this is a new file or empty file
                write_header = not file_exists or f.tell() == 0

                # Append to CSV
                df = pd.DataFrame([new_row])
                df.to_csv(f, header=write_header, index=False)

                # Lock is automatically released when file is closed
        except Exception as e:
            logger = get_logger(__name__)
            logger.warning(f"Failed to append to processed.csv: {e}")

    def load_sample_embedding(self, accession: str) -> Optional[np.ndarray]:
        """Load a sample's embedding from .npz file."""
        embedding_file = self.embeddings_dir / f"{str(accession)}.npz"
        if not embedding_file.exists():
            return None

        try:
            data = np.load(embedding_file)
            return data["embedding"]
        except Exception as e:
            logger = get_logger(__name__)
            logger.warning(f"Failed to load embedding for {accession}: {e}")
            return None

    def get_all_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """Load all embeddings for this model/dataset/split combination."""
        embeddings = []
        accessions = []

        for accession in self.processed_samples:
            embedding = self.load_sample_embedding(accession)
            if embedding is not None:
                embeddings.append(embedding)
                accessions.append(accession)

        if embeddings:
            return np.stack(embeddings), accessions
        else:
            return np.array([]), []

    def get_all_embeddings_from_directory(self) -> Tuple[np.ndarray, List[str]]:
        """Load ALL embeddings from the directory, ignoring model/dataset filtering."""
        embeddings = []
        accessions = []

        logger = get_logger(__name__)

        # Get all .npz files in the embeddings directory
        for embedding_file in self.embeddings_dir.glob("*.npz"):
            accession = embedding_file.stem  # filename without extension
            try:
                data = np.load(embedding_file)
                embedding = data["embedding"]

                # Debug: check shape
                if len(embeddings) < 5:  # Only log first few
                    logger.debug(f"Loading {accession}: shape {embedding.shape}")

                embeddings.append(embedding)
                accessions.append(accession)
            except Exception as e:
                logger.warning(f"Failed to load embedding from {embedding_file}: {e}")
                continue

        if embeddings:
            # Check for shape consistency before stacking
            shapes = [emb.shape for emb in embeddings]
            unique_shapes = set(shapes)
            if len(unique_shapes) > 1:
                logger.error(f"Shape mismatch detected! Found shapes: {unique_shapes}")
                logger.error(f"First few shapes: {shapes[:10]}")
                # Try to fix by squeezing extra dimensions
                fixed_embeddings = []
                for emb in embeddings:
                    if emb.ndim == 3 and emb.shape[0] == 1:
                        fixed_embeddings.append(emb.squeeze(0))
                    else:
                        fixed_embeddings.append(emb)
                embeddings = fixed_embeddings

            return np.stack(embeddings), accessions
        else:
            return np.array([]), []

    def refresh_from_disk(self) -> None:
        """Refresh the processed samples set by reloading from disk.

        This is needed in multi-GPU scenarios where worker processes have saved
        embeddings to disk, but the main process's checkpoint manager needs to
        be updated to reflect the actual state on disk.
        """
        logger = get_logger(__name__)

        # Reload from processed.csv if it exists
        if self.processed_csv.exists():
            try:
                df = pd.read_csv(self.processed_csv)
                # Filter for this specific model/dataset/split combination
                mask = (
                    (df["model_name"] == self.model_name)
                    & (df["dataset_name"] == self.dataset_name)
                    & (df["split"] == self.split)
                )
                filtered_df = df[mask]

                if self.skip_existing_cache and not filtered_df.empty:
                    filtered_df = filtered_df[
                        ~filtered_df["accession"].astype(str).isin(self._initial_processed_samples)
                    ]

                # Update processed samples set
                old_count = len(self.processed_samples)
                self.processed_samples = set(filtered_df["accession"].astype(str))
                new_count = len(self.processed_samples)

                logger.debug(
                    f"Checkpoint: Refreshed from disk - {old_count} -> {new_count} processed samples"
                )

            except Exception as e:
                logger.warning(f"Failed to refresh checkpoint from disk: {e}")
        else:
            logger.debug("Checkpoint: No processed.csv found, no refresh needed")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about processed samples."""
        return {
            "total_processed": len(self.processed_samples),
            "embeddings_dir": str(self.embeddings_dir),
            "processed_csv": str(self.processed_csv),
            "has_processed_samples": len(self.processed_samples) > 0,
            "ignored_processed": (
                len(self._initial_processed_samples) if self.skip_existing_cache else 0
            ),
            "skip_existing_cache": self.skip_existing_cache,
        }


class SimpleResumableDataset(Dataset):
    """Simple wrapper for datasets that supports resumable iteration."""

    def __init__(self, dataset, checkpoint_manager: SimpleCheckpointManager):
        self.dataset = dataset
        self.checkpoint_manager = checkpoint_manager

        # Build filtered indices for unprocessed samples
        logger = get_logger(__name__)
        logger.info("Filtering dataset for unprocessed samples...")

        self.unprocessed_indices: List[int] = []
        self.sample_id_map: Dict[int, str] = {}  # maps filtered index to original accession

        total_samples = len(dataset)

        # All datasets must implement get_all_accessions() for efficient filtering
        logger.info("Loading all accessions for efficient filtering...")
        all_accessions = dataset.get_all_accessions()

        for idx, accession in enumerate(all_accessions):
            if not checkpoint_manager.is_sample_processed(accession):
                filtered_idx = len(self.unprocessed_indices)
                self.unprocessed_indices.append(idx)
                self.sample_id_map[filtered_idx] = accession

        self._remaining = len(self.unprocessed_indices)

        already_processed = total_samples - self._remaining
        logger.info(
            "Dataset filtering complete: %d unprocessed / %d total samples (%d already processed)",
            self._remaining,
            total_samples,
            already_processed,
        )

    def __len__(self):
        return self._remaining

    def __getitem__(self, idx):
        original_idx = self.unprocessed_indices[idx]
        return self.dataset[original_idx]

    def get_accession(self, idx):
        return self.sample_id_map[idx]

    def get_sample_info(self, idx):
        original_idx = self.unprocessed_indices[idx]
        return self.dataset.get_sample_info(original_idx)
