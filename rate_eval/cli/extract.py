"""CLI for extracting embeddings from models with multi-GPU support and graceful resume."""

import argparse
import torch
import torch.multiprocessing as mp
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import time
import warnings
import os

# Suppress warnings early
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*cuBLAS.*")
warnings.filterwarnings("ignore", message=".*cuDNN.*")
warnings.filterwarnings("ignore", message=".*CUDA.*factory.*")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Suppress CUDA initialization warnings that appear on stderr
import sys
import contextlib


@contextlib.contextmanager
def suppress_cuda_warnings():
    """Context manager to suppress CUDA stderr warnings."""
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


from .. import setup_pipeline, create_model, create_dataset
from ..common import get_logger, SimpleCheckpointManager, SimpleResumableDataset

logger = get_logger(__name__)


class ModelPreprocessor:
    """A picklable wrapper for model preprocessing."""

    def __init__(self, model_cls, model_config):
        self.model_cls = model_cls
        self.model_config = model_config
        self._model_instance = None

    def __call__(self, image, **kwargs):
        preprocess_method = getattr(self.model_cls, "preprocess_single", None)
        if preprocess_method is None:
            raise AttributeError(
                f"Model {self.model_cls.__name__} does not have preprocess_single method"
            )

        return self.model_cls.preprocess_single(image, model_config=self.model_config, **kwargs)


def extract_embeddings_single_gpu(
    rank, world_size, args, config, dataset_size, checkpoint_manager=None
):
    """Extract embeddings on a single GPU with checkpoint support."""
    try:
        # Setup logging in worker process (needed for multiprocessing)
        if world_size > 1:
            from ..common import setup_logging

            log_level = "DEBUG" if args.debug else "INFO"
            setup_logging(level=log_level, rank=rank, world_size=world_size, debug=args.debug)

        # Set GPU device for multi-GPU, or use specified device for single GPU
        if world_size > 1:
            device = f"cuda:{rank}"
            torch.cuda.set_device(rank)
        else:
            device = args.device or getattr(getattr(config, "hardware", None), "device", "cuda:0")

        # Update config for this GPU
        from omegaconf import OmegaConf

        local_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        OmegaConf.update(local_config, "hardware.device", device)

        logger.info(f"GPU {rank}: Loading model {args.model} on {device}")
        logger.debug(f"GPU {rank}: Model config for {args.model}: device={device}")

        # Setup model
        model_load_start = time.time()
        model = create_model(args.model, local_config)
        model_load_time = time.time() - model_load_start
        print(f"[TIMING] GPU {rank}: Model loading took {model_load_time:.2f}s")
        logger.debug(f"GPU {rank}: Model {args.model} loaded successfully")

        # Setup dataset - use resumable dataset if checkpoint manager provided
        # Create a picklable preprocessor if the model supports it
        dataset_create_start = time.time()
        if hasattr(model, "preprocess_single") and hasattr(model, "model_config"):
            model_preprocess = ModelPreprocessor(model.__class__, model.model_config)
        else:
            model_preprocess = None

        original_dataset = create_dataset(
            args.dataset, local_config, args.split, model_preprocess=model_preprocess
        )
        dataset_create_time = time.time() - dataset_create_start
        print(f"[TIMING] GPU {rank}: Dataset creation took {dataset_create_time:.2f}s")

        # Use modality from config if provided, otherwise use dataset default
        modality = getattr(getattr(config, "dataset", None), "modality", original_dataset.modality)
        if (
            hasattr(config, "dataset")
            and hasattr(config.dataset, "modality")
            and config.dataset.modality != original_dataset.modality
        ):
            logger.info(
                f"GPU {rank}: Overriding dataset modality '{original_dataset.modality}' with '{modality}'"
            )
        original_dataset.modality = modality

        # Apply resume logic if checkpoint manager is available
        # Note: Each GPU will independently check which samples are unprocessed
        if checkpoint_manager:
            dataset = SimpleResumableDataset(original_dataset, checkpoint_manager)
            logger.info(f"GPU {rank}: Resume mode: {len(dataset)} unprocessed samples remaining")
        else:
            dataset = original_dataset

        # Calculate subset for this GPU
        if world_size > 1:
            samples_per_gpu = len(dataset) // world_size
            start_idx = rank * samples_per_gpu

            # Handle remainder samples on last GPU
            if rank == world_size - 1:
                end_idx = len(dataset)
            else:
                end_idx = start_idx + samples_per_gpu

            # Apply max_samples limit if specified
            if args.max_samples:
                total_samples = min(len(dataset), args.max_samples)
                samples_per_gpu = total_samples // world_size
                start_idx = rank * samples_per_gpu

                if rank == world_size - 1:
                    end_idx = total_samples
                else:
                    end_idx = start_idx + samples_per_gpu

                # Skip if this GPU has no samples to process
                if start_idx >= total_samples:
                    logger.info(f"GPU {rank}: No samples to process")
                    return [], []

            # Create subset for this GPU
            subset_indices = list(range(start_idx, min(end_idx, len(dataset))))
            if not subset_indices:
                logger.info(f"GPU {rank}: No samples in subset")
                return [], []

            subset = Subset(dataset, subset_indices)

            logger.info(
                f"GPU {rank}: Processing {len(subset)} samples (indices {start_idx}-{end_idx-1})"
            )
        else:
            # Single GPU mode - process all samples
            subset = dataset
            subset_indices = list(range(len(dataset)))

            # Apply max_samples limit
            if args.max_samples and args.max_samples < len(dataset):
                subset_indices = subset_indices[: args.max_samples]
                subset = Subset(dataset, subset_indices)

            logger.info(f"Processing {len(subset)} samples on {device}")

        # Create data loader
        batch_size = getattr(getattr(config, "hardware", None), "batch_size_per_gpu", 4)
        num_workers = min(getattr(getattr(config, "hardware", None), "num_workers_per_gpu", 4), 8)

        data_loader = DataLoader(
            subset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True
        )

        # Extract embeddings with checkpoint saving
        embeddings = []
        accessions = []

        if hasattr(model, "eval"):
            model.eval()

        with torch.no_grad():
            if world_size > 1:
                pbar = tqdm(data_loader, desc=f"GPU {rank}", position=rank, leave=True)
            else:
                pbar = tqdm(data_loader, desc="Extracting embeddings")

            for batch_idx, volumes in enumerate(pbar):
                batch_start_time = time.time()
                try:
                    if isinstance(volumes, list):
                        extra_infos = volumes[1]
                        volumes = volumes[0]
                    else:
                        extra_infos = None

                    data_load_time = time.time() - batch_start_time
                    print(
                        f"[TIMING] GPU {rank} Batch {batch_idx}: Data loading took {data_load_time:.2f}s"
                    )

                    logger.debug(
                        f"GPU {rank}: Starting batch {batch_idx}, received volume shape: {volumes.shape}, dtype: {volumes.dtype}"
                    )

                    # Check for empty batch
                    if volumes.shape[0] == 0:
                        logger.warning(
                            f"GPU {rank}: Batch {batch_idx} - Empty batch received, skipping"
                        )
                        continue

                    # Check volume dimensions before processing
                    logger.debug(
                        f"GPU {rank}: Batch {batch_idx} - Raw volume shape: {volumes.shape}"
                    )
                    if len(volumes.shape) < 4:
                        logger.error(
                            f"GPU {rank}: Batch {batch_idx} - Invalid volume shape {volumes.shape}, expected at least 4D"
                        )
                        if args.continue_on_error:
                            logger.warning(
                                f"⚠️  CONTINUING after invalid shape (--continue-on-error enabled)"
                            )
                            continue
                        else:
                            raise ValueError(f"Invalid volume shape: {volumes.shape}")

                    # Get accession IDs for this batch
                    batch_accessions = []
                    for i in range(len(volumes)):
                        if world_size > 1:
                            # Multi-GPU: calculate original dataset index
                            if hasattr(dataset, "get_accession"):
                                original_idx = subset_indices[batch_idx * batch_size + i]
                                batch_accessions.append(dataset.get_accession(original_idx))
                            else:
                                # Single GPU: direct calculation
                                original_idx = batch_idx * batch_size + i
                                if original_idx >= len(dataset):
                                    break
                                batch_accessions.append(dataset.get_accession(original_idx))
                        else:
                            # Single GPU case
                            original_idx = batch_idx * batch_size + i
                            if original_idx >= len(dataset):
                                break
                            batch_accessions.append(dataset.get_accession(original_idx))

                    logger.debug(
                        f"GPU {rank}: Batch {batch_idx} - Processing {len(batch_accessions)} samples: {batch_accessions}"
                    )

                    # Handle different input shapes based on preprocessing
                    if len(volumes.shape) == 4:  # (B, C, H, W) - 2D images
                        logger.debug(f"GPU {rank}: Batch {batch_idx} - 2D input format")
                    elif len(volumes.shape) == 5:  # (B, C, D, H, W) - 3D volumes
                        logger.debug(f"GPU {rank}: Batch {batch_idx} - 3D volume format")
                    elif len(volumes.shape) == 6:  # (B, K, C, D, H, W) - multiple 3D volumes
                        logger.debug(f"GPU {rank}: Batch {batch_idx} - Multiple 3D volumes format")
                    else:
                        logger.error(
                            f"GPU {rank}: Batch {batch_idx} - Unexpected volume shape: {volumes.shape}"
                        )
                        if args.continue_on_error:
                            logger.warning(
                                f"⚠️  CONTINUING after unexpected shape (--continue-on-error enabled)"
                            )
                            continue
                        else:
                            raise ValueError(f"Unexpected volume shape: {volumes.shape}")

                    # Move to device and ensure float type
                    try:
                        transfer_start = time.time()
                        volumes = volumes.to(device)
                        transfer_time = time.time() - transfer_start
                        print(
                            f"[TIMING] GPU {rank} Batch {batch_idx}: GPU transfer took {transfer_time:.2f}s"
                        )
                        logger.debug(f"GPU {rank}: Batch {batch_idx} - Moved to device {device}")

                        # Check if tensor is empty after moving to device
                        if volumes.numel() == 0:
                            logger.error(
                                f"GPU {rank}: Batch {batch_idx} - Empty tensor after moving to device"
                            )
                            if args.continue_on_error:
                                logger.warning(
                                    f"⚠️  CONTINUING after empty tensor (--continue-on-error enabled)"
                                )
                                continue
                            else:
                                raise RuntimeError("Empty tensor after moving to device")

                        volumes = volumes.float()
                        logger.debug(f"GPU {rank}: Batch {batch_idx} - Converted to float")

                        # Log basic statistics (only if tensor has elements)
                        if volumes.numel() > 0:
                            logger.debug(
                                f"GPU {rank}: Batch {batch_idx} - Input stats: min={volumes.min():.4f}, max={volumes.max():.4f}, mean={volumes.mean():.4f}"
                            )
                        else:
                            logger.error(
                                f"GPU {rank}: Batch {batch_idx} - Tensor is empty, cannot compute stats"
                            )
                            continue
                    except Exception as tensor_error:
                        logger.error(
                            f"GPU {rank}: Batch {batch_idx} - Error processing tensor: {tensor_error}"
                        )
                        if args.continue_on_error:
                            logger.warning(
                                f"⚠️  CONTINUING after tensor error (--continue-on-error enabled)"
                            )
                            continue
                        else:
                            raise

                    # NaN checking if enabled
                    if args.check_nan and volumes.numel() > 0:
                        input_nan_count = torch.isnan(volumes).sum().item()
                        logger.debug(
                            f"GPU {rank}: Batch {batch_idx} - Input NaN check: {input_nan_count} NaN values found"
                        )

                        if input_nan_count > 0:
                            logger.error(
                                f"🚨 INPUT NaN DETECTED: {input_nan_count} NaN values found in input batch {batch_idx} on GPU {rank}"
                            )
                            logger.error(f"Input batch shape: {volumes.shape}")
                            logger.error(f"Problematic samples in this batch: {batch_accessions}")

                            if args.continue_on_error:
                                logger.warning(
                                    f"⚠️  CONTINUING despite INPUT NaN (--continue-on-error enabled)"
                                )
                                logger.warning(
                                    f"This may result in corrupted embeddings being saved!"
                                )
                            else:
                                raise RuntimeError(
                                    f"INPUT NaN DETECTED: Found {input_nan_count} NaN values in input data at batch {batch_idx}. Problematic samples: {batch_accessions}"
                                )

                    # Extract features
                    logger.debug(
                        f"GPU {rank}: Batch {batch_idx} - Calling model.extract_features with modality='{modality}'"
                    )
                    inference_start = time.time()
                    if extra_infos is not None:
                        batch_embeddings = model.extract_features(
                            volumes, modality=modality, extra_infos=extra_infos
                        )
                    else:
                        batch_embeddings = model.extract_features(volumes, modality=modality)
                    inference_time = time.time() - inference_start
                    print(
                        f"[TIMING] GPU {rank} Batch {batch_idx}: Model inference took {inference_time:.2f}s"
                    )
                    logger.debug(
                        f"GPU {rank}: Batch {batch_idx} - Extracted embeddings shape: {batch_embeddings.shape}, dtype: {batch_embeddings.dtype}"
                    )

                    # Check embeddings validity
                    if batch_embeddings.size == 0:
                        logger.error(
                            f"GPU {rank}: Batch {batch_idx} - Model returned empty embeddings"
                        )
                        if args.continue_on_error:
                            logger.warning(
                                f"⚠️  CONTINUING after empty embeddings (--continue-on-error enabled)"
                            )
                            continue
                        else:
                            raise RuntimeError("Model returned empty embeddings")

                    # Log embedding statistics (only if embeddings exist)
                    if batch_embeddings.size > 0:
                        logger.debug(
                            f"GPU {rank}: Batch {batch_idx} - Embedding stats: min={batch_embeddings.min():.4f}, max={batch_embeddings.max():.4f}, mean={batch_embeddings.mean():.4f}"
                        )

                        # Check for NaN values in embeddings
                        nan_count = np.isnan(batch_embeddings).sum()
                        logger.debug(
                            f"GPU {rank}: Batch {batch_idx} - NaN check: {nan_count} out of {batch_embeddings.size} elements are NaN"
                        )

                        if nan_count > 0:
                            logger.error(
                                f"🚨 MODEL OUTPUT NaN DETECTED: {nan_count} NaN values in embeddings at batch {batch_idx} on GPU {rank}"
                            )
                            logger.error(
                                f"❗ INPUT WAS CLEAN - NaN originated from MODEL PROCESSING"
                            )
                            logger.error(f"Output batch shape: {batch_embeddings.shape}")
                            logger.error(f"Affected samples in this batch: {batch_accessions}")

                            if args.continue_on_error:
                                logger.warning(
                                    f"⚠️  CONTINUING despite MODEL NaN (--continue-on-error enabled)"
                                )
                                logger.warning(
                                    f"This may result in corrupted embeddings being saved!"
                                )
                            else:
                                raise RuntimeError(
                                    f"MODEL NaN DETECTED: Model generated {nan_count} NaN values in embeddings. Input was clean - issue is in model processing."
                                )

                    # Store results
                    embeddings.append(batch_embeddings)
                    accessions.extend(batch_accessions)

                    # Save individual samples if checkpoint manager available
                    if checkpoint_manager:
                        checkpoint_start = time.time()
                        logger.debug(
                            f"GPU {rank}: Batch {batch_idx} - Saving {len(batch_accessions)} samples to checkpoint"
                        )
                        for i, accession in enumerate(batch_accessions):
                            if not checkpoint_manager.is_sample_processed(accession):
                                sample_embedding = batch_embeddings[
                                    i : i + 1
                                ]  # Keep batch dimension
                                logger.debug(
                                    f"GPU {rank}: Batch {batch_idx} - Saving sample {i} (accession: {accession}), embedding shape: {sample_embedding.shape}"
                                )
                                checkpoint_manager.save_sample_embedding(
                                    accession,
                                    sample_embedding,
                                    continue_on_error=args.continue_on_error,
                                )
                            else:
                                logger.debug(
                                    f"GPU {rank}: Batch {batch_idx} - Skipping already processed sample {i} (accession: {accession})"
                                )
                        checkpoint_time = time.time() - checkpoint_start
                        print(
                            f"[TIMING] GPU {rank} Batch {batch_idx}: Checkpoint saving took {checkpoint_time:.2f}s"
                        )

                    batch_total_time = time.time() - batch_start_time
                    print(
                        f"[TIMING] GPU {rank} Batch {batch_idx}: Total batch time {batch_total_time:.2f}s"
                    )
                    logger.debug(
                        f"GPU {rank}: Batch {batch_idx} - Successfully processed {len(batch_accessions)} samples"
                    )
                except Exception as batch_error:
                    logger.error(f"GPU {rank}: Error processing batch {batch_idx}: {batch_error}")
                    if hasattr(batch_error, "__traceback__"):
                        import traceback

                        logger.debug(
                            f"GPU {rank}: Batch {batch_idx} traceback: {''.join(traceback.format_tb(batch_error.__traceback__))}"
                        )

                    if args.continue_on_error:
                        logger.warning(
                            f"⚠️  CONTINUING after batch error (--continue-on-error enabled)"
                        )
                        logger.warning(
                            f"Batch {batch_idx} will be skipped, but extraction continues"
                        )
                        continue  # Skip this batch and continue with next one
                    else:
                        raise  # Re-raise the exception to stop processing

        # Concatenate embeddings for this GPU
        if embeddings:
            gpu_embeddings = np.concatenate(embeddings, axis=0)
            logger.info(f"GPU {rank}: Extracted {gpu_embeddings.shape[0]} embeddings")
            return gpu_embeddings, accessions
        else:
            logger.error(f"GPU {rank}: No embeddings extracted")
            return np.array([]), []

    except Exception as e:
        logger.error(f"GPU {rank}: Error during extraction: {e}")
        import traceback

        traceback.print_exc()
        return np.array([]), []


def _worker_process(rank, num_gpus, args, config, dataset_size, checkpoint_manager, results_queue):
    """Worker function for multiprocessing - must be picklable."""
    result = extract_embeddings_single_gpu(
        rank, num_gpus, args, config, dataset_size, checkpoint_manager
    )
    results_queue.put((rank, *result))


def extract_embeddings(args, config, checkpoint_manager=None):
    """Extract embeddings using available GPUs with checkpoint support."""
    # Get available GPUs (inline the simple function)
    if torch.cuda.is_available():
        available_gpus = list(range(torch.cuda.device_count()))
    else:
        available_gpus = []

    # Determine number of GPUs to use
    if not available_gpus:
        logger.warning("No GPUs available, using CPU")
        num_gpus = 1
        gpu_ids = ["cpu"]
    else:
        if args.num_gpus:
            num_gpus = min(args.num_gpus, len(available_gpus))
        else:
            num_gpus = len(available_gpus)
        gpu_ids = available_gpus[:num_gpus]

    logger.info(f"Using {num_gpus} {'GPU' if num_gpus == 1 else 'GPUs'}: {gpu_ids}")

    # Load dataset to get size (without loading data)
    # Note: We don't pass model_preprocess here since this is just for getting size
    dataset = create_dataset(args.dataset, config, args.split)
    dataset_size = len(dataset)

    if args.max_samples:
        dataset_size = min(dataset_size, args.max_samples)

    logger.info(f"Total samples to process: {dataset_size}")

    # Single GPU/CPU case - no multiprocessing needed
    if num_gpus == 1:
        return extract_embeddings_single_gpu(0, 1, args, config, dataset_size, checkpoint_manager)

    # Multi-GPU case - use multiprocessing (all GPUs will save their embeddings)
    if checkpoint_manager:
        logger.info("Multi-GPU checkpoint mode: All GPUs will save their processed embeddings")

    mp.set_start_method("spawn", force=True)

    # Create processes for each GPU
    processes = []
    results_queue = mp.Queue()

    for rank in range(num_gpus):
        # Pass checkpoint manager to all GPUs so they can save their embeddings
        p = mp.Process(
            target=_worker_process,
            args=(rank, num_gpus, args, config, dataset_size, checkpoint_manager, results_queue),
        )
        p.start()
        processes.append(p)

    # Collect results
    all_embeddings = []
    all_accessions = []

    for _ in range(num_gpus):
        rank, embeddings, accessions = results_queue.get()
        if len(embeddings) > 0:
            all_embeddings.append(embeddings)
            all_accessions.extend(accessions)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Combine results from all GPUs
    if all_embeddings:
        combined_embeddings = np.concatenate(all_embeddings, axis=0)
        logger.info(f"Combined embeddings from all GPUs: {combined_embeddings.shape}")
        return combined_embeddings, all_accessions
    else:
        return np.array([]), []


def extract_embeddings_cli():
    """CLI entry point for extracting embeddings with checkpoint support."""
    # Import Hydra utilities
    from ..hydra_config import (
        get_hydra_overrides_from_env,
        load_config_with_hydra,
        parse_hydra_overrides_from_args,
    )

    parser = argparse.ArgumentParser(
        description="Extract embeddings from vision-language (VLM) models with graceful resume. "
        "Supports Hydra-style overrides: key.subkey=value",
        epilog="Examples:\n"
        "  rate-extract --model medgemma --dataset dummy\n"
        "  rate-extract --model medgemma --dataset dummy hardware.batch_size_per_gpu=32\n"
        "  rate-extract --model medgemma --dataset dummy model.extraction.pool_op=max",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., 'medgemma')")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name (e.g., 'abd_ct_merlin', 'dummy')"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "valid", "test"],
        help="Dataset split to process",
    )
    parser.add_argument(
        "--all-splits",
        action="store_true",
        help="Process all available splits (train, valid, test)",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for embeddings (default: cache/{model}_{dataset})",
    )
    parser.add_argument(
        "--skip-existing-cache",
        action="store_true",
        default=False,
        help="Ignore previously cached embeddings and process every sample from scratch",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size per GPU (overrides config)"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use for single GPU mode"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=None, help="Number of GPUs to use (default: all available)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable DEBUG level logging for detailed tracing (helps debug NaN issues)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=False,
        help="Continue extraction even when errors occur (logs error but doesn't stop). WARNING: May result in incomplete or corrupted data!",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of dataloader workers per GPU (overrides config, default: 4)",
    )
    parser.add_argument(
        "--check-nan",
        action="store_true",
        default=False,
        help="Enable detailed NaN checking and logging (may impact performance)",
    )
    parser.add_argument(
        "--model-repo-id",
        type=str,
        default=None,
        help="Override model repository ID (e.g., 'YalaLab/model_name')",
    )
    parser.add_argument(
        "--model-revision",
        type=str,
        default=None,
        help="Override model revision (e.g., 'epoch_16')",
    )
    parser.add_argument(
        "--ct-window-type",
        type=str,
        default=None,
        help="CT window type for CT windowing (e.g., 'minmax', 'lung', 'all') (default: use config default)",
    )
    parser.add_argument(
        "--pool-op",
        type=str,
        default=None,
        help="Pooling operation to use (e.g., 'mean', 'max', 'cls') (default: use config default)",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default=None,
        help="Modality to use for feature extraction (e.g., 'abdomen_ct', 'chest_xray_two_view') (default: use dataset default)",
    )

    # Support Hydra-style overrides passed directly on the command line by
    # separating trailing key=value pairs before argparse runs.
    import sys

    regular_args, cli_hydra_overrides = parse_hydra_overrides_from_args(sys.argv[1:])
    args = parser.parse_args(regular_args)

    # Setup logging level based on debug flag or environment variable
    from ..common import setup_logging

    # Check for environment variable first, then command line flag
    log_level = os.environ.get("RATE_LOG_LEVEL", "INFO").upper()
    if args.debug:
        log_level = "DEBUG"

    # For main process, assume rank 0 behavior (INFO on rank 0, WARN on others handled in workers)
    setup_logging(level=log_level, rank=0, world_size=1, debug=args.debug)

    if log_level == "DEBUG":
        logger.debug("Debug mode active")

    # Warn about continue-on-error flag
    if args.continue_on_error:
        logger.warning("⚠️  WARNING: --continue-on-error flag is enabled!")
        logger.warning("Extraction will continue even when errors occur.")
        logger.warning("This may result in incomplete or corrupted data being saved.")
        logger.warning("Use this flag only for debugging purposes.")

    # Set default output directory if not provided
    if args.output_dir is None:
        args.output_dir = f"cache/{args.model}_{args.dataset}"

    # Determine which splits to process
    if args.all_splits:
        splits_to_process = ["train", "valid", "test"]
        logger.info("Processing all splits: train, valid, test")
    else:
        splits_to_process = [args.split]

    # Combine overrides from the Hydra wrapper environment variable with any
    # CLI trailing overrides captured above.
    hydra_overrides = list(get_hydra_overrides_from_env())
    for override in cli_hydra_overrides:
        if override not in hydra_overrides:
            hydra_overrides.append(override)

    # Initialize pipeline with model and dataset overrides plus Hydra overrides
    if hydra_overrides:
        logger.info(f"Applying Hydra overrides: {hydra_overrides}")
        config = load_config_with_hydra(
            config_name="config" if args.config is None else Path(args.config).stem,
            model_name=args.model,
            dataset_name=args.dataset,
            overrides=hydra_overrides,
        )
    else:
        # Fallback to standard setup_pipeline for backward compatibility
        config = setup_pipeline(config_path=args.config, model=args.model, dataset=args.dataset)

    # Print config for debugging if requested
    from omegaconf import OmegaConf

    if log_level == "DEBUG":
        logger.debug("=" * 80)
        logger.debug("FULL CONFIGURATION:")
        logger.debug("=" * 80)
        logger.debug(OmegaConf.to_yaml(config))
        logger.debug("=" * 80)

    # Override config with CLI arguments using OmegaConf
    if args.batch_size:
        OmegaConf.update(config, "hardware.batch_size_per_gpu", args.batch_size)
    if args.device:
        OmegaConf.update(config, "hardware.device", args.device)
    if args.num_workers is not None:
        OmegaConf.update(config, "hardware.num_workers_per_gpu", args.num_workers)
    if args.model_repo_id is not None:
        # Unified repo_id path for all models
        OmegaConf.update(config, "model.repo_id", args.model_repo_id)
    if args.model_revision is not None:
        # Unified revision path for all models
        OmegaConf.update(config, "model.revision", args.model_revision)
    if args.ct_window_type is not None:
        OmegaConf.update(config, "model.preprocessing.ct.window_type", args.ct_window_type)
    if args.pool_op is not None:
        OmegaConf.update(config, "model.extraction.pool_op", args.pool_op)
    if args.modality is not None:
        OmegaConf.update(config, "dataset.modality", args.modality)

    # Process each split
    for split in splits_to_process:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing split: {split}")
        logger.info(f"{'='*60}")

        # Create args copy with current split
        current_args = argparse.Namespace(**vars(args))
        current_args.split = split

        # Initialize checkpoint manager (always enabled)
        checkpoint_manager = SimpleCheckpointManager(
            model_name=args.model,
            dataset_name=args.dataset,
            split=split,
            cache_dir=args.output_dir,
            skip_existing_cache=args.skip_existing_cache,
        )

        # Show resume stats
        stats = checkpoint_manager.get_stats()
        logger.info(f"Resume stats for {split}: {stats}")

        if stats["has_processed_samples"]:
            logger.info(
                f"Resuming {split} extraction: {stats['total_processed']} samples already processed"
            )
        elif stats.get("ignored_processed"):
            logger.info(
                "Skip-existing-cache enabled: ignoring %d cached samples for this run",
                stats["ignored_processed"],
            )
        else:
            logger.info(
                f"No previous processed samples found for {split}, starting fresh extraction"
            )

        start_time = time.time()

        # Extract embeddings for current split
        try:
            all_embeddings, accessions = extract_embeddings(
                current_args, config, checkpoint_manager
            )
        except Exception as e:
            logger.error(f"Failed to extract embeddings for split {split}: {e}")
            continue

        extraction_time = time.time() - start_time

        # Refresh checkpoint manager from disk to get accurate stats after multi-GPU processing
        checkpoint_manager.refresh_from_disk()

        # Report extraction results
        stats = checkpoint_manager.get_stats()
        logger.info(f"Extraction completed for {split} in {extraction_time:.2f} seconds")
        logger.info(f"Total samples processed: {stats['total_processed']}")
        logger.info(f"Embeddings saved to checkpoint directory: {args.output_dir}")
        if extraction_time > 0:
            logger.info(f"Throughput: {stats['total_processed']/extraction_time:.2f} samples/sec")

        if stats["total_processed"] == 0:
            logger.error(f"No embeddings were extracted for split {split}!")

    logger.info(f"\n{'='*60}")
    logger.info("All splits processing completed!")
    logger.info(f"{'='*60}")


def main():
    """Main entry point that supports Hydra overrides."""
    from ..hydra_config import create_hydra_compatible_cli

    hydra_cli = create_hydra_compatible_cli(extract_embeddings_cli)
    hydra_cli()


if __name__ == "__main__":
    main()
