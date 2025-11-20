"""CLI for evaluating embeddings using cached features."""

import argparse
from pathlib import Path
import wandb

# load_config no longer needed - using Hydra loading exclusively
from ..evaluation import EmbeddingEvaluator
from ..common import get_logger, setup_logging

logger = get_logger(__name__)


def evaluate_embeddings_cli():
    """CLI entry point for evaluating embeddings with disease finding classification."""
    # Import Hydra utilities
    from ..hydra_config import (
        get_hydra_overrides_from_env,
        load_config_with_hydra,
        parse_hydra_overrides_from_args,
    )

    parser = argparse.ArgumentParser(
        description="Evaluate embeddings from vision-language (VLM) models using cached features and logistic regression. "
        "Supports Hydra-style overrides: key.subkey=value",
        epilog="Examples:\n"
        "  rate-evaluate --checkpoint-dir cache/model_dataset --dataset-name dataset\n"
        "  rate-evaluate --checkpoint-dir cache/model_dataset --dataset-name dataset evaluation.use_wandb=false\n"
        "  rate-evaluate --checkpoint-dir cache/model_dataset --dataset-name dataset evaluation.pytorch.batch_size=4096",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--model", type=str, required=False, default=None, help="Model name (e.g., 'medgemma')"
    )

    parser.add_argument(
        "--dataset-name", type=str, required=True, help="Name of the dataset used for extraction"
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Path to checkpoint directory containing embeddings (default: cache/{model}_{dataset})",
    )
    parser.add_argument(
        "--labels-json",
        type=str,
        required=False,
        help="Path to JSON file with labels (qa_results format). If not provided, will load from dataset config.",
    )

    # Optional arguments
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--pool-op",
        type=str,
        default="mean",
        choices=["mean", "max", "median"],
        help="Pooling operation used for feature extraction",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--check-nan",
        action="store_true",
        default=False,
        help="Enable detailed NaN checking and logging during evaluation (may impact performance)",
    )

    # Allow trailing Hydra-style overrides (key=value) exactly as shown in the
    # CLI examples by stripping them before running argparse.
    import sys

    regular_args, cli_hydra_overrides = parse_hydra_overrides_from_args(sys.argv[1:])
    args = parser.parse_args(regular_args)

    # Set default checkpoint directory if not provided
    if args.checkpoint_dir is None:
        assert (
            args.model is not None
        ), "Model name is required when checkpoint directory is not provided"
        args.checkpoint_dir = f"cache/{args.model}_{args.dataset_name}"

    # Determine model name: use CLI argument if provided, otherwise extract from checkpoint directory
    if args.model is not None:
        model_name = args.model
        logger.info(f"Using model name from CLI argument: {model_name}")
    else:
        # Extract model name from checkpoint directory path
        # Checkpoint directory follows pattern: cache/{model}_{dataset}
        checkpoint_path = Path(args.checkpoint_dir)

        # Known model names to check against (in order of preference)
        known_models = [
            "medgemma",
            "medimageinsight",
            "pillar0",
            "merlin",
            "ctclip",
            "lingshu",
        ]

        # Try to find the model name by checking against known models
        model_name = "unknown_model"
        checkpoint_name = checkpoint_path.name

        for model in known_models:
            if checkpoint_name.startswith(f"{model}_"):
                model_name = model
                break

        # Fallback: if no known model found, try to extract from first underscore
        if model_name == "unknown_model" and "_" in checkpoint_name:
            model_name = checkpoint_name.split("_")[0]

        logger.info(
            f"Extracted model name: {model_name} from checkpoint directory: {checkpoint_name}"
        )

    # Generate dynamic experiment name from model and dataset (only if not specified in config)
    dynamic_exp_name = f"{model_name}_{args.dataset_name}"
    logger.info(f"Generated dynamic experiment name: {dynamic_exp_name}")

    # Validate input directory
    checkpoint_dir = Path(args.checkpoint_dir)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise ValueError(f"Path is not a directory: {checkpoint_dir}")

    logger.info("Starting embedding evaluation")
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Pooling: {args.pool_op}")
    logger.info(f"Checkpoint directory: {args.checkpoint_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    # Combine overrides provided via the Hydra wrapper environment variable with
    # any trailing CLI overrides collected above.
    hydra_overrides = list(get_hydra_overrides_from_env())
    for override in cli_hydra_overrides:
        if override not in hydra_overrides:
            hydra_overrides.append(override)

    # Add check_nan argument as Hydra override
    if hasattr(args, "check_nan") and args.check_nan:
        hydra_overrides.append("evaluation.check_nan=true")

    # Load configuration with Hydra to ensure proper config merging
    from omegaconf import OmegaConf

    # Always use Hydra loading to get proper base config merging
    if hydra_overrides:
        logger.info(f"Applying Hydra overrides: {hydra_overrides}")
    else:
        logger.info("Loading configuration without overrides")

    config = load_config_with_hydra(
        config_name="config" if args.config == "configs/default.yaml" else Path(args.config).stem,
        overrides=hydra_overrides,
    )

    # check_nan is handled via Hydra overrides if needed

    # Log final merged config
    logger.info("Final merged configuration:")
    logger.info("\n" + OmegaConf.to_yaml(config))

    # Setup logging from config
    logging_config = getattr(config, "logging", {})
    setup_logging(
        level=getattr(logging_config, "level", "INFO"),
        format_str=getattr(logging_config, "format", None),
    )

    # Load labels path from dataset config if not provided
    labels_json_path = args.labels_json
    if not labels_json_path:
        from ..config import load_dataset_config

        dataset_config = load_dataset_config(args.dataset_name)
        labels_json_path = getattr(getattr(dataset_config, "labels", None), "labels_json", None)
        if not labels_json_path:
            raise ValueError(
                f"No labels_json found in dataset config for {args.dataset_name}. "
                f"Please specify --labels-json or add 'data.labels_json' to configs/datasets/{args.dataset_name}.yaml"
            )
        logger.info(f"Using labels from dataset config: {labels_json_path}")

    # Initialize WandB if enabled
    eval_config = getattr(config, "evaluation", {})
    use_wandb = getattr(eval_config, "use_wandb", True)
    entity = getattr(eval_config, "wandb_entity", "yala-lab")
    wandb_project = getattr(eval_config, "wandb_project", "rate-eval")

    # Use config exp_name if specified, otherwise use dynamic generation
    exp_name = getattr(eval_config, "exp_name", None)
    if exp_name is None:
        exp_name = dynamic_exp_name
        logger.info(f"Using dynamically generated experiment name: {exp_name}")
    else:
        logger.info(f"Using configured experiment name: {exp_name}")

    if use_wandb:
        wandb.init(
            entity=entity,
            project=wandb_project,
            name=exp_name,
            config={
                "model_name": model_name,
                "dataset_name": args.dataset_name,
                "pool_op": args.pool_op,
                "checkpoint_dir": str(checkpoint_dir),
                "output_dir": args.output_dir,
                "config_file": args.config,
                "labels_json": labels_json_path,
            },
            reinit=True,
        )
        logger.info(f"WandB initialized for evaluation pipeline with experiment name: {exp_name}")

    # Initialize evaluator
    evaluator = EmbeddingEvaluator(config)

    # Run full evaluation pipeline
    try:
        results = evaluator.run_full_evaluation(
            checkpoint_dir=str(checkpoint_dir),
            dataset_name=args.dataset_name,
            labels_json_path=labels_json_path,
            pool_op=args.pool_op,
            output_dir=args.output_dir,
        )

        # Print final summary
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Total findings: {results['summary_stats']['total_findings']}")
        logger.info(f"Evaluated findings: {results['summary_stats']['evaluated_findings']}")

        if results["summary_stats"]:
            mean_auc = results["summary_stats"].get("avg_auc", 0)
            mean_f1 = results["summary_stats"].get("avg_f1", 0)
            logger.info(f"Average AUC: {mean_auc:.3f}")
            logger.info(f"Average F1: {mean_f1:.3f}")

        logger.info(f"Results saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise
    finally:
        if use_wandb:
            wandb.finish()
            logger.info("WandB finished for evaluation pipeline")


def main():
    """Main entry point that supports Hydra overrides."""
    from ..hydra_config import create_hydra_compatible_cli

    hydra_cli = create_hydra_compatible_cli(evaluate_embeddings_cli)
    hydra_cli()


if __name__ == "__main__":
    main()
