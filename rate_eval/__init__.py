"""RATE Evaluation Package

A comprehensive toolkit for evaluating Vision-Language Models on medical imaging tasks.
"""

from .config import load_config, get_config_value, setup_pipeline as setup_pipeline_new
from .components import create_model, create_dataset
from .evaluation import EmbeddingEvaluator
from .common import setup_logging, get_logger, setup_device

# Import models and datasets for component factory
from . import models
from . import datasets

__version__ = "2.0.0"
__author__ = "RATE Evaluation Team"

# Public API
__all__ = [
    # Configuration
    "load_config",
    "get_config_value",
    # Component factories
    "create_model",
    "create_dataset",
    # Evaluation
    "EmbeddingEvaluator",
    # Common utilities
    "setup_logging",
    "get_logger",
    "setup_device",
]


def setup_pipeline(config_path=None, log_level="INFO", **overrides):
    """
    Initialize the RATE evaluation pipeline with OmegaConf support.

    Args:
        config_path: Path to configuration file
        log_level: Logging level (deprecated - use config file)
        **overrides: Configuration overrides

    Returns:
        Loaded OmegaConf configuration
    """
    # Use new OmegaConf-based setup
    config = setup_pipeline_new(config_path=config_path, **overrides)

    # Setup logging from config
    logging_config = getattr(config, "logging", {})
    setup_logging(
        level=getattr(logging_config, "level", log_level),
        format_str=getattr(logging_config, "format", None),
    )

    logger = get_logger(__name__)
    logger.info(f"RATE Evaluation Pipeline v{__version__} initialized with OmegaConf")

    return config
