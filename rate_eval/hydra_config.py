"""Hydra-enabled configuration management for rate-eval."""

import os
from pathlib import Path
from typing import List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from .common import get_logger

logger = get_logger(__name__)


def setup_hydra_config_dir():
    """Set up Hydra configuration directory."""
    project_root = Path(__file__).parent.parent
    config_dir = project_root / "configs"
    return str(config_dir)


def load_config_with_hydra(
    config_name: str = "config",
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    return_hydra_cfg: bool = False,
) -> DictConfig:
    """
    Load configuration using Hydra with full override support.

    Args:
        config_name: Name of the main config file (without .yaml)
        model_name: Model to use (overrides default in config)
        dataset_name: Dataset to use (overrides default in config)
        overrides: List of Hydra-style overrides like ['key=value', 'nested.key=value']
        return_hydra_cfg: If True, return raw Hydra config, else process for rate-eval

    Returns:
        OmegaConf DictConfig object
    """
    config_dir = setup_hydra_config_dir()
    override_list = list(overrides or [])

    if model_name:
        override_list.append(f"model={model_name}")
    if dataset_name:
        override_list.append(f"dataset={dataset_name}")

    _clear_global_hydra()

    try:
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name=config_name, overrides=override_list)

            if return_hydra_cfg:
                return cfg

            cfg_dict = OmegaConf.to_container(cfg, resolve=False)
            if isinstance(cfg_dict, dict):
                cfg_dict.pop("hydra", None)
                return OmegaConf.create(cfg_dict)

            return OmegaConf.create(cfg_dict)

    except Exception as exc:
        logger.warning(
            "Hydra loading failed for config '%s' (%s); falling back to OmegaConf",
            config_name,
            exc,
        )
        from .config import load_config

        return load_config(
            config_path=Path(config_dir) / f"{config_name}.yaml",
            model_name=model_name,
            dataset_name=dataset_name,
            overrides=overrides,
        )
    finally:
        _clear_global_hydra()


def parse_hydra_overrides_from_args(args: List[str]) -> Tuple[List[str], List[str]]:
    """
    Parse command line arguments to separate regular args from Hydra overrides.

    Args:
        args: List of command line arguments

    Returns:
        Tuple of (regular_args, hydra_overrides)
    """
    regular_args = []
    hydra_overrides = []

    for arg in args:
        # Check if this looks like a Hydra override (key=value without leading dashes)
        if "=" in arg and not arg.startswith("-"):
            hydra_overrides.append(arg)
        else:
            regular_args.append(arg)

    return regular_args, hydra_overrides


def create_hydra_compatible_cli(original_main_func):
    """
    Decorator to make a CLI function Hydra-compatible while maintaining backward compatibility.

    Args:
        original_main_func: The original CLI main function

    Returns:
        Wrapped function that supports both argparse and Hydra overrides
    """

    def wrapper():
        import sys

        args = sys.argv[1:]
        regular_args, hydra_overrides = parse_hydra_overrides_from_args(args)

        original_argv = sys.argv[:]
        sys.argv = [sys.argv[0]] + regular_args

        try:
            if hydra_overrides:
                os.environ["RATE_EVAL_HYDRA_OVERRIDES"] = "|".join(hydra_overrides)

            return original_main_func()

        finally:
            sys.argv = original_argv
            os.environ.pop("RATE_EVAL_HYDRA_OVERRIDES", None)

    return wrapper


def get_hydra_overrides_from_env() -> List[str]:
    """Get Hydra overrides from environment variable set by CLI wrapper."""
    overrides_str = os.environ.get("RATE_EVAL_HYDRA_OVERRIDES", "")
    if overrides_str:
        return overrides_str.split("|")
    return []


def _clear_global_hydra() -> None:
    """Reset Hydra's global state if it was previously initialised."""
    instance = GlobalHydra.instance()
    if instance.is_initialized():
        instance.clear()
