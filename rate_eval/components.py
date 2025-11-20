"""Component factory for creating models and datasets.

This module replaces the complex registry system with simple factory functions.
"""

from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf

from . import datasets
from .common import DatasetError, ModelError, get_logger
from .config import load_dataset_config, load_model_config, merge_configs

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_MODEL_REGISTRY: Dict[str, Tuple[str, str]] = {
    "medgemma": ("rate_eval.models.medgemma", "MedGemma"),
    "medimageinsight": ("rate_eval.models.medimageinsight", "MedImageInsight"),
    "pillar0": ("rate_eval.models.pillar0", "Pillar0"),
    "merlin": ("rate_eval.models.merlin", "Merlin"),
    "lingshu": ("rate_eval.models.lingshu", "Lingshu"),
    "ctclip": ("rate_eval.models.ctclip", "CTCLIP"),
}


def _to_dict(config_section: Any) -> Dict[str, Any]:
    """Convert OmegaConf sections to regular dictionaries."""
    if isinstance(config_section, DictConfig):
        return OmegaConf.to_container(config_section, resolve=True)  # type: ignore[return-value]
    if config_section is None:
        return {}
    if isinstance(config_section, dict):
        return config_section
    return {}


def _resolve_path(path_str: Union[str, Path]) -> Path:
    """Resolve a potentially relative path against the project root."""
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _load_dataset_class(name: str, loader_spec: Dict[str, Any] = None) -> type:
    """Load a dataset class from datasets module using __dict__."""
    loader_spec = loader_spec or {}
    class_name = loader_spec.get("class")

    if class_name and hasattr(datasets, class_name):
        return getattr(datasets, class_name)

    # List available dataset classes
    available_classes = [
        attr
        for attr in datasets.__dict__
        if isinstance(datasets.__dict__[attr], type)
        and hasattr(datasets.__dict__[attr], "__module__")
        and datasets.__dict__[attr].__module__.startswith("rate_eval.datasets")
    ]

    raise DatasetError(
        f"Dataset class for '{name}' not found. Available classes: {available_classes}"
    )


def _resolve_model_class(name: str) -> type:
    """Resolve the concrete model class from the registry."""
    try:
        module_path, class_name = _MODEL_REGISTRY[name]
    except KeyError as exc:
        available_models = sorted(_MODEL_REGISTRY)
        raise ModelError(f"Unknown model '{name}'. Available models: {available_models}") from exc

    try:
        module = import_module(module_path)
        model_cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ModelError(
            f"Failed to import model '{name}' from '{module_path}.{class_name}': {exc}"
        ) from exc

    return model_cls


def _get_loader_spec(
    name: str, config: dict
) -> Tuple[Dict[str, Any], Optional[DictConfig], Dict[str, Any]]:
    """Return dataset loader spec along with any pre-loaded dataset config."""
    dataset_registry = _to_dict(getattr(config, "datasets", None))
    loader_spec = _to_dict(dataset_registry.get(name))
    if loader_spec:
        return loader_spec, None, dataset_registry

    try:
        dataset_config = load_dataset_config(name)
    except FileNotFoundError:
        return {}, None, dataset_registry

    loader_spec = _to_dict(OmegaConf.select(dataset_config, "loader"))
    return loader_spec, dataset_config, dataset_registry


def _load_dataset_config_for_loader(
    name: str,
    loader_spec: Dict[str, Any],
    fallback_config: Optional[DictConfig],
) -> DictConfig:
    """Load the dataset config indicated by the loader spec."""
    dataset_config_path = loader_spec.get("config") or loader_spec.get("config_file")
    if dataset_config_path:
        resolved_path = _resolve_path(dataset_config_path)
        if not resolved_path.exists():
            raise DatasetError(
                f"Config file '{resolved_path}' for dataset '{name}' does not exist."
            )
        return OmegaConf.load(resolved_path)

    if fallback_config is not None:
        return fallback_config

    return load_dataset_config(name)


def create_model(name: str, config: dict) -> Any:
    """
    Create a model instance by name.

    Args:
        name: Model name (e.g., 'medgemma', 'medimageinsight')
        config: Base configuration dictionary

    Returns:
        Model instance

    Raises:
        ModelError: If model creation fails
    """
    try:
        model_config = load_model_config(name)
        merged_config = merge_configs(model_config, config)

        model_cls = _resolve_model_class(name)

        logger.info("Creating %s model", model_cls.__name__)
        return model_cls(merged_config)

    except Exception as exc:
        if isinstance(exc, ModelError):
            raise
        raise ModelError(f"Failed to create model '{name}': {exc}") from exc


def create_dataset(
    name: str, config: dict, split: str = "train", transforms=None, model_preprocess=None
) -> Any:
    """
    Create a dataset instance by name.

    Args:
        name: Dataset name (e.g., 'abd_ct_merlin', 'dummy')
        config: Base configuration dictionary
        split: Dataset split ('train', 'valid', 'test')
        transforms: Optional transforms to apply
        model_preprocess: Optional model preprocessing method

    Returns:
        Dataset instance

    Raises:
        DatasetError: If dataset creation fails
    """
    try:
        loader_spec, dataset_config_hint, dataset_registry = _get_loader_spec(name, config)

        if not loader_spec:
            available = sorted(dataset_registry.keys()) if dataset_registry else []
            raise DatasetError(
                f"Unknown dataset '{name}'. Provide loader configuration in YAML."
                f" Available datasets: {available}"
            )

        dataset_config = _load_dataset_config_for_loader(
            name,
            loader_spec,
            dataset_config_hint,
        )

        merged_config = merge_configs(dataset_config, config)

        dataset_class = _load_dataset_class(name, loader_spec)

        init_kwargs = loader_spec.get("init_args", {})
        if init_kwargs and not isinstance(init_kwargs, dict):
            raise DatasetError(f"init_args for dataset '{name}' must be a dictionary if provided.")

        logger.info(
            "Creating dataset '%s' (%s) for split '%s'",
            name,
            dataset_class.__name__,
            split,
        )

        return dataset_class(
            merged_config,
            split,
            transforms,
            model_preprocess,
            **init_kwargs,
        )

    except Exception as exc:
        if isinstance(exc, DatasetError):
            raise
        raise DatasetError(f"Failed to create dataset '{name}': {exc}") from exc
