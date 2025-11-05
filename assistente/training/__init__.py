"""Pipeline di training dedicate al dataset MASSIVE (Amazon Science)."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DEFAULT_DATASET_DIR",
    "LEGACY_DATASET_DIR",
    "Sample",
    "iter_samples",
    "load_samples",
    "MODEL_FILENAME",
    "build_pipeline",
    "save_model",
    "train_model",
    "LabelStats",
    "describe_labels",
    "plot_distribution",
    "print_summary",
    "visualize_dataset",
]

_DATASET_EXPORTS = {
    "DEFAULT_DATASET_DIR",
    "LEGACY_DATASET_DIR",
    "Sample",
    "iter_samples",
    "load_samples",
}
_PIPELINE_EXPORTS = {
    "MODEL_FILENAME",
    "build_pipeline",
    "save_model",
    "train_model",
}
_VISUALIZER_EXPORTS = {
    "LabelStats",
    "describe_labels",
    "plot_distribution",
    "print_summary",
    "visualize_dataset",
}


def __getattr__(name: str) -> Any:
    if name in _DATASET_EXPORTS:
        module = import_module('.dataset', __name__)
    elif name in _PIPELINE_EXPORTS:
        module = import_module('.pipeline', __name__)
    elif name in _VISUALIZER_EXPORTS:
        module = import_module('.data_visualizer', __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals().keys()))
