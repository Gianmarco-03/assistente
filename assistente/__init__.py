"""Assistente package per visualizzazione audio, TTS e training."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["audio", "visualization", "tts", "training"]


def __getattr__(name: str) -> Any:
    """Risoluzione pigra dei sottopacchetti principali."""

    if name in __all__:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
