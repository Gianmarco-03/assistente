"""Dispatcher per gli intent gestiti dal pacchetto ``actions``."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import Callable

Handler = Callable[[], str]


@lru_cache(maxsize=None)
def _load_handler(intent: str) -> Handler | None:
    """Restituisce l'handler associato all'intent, se disponibile."""

    module_name = f"{__name__}.{intent}"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return None
        raise
    handler = getattr(module, "handle", None)
    if callable(handler):
        return handler  # type: ignore[return-value]
    return None


def handle(slots : dict,intent: str) -> str:
    """Invoca l'handler associato all'intent oppure restituisce un fallback."""
    print(intent)
    handler = _load_handler(intent)
    if handler is not None:
        return handler(slots)
    normalized = intent.replace("_", " ")
    return f"Questa richiesta appartiene alla classe '{normalized}'."


__all__ = ["handle"]