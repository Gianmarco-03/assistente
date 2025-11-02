"""Interfacce comuni per i motori di sintesi vocale."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TextToSpeechEngine(ABC):
    """Definisce le operazioni minime di un motore TTS."""

    name: str = "generic"

    @abstractmethod
    def synthesize_to_file(self, text: str, output_path: str) -> None:
        """Genera un file audio contenente ``text`` pronunciato."""

    def shutdown(self) -> None:
        """Permette ai motori di rilasciare risorse opzionali."""

    def __enter__(self) -> "TextToSpeechEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()
