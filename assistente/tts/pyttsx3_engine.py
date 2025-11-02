"""Motore TTS basato su :mod:`pyttsx3`."""

from __future__ import annotations

from typing import Iterable

from .base import TextToSpeechEngine

try:  # pragma: no cover - dipendenza opzionale
    import pyttsx3
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "È necessario installare la libreria 'pyttsx3' per utilizzare la sintesi vocale.\n"
        "Puoi installarla con: pip install pyttsx3"
    ) from exc


class Pyttsx3Engine(TextToSpeechEngine):
    """Implementazione concreta di :class:`TextToSpeechEngine` basata su pyttsx3."""

    name = "pyttsx3"

    def __init__(self, voice: str | None = None, rate: int | None = None) -> None:
        self._engine = pyttsx3.init()
        if voice:
            self.set_voice(voice)
        if rate is not None:
            self._engine.setProperty("rate", rate)

    # ------------------------------------------------------------------
    def available_voices(self) -> Iterable[str]:  # pragma: no cover - interrogazione runtime
        for voice in self._engine.getProperty("voices"):
            yield getattr(voice, "id", "")

    def set_voice(self, voice_id: str) -> None:
        voices = {getattr(voice, "id", ""): voice for voice in self._engine.getProperty("voices")}
        if voice_id not in voices:
            raise ValueError(
                "Voce inesistente." " Usa available_voices() per ottenere la lista delle voci installate."
            )
        self._engine.setProperty("voice", voice_id)

    # ------------------------------------------------------------------
    def synthesize_to_file(self, text: str, output_path: str) -> None:
        self._engine.save_to_file(text, output_path)
        self._engine.runAndWait()

    def shutdown(self) -> None:
        self._engine.stop()
