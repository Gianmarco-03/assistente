"""Gestione delle risposte vocali basate su text-to-speech."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping

from .base import TextToSpeechEngine


@dataclass(slots=True)
class TextToSpeechResponder:
    """Genera una risposta vocale predefinita per un determinato input."""

    engine: TextToSpeechEngine
    responses: Mapping[str, str] = field(default_factory=dict)
    default_response: str = "Mi spiace, non ho ancora imparato come rispondere a questa richiesta."
    output_dir: Path = field(default_factory=lambda: Path("tts_output"))
    _normalized: Dict[str, str] = field(init=False, repr=False, default_factory=dict)


    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._normalized = {self._normalize(key): value for key, value in self.responses.items()}

    # ------------------------------------------------------------------
    def _normalize(self, text: str) -> str:
        return " ".join(text.strip().lower().split())

    def register(self, trigger: str, response: str) -> None:
        """Aggiunge o aggiorna un'associazione ``trigger -> response``."""

        self._normalized[self._normalize(trigger)] = response

    def response_for(self, text: str) -> str:
        """Restituisce il testo della risposta associata a ``text``."""

        return self._normalized.get(self._normalize(text), self.default_response)

    def respond(self, text: str) -> tuple[str, Path]:
        """Genera il file audio corrispondente alla risposta."""

        response_text = self.response_for(text)
        filename = f"response_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex}.wav"
        output_path = self.output_dir / filename
        self.engine.synthesize_to_file(response_text, str(output_path))
        return response_text, output_path

    # ------------------------------------------------------------------
    @classmethod
    def from_json(
        cls,
        engine: TextToSpeechEngine,
        json_path: str | Path,
        *,
        default_response: str | None = None,
        output_dir: str | Path | None = None,
    ) -> "TextToSpeechResponder":
        data = cls._load_json(json_path)
        return cls(
            engine=engine,
            responses=data,
            default_response=default_response or cls.default_response,  # type: ignore[arg-type]
            output_dir=output_dir or Path(json_path).parent / "tts_output",
        )

    @staticmethod
    def _load_json(json_path: str | Path) -> Dict[str, str]:
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"File JSON non trovato: {path}")
        with path.open("r", encoding="utf8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):  # pragma: no cover - validazione semplice
            raise ValueError("Il file JSON deve contenere un dizionario 'input' -> 'response'.")
        return {str(key): str(value) for key, value in data.items()}


__all__ = ["TextToSpeechResponder"]
