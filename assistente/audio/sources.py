"""Sorgenti audio compatibili con :class:`~assistente.audio.analyzer.AudioAnalyzer`."""

from __future__ import annotations

from typing import Protocol

import numpy as np

try:  # pragma: no cover - dipendenza opzionale
    import soundfile as sf
except ImportError as exc:  # pragma: no cover - dipendenza opzionale
    raise SystemExit(
        "È necessario installare la libreria 'soundfile' per leggere la traccia audio.\n"
        "Puoi installarla con: pip install soundfile"
    ) from exc


class AudioSource(Protocol):
    """Interfaccia minima per fornire blocchi audio all'analizzatore."""

    samplerate: float

    def reset(self) -> None:
        """Riporta la sorgente all'inizio del flusso audio."""

    def read(self, blocksize: int) -> np.ndarray:
        """Restituisce un array 1D di campioni (può essere vuoto a EOF)."""

    def close(self) -> None:
        """Rilascia le risorse associate alla sorgente."""


class SoundFileSource:
    """Legge blocchi da un file su disco usando :mod:`soundfile`."""

    def __init__(self, path: str) -> None:
        self._path = path
        try:
            self._file = sf.SoundFile(path)
        except FileNotFoundError as exc:
            raise SystemExit(f"File audio non trovato: {path}") from exc
        except sf.SoundFileError as exc:
            raise SystemExit(
                "Impossibile aprire il file audio specificato." " Assicurati che il formato sia supportato."
            ) from exc

        self.samplerate = float(self._file.samplerate)

    def _ensure_open(self) -> None:
        if self._file.closed:
            self._file = sf.SoundFile(self._path)
            self.samplerate = float(self._file.samplerate)

    def reset(self) -> None:
        self._ensure_open()
        self._file.seek(0)

    def read(self, blocksize: int) -> np.ndarray:
        self._ensure_open()
        frames = self._file.read(blocksize, dtype="float32", always_2d=True)
        if frames.size == 0:
            return np.empty(0, dtype=float)
        return frames.mean(axis=1)

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
