"""Configurazioni per l'analisi audio."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AudioConfig:
    """Parametri base per l'analisi audio."""

    blocksize: int = 1_024
    response_strength: float = 1.2
    volume_strength: float = 0.45
    smoothing: float = 0.35
    track_path: str = ""
    loop_track: bool = True
    use_output_loopback: bool = False
    loopback_device: str = ""

