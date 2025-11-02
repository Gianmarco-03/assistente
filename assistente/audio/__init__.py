"""Componenti audio per l'assistente."""

from .config import AudioConfig
from .analyzer import AudioAnalyzer
from .sources import AudioSource, SoundFileSource

__all__ = ["AudioConfig", "AudioAnalyzer", "AudioSource", "SoundFileSource"]
