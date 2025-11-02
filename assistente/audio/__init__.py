"""Componenti audio per l'assistente."""

from .config import AudioConfig
from .analyzer import AudioAnalyzer
from .sources import AudioSource, SoundFileSource
from .microphone import InteractiveAudioSource, MicrophoneSource
from .recognizer import BackgroundRecognizer, RecognitionConfig

__all__ = [
    "AudioConfig",
    "AudioAnalyzer",
    "AudioSource",
    "SoundFileSource",
    "SoundDeviceLoopbackSource",
    "MicrophoneSource",
    "InteractiveAudioSource",
    "RecognitionConfig",
    "BackgroundRecognizer",
]