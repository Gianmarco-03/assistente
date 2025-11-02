"""Strumenti di sintesi vocale per l'assistente."""

from .base import TextToSpeechEngine
from .pyttsx3_engine import Pyttsx3Engine
from .responder import TextToSpeechResponder

__all__ = ["TextToSpeechEngine", "Pyttsx3Engine", "TextToSpeechResponder"]
