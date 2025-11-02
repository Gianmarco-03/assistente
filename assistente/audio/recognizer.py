"""Riconoscimento vocale di base su thread separato."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import queue

import numpy as np

try:  # pragma: no cover - dipendenza opzionale
    import speech_recognition as sr
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "È necessario installare la libreria 'speech_recognition' per l'ascolto vocale.\n"
        "Puoi installarla con: pip install SpeechRecognition"
    ) from exc


@dataclass(slots=True)
class RecognitionConfig:
    language: str = "it-IT"
    start_threshold: float = 0.02
    stop_threshold: float = 0.01
    silence_duration: float = 0.6
    min_phrase_duration: float = 0.35
    max_phrase_duration: float = 6.0


class BackgroundRecognizer:
    """Esegue il riconoscimento vocale continuo utilizzando una coda di campioni."""

    def __init__(
        self,
        sample_queue: "queue.Queue[np.ndarray]",
        samplerate: float,
        *,
        config: Optional[RecognitionConfig] = None,
    ) -> None:
        self._queue = sample_queue
        self._samplerate = float(samplerate)
        self._config = config or RecognitionConfig()

        self._recognizer = sr.Recognizer()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._running = threading.Event()
        self._paused = threading.Event()

        self.on_text: list[callable[[str], None]] = []
        self.on_error: list[callable[[str], None]] = []

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._thread.start()

    def stop(self) -> None:
        if not self._running.is_set():
            return
        self._running.clear()
        try:
            self._queue.put_nowait(np.empty(0, dtype=float))
        except queue.Full:  # pragma: no cover - coda satura durante lo stop
            pass
        self._thread.join(timeout=1.5)

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._paused.set()
        else:
            self._paused.clear()

    def _emit_text(self, text: str) -> None:
        for callback in list(self.on_text):
            try:
                callback(text)
            except Exception as exc:  # pragma: no cover - logica callback esterna
                self._emit_error(f"Callback testo fallita: {exc}")

    def _emit_error(self, message: str) -> None:
        for callback in list(self.on_error):
            try:
                callback(message)
            except Exception:  # pragma: no cover - logica callback esterna
                pass

    def _worker(self) -> None:  # pragma: no cover - thread
        cfg = self._config
        silence_samples = int(cfg.silence_duration * self._samplerate)
        min_samples = int(cfg.min_phrase_duration * self._samplerate)
        max_samples = int(cfg.max_phrase_duration * self._samplerate) if cfg.max_phrase_duration else 0

        capturing = False
        silence_counter = 0
        collected: list[np.ndarray] = []

        while self._running.is_set():
            try:
                chunk = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if not self._running.is_set():
                break

            if not chunk.size:
                continue

            if self._paused.is_set():
                capturing = False
                silence_counter = 0
                collected.clear()
                continue

            amplitude = float(np.max(np.abs(chunk)))

            if not capturing:
                if amplitude >= cfg.start_threshold:
                    capturing = True
                    silence_counter = 0
                    collected = [chunk]
                continue

            collected.append(chunk)
            if amplitude >= cfg.stop_threshold:
                silence_counter = 0
            else:
                silence_counter += chunk.size

            total_samples = sum(block.size for block in collected)
            if max_samples and total_samples >= max_samples:
                self._process_phrase(collected, min_samples, cfg.language)
                capturing = False
                silence_counter = 0
                collected = []
                continue

            if silence_samples and silence_counter >= silence_samples:
                self._process_phrase(collected, min_samples, cfg.language)
                capturing = False
                silence_counter = 0
                collected = []

    def _process_phrase(self, blocks: list[np.ndarray], min_samples: int, language: str) -> None:
        if not blocks:
            return

        audio = np.concatenate(blocks)
        if audio.size < min_samples:
            return

        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767.0).astype(np.int16)
        audio_data = sr.AudioData(pcm.tobytes(), sample_rate=int(self._samplerate), sample_width=2)

        try:
            text = self._recognizer.recognize_google(audio_data, language=language)
        except sr.UnknownValueError:
            return
        except sr.RequestError as exc:
            self._emit_error(f"Errore dal servizio di riconoscimento: {exc}")
            return

        self._emit_text(text)


__all__ = ["RecognitionConfig", "BackgroundRecognizer"]