"""Sorgenti audio basate su microfono per l'analizzatore."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional

import queue

import numpy as np

try:  # pragma: no cover - dipendenza opzionale
    import sounddevice as sd
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "È necessario installare la libreria 'sounddevice' per utilizzare il microfono.\n"
        "Puoi installarla con: pip install sounddevice"
    ) from exc

try:  # pragma: no cover - dipendenza opzionale
    import soundfile as sf
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "È necessario installare la libreria 'soundfile' per elaborare i file audio.\n"
        "Puoi installarla con: pip install soundfile"
    ) from exc

from .sources import AudioSource


class MicrophoneSource(AudioSource):
    """Flusso audio continuo proveniente da un microfono."""

    def __init__(
        self,
        *,
        samplerate: Optional[float] = None,
        device: Optional[int | str] = None,
        channels: int = 1,
        stream_blocksize: int | None = None,
    ) -> None:
        self._device = device
        self._channels = channels
        self._stream_blocksize = stream_blocksize or 0

        if samplerate is not None:
            self.samplerate = float(samplerate)
        else:
            device_info = sd.query_devices(device, "input")
            default_samplerate = float(device_info.get("default_samplerate", 44_100.0))
            self.samplerate = default_samplerate

        self._queue: "queue.Queue[np.ndarray]"
        self._extra_queues: List["queue.Queue[np.ndarray]"]
        self._queue = queue.Queue()
        self._extra_queues = []
        self._lock = threading.Lock()
        self._buffer = np.empty(0, dtype=float)
        self._stream: sd.InputStream | None = None

    def _callback(self, indata: np.ndarray, frames: int, time, status) -> None:  # pragma: no cover
        if status:
            print(f"[microfono] stato stream: {status}")

        mono = indata.mean(axis=1).copy()
        try:
            self._queue.put_nowait(mono)
        except queue.Full:
            pass

        with self._lock:
            for extra_queue in self._extra_queues:
                try:
                    extra_queue.put_nowait(mono.copy())
                except queue.Full:
                    pass

    def register_listener(self, maxsize: int = 0) -> "queue.Queue[np.ndarray]":
        """Restituisce una coda che riceve una copia dei campioni del microfono."""

        listener_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._extra_queues.append(listener_queue)
        return listener_queue

    def reset(self) -> None:
        self._buffer = np.empty(0, dtype=float)
        self._drain_queue()
        if self._stream is None:
            self._stream = sd.InputStream(
                device=self._device,
                samplerate=self.samplerate,
                channels=self._channels,
                blocksize=self._stream_blocksize,
                dtype="float32",
                callback=self._callback,
            )
        if not self._stream.active:
            self._stream.start()

    def read(self, blocksize: int) -> np.ndarray:
        if self._stream is None:
            return np.empty(0, dtype=float)

        chunks = []
        total = 0

        if self._buffer.size:
            chunks.append(self._buffer)
            total += self._buffer.size
            self._buffer = np.empty(0, dtype=float)

        while total < blocksize:
            try:
                chunk = self._queue.get(timeout=0.2)
            except queue.Empty:
                break

            if not chunk.size:
                continue

            chunks.append(chunk)
            total += chunk.size

        if not chunks:
            return np.empty(0, dtype=float)

        audio = np.concatenate(chunks)
        if audio.size > blocksize:
            self._buffer = audio[blocksize:]
            audio = audio[:blocksize]

        return audio.astype(float, copy=False)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            self._extra_queues.clear()

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


class InteractiveAudioSource(AudioSource):
    """Sorgente che combina il microfono con eventuali riproduzioni audio."""

    def __init__(self, microphone: MicrophoneSource) -> None:
        self._microphone = microphone
        self.samplerate = microphone.samplerate
        self._lock = threading.Lock()
        self._playback_buffer = np.empty(0, dtype=float)

    def reset(self) -> None:
        self._microphone.reset()

    def read(self, blocksize: int) -> np.ndarray:
        with self._lock:
            if self._playback_buffer.size:
                chunk = self._playback_buffer[:blocksize]
                self._playback_buffer = self._playback_buffer[blocksize:]
                return chunk

        return self._microphone.read(blocksize)

    def close(self) -> None:
        self._microphone.close()

    def queue_playback_file(self, audio_path: str | Path) -> None:
        data, samplerate = sf.read(str(audio_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if samplerate != self.samplerate:
            data = self._resample(data, samplerate, self.samplerate)
        self.queue_playback_samples(data)

    def queue_playback_samples(self, samples: np.ndarray) -> None:
        if not samples.size:
            return
        with self._lock:
            if not self._playback_buffer.size:
                self._playback_buffer = samples.astype(float, copy=False)
            else:
                self._playback_buffer = np.concatenate(
                    [self._playback_buffer, samples.astype(float, copy=False)]
                )

    def microphone(self) -> MicrophoneSource:
        return self._microphone

    @staticmethod
    def _resample(data: np.ndarray, source_rate: float, target_rate: float) -> np.ndarray:
        if not data.size or source_rate == target_rate:
            return data

        duration = data.size / float(source_rate)
        target_length = max(int(duration * float(target_rate)), 1)
        old_axis = np.linspace(0.0, 1.0, data.size, endpoint=False)
        new_axis = np.linspace(0.0, 1.0, target_length, endpoint=False)
        return np.interp(new_axis, old_axis, data)


__all__ = ["MicrophoneSource", "InteractiveAudioSource"]