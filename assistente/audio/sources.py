"""Sorgenti audio compatibili con :class:`~assistente.audio.analyzer.AudioAnalyzer`."""

from __future__ import annotations

import queue
import sys
import time

from typing import Protocol

import numpy as np

try:  # pragma: no cover - dipendenza opzionale
    import soundfile as sf
except ImportError as exc:  # pragma: no cover - dipendenza opzionale
    raise SystemExit(
        "È necessario installare la libreria 'soundfile' per leggere la traccia audio.\n"
        "Puoi installarla con: pip install soundfile"
    ) from exc
try:  # pragma: no cover - dipendenza opzionale
    import sounddevice as sd
except ImportError:  # pragma: no cover
    sd = None  # type: ignore[assignment]

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


class SoundDeviceLoopbackSource:
    """Cattura l'audio di uscita del sistema tramite :mod:`sounddevice`."""

    def __init__(self, blocksize: int, *, device: str | int | None = None) -> None:
        if sd is None:  # pragma: no cover - dipendenza opzionale
            raise SystemExit(
                "È necessario installare la libreria 'sounddevice' per monitorare l'audio di uscita.\n"
                "Puoi installarla con: pip install sounddevice"
            )

        self._blocksize = int(blocksize)
        if self._blocksize <= 0:
            raise ValueError("blocksize deve essere un intero positivo")

        self._device = device
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=16)
        self._pending = np.empty((0, 1), dtype=np.float32)
        self._closed = False

        device_info = self._resolve_device(device)
        self._channels = max(1, min(2, int(device_info.get("max_output_channels", 2) or 1)))
        self.samplerate = float(device_info.get("default_samplerate", 44_100.0))

        extra_settings = None
        if hasattr(sd, "WasapiSettings"):
            try:  # pragma: no cover - specifico Windows
                extra_settings = sd.WasapiSettings(loopback=True)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - fallback in caso di host API diversa
                extra_settings = None

        self._stream = sd.InputStream(  # type: ignore[call-arg]
            samplerate=self.samplerate,
            blocksize=self._blocksize,
            dtype="float32",
            channels=self._channels,
            device=self._device,
            callback=self._callback,
            extra_settings=extra_settings,
        )
        self._stream.start()

    def _resolve_device(self, device: str | int | None) -> dict:
        target = device
        if isinstance(target, str) and not target:
            target = None
        try:
            return sd.query_devices(target, "output")  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - propagazione chiara
            raise SystemExit(
                "Impossibile individuare il dispositivo di uscita richiesto."
                " Verifica il nome/indice passato a loopback_device."
            ) from exc

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:  # pragma: no cover - thread audio
        if status:
            print(f"[loopback] {status}", file=sys.stderr)
        chunk = np.asarray(indata, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk[:, np.newaxis]
        try:
            self._queue.put_nowait(chunk.copy())
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(chunk.copy())

    def reset(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._pending = np.empty((0, self._channels), dtype=np.float32)

    def read(self, blocksize: int) -> np.ndarray:
        if self._closed:
            return np.empty(0, dtype=float)

        requested = int(blocksize)
        if requested <= 0:
            return np.empty(0, dtype=float)

        buffers: list[np.ndarray] = []
        total_frames = 0

        if self._pending.size:
            buffers.append(self._pending)
            total_frames += self._pending.shape[0]
            self._pending = np.empty((0, self._channels), dtype=np.float32)

        deadline = time.perf_counter() + 0.5
        while total_frames < requested:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                chunk = self._queue.get(timeout=remaining)
            except queue.Empty:
                break
            buffers.append(chunk)
            total_frames += chunk.shape[0]

        if not buffers:
            return np.empty(0, dtype=float)

        data = np.concatenate(buffers, axis=0)
        if data.shape[0] > requested:
            self._pending = data[requested:]
            data = data[:requested]
        else:
            self._pending = np.empty((0, self._channels), dtype=np.float32)

        if data.ndim == 1 or data.shape[1] == 1:
            mono = data.reshape(-1)
        else:
            mono = data.mean(axis=1)
        return mono.astype(float, copy=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.stop()
        finally:
            self._stream.close()
        self.reset()
