"""Analizzatore audio che calcola lo spettro per animare la sfera."""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from .config import AudioConfig
from .sources import AudioSource, SoundFileSource


class AudioAnalyzer:
    """Elabora blocchi audio da una sorgente e ne calcola lo spettro."""

    def __init__(self, config: Optional[AudioConfig] = None, source: Optional[AudioSource] = None) -> None:
        self.config = config or AudioConfig()
        self._source: AudioSource | None = None
        self._samplerate = 44_100.0
        self._latest_spectrum = np.zeros(self.config.blocksize // 2 + 1, dtype=float)
        self._latest_level = 0.0
        self.attach_source(source)
        self._lock = threading.Lock()
        self._window = np.hanning(self.config.blocksize)

        self._running = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Gestione sorgenti
    def attach_source(self, source: Optional[AudioSource]) -> None:
        """Fornisce o sostituisce dinamicamente la sorgente audio."""

        if source is None:
            if self.config.track_path:
                source = SoundFileSource(self.config.track_path)
            else:
                self._source = None
                self._latest_spectrum.fill(0.0)
                self._latest_level = 0.0
                return

        if self._running:
            raise RuntimeError("Impossibile cambiare sorgente mentre l'analisi è in corso.")

        self._source = source
        self._samplerate = float(source.samplerate)
        self._latest_spectrum = np.zeros(self.config.blocksize // 2 + 1, dtype=float)
        self._latest_level = 0.0

    def replace_source(self, source: AudioSource) -> None:
        """Sostituisce la sorgente in modo sicuro riavviando l'analisi se necessario."""

        was_running = self._running
        if was_running:
            self.stop()
        self.attach_source(source)
        if was_running:
            self.start()

    # ------------------------------------------------------------------
    # Controllo esecuzione
    def start(self) -> None:
        if self._running:
            return

        if self._source is None:
            raise SystemExit(
                "Nessuna sorgente audio fornita all'AudioAnalyzer. Usa attach_source()"
                " oppure specifica track_path."
            )

        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._running:
            self._running = False
            if self._thread is not None:
                self._thread.join(timeout=1.0)
        else:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=0.5)

        if self._source is not None:
            self._source.close()

    @property
    def is_running(self) -> bool:
        """Indica se il thread di analisi è attivo."""

        return self._running

    # ------------------------------------------------------------------
    # API per la visualizzazione
    def spectrum_for_points(self, n_points: int) -> np.ndarray:
        """Restituisce lo spettro ridimensionato al numero di punti richiesto."""

        with self._lock:
            spectrum = self._latest_spectrum.copy()

        if not spectrum.size:
            return np.zeros(n_points)

        base_axis = np.linspace(0, 1, spectrum.size)
        target_axis = np.linspace(0, 1, n_points)
        interpolated = np.interp(target_axis, base_axis, spectrum)

        interpolated -= interpolated.min()
        peak = interpolated.max()
        if peak > 1e-6:
            interpolated /= peak

        interpolated = np.power(interpolated, 0.6)

        return interpolated

    def volume_level(self) -> float:
        """Restituisce un livello di volume normalizzato e smussato."""

        with self._lock:
            return float(self._latest_level)

    # ------------------------------------------------------------------
    # Implementazione interna
    def _process_block(self, audio: np.ndarray) -> None:
        if not audio.size:
            return

        block = audio.astype(float, copy=False)
        rms = float(np.sqrt(np.mean(block ** 2))) if block.size else 0.0

        if block.size < self.config.blocksize:
            padded = np.zeros(self.config.blocksize, dtype=float)
            padded[: block.size] = block
        else:
            padded = block[: self.config.blocksize]

        windowed = padded * self._window
        spectrum = np.abs(np.fft.rfft(windowed))
        spectrum = np.log1p(spectrum)

        level = np.log1p(8.0 * rms)

        with self._lock:
            self._latest_spectrum = (
                self.config.smoothing * self._latest_spectrum
                + (1 - self.config.smoothing) * spectrum
            )
            self._latest_level = (
                self.config.smoothing * self._latest_level
                + (1 - self.config.smoothing) * level
            )

    def _worker(self) -> None:  # pragma: no cover - eseguito su thread separato
        assert self._source is not None

        self._source.reset()
        next_tick = time.perf_counter()

        while self._running:
            audio = self._source.read(self.config.blocksize)

            if audio.size == 0:
                if self.config.loop_track:
                    self._source.reset()
                    continue
                with self._lock:
                    self._latest_spectrum.fill(0.0)
                    self._latest_level = 0.0
                break

            self._process_block(audio)

            next_tick += len(audio) / self._samplerate
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

        self._running = False
