"""Visualizza una sfera di micro puntini che reagisce a una traccia audio.

Questo script usa PyQtGraph (con backend OpenGL) per ottenere un'animazione
molto reattiva rispetto alla versione basata su Matplotlib. I puntini sono
proiettati sulla superficie di una sfera e la loro distanza dal centro è
modulata in base allo spettro calcolato su blocchi di una traccia audio
selezionata dall'utente.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass

from numpy.random import default_rng

import numpy as np

try:
    import soundfile as sf
except ImportError as exc:  # pragma: no cover - dipendenza opzionale
    raise SystemExit(
        "È necessario installare la libreria 'soundfile' per leggere la traccia audio.\n"
        "Puoi installarla con: pip install soundfile"
    ) from exc

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError as exc:  # pragma: no cover - dipendenza opzionale
    raise SystemExit(
        "È necessario installare la libreria 'PyQt5' per l'interfaccia grafica.\n"
        "Puoi installarla con: pip install PyQt5"
    ) from exc

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
except ImportError as exc:  # pragma: no cover - dipendenza opzionale
    raise SystemExit(
        "È necessario installare la libreria 'pyqtgraph' (con supporto OpenGL).\n"
        "Puoi installarla con: pip install pyqtgraph"
    ) from exc


@dataclass
class AudioConfig:
    """Parametri base per l'analisi audio."""

    blocksize: int = 1_024
    response_strength: float = 1.2
    volume_strength: float = 0.45
    smoothing: float = 0.35
    track_path: str = ""
    loop_track: bool = True


class AudioAnalyzer:
    """Legge blocchi di una traccia audio e ne calcola lo spettro di ampiezza."""

    def __init__(self, config: AudioConfig | None = None) -> None:
        self.config = config or AudioConfig()
        if not self.config.track_path:
            raise SystemExit(
                "Devi specificare il percorso della traccia audio da analizzare."
            )

        try:
            self._file = sf.SoundFile(self.config.track_path)
        except FileNotFoundError as exc:
            raise SystemExit(
                f"File audio non trovato: {self.config.track_path}"
            ) from exc
        except sf.SoundFileError as exc:
            raise SystemExit(
                "Impossibile aprire il file audio specificato."
                " Assicurati che il formato sia supportato."
            ) from exc

        self._samplerate = float(self._file.samplerate)
        self._latest_spectrum = np.zeros(self.config.blocksize // 2 + 1, dtype=float)
        self._latest_level = 0.0
        self._lock = threading.Lock()
        self._window = np.hanning(self.config.blocksize)

        self._running = False
        self._thread: threading.Thread | None = None

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
        self._file.seek(0)
        next_tick = time.perf_counter()

        while self._running:
            frames = self._file.read(
                self.config.blocksize, dtype="float32", always_2d=True
            )

            if frames.size == 0:
                if self.config.loop_track:
                    self._file.seek(0)
                    continue
                with self._lock:
                    self._latest_spectrum.fill(0.0)
                    self._latest_level = 0.0
                break

            audio = frames.mean(axis=1)
            self._process_block(audio)

            next_tick += len(audio) / self._samplerate
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

        self._running = False

    def start(self) -> None:
        if self._running:
            return

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

        if not self._file.closed:
            self._file.close()

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

        # Accentua le differenze mantenendo una base minima
        interpolated = np.power(interpolated, 0.6)

        return interpolated

    def volume_level(self) -> float:
        """Restituisce un livello di volume normalizzato e smussato."""
        with self._lock:
            return float(self._latest_level)


def fibonacci_sphere(n_points: int, radius: float = 1.0) -> np.ndarray:
    """Genera punti quasi uniformi sulla superficie di una sfera."""

    indices = np.arange(n_points, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))

    y = 1 - (indices / (n_points - 1)) * 2
    radius_xy = np.sqrt(1 - y * y)
    theta = phi * indices

    x = np.cos(theta) * radius_xy
    z = np.sin(theta) * radius_xy

    points = np.column_stack((x, y, z)) * radius
    return points


class SphereVisualizer(QtWidgets.QWidget):
    """Widget che rende la sfera di puntini e la anima in base all'audio."""

    def __init__(self, analyzer: AudioAnalyzer, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self.analyzer = analyzer
        self.n_points = 4_000
        self.base_radius = 1.0
        self.horizontal_strength = 0.45
        self.rotation_speed = np.deg2rad(0.35)
        self.rotation_angle = 0.0

        self.base_points = fibonacci_sphere(self.n_points, self.base_radius)
        self.normals = self.base_points / np.linalg.norm(
            self.base_points, axis=1, keepdims=True
        )
        # Colatitudine normalizzata (0 al polo nord, 1 al polo sud) per effetti radiali
        self.profile_samples = 256
        self.colatitude_norm = np.arccos(self.normals[:, 1]) / np.pi

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.view = gl.GLViewWidget()
        layout.addWidget(self.view)

        self.view.setBackgroundColor(QtGui.QColor("black"))
        self.view.opts["distance"] = 4
        self.view.opts["elevation"] = 18
        self.view.opts["azimuth"] = 35

        color = (0.43, 0.90, 1.0, 0.9)
        self.scatter = gl.GLScatterPlotItem(
            pos=self.base_points,
            color=color,
            size=0.026,
            pxMode=False,
        )
        self.view.addItem(self.scatter)

        rng = default_rng(42)
        (
            self.cloud_directions,
            self.cloud_base_radii,
            self.cloud_base,
        ) = self._generate_cloud(rng, 6_000, 1.45, 0.55)
        self.cloud_colatitude = np.arccos(self.cloud_directions[:, 1]) / np.pi
        cloud_color = (0.43, 0.90, 1.0, 0.18)
        self.cloud = gl.GLScatterPlotItem(
            pos=self.cloud_base,
            color=cloud_color,
            size=0.018,
            pxMode=False,
        )
        self.view.addItem(self.cloud)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_points)
        self.timer.start(16)

    def update_points(self) -> None:
        response_profile = self.analyzer.spectrum_for_points(self.profile_samples)
        phi_axis = np.linspace(0.0, 1.0, self.profile_samples)
        radial_profile = np.interp(self.colatitude_norm, phi_axis, response_profile)

        global_level = response_profile.mean()
        combined = 0.7 * radial_profile + 0.3 * global_level
        volume = self.analyzer.volume_level()
        scale = 1.0 + self.analyzer.config.volume_strength * volume
        radial_offset = scale * (
            self.base_radius + self.analyzer.config.response_strength * combined
        )

        half = self.profile_samples // 2
        phi_half_axis = np.linspace(0.0, 1.0, half)
        profile_x = np.interp(
            self.colatitude_norm,
            phi_half_axis,
            response_profile[:half],
        )
        profile_z = np.interp(
            self.colatitude_norm,
            phi_half_axis,
            response_profile[half:],
        )

        profile_x -= profile_x.mean()
        profile_z -= profile_z.mean()

        new_positions = self.normals * radial_offset[:, np.newaxis]
        new_positions[:, 0] += self.horizontal_strength * profile_x
        new_positions[:, 2] += self.horizontal_strength * profile_z

        self.rotation_angle = (self.rotation_angle + self.rotation_speed) % (2 * np.pi)
        cos_a = np.cos(self.rotation_angle)
        sin_a = np.sin(self.rotation_angle)
        rotation_matrix = np.array(
            [[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]]
        )

        rotated_points = new_positions @ rotation_matrix.T

        cloud_profile = np.interp(
            self.cloud_colatitude,
            phi_axis,
            response_profile,
        )
        cloud_profile_x = np.interp(
            self.cloud_colatitude,
            phi_half_axis,
            response_profile[:half],
        )
        cloud_profile_z = np.interp(
            self.cloud_colatitude,
            phi_half_axis,
            response_profile[half:],
        )

        cloud_profile_x -= cloud_profile_x.mean()
        cloud_profile_z -= cloud_profile_z.mean()

        cloud_gain = 1.0 + 0.35 * global_level + 0.55 * volume
        dynamic_cloud_radius = (
            self.cloud_base_radii * cloud_gain
            + self.analyzer.config.response_strength * 0.28 * cloud_profile
        )
        cloud_positions = self.cloud_directions * dynamic_cloud_radius[:, np.newaxis]
        cloud_positions[:, 0] += self.horizontal_strength * 0.35 * cloud_profile_x
        cloud_positions[:, 2] += self.horizontal_strength * 0.35 * cloud_profile_z

        rotated_cloud = cloud_positions @ rotation_matrix.T

        self.scatter.setData(pos=rotated_points)
        self.cloud.setData(pos=rotated_cloud)

    @staticmethod
    def _generate_cloud(
        rng: np.random.Generator, n_points: int, radius: float, thickness: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Crea un alone nebuloso di punti attorno alla sfera."""

        directions = rng.normal(size=(n_points, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radii = radius + thickness * rng.random(n_points)
        base_positions = directions * radii[:, np.newaxis]
        return directions, radii, base_positions

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover - evento GUI
        self.timer.stop()
        self.analyzer.stop()
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualizza una sfera di puntini reattiva basata su una traccia audio."
        )
    )
    parser.add_argument(
        "audio_path",
        help="Percorso del file audio da utilizzare come sorgente (wav, flac, ogg, ...).",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=1_024,
        help="Numero di campioni per blocco di analisi (default: 1024).",
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Disabilita il loop della traccia audio una volta terminata.",
    )

    args = parser.parse_args(argv)

    if args.blocksize <= 0:
        raise SystemExit("Il parametro --blocksize deve essere un intero positivo.")

    config = AudioConfig(
        blocksize=args.blocksize,
        track_path=args.audio_path,
        loop_track=not args.no_loop,
    )

    analyzer = AudioAnalyzer(config)
    analyzer.start()

    app = QtWidgets.QApplication([sys.argv[0]])
    pg.setConfigOptions(antialias=True)

    window = SphereVisualizer(analyzer)
    window.resize(900, 900)
    window.setWindowTitle("Sfera Spettro 3D")
    window.show()

    try:
        exit_code = app.exec_()
    finally:
        analyzer.stop()

    sys.exit(exit_code)


if __name__ == "__main__":
    main(sys.argv[1:])