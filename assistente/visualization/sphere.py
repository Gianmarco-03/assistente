"""Widget Qt responsabile della visualizzazione della sfera animata."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.random import default_rng

try:  # pragma: no cover - dipendenze opzionali
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "È necessario installare la libreria 'PyQt5' per l'interfaccia grafica.\n"
        "Puoi installarla con: pip install PyQt5"
    ) from exc

try:  # pragma: no cover
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "È necessario installare la libreria 'pyqtgraph' (con supporto OpenGL).\n"
        "Puoi installarla con: pip install pyqtgraph"
    ) from exc

from audio.analyzer import AudioAnalyzer


@dataclass(slots=True)
class SphereStyle:
    """Parametri estetici del visualizzatore."""

    n_points: int = 4_000
    base_radius: float = 1.0
    horizontal_strength: float = 0.45
    rotation_speed_deg: float = 0.35
    cloud_points: int = 6_000
    cloud_radius: float = 1.45
    cloud_thickness: float = 0.55


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

    def __init__(
        self,
        analyzer: AudioAnalyzer,
        parent: QtWidgets.QWidget | None = None,
        style: SphereStyle | None = None,
    ) -> None:
        super().__init__(parent)

        self.analyzer = analyzer
        self.style = style or SphereStyle()
        self.n_points = self.style.n_points
        self.base_radius = self.style.base_radius
        self.horizontal_strength = self.style.horizontal_strength
        self.rotation_speed = 0.5
        self.rotation_angle = 0.0
        self.rotation_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)


        self.base_points = fibonacci_sphere(self.n_points, self.base_radius)
        self.normals = self.base_points / np.linalg.norm(
            self.base_points, axis=1, keepdims=True
        )
        self.reactive = False
        self.profile_samples = 256
        self.colatitude_norm = np.arccos(self.normals[:, 1]) / np.pi
        self._last_frame_time = time.perf_counter()
        self._idle_pulse_phase = 0.0
        self._audio_pulse_phase = 0.0
        self.idle_pulse_frequency = 0.45
        self.idle_pulse_amplitude = 0.08
        self.idle_cloud_amplitude = 0.12
        self.audio_pulse_base_frequency = 1.0

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
        ) = self._generate_cloud(rng, self.style.cloud_points, self.style.cloud_radius, self.style.cloud_thickness)
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


    def _rotation_matrix(self, axis, angle_degrees):
        """Crea una matrice di rotazione 3D intorno all'asse dato (x,y,z)."""
        import numpy as np
        angle = np.radians(angle_degrees)
        x, y, z = axis
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c + (1 - c) * x * x,     (1 - c) * x * y - s * z, (1 - c) * x * z + s * y],
            [(1 - c) * y * x + s * z, c + (1 - c) * y * y,     (1 - c) * y * z - s * x],
            [(1 - c) * z * x - s * y, (1 - c) * z * y + s * x, c + (1 - c) * z * z]
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    def update_points(self) -> None:
        # 1) ruota SEMPRE
        self.rotation_angle = (self.rotation_angle + self.rotation_speed) % 360.0
        rotation_matrix = self._rotation_matrix(self.rotation_axis, self.rotation_angle)

        # 2) se non deve reagire all'audio, applica una pulsazione morbida
        if not self.reactive:
            now = time.perf_counter()
            elapsed = now - self._last_frame_time
            self._last_frame_time = now
            self._idle_pulse_phase = (
                self._idle_pulse_phase
                + elapsed * self.idle_pulse_frequency * 2.0 * np.pi
            ) % (2.0 * np.pi)
            pulse_wave = np.sin(self._idle_pulse_phase)
            sphere_scale = 1.0 + self.idle_pulse_amplitude * pulse_wave
            cloud_scale = 1.0 + self.idle_cloud_amplitude 

            scaled_points = self.base_points * sphere_scale
            scaled_cloud = self.cloud_base * cloud_scale
            rotated_points = scaled_points @ rotation_matrix.T
            rotated_cloud = scaled_cloud @ rotation_matrix.T
            self.scatter.setData(pos=rotated_points)
            self.cloud.setData(pos=rotated_cloud)
            return

        # 3) altrimenti: usa l'audio come prima
        now = time.perf_counter()
        elapsed = now - self._last_frame_time
        self._last_frame_time = now
        self._idle_pulse_phase = (
            self._idle_pulse_phase
            + elapsed * self.idle_pulse_frequency * 2.0 * np.pi
        ) % (2.0 * np.pi)
        response_profile = self.analyzer.spectrum_for_points(self.profile_samples)
        phi_axis = np.linspace(0.0, 1.0, self.profile_samples)
        radial_profile = np.interp(self.colatitude_norm, phi_axis, response_profile)

        global_level = response_profile.mean()
        combined = 0.7 * radial_profile + 0.3 * global_level
        volume = self.analyzer.volume_level()
        global_level_clamped = np.clip(global_level, 0.0, 1.0)
        volume_clamped = np.clip(volume, 0.0, 1.0)

        pulse_speed = self.audio_pulse_base_frequency + 3.5 * volume_clamped
        self._audio_pulse_phase = (
            self._audio_pulse_phase + elapsed * pulse_speed * 2.0 * np.pi
        ) % (2.0 * np.pi)
        pulse_wave = np.sin(self._audio_pulse_phase)

        base_radius = self.base_radius + self.analyzer.config.response_strength * combined
        audio_pulse_strength = 0.18 + 0.55 * volume_clamped + 0.35 * global_level_clamped
        radial_offset = base_radius * (1.0 + audio_pulse_strength * 0.28 * pulse_wave)
        radial_offset = np.clip(radial_offset, 0.05, None)

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
        lateral_gain = self.horizontal_strength * (0.25 + 0.55 * audio_pulse_strength)
        new_positions[:, 0] += lateral_gain * profile_x
        new_positions[:, 2] += lateral_gain * profile_z

        # applica la rotazione calcolata all'inizio
        rotated_points = new_positions @ rotation_matrix.T

        # anche la cloud segue l'audio
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
        cloud_pulse = 1.0 + audio_pulse_strength * 0.35 * pulse_wave
        dynamic_cloud_radius = (
            self.cloud_base_radii * cloud_gain * cloud_pulse
            + self.analyzer.config.response_strength * 0.28 * cloud_profile
        )
        cloud_positions = self.cloud_directions * dynamic_cloud_radius[:, np.newaxis]
        cloud_positions[:, 0] += self.horizontal_strength * 0.35 * cloud_profile_x
        cloud_positions[:, 2] += self.horizontal_strength * 0.35 * cloud_profile_z

        rotated_cloud = cloud_positions @ rotation_matrix.T

        self.scatter.setData(pos=rotated_points)
        self.cloud.setData(pos=rotated_cloud)


    def set_reactive(self, on: bool) -> None:
        """Accende o spegne l'animazione guidata dall'audio."""
        self.reactive = on
        self._last_frame_time = time.perf_counter()
        if not on:
            # svuota i dati dell'audio, così al prossimo frame non si deforma
            self.analyzer.mute()
            # rimetti forma/base e alone base
            self.scatter.setData(pos=self.base_points)
            self.cloud.setData(pos=self.cloud_base)

    # ------------------------------------------------------------------
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
    # ------------------------------------------------------------------
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover - evento GUI
        self.timer.stop()
        self.analyzer.stop()
        super().closeEvent(event)
    

__all__ = ["SphereVisualizer", "fibonacci_sphere", "SphereStyle", "pg"]
