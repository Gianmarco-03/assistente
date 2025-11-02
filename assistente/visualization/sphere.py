"""Widget Qt responsabile della visualizzazione della sfera animata."""

from __future__ import annotations

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

from assistente.audio.analyzer import AudioAnalyzer


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
        self.rotation_speed = np.deg2rad(self.style.rotation_speed_deg)
        self.rotation_angle = 0.0

        self.base_points = fibonacci_sphere(self.n_points, self.base_radius)
        self.normals = self.base_points / np.linalg.norm(
            self.base_points, axis=1, keepdims=True
        )
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

    # ------------------------------------------------------------------
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
