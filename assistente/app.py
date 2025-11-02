"""Utility di alto livello per avviare la visualizzazione della sfera."""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np

try:  # pragma: no cover - dipendenza opzionale
    import soundfile as sf
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "È necessario installare la libreria 'soundfile' per riprodurre i file audio.\n"
        "Puoi installarla con: pip install soundfile"
    ) from exc

try:  # pragma: no cover - riproduzione opzionale
    import sounddevice as sd
except ImportError:  # pragma: no cover
    sd = None  # type: ignore[assignment]

from PyQt5 import QtWidgets

from assistente.audio.analyzer import AudioAnalyzer
from assistente.audio.config import AudioConfig
from assistente.audio.sources import SoundFileSource
from assistente.visualization.sphere import SphereVisualizer, pg


def create_analyzer_for_file(audio_path: str, config: Optional[AudioConfig] = None) -> AudioAnalyzer:
    """Restituisce un :class:`AudioAnalyzer` configurato per ``audio_path``."""

    cfg = config or AudioConfig()
    source = SoundFileSource(audio_path)
    analyzer = AudioAnalyzer(cfg, source=source)
    return analyzer


def create_qt_application() -> QtWidgets.QApplication:
    """Crea (o riutilizza) l'istanza di :class:`QApplication`."""

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([sys.argv[0]])
    pg.setConfigOptions(antialias=True)
    return app


def run_visualizer(analyzer: AudioAnalyzer, *, window_title: str = "Sfera Spettro 3D", size: int = 900) -> int:
    """Avvia l'interfaccia grafica che visualizza la sfera."""

    app = create_qt_application()
    window = SphereVisualizer(analyzer)
    window.resize(size, size)
    window.setWindowTitle(window_title)
    window.show()

    analyzer.start()
    try:
        return app.exec_()
    finally:
        analyzer.stop()


def playback_audio_file(audio_path: str) -> None:
    """Riproduce ``audio_path`` se la dipendenza ``sounddevice`` è disponibile."""

    if sd is None:
        print("[avviso] Impossibile riprodurre il file audio: installa 'sounddevice' per l'ascolto.")
        return

    data, samplerate = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    sd.play(data, samplerate)
    sd.wait()


__all__ = [
    "create_analyzer_for_file",
    "create_qt_application",
    "run_visualizer",
    "playback_audio_file",
]
