"""Gestione della classificazione dell'input e instradamento verso le azioni."""

from __future__ import annotations
import joblib

from dataclasses import dataclass

from sklearn.calibration import LabelEncoder
from training.pipeline import build_pipeline
from pathlib import Path
from typing import Tuple
from sklearn.pipeline import Pipeline

MODEL_PATH = Path("models/text_response_model.joblib")
import actions

class IntentRouterError(RuntimeError):
    """Errore generico sollevato durante l'analisi dell'input."""


@dataclass(slots=True)
class IntentRouter:
    """Carica il modello addestrato e restituisce l'azione associata all'intent."""

    model_path: Path | None = None
    _pipeline: Pipeline | None = None
    _label_encoder : LabelEncoder | None = None

    def __post_init__(self) -> None:
        if self.model_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            self.model_path = base_dir / "models" / "text_response_model.joblib"
        else:
            self.model_path = Path(self.model_path)
        bundle = joblib.load(MODEL_PATH)
        self._pipeline = bundle['pipeline']
        self._label_encoder = bundle["label_encoder"]

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None and self._label_encoder is not None:
            return
        if self.model_path is None or not self.model_path.exists():
            raise IntentRouterError(
                f"Modello di classificazione non trovato: {self.model_path}"
            )
        try:
            from joblib import load
        except ImportError as exc:  # pragma: no cover - dipendenza opzionale
            raise IntentRouterError(
                "È necessario installare joblib per utilizzare il classificatore"
            ) from exc
        try:
            bundle = load(self.model_path)
        except Exception as exc:  # pragma: no cover - errori di I/O imprevedibili
            raise IntentRouterError(
                f"Impossibile caricare il modello da {self.model_path}: {exc}"
            ) from exc
        try:
            self._pipeline = bundle["pipeline"]
            self._label_encoder = bundle["label_encoder"]
        except Exception as exc:
            raise IntentRouterError(
                "Il file del modello non contiene pipeline e label encoder."
            ) from exc

    def predict_intent(self, text: str) -> str:
        """Restituisce la label dell'intent stimata per ``text``."""

        self._ensure_loaded()
        if self._pipeline is None or self._label_encoder is None:  # pragma: no cover
            raise IntentRouterError("Il modello non è stato inizializzato correttamente.")
        prediction = self._pipeline.predict([text])
        label = self._label_encoder.inverse_transform(prediction)[0]
        return str(label)

    def describe(self, text: str) -> Tuple[str, str]:
        """Restituisce la coppia ``(intent, messaggio)`` per l'input fornito."""

        intent = self.predict_intent(text)
        message = actions.handle(intent)
        return intent, message


__all__ = ["IntentRouter", "IntentRouterError"]
