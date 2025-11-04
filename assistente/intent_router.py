"""Gestione della classificazione dell'input, degli slot e instradamento."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
from sklearn.calibration import LabelEncoder
from sklearn.pipeline import Pipeline

import actions
from training.pipeline_TR import (
    TOKEN_MODEL_FILENAME,
    extract_token_features,
    tokenize,
)


class IntentRouterError(RuntimeError):
    """Errore generico sollevato durante l'analisi dell'input."""


class SlotRecognizerError(IntentRouterError):
    """Errore sollevato durante l'estrazione dei parametri (slot)."""


@dataclass(slots=True)
class IntentRouter:
    """Carica il modello addestrato e restituisce l'azione associata all'intent."""

    model_path: Path | None = None
    _pipeline: Pipeline | None = None
    _label_encoder: LabelEncoder | None = None

    def __post_init__(self) -> None:
        if self.model_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            self.model_path = base_dir / "models" / "text_response_model.joblib"
        else:
            self.model_path = Path(self.model_path)
        bundle = joblib.load(self.model_path)
        self._pipeline = bundle["pipeline"]
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


def _resolve_token_model_path(model_path: Path | None = None) -> Path:
    if model_path is not None:
        return Path(model_path).resolve()
    base_dir = Path(__file__).resolve().parent.parent
    return (base_dir / "models" / TOKEN_MODEL_FILENAME).resolve()


@lru_cache(maxsize=None)
def _load_token_bundle(model_path: Path) -> Tuple[object, object, LabelEncoder]:
    try:
        bundle = joblib.load(model_path)
    except FileNotFoundError as exc:  # pragma: no cover - I/O dipendente
        raise SlotRecognizerError(
            f"Modello parametri non trovato: {model_path}"
        ) from exc
    except Exception as exc:  # pragma: no cover - errori joblib
        raise SlotRecognizerError(
            f"Impossibile caricare il modello dei parametri: {exc}"
        ) from exc
    try:
        vectorizer = bundle["vectorizer"]
        classifier = bundle["classifier"]
        label_encoder: LabelEncoder = bundle["label_encoder"]
    except KeyError as exc:  # pragma: no cover - bundle malformato
        raise SlotRecognizerError(
            "Il modello parametri non contiene gli oggetti necessari."
        ) from exc
    return vectorizer, classifier, label_encoder


def _parse_label(label: str) -> Tuple[str | None, str | None]:
    if not label or label == "O":
        return None, None
    prefix, sep, name = label.partition("-")
    if not sep or prefix not in {"B", "I"} or not name:
        return None, None
    return prefix, name


def _format_tokens(tokens: Sequence[str]) -> str:
    if not tokens:
        return ""
    text = " ".join(tokens)
    text = re.sub(r"\s+([?!.,;:])", r"\1", text)
    text = re.sub(r"\s+'", "'", text)
    text = re.sub(r"'\s+", "'", text)
    text = re.sub(r"\s+([)\]\}])", r"\1", text)
    text = re.sub(r"([(\[\{])\s+", r"\1", text)
    return text.strip()


def _collect_slots(tokens: Sequence[str], labels: Sequence[str]) -> Dict[str, List[str]]:
    slots: Dict[str, List[str]] = defaultdict(list)
    current_name: str | None = None
    current_tokens: List[str] = []

    def _flush() -> None:
        nonlocal current_name, current_tokens
        if current_name and current_tokens:
            slots[current_name].append(_format_tokens(current_tokens))
        current_name = None
        current_tokens = []

    for token, raw_label in zip(tokens, labels):
        prefix, name = _parse_label(str(raw_label))
        if prefix is None:
            _flush()
            continue
        if prefix == "B" or name != current_name:
            _flush()
            current_name = name
            current_tokens = [token]
        else:
            current_tokens.append(token)

    _flush()
    return {key: values for key, values in slots.items() if values}


def _build_annotation(tokens: Sequence[str], labels: Sequence[str]) -> str:
    parts: List[str] = []
    plain_tokens: List[str] = []
    slot_tokens: List[str] = []
    slot_name: str | None = None

    def _flush_plain() -> None:
        nonlocal plain_tokens
        if plain_tokens:
            parts.append(_format_tokens(plain_tokens))
            plain_tokens = []

    def _flush_slot() -> None:
        nonlocal slot_tokens, slot_name
        if slot_name and slot_tokens:
            parts.append(f"[{slot_name}: {_format_tokens(slot_tokens)}]")
        slot_name = None
        slot_tokens = []

    for token, raw_label in zip(tokens, labels):
        prefix, name = _parse_label(str(raw_label))
        if prefix is None:
            _flush_slot()
            plain_tokens.append(token)
            continue
        if prefix == "B" or name != slot_name:
            _flush_slot()
            _flush_plain()
            slot_name = name
            slot_tokens = [token]
        else:
            slot_tokens.append(token)

    _flush_slot()
    _flush_plain()

    annotated = " ".join(part for part in parts if part)
    annotated = re.sub(r"\s+([?!.,;:])", r"\1", annotated)
    annotated = re.sub(r"\s+'", "'", annotated)
    annotated = re.sub(r"'\s+", "'", annotated)
    return annotated.strip()


def get_annot_utt(frase: str, *, model_path: Path | None = None) -> Tuple[str, Dict[str, List[str]]]:
    """Restituisce ``annot_utt`` e slot estratti dal testo fornito."""

    text = frase or ""
    tokens = tokenize(text)
    if not tokens:
        return "", {}

    resolved_path = _resolve_token_model_path(model_path)
    vectorizer, classifier, label_encoder = _load_token_bundle(resolved_path)

    features = [extract_token_features(tokens, index) for index in range(len(tokens))]
    matrix = vectorizer.transform(features)
    predicted = classifier.predict(matrix)
    labels = label_encoder.inverse_transform(predicted)

    annotated = _build_annotation(tokens, labels)
    slots = _collect_slots(tokens, labels)
    return annotated or text, slots


__all__ = [
    "IntentRouter",
    "IntentRouterError",
    "SlotRecognizerError",
    "get_annot_utt",
]