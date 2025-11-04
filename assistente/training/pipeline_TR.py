"""Pipeline dedicata al riconoscimento dei parametri (token classification)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import re

try:  # pragma: no cover - dipendenze opzionali
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import LabelEncoder
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "È necessario installare scikit-learn per eseguire il training dei parametri.\n"
        "Puoi installarlo con: pip install scikit-learn"
    ) from exc

try:  # pragma: no cover
    import joblib
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "È necessario installare joblib per salvare il modello addestrato.\n"
        "Puoi installarla con: pip install joblib"
    ) from exc

try:
    from .dataset import (
        DEFAULT_DATASET_DIR,
        LEGACY_DATASET_DIR,
        Sample,
        load_samples,
    )
except ImportError:  # pragma: no cover - compatibilità con esecuzione diretta
    from dataset import (  # type: ignore
        DEFAULT_DATASET_DIR,
        LEGACY_DATASET_DIR,
        Sample,
        load_samples,
    )

try:
    from .pipeline import (  # type: ignore
        DEFAULT_CONFIG,
        DEFAULT_EVAL_SPLIT,
        DEFAULT_TRAIN_SPLIT,
    )
except Exception:  # pragma: no cover - compatibilità se pipeline non disponibile
    DEFAULT_CONFIG = "massive"
    DEFAULT_TRAIN_SPLIT = "train"
    DEFAULT_EVAL_SPLIT = "validation"


TOKEN_MODEL_FILENAME = "token_response_model.joblib"
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Tokenizzazione di base condivisa tra training e inference."""

    text = text or ""
    return _TOKEN_PATTERN.findall(text)


def extract_token_features(tokens: Sequence[str], index: int) -> Dict[str, object]:
    """Genera un dizionario di feature contestuali per il token dato."""

    token = tokens[index]
    lower = token.lower()
    features: Dict[str, object] = {
        "bias": 1.0,
        "token.lower": lower,
        "token.isalpha": token.isalpha(),
        "token.isdigit": token.isdigit(),
        "token.istitle": token.istitle(),
        "prefix3": lower[:3],
        "suffix3": lower[-3:],
    }

    if index > 0:
        prev = tokens[index - 1]
        features.update(
            {
                "-1.token.lower": prev.lower(),
                "-1.isupper": prev.isupper(),
            }
        )
    else:
        features["BOS"] = True  # Begin of sentence

    if index + 1 < len(tokens):
        nxt = tokens[index + 1]
        features.update(
            {
                "+1.token.lower": nxt.lower(),
                "+1.isupper": nxt.isupper(),
            }
        )
    else:
        features["EOS"] = True  # End of sentence

    return features


def _ensure_alignment(tokens: List[str], tags: List[str]) -> Tuple[List[str], List[str]]:
    """Si assicura che lunghezza dei tag corrisponda ai token."""

    if len(tokens) == len(tags):
        return tokens, tags
    if not tokens:
        return [], []
    if not tags:
        return tokens, ["O"] * len(tokens)
    if len(tags) < len(tokens):
        tags = tags + ["O"] * (len(tokens) - len(tags))
    else:
        tags = tags[: len(tokens)]
    return tokens, tags


def prepare_token_dataset(samples: Sequence[Sample]) -> Tuple[List[Dict[str, object]], List[str]]:
    """Converte gli esempi annotati in feature per token e relative etichette."""

    feature_dicts: List[Dict[str, object]] = []
    labels: List[str] = []

    for sample in samples:
        tokens = list(sample.tokens or tokenize(sample.utterance))
        tags = list(sample.slot_tags or [])
        tokens, tags = _ensure_alignment(tokens, tags)
        for index, tag in enumerate(tags):
            feature = extract_token_features(tokens, index)
            feature_dicts.append(feature)
            labels.append(tag)

    return feature_dicts, labels


def build_token_classifier() -> LogisticRegression:
    """Restituisce il classificatore per i tag di slot."""

    return LogisticRegression(max_iter=200, class_weight="balanced")


def train_token_model(
    dataset_dir: str | Path,
    *,
    config: str = DEFAULT_CONFIG,
    train_split: str = DEFAULT_TRAIN_SPLIT,
    eval_split: str = DEFAULT_EVAL_SPLIT,
) -> Tuple[Dict[str, object], str]:
    """Addestra il modello di token classification sui parametri."""

    dataset_dir = Path(dataset_dir)

    train_samples = load_samples(dataset_dir, config=config, split=train_split)
    train_features, train_labels = prepare_token_dataset(train_samples)

    vectorizer = DictVectorizer(sparse=True)
    X_train = vectorizer.fit_transform(train_features)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)

    classifier = build_token_classifier()
    classifier.fit(X_train, y_train)

    eval_samples = load_samples(dataset_dir, config=config, split=eval_split)
    eval_features, eval_labels = prepare_token_dataset(eval_samples)
    X_eval = vectorizer.transform(eval_features)
    y_eval = label_encoder.transform(eval_labels)
    y_pred = classifier.predict(X_eval)
    predicted_labels = label_encoder.inverse_transform(y_pred)

    report = classification_report(
        eval_labels,
        predicted_labels,
        zero_division=0,
    )

    bundle: Dict[str, object] = {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "label_encoder": label_encoder,
        "config": {
            "dataset_dir": str(dataset_dir),
            "config": config,
            "train_split": train_split,
            "eval_split": eval_split,
        },
    }

    return bundle, report


def save_token_model(bundle: Dict[str, object], output_dir: str | Path, *, filename: str = TOKEN_MODEL_FILENAME) -> Path:
    """Salva su disco il modello addestrato per il riconoscimento dei parametri."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / filename
    joblib.dump(bundle, file_path)
    return file_path


__all__ = [
    "TOKEN_MODEL_FILENAME",
    "build_token_classifier",
    "prepare_token_dataset",
    "train_token_model",
    "save_token_model",
    "tokenize",
    "extract_token_features",
]

