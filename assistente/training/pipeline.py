"""Pipeline di addestramento basata sul dataset MASSIVE (Amazon Science)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
from collections import Counter
import random

try:  # pragma: no cover - dipendenze opzionali
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "È necessario installare scikit-learn per eseguire il training.\n"
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
    from .dataset import load_samples
except ImportError:
    from dataset import load_samples

DEFAULT_CONFIG = "massive_it"
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_EVAL_SPLIT = "validation"


MODEL_FILENAME = "text_response_model.joblib"

def oversample(texts, labels, min_count=30, random_state=42):
    """
    Aumenta artificialmente il numero di esempi per le classi più rare.

    Parametri:
        texts (list[str]): frasi di input.
        labels (list[str]): etichette corrispondenti.
        min_count (int): numero minimo desiderato di esempi per classe.
        random_state (int): seme per la riproducibilità.

    Ritorna:
        new_texts, new_labels: liste con esempi bilanciati.
    """
    random.seed(random_state)
    counter = Counter(labels)
    new_texts, new_labels = list(texts), list(labels)

    for cls, count in counter.items():
        if count < min_count:
            needed = min_count - count
            # campiona esempi esistenti di quella classe
            samples = [t for t, l in zip(texts, labels) if l == cls]
            for _ in range(needed):
                s = random.choice(samples)
                new_texts.append(s)
                new_labels.append(cls)

    print(f"✅ Oversampling completato ({len(labels)} → {len(new_labels)} esempi)")
    return new_texts, new_labels



def build_pipeline() -> Pipeline:
    """Restituisce una pipeline TF-IDF + MLPClassifier."""

    """TF-IDF + MLP (Adam)."""
    return Pipeline([
        ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            learning_rate_init=5e-4,    # prev: 1e-3
            batch_size=64,
            max_iter=100,                # prev: 60
#            early_stopping=True,        # si ferma da solo [new]
            n_iter_no_change=5,         #prev: 10
#            validation_fraction=0.1,
            random_state= 23,
            verbose=True,
        )),
    ])

def _load_split(
    dataset_dir: str | Path,
    *,
    config: str,
    split: str,
) -> Optional[Tuple[list[str], list[str]]]:
    """Carica un singolo split del dataset restituendo testi e intent."""

    try:
        samples = load_samples(dataset_dir, config=config, split=split)
    except FileNotFoundError:
        return None
    except ValueError:
        return None

    prompts = [sample.prompt for sample in samples]
    responses = [sample.response for sample in samples]
    if not prompts or not responses:
        return None
    return prompts, responses


def train_model(
    dataset_dir: str | Path,
    *,
    config: str = DEFAULT_CONFIG,
    train_split: str = DEFAULT_TRAIN_SPLIT,
    eval_split: str = DEFAULT_EVAL_SPLIT,
    Oversample: bool = False
) -> Tuple[Dict[str, Any], str]:
    """Addestra la pipeline sul dataset e restituisce un report di valutazione."""

    dataset_dir = Path(dataset_dir)

    # 1) --- CARICA TRAIN ---
    train_samples = load_samples(dataset_dir, config=config, split=train_split)
    train_texts = [s.prompt for s in train_samples]
    train_labels = [s.intent for s in train_samples]

    # 1.5) --- TENTATOVP DI OVERSAMPLEING
    if(Oversample):
        train_texts, train_labels = oversample(train_texts, train_labels, min_count=30)


    # 2) --- ENCODER SULLE LABEL ---
    #    qui trasformiamo "alarm_query" -> 0, "calendar_set" -> 1, ...
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)

    # 3) --- COSTRUISCI E ALLENA PIPELINE ---
    pipeline = build_pipeline()
    pipeline.fit(train_texts, y_train)

    # 4) --- CARICA EVAL ---
    eval_samples = load_samples(dataset_dir, config=config, split=eval_split)
    eval_texts = [s.prompt for s in eval_samples]
    eval_labels = [s.intent for s in eval_samples]

    # le etichette di valutazione le convertiamo con lo stesso encoder
    y_eval = label_encoder.transform(eval_labels)
    y_pred_enc = pipeline.predict(eval_texts)

    # 5) --- RICONVERTI LE PREDIZIONI IN TESTO ---
    y_pred_labels = label_encoder.inverse_transform(y_pred_enc)

    # 6) --- REPORT TESTUALE ---
    report = classification_report(
        eval_labels,
        y_pred_labels,
        zero_division=0,
    )

    # 7) --- RESTITUISCI UN "BUNDLE" ---
    # perché in fase di inference ci serve ANCHE il label encoder
    bundle: Dict[str, Any] = {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
    }

    return bundle, report



def save_model(pipeline: Pipeline, output_dir: str | Path, *, filename: str = MODEL_FILENAME, Oversample : bool = False) -> Path:
    """Salva la pipeline addestrata su disco."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / filename
    joblib.dump(pipeline, file_path)
    return file_path


__all__ = ["build_pipeline", "train_model", "save_model", "MODEL_FILENAME"]