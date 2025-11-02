"""Pipeline di addestramento basata sul dataset ``zendod_dataset``."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

try:  # pragma: no cover - dipendenze opzionali
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
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

from .dataset import load_samples


MODEL_FILENAME = "text_response_model.joblib"


def build_pipeline() -> Pipeline:
    """Restituisce una pipeline TF-IDF + LinearSVC."""

    return Pipeline(
        [
            ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("classifier", LinearSVC()),
        ]
    )


def train_model(dataset_dir: str | Path) -> Tuple[Pipeline, str]:
    """Addestra la pipeline sul dataset e restituisce un report di valutazione."""

    samples = load_samples(dataset_dir)
    prompts = [sample.prompt for sample in samples]
    responses = [sample.response for sample in samples]

    if len(samples) < 4:  # pragma: no cover - dataset molto piccolo
        raise ValueError(
            "Il dataset è troppo piccolo per una suddivisione train/test."
            " Aggiungi almeno quattro esempi in zendod_dataset/responses.csv."
        )

    train_prompts, test_prompts, train_responses, test_responses = train_test_split(
        prompts, responses, test_size=0.25, random_state=42, stratify=responses
    )

    pipeline = build_pipeline()
    pipeline.fit(train_prompts, train_responses)

    predicted = pipeline.predict(test_prompts)
    report = classification_report(test_responses, predicted)

    return pipeline, report


def save_model(pipeline: Pipeline, output_dir: str | Path, *, filename: str = MODEL_FILENAME) -> Path:
    """Salva la pipeline addestrata su disco."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / filename
    joblib.dump(pipeline, file_path)
    return file_path


__all__ = ["build_pipeline", "train_model", "save_model", "MODEL_FILENAME"]
