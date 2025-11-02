"""Pipeline di addestramento basata sul dataset ``zendod_dataset``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

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

from dataset import load_samples


DEFAULT_CONFIG = "massive"
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_EVAL_SPLIT = "validation"


MODEL_FILENAME = "text_response_model.joblib"


def build_pipeline() -> Pipeline:
    """Restituisce una pipeline TF-IDF + LinearSVC."""

    return Pipeline(
        [
            ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("classifier", LinearSVC()),
        ]
    )


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
) -> Tuple[Pipeline, str]:
    """Addestra la pipeline sul dataset e restituisce un report di valutazione."""

    train_data = _load_split(dataset_dir, config=config, split=train_split)
    if train_data is None:
        raise ValueError(
            "Impossibile caricare lo split di training richiesto. "
            "Verifica che i file JSON del dataset siano presenti."
        )

    train_prompts, train_responses = train_data
    if len(train_prompts) < 4:  # pragma: no cover - dataset molto piccolo
        raise ValueError(
            "Il dataset di training è troppo piccolo per addestrare un modello affidabile."
        )

    eval_data = _load_split(dataset_dir, config=config, split=eval_split)
    if eval_data is None:
        # fallback su suddivisione interna dello split di training
        train_prompts, test_prompts, train_responses, test_responses = train_test_split(
            train_prompts,
            train_responses,
            test_size=0.25,
            random_state=42,
            stratify=train_responses,
        )
    else:
        test_prompts, test_responses = eval_data

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