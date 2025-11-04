"""Tester per il modello di riconoscimento dei parametri (token classification)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import joblib
from sklearn.metrics import classification_report

from dataset import DEFAULT_DATASET_DIR, LEGACY_DATASET_DIR, load_samples
from pipeline_TR import (
    TOKEN_MODEL_FILENAME,
    extract_token_features,
    prepare_token_dataset,
    tokenize,
)


MODEL_PATH = Path("models") / TOKEN_MODEL_FILENAME
DATASET_DIR = Path(DEFAULT_DATASET_DIR)
if not DATASET_DIR.exists() and LEGACY_DATASET_DIR.exists():
    DATASET_DIR = LEGACY_DATASET_DIR

CONFIG = "massive"
SPLIT = "test"


def _predict_slots(
    text: str,
    *,
    vectorizer,
    classifier,
    label_encoder,
) -> List[Tuple[str, str]]:
    tokens = tokenize(text)
    if not tokens:
        return []
    features = [extract_token_features(tokens, idx) for idx in range(len(tokens))]
    X = vectorizer.transform(features)
    predicted = classifier.predict(X)
    labels = label_encoder.inverse_transform(predicted)
    return list(zip(tokens, labels))


def tester(ask: bool = False, model_path: Path = MODEL_PATH) -> None:
    bundle = joblib.load(model_path)
    vectorizer = bundle["vectorizer"]
    classifier = bundle["classifier"]
    label_encoder = bundle["label_encoder"]

    samples = load_samples(DATASET_DIR, config=CONFIG, split=SPLIT)
    features, labels = prepare_token_dataset(samples)
    X = vectorizer.transform(features)
    predicted_ids = classifier.predict(X)
    predicted_labels = label_encoder.inverse_transform(predicted_ids)
    print("\n📊 Risultati sullo split:", SPLIT)
    print(classification_report(labels, predicted_labels, zero_division=0))

    if ask:
        print("\n🤖 Modello parametri pronto! Inserisci una frase (" "'exit' per terminare).\n")
        while True:
            try:
                text = input("🎙️  Tu: ").strip()
                if not text:
                    continue
                if text.lower() in {"exit", "quit", "esci"}:
                    print("👋 Ciao!")
                    break

                predictions = _predict_slots(
                    text,
                    vectorizer=vectorizer,
                    classifier=classifier,
                    label_encoder=label_encoder,
                )
                if not predictions:
                    print("🤖 Nessun token da analizzare.\n")
                    continue

                formatted = " ".join(
                    f"{token}<{label}>" for token, label in predictions
                )
                print(f"🤖 Token etichettati: {formatted}\n")
            except KeyboardInterrupt:
                print("\n👋 Interrotto dall’utente.")
                break


def main() -> None:  # pragma: no cover - esecuzione diretta
    tester(ask=True)


if __name__ == "__main__":  # pragma: no cover - esecuzione diretta
    main()

