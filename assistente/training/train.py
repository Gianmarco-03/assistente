"""Entry point per l'addestramento del modello basato su ``zendod_dataset``."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import MODEL_FILENAME, save_model, train_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Addestra un modello di classificazione delle risposte a partire dal dataset zendod_dataset."
        )
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=str(Path("zendod_dataset")),
        help="Percorso della cartella contenente responses.csv (default: zendod_dataset).",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Cartella dove salvare il modello addestrato (default: models).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pipeline, report = train_model(args.dataset_dir)
    model_path = save_model(pipeline, args.output_dir, filename=MODEL_FILENAME)
    print("Training completato. Report di valutazione:\n")
    print(report)
    print(f"\nModello salvato in: {model_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - esecuzione diretta
    raise SystemExit(main())
