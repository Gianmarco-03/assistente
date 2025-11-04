"""Entry point per l'addestramento sul dataset MASSIVE (Amazon Science)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from loss_visualizaer import plot_loss_from_list
from dataset import DEFAULT_DATASET_DIR, LEGACY_DATASET_DIR

from pipeline import (
    DEFAULT_CONFIG,
    DEFAULT_EVAL_SPLIT,
    DEFAULT_TRAIN_SPLIT,
    MODEL_FILENAME,
    save_model,
    train_model,
)

OVERSAMPLE_SWICH = False
DATASET_DIR = 'MASSIVE_dataset'


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Addestra un modello di classificazione delle risposte a partire dal dataset {DATASET_DIR}."
        )
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
 default=str(DEFAULT_DATASET_DIR),
        help=(
            "Percorso della cartella contenente i file JSON MASSIVE (default: massive_dataset). "
            "Se la cartella non esiste verrà usata automaticamente la directory legacy "
            f"{LEGACY_DATASET_DIR}."
        ),    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Cartella dove salvare il modello addestrato (default: models).",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=(
            "Nome della configurazione del dataset (ad es. massive, hard_noisy, hard_speaker)."
        ),
    )
    parser.add_argument(
        "--train-split",
        default=DEFAULT_TRAIN_SPLIT,
        help="Nome dello split di training da utilizzare (default: train).",
    )
    parser.add_argument(
        "--eval-split",
        default=DEFAULT_EVAL_SPLIT,
        help="Nome dello split di valutazione da utilizzare (default: validation).",
    )
    return parser.parse_args(argv)


def train(argv: list[str] | None = None, Oversample : bool = False) -> int:
    # 1) allenamento
    args = parse_args(argv)
    bundle, report = train_model(
        args.dataset_dir,
        config=args.config,
        train_split=args.train_split,
        eval_split=args.eval_split,
        Oversample=OVERSAMPLE_SWICH
    )
     # 2) salva modello
    output_dir = Path(args.output_dir)
    filename = MODEL_FILENAME
    if Oversample:
        filename = "text_response_model_oversample.joblib"
    model_path = save_model(bundle, output_dir, filename=filename,Oversample=OVERSAMPLE_SWICH)

    # 3)  salva le loss se il classificatore le espone
    #    (succede se in pipeline.py MLPClassifier(solver="adam", ...))
    pipeline = bundle['pipeline']
    if Oversample:
        log_path = output_dir / "training_oversample_log.json"    
    else:
        log_path = output_dir / "training_log.json"
    losses: list[dict] = []

    clf = pipeline.named_steps.get("classifier", None)
    if clf is not None and hasattr(clf, "loss_curve_"):
        losses = [
            {"epoch": i + 1, "train_loss": loss}
            for i, loss in enumerate(clf.loss_curve_)
        ]
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(losses, f, indent=2, ensure_ascii=False)
        print(f"\n✅ File delle loss salvato in: {log_path}")
    else:
        print("\n⚠️ Nessuna loss da salvare (il classifier non espone 'loss_curve_').")

 # 4) Mostra / salva il plot usando la funzione importata
    if losses:
        # mostra il grafico e salvalo anche in PNG
        plot_loss_from_list(
            losses,
            output_file=output_dir / "loss_plot.png",
            show=True,
        )

    # 5) Report    
    print("Training completato. Report di valutazione:\n")
    print(report)
    print(f"\nModello salvato in: {model_path}")
    return 0

def main(argv: list[str] | None = None) -> int:
    train(argv, Oversample=OVERSAMPLE_SWICH)

if __name__ == "__main__":  # pragma: no cover - esecuzione diretta
    raise SystemExit(main())