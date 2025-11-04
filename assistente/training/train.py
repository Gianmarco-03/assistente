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
from pipeline_TR import (
    TOKEN_MODEL_FILENAME,
    save_token_model,
    train_token_model,
)

OVERSAMPLE_SWICH = False
DATASET_DIR = "MASSIVE_dataset"

PARAM: dict[str, object] = {
    "dataset_dir": str(DATASET_DIR),
    "output_dir": "models",
    "config": DEFAULT_CONFIG,
    "train_split": DEFAULT_TRAIN_SPLIT,
    "eval_split": DEFAULT_EVAL_SPLIT,
    "task": "slots",
    "oversample": False,
}



def train() -> int:
    # 1) allenamento
    params = PARAM
    task = params["task"]

    output_dir = Path(params["output_dir"])
    losses: list[dict] = []

    if task == "intent":
        print("====================================")
        print("starting Intent Classifier training")
        print("====================================")
        bundle, report = train_model(
            params["dataset_dir"],
            config=params["config"],
            train_split=params["train_split"],
            eval_split=params["eval_split"],
            Oversample=params.get("oversample", False),
        )

        filename = MODEL_FILENAME
        oversample = params.get("oversample", False)
        if oversample:
            filename = "text_response_model_oversample.joblib"
        model_path = save_model(
            bundle,
            output_dir,
            filename=filename,
            Oversample=oversample,
        )

        pipeline = bundle.get("pipeline")
        if oversample:
            log_path = output_dir / "training_oversample_log.json"
        else:
            log_path = output_dir / "training_log.json"

        if pipeline is not None:
            clf = getattr(pipeline, "named_steps", {}).get("classifier")
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
    else:
        print("====================================")
        print("starting Token Recognizer training")
        print("====================================")
        bundle, report = train_token_model(
            params["dataset_dir"],
            config=params["config"],
            train_split=params["train_split"],
            eval_split=params["eval_split"],
        )
        model_path = save_token_model(
            bundle,
            output_dir,
            filename=TOKEN_MODEL_FILENAME,
        )
        print(
            "\nℹ️ Modello di riconoscimento parametri addestrato senza loss curve disponibile."
        )

    if losses:
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
    return train()

if __name__ == "__main__":  # pragma: no cover - esecuzione diretta
    raise SystemExit(main())