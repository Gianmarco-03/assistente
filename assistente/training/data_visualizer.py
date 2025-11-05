"""Utility per esplorare la distribuzione delle intent nel dataset MASSIVE.

Il modulo fornisce sia un'API programmatica sia una piccola CLI:

    python -m assistente.training.data_visualizer --show

che stampa alcune statistiche riassuntive e (opzionalmente) genera
un grafico della distribuzione delle etichette.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - dipendenza opzionale
    plt = None  # type: ignore[assignment]

from dataset import (
    DEFAULT_DATASET_DIR,
    LEGACY_DATASET_DIR,
    load_samples,
)
try:
    from .pipeline import DEFAULT_CONFIG as _DEFAULT_CONFIG, DEFAULT_TRAIN_SPLIT as _DEFAULT_TRAIN_SPLIT, oversample as _pipeline_oversample
except (ImportError, SystemExit):  # pragma: no cover - dipendenze opzionali assenti
    _pipeline_oversample = None
    _DEFAULT_CONFIG = "massive_it"
    _DEFAULT_TRAIN_SPLIT = "train"

DEFAULT_CONFIG = _DEFAULT_CONFIG
DEFAULT_TRAIN_SPLIT = _DEFAULT_TRAIN_SPLIT


def _local_oversample(texts: Sequence[str], labels: Sequence[str], *, min_count: int, random_state: int = 42) -> tuple[list[str], list[str]]:
    counter = Counter(labels)
    new_texts = list(texts)
    new_labels = list(labels)
    if not counter:
        return new_texts, new_labels
    rng = random.Random(random_state)
    for cls, count in counter.items():
        if count >= min_count:
            continue
        needed = min_count - count
        samples = [text for text, label in zip(texts, labels) if label == cls]
        if not samples:
            continue
        for _ in range(needed):
            new_texts.append(rng.choice(samples))
            new_labels.append(cls)
    return new_texts, new_labels


def _apply_oversample(texts: list[str], labels: list[str], *, min_count: int, random_state: int = 42) -> tuple[list[str], list[str]]:
    if _pipeline_oversample is not None:
        new_texts, new_labels = _pipeline_oversample(texts, labels, min_count=min_count)
        return list(new_texts), list(new_labels)

    new_texts, new_labels = _local_oversample(texts, labels, min_count=min_count, random_state=random_state)
    if len(new_labels) != len(labels):
        print(f"✅ Oversampling completato ({len(labels)} → {len(new_labels)} esempi)")
    return new_texts, new_labels


@dataclass(slots=True)
class LabelStats:
    """Statistiche per una singola etichetta."""

    label: str
    count: int
    percentage: float


def _resolve_dataset_dir(dataset_dir: str | Path | None) -> Path:
    """Restituisce la directory del dataset, provando valori di fallback."""

    candidates: list[Path] = []
    if dataset_dir is not None:
        candidates.append(Path(dataset_dir).expanduser())
    candidates.extend([DEFAULT_DATASET_DIR, LEGACY_DATASET_DIR])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Impossibile trovare il dataset. Specifica --dataset-dir oppure copia i "
        "file in 'massive_dataset/' (o nella cartella legacy 'zendod_dataset/')."
    )


def _compute_distribution(labels: Sequence[str]) -> list[LabelStats]:
    """Calcola conteggi e percentuali a partire da una sequenza di etichette."""

    counter = Counter(labels)
    total = sum(counter.values())
    if total == 0:
        return []

    stats = [
        LabelStats(label=label, count=count, percentage=count / total * 100)
        for label, count in counter.most_common()
    ]
    return stats


def describe_labels(labels: Sequence[str]) -> list[LabelStats]:
    """Restituisce statistiche ordinate per frequenza decrescente."""

    return _compute_distribution(labels)


def print_summary(stats: Iterable[LabelStats]) -> None:
    """Stampa una tabella con conteggi e percentuali."""

    stats = list(stats)
    if not stats:
        print("⚠️ Nessuna etichetta disponibile per lo split selezionato.")
        return

    total = sum(entry.count for entry in stats)
    print(f"\n📊 Esempi totali: {total}\n")
    print("Etichetta                          Conteggio    Percentuale")
    print("-" * 60)
    for entry in stats:
        print(f"{entry.label:30s} {entry.count:10d}   {entry.percentage:6.2f}%")


def plot_distribution(
    stats: Sequence[LabelStats],
    *,
    title: str = "Distribuzione delle intent",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Visualizza (e opzionalmente salva) un grafico a barre della distribuzione."""

    if plt is None:
        print("⚠️ Matplotlib non è installato: impossibile generare il grafico.")
        if save_path is not None:
            print("   (Parametro --save-plot ignorato)")
        return

    if not stats:
        print("⚠️ Distribuzione vuota: nessun grafico generato.")
        return

    labels = [entry.label for entry in stats]
    counts = [entry.count for entry in stats]

    plt.figure(figsize=(max(10, len(labels) * 0.3), 6))
    bars = plt.bar(labels, counts)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.xlabel("Intent")
    plt.ylabel("Numero di esempi")

    for bar, entry in zip(bars, stats):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{entry.percentage:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output)
        print(f"💾 Grafico salvato in: {output}")

    if show:
        plt.show()
    else:
        plt.close()


def _load_labels(
    *,
    dataset_dir: Path,
    config: str,
    split: str,
) -> tuple[list[str], list[str]]:
    samples = load_samples(dataset_dir, config=config, split=split)
    prompts = [sample.prompt for sample in samples]
    labels = [sample.intent for sample in samples]
    return prompts, labels


def visualize_dataset(
    *,
    dataset_dir: str | Path | None = None,
    config: str = DEFAULT_CONFIG,
    split: str = DEFAULT_TRAIN_SPLIT,
    oversample_min_count: int | None = None,
    save_plot: str | Path | None = None,
    show_plot: bool = True,
) -> list[LabelStats]:
    """Carica un dataset e restituisce le statistiche della distribuzione."""

    resolved_dir = _resolve_dataset_dir(dataset_dir)
    prompts, labels = _load_labels(dataset_dir=resolved_dir, config=config, split=split)

    if oversample_min_count is not None and oversample_min_count > 0:
        prompts, labels = _apply_oversample(prompts, labels, min_count=oversample_min_count)
        title_suffix = f" (oversample≥{oversample_min_count})"
    else:
        title_suffix = ""

    stats = describe_labels(labels)
    print_summary(stats)
    plot_distribution(
        stats,
        title=f"Distribuzione intent – {config}/{split}{title_suffix}",
        save_path=save_plot,
        show=show_plot,
    )
    return stats


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Esplora la distribuzione delle intent nel dataset MASSIVE.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Percorso della cartella del dataset (di default prova massive_dataset/ e zendod_dataset/).",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Configurazione del dataset da caricare (es. massive).")
    parser.add_argument(
        "--split",
        default=DEFAULT_TRAIN_SPLIT,
        help="Split da analizzare (train, validation, test, ...).",
    )
    parser.add_argument(
        "--oversample-min-count",
        type=int,
        default=None,
        help="Esegue l'oversampling portando ogni classe almeno a questo numero di esempi.",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        help="Se specificato salva il grafico in questo percorso.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Non mostrare la finestra interattiva di matplotlib.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    visualize_dataset(
        dataset_dir=args.dataset_dir,
        config=args.config,
        split=args.split,
        oversample_min_count=args.oversample_min_count,
        save_plot=args.save_plot,
        show_plot=not args.no_show,
    )


if __name__ == "__main__":  # pragma: no cover - entry point a riga di comando
    main()
