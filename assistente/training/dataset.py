"""Utility per caricare il dataset ``zendod_dataset``."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(slots=True)
class Sample:
    """Singolo esempio input/risposta."""

    prompt: str
    response: str


def load_samples(dataset_dir: str | Path, *, filename: str = "responses.csv") -> List[Sample]:
    """Carica il dataset atteso in formato CSV."""

    path = Path(dataset_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"La cartella del dataset non esiste: {path}. Copia i dati in 'zendod_dataset/'."
        )

    csv_path = path / filename
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Impossibile trovare il file {filename} nella cartella {path}."
            " Il file deve contenere le colonne 'prompt' e 'response'."
        )

    samples: List[Sample] = []
    with csv_path.open("r", encoding="utf8") as handle:
        reader = csv.DictReader(handle)
        missing = {"prompt", "response"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Colonne mancanti nel dataset: {', '.join(sorted(missing))}."
                " Assicurati che il CSV contenga le intestazioni corrette."
            )
        for row in reader:
            prompt = row.get("prompt", "").strip()
            response = row.get("response", "").strip()
            if not prompt or not response:
                continue
            samples.append(Sample(prompt=prompt, response=response))

    if not samples:
        raise ValueError("Il dataset è vuoto: aggiungi almeno un esempio valido.")

    return samples


def iter_samples(dataset_dir: str | Path, *, filename: str = "responses.csv") -> Iterable[Sample]:
    """Generator lineare sui campioni del dataset."""

    for sample in load_samples(dataset_dir, filename=filename):
        yield sample


__all__ = ["Sample", "load_samples", "iter_samples"]
