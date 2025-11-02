"""Utility per caricare il dataset ``zendod_dataset``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(slots=True)
class Sample:
    """Singolo esempio ``utt``/``intent`` dal dataset ITALIC."""

    utterance: str
    intent: str

    @property
    def prompt(self) -> str:
        """Compatibilità con la vecchia interfaccia basata su ``prompt``."""

        return self.utterance

    @property
    def response(self) -> str:
        """Compatibilità con la vecchia interfaccia basata su ``response``."""

        return self.intent


def _resolve_split_file(dataset_dir: Path, config: str, split: str) -> Path:
    """Restituisce il percorso del file JSON lines per la configurazione richiesta."""

    filename = f"{config}_{split}.json"
    file_path = dataset_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(
            "Impossibile trovare il file dello split richiesto: "
            f"{file_path}. Assicurati che il dataset sia stato estratto in 'zendod_dataset/'."
        )
    return file_path


def load_samples(
    dataset_dir: str | Path,
    *,
    config: str = "massive",
    split: str = "train",
) -> List[Sample]:
    """Carica gli esempi ``utt``/``intent`` da un file JSON lines dello ITALIC dataset."""

    path = Path(dataset_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"La cartella del dataset non esiste: {path}. Copia i dati in 'zendod_dataset/'."
        )

    json_path = _resolve_split_file(path, config=config, split=split)

    samples: List[Sample] = []
    with json_path.open("r", encoding="utf8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Impossibile decodificare la riga "
                    f"{line_number} di {json_path}: {exc}"
                ) from exc

            utterance = str(record.get("utt", "")).strip()
            intent = str(record.get("intent", "")).strip()
            if not utterance or not intent:
                # Salta esempi incompleti
                continue
            samples.append(Sample(utterance=utterance, intent=intent))

    if not samples:
        raise ValueError(
            "Il dataset è vuoto o non contiene esempi validi per lo split richiesto."
        )

    return samples


def iter_samples(
    dataset_dir: str | Path,
    *,
    config: str = "massive",
    split: str = "train",
) -> Iterable[Sample]:
    """Generator lineare sui campioni del dataset."""

    yield from load_samples(dataset_dir, config=config, split=split)


__all__ = ["Sample", "load_samples", "iter_samples"]