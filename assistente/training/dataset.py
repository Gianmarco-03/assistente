"""Utility per caricare il dataset MASSIVE (Amazon Science)."""

from __future__ import annotations

import json
import csv
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any


DEFAULT_DATASET_DIR = Path("massive_dataset")
LEGACY_DATASET_DIR = Path("zendod_dataset")


@dataclass(slots=True)
class Sample:
    """Singolo esempio ``utt``/``intent`` dal dataset MASSIVE."""

    utterance: str
    intent: str
    slots: Dict[str, Any] | None = None


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

    filename = f"{config}_{split}.csv"
    file_path = dataset_dir / filename
    if not file_path.exists():
        raise FileNotFoundError(
            "Impossibile trovare il file dello split richiesto: "
            f"{file_path}. Assicurati che il dataset MASSIVE sia stato estratto in 'massive_dataset/' "
            "(oppure nella cartella legacy 'zendod_dataset/')."
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

    dataset_root = path
    massive_dir = dataset_root / "MASSIVE_dataset"
    if massive_dir.exists() and massive_dir.is_dir():
        dataset_root = massive_dir

    csv_candidates = []
    csv_candidates.append(dataset_root / "massive_it_train.csv")
    candidate_by_config = dataset_root / f"{config}_{split}.csv"
    if candidate_by_config.name != csv_candidates[0].name:
        csv_candidates.append(candidate_by_config)

    csv_path = next((candidate for candidate in csv_candidates if candidate.exists()), None)

    samples: List[Sample] = []
    if csv_path is not None:
        try:
            dataframe = pd.read_csv(csv_path)
        except Exception as exc:  # pragma: no cover - delega alla libreria
            raise ValueError(
                f"Impossibile leggere il file CSV {csv_path}: {exc}"
            ) from exc

        normalized_columns = {col.strip().lower(): col for col in dataframe.columns}

        def _resolve_column(*aliases: str) -> str | None:
            for alias in aliases:
                if alias in normalized_columns:
                    return normalized_columns[alias]
            return None

        utterance_column = _resolve_column("utt", "utterance", "text")
        intent_column = _resolve_column("intent", "label", "response")

        if utterance_column is None or intent_column is None:
            raise ValueError(
                "Il file CSV non contiene colonne compatibili per 'utt'/'intent'."
            )

        for utterance_value, intent_value in dataframe[[utterance_column, intent_column]].itertuples(
            index=False,
            name=None,
        ):
            utterance = str(utterance_value).strip()
            intent = str(intent_value).strip()
            if not utterance or not intent:
                continue
            samples.append(Sample(utterance=utterance, intent=intent))
    else:
        json_path = _resolve_split_file(path, config=config, split=split)

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


__all__ = [
    "Sample",
    "load_samples",
    "iter_samples",
    "DEFAULT_DATASET_DIR",
    "LEGACY_DATASET_DIR",
]