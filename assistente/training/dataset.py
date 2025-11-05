"""Utility per caricare il dataset MASSIVE (Amazon Science)."""

from __future__ import annotations

import json
import re
try:
    import pandas as pd
except ImportError:  # pragma: no cover - dipendenza opzionale
    pd = None  # type: ignore[assignment]
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple


DEFAULT_DATASET_DIR = Path("massive_dataset")
LEGACY_DATASET_DIR = Path("zendod_dataset")


@dataclass(slots=True)
class Sample:
    """Singolo esempio ``utt``/``intent`` dal dataset MASSIVE."""

    utterance: str
    intent: str
    slots: Dict[str, Any] | None = None
    annotated_utterance: str | None = None
    tokens: List[str] | None = None
    slot_tags: List[str] | None = None


    @property
    def prompt(self) -> str:
        """Compatibilità con la vecchia interfaccia basata su ``prompt``."""

        return self.utterance

    @property
    def response(self) -> str:
        """Compatibilità con la vecchia interfaccia basata su ``response``."""

        return self.intent


_SLOT_PATTERN = re.compile(r"\[([^:\]]+)\s*:\s*([^\]]+)\]")


def _tokenize(text: str) -> List[str]:
    """Suddivide una stringa in token (parole e punteggiatura)."""

    if not text:
        return []
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def parse_annotated_utterance(
    annotated_text: str | None,
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """Estrae token, tag BIO e valori slot da ``annot_utt``.

    Restituisce tre liste parallele: ``tokens`` e ``slot_tags``
    (uno per token) e un dizionario ``slots`` che mappa ogni
    tipo di parametro ai valori individuati.
    """

    if not annotated_text:
        return [], [], {}

    tokens: List[str] = []
    tags: List[str] = []
    slot_values: Dict[str, List[str]] = {}

    cursor = 0
    for match in _SLOT_PATTERN.finditer(annotated_text):
        prefix = annotated_text[cursor : match.start()]
        prefix_tokens = _tokenize(prefix)
        tokens.extend(prefix_tokens)
        tags.extend(["O"] * len(prefix_tokens))

        slot_name = match.group(1).strip()
        slot_text = match.group(2).strip()
        slot_tokens = _tokenize(slot_text)
        if slot_tokens:
            tag_sequence = [f"B-{slot_name}"] + [f"I-{slot_name}"] * (len(slot_tokens) - 1)
            tags.extend(tag_sequence)
            tokens.extend(slot_tokens)
            slot_values.setdefault(slot_name, []).append(slot_text)
        cursor = match.end()

    suffix = annotated_text[cursor:]
    suffix_tokens = _tokenize(suffix)
    tokens.extend(suffix_tokens)
    tags.extend(["O"] * len(suffix_tokens))

    return tokens, tags, slot_values


def _resolve_split_file(dataset_dir: Path, config: str, split: str) -> Path:
    """Restituisce il percorso del file di uno split del dataset."""

    base_name = f"{config}_{split}"
    candidates = [
        dataset_dir / f"{base_name}.csv",
        dataset_dir / f"{base_name}.json",
        dataset_dir / f"{base_name}.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Impossibile trovare il file dello split richiesto. "
        "Assicurati che il dataset MASSIVE sia stato estratto in 'massive_dataset/' "
        "(oppure nella cartella legacy 'zendod_dataset/')."
    )


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

    csv_candidates: list[Path] = []
    csv_path: Path | None = None
    if pd is not None:
        csv_candidates.append(dataset_root / "massive_it_train.csv")
        candidate_by_config = dataset_root / f"{config}_{split}.csv"
        if candidate_by_config.name != csv_candidates[0].name:
            csv_candidates.append(candidate_by_config)
        csv_path = next((candidate for candidate in csv_candidates if candidate.exists()), None)

    samples: List[Sample] = []
    if csv_path is not None and pd is not None:
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
        annot_column = _resolve_column("annot_utt", "annotated_utt", "slots")

        if utterance_column is None or intent_column is None:
            raise ValueError(
                "Il file CSV non contiene colonne compatibili per 'utt'/'intent'."
            )

        columns = [utterance_column, intent_column]
        if annot_column:
            columns.append(annot_column)

        for row in dataframe[columns].itertuples(index=False, name=None):
            if annot_column:
                utterance_value, intent_value, annot_value = row
            else:
                utterance_value, intent_value = row
                annot_value = None

            utterance = str(utterance_value).strip()
            intent = str(intent_value).strip()
            annot_text = None if annot_value is None else str(annot_value).strip()
            if not utterance or not intent:
                continue

            tokens, slot_tags, slots_dict = parse_annotated_utterance(annot_text)
            if not tokens:
                tokens = _tokenize(utterance)
                slot_tags = ["O"] * len(tokens)
            samples.append(
                Sample(
                    utterance=utterance,
                    intent=intent,
                    slots=slots_dict or None,
                    annotated_utterance=annot_text,
                    tokens=tokens,
                    slot_tags=slot_tags,
                )
            )
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
                annot_text = str(record.get("annot_utt", "")).strip() or None
                if not utterance or not intent:
                    # Salta esempi incompleti
                    continue
                tokens, slot_tags, slots_dict = parse_annotated_utterance(annot_text)
                if not tokens:
                    tokens = _tokenize(utterance)
                    slot_tags = ["O"] * len(tokens)
                samples.append(
                    Sample(
                        utterance=utterance,
                        intent=intent,
                        slots=slots_dict or None,
                        annotated_utterance=annot_text,
                        tokens=tokens,
                        slot_tags=slot_tags,
                    )
                )

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
    "parse_annotated_utterance",
    "DEFAULT_DATASET_DIR",
    "LEGACY_DATASET_DIR",
]