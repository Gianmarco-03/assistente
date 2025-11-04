"""Pipeline dedicata al riconoscimento dei parametri (token classification con CRF)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import re

# ------------------------
# dipendenze
# ------------------------
try:  # pragma: no cover - dipendenze opzionali
    from sklearn.metrics import classification_report
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "È necessario installare scikit-learn per eseguire la valutazione.\n"
        "Puoi installarlo con: pip install scikit-learn"
    ) from exc

try:  # pragma: no cover
    import joblib
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "È necessario installare joblib per salvare il modello addestrato.\n"
        "Puoi installarla con: pip install joblib"
    ) from exc

try:  # pragma: no cover
    import sklearn_crfsuite
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "È necessario installare sklearn-crfsuite per il training dei parametri.\n"
        "Puoi installarlo con: pip install sklearn-crfsuite"
    ) from exc

# import dal tuo progetto
try:
    from .dataset import (
        DEFAULT_DATASET_DIR,
        LEGACY_DATASET_DIR,
        Sample,
        load_samples,
    )
except ImportError:  # pragma: no cover - compatibilità con esecuzione diretta
    from dataset import (  # type: ignore
        DEFAULT_DATASET_DIR,
        LEGACY_DATASET_DIR,
        Sample,
        load_samples,
    )

try:
    from .pipeline import (  # type: ignore
        DEFAULT_CONFIG,
        DEFAULT_EVAL_SPLIT,
        DEFAULT_TRAIN_SPLIT,
    )
except Exception:  # pragma: no cover - compatibilità se pipeline non disponibile
    DEFAULT_CONFIG = "massive"
    DEFAULT_TRAIN_SPLIT = "train"
    DEFAULT_EVAL_SPLIT = "validation"


TOKEN_MODEL_FILENAME = "token_response_model.joblib"
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


# ---------------------------------------------------------------------------
# tokenizzazione
# ---------------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    """Tokenizzazione di base condivisa tra training e inference."""
    text = text or ""
    return _TOKEN_PATTERN.findall(text)


# ---------------------------------------------------------------------------
# feature engineering
# ---------------------------------------------------------------------------
def _shape(token: str) -> str:
    """
    Rappresenta il "pattern" del token, es.:
    - "Roma" -> "Xxxx"
    - "12" -> "dd"
    - "A12" -> "Xdd"
    Serve a dare indizi al CRF.
    """
    out = []
    for ch in token:
        if ch.isdigit():
            out.append("d")
        elif ch.isupper():
            out.append("X")
        elif ch.islower():
            out.append("x")
        else:
            out.append(ch)
    return "".join(out)


def extract_token_features(tokens: Sequence[str], index: int) -> Dict[str, object]:
    """Genera un dizionario di feature contestuali per il token dato."""
    token = tokens[index]
    lower = token.lower()
    features: Dict[str, object] = {
        "bias": 1.0,
        "token": token,
        "token.lower": lower,
        "token.isalpha": token.isalpha(),
        "token.isdigit": token.isdigit(),
        "token.istitle": token.istitle(),
        "token.len": len(token),
        "token.hasdigit": any(ch.isdigit() for ch in token),
        "token.contains-hyphen": "-" in token,
        "prefix2": lower[:2],
        "prefix3": lower[:3],
        "suffix2": lower[-2:],
        "suffix3": lower[-3:],
        "shape": _shape(token),
    }

    # token precedente
    if index > 0:
        prev = tokens[index - 1]
        prev_lower = prev.lower()
        features.update(
            {
                "-1.token.lower": prev_lower,
                "-1.token.isdigit": prev.isdigit(),
                "-1.shape": _shape(prev),
            }
        )
    else:
        features["BOS"] = True  # Begin of sentence

    # token successivo
    if index + 1 < len(tokens):
        nxt = tokens[index + 1]
        nxt_lower = nxt.lower()
        features.update(
            {
                "+1.token.lower": nxt_lower,
                "+1.token.isdigit": nxt.isdigit(),
                "+1.shape": _shape(nxt),
            }
        )
    else:
        features["EOS"] = True  # End of sentence

    return features


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------
def _ensure_alignment(
    sample_id: str,
    tokens: List[str],
    tags: List[str],
) -> Tuple[List[str], List[str]]:
    """Si assicura che lunghezza dei tag corrisponda ai token, loggando i mismatch."""
    if len(tokens) == len(tags):
        return tokens, tags

    print(
        f"[WARN] mismatch token/tag per sample {sample_id}: "
        f"{len(tokens)} token vs {len(tags)} tag. Completo con 'O'."
    )

    if not tokens:
        return [], []
    if not tags:
        return tokens, ["O"] * len(tokens)
    if len(tags) < len(tokens):
        tags = tags + ["O"] * (len(tokens) - len(tags))
    else:
        tags = tags[: len(tokens)]
    return tokens, tags


# ---------------------------------------------------------------------------
# dataset preparation (in sequenza!)
# ---------------------------------------------------------------------------
def prepare_token_dataset(
    samples: Sequence[Sample],
) -> Tuple[List[List[Dict[str, object]]], List[List[str]]]:
    """
    Converte gli esempi annotati in:
    - lista di frasi, ciascuna = lista di feature-dict per token
    - lista di frasi, ciascuna = lista di tag
    Questo formato è quello atteso dal CRF.
    """
    X_seq: List[List[Dict[str, object]]] = []
    y_seq: List[List[str]] = []

    for sample in samples:
        tokens = list(sample.tokens or tokenize(sample.utterance))
        tags = list(sample.slot_tags or [])
        tokens, tags = _ensure_alignment(sample.id if hasattr(sample, "id") else "<no-id>", tokens, tags)

        sent_features: List[Dict[str, object]] = []
        sent_labels: List[str] = []

        intent = getattr(sample, "intent", None)
        for idx, tag in enumerate(tags):
            feature = extract_token_features(tokens, idx)
            feature['intent'] = intent
            sent_features.append(feature)
            sent_labels.append(tag)

        X_seq.append(sent_features)
        y_seq.append(sent_labels)

    return X_seq, y_seq


# ---------------------------------------------------------------------------
# modello
# ---------------------------------------------------------------------------
def build_token_classifier() -> "sklearn_crfsuite.CRF":
    """Restituisce il classificatore CRF per i tag di slot."""
    # L-BFGS va bene, bilanciamo con all_possible_transitions
    return sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        max_iterations=200,
        all_possible_transitions=True,
    )


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def train_token_model(
    dataset_dir: str | Path,
    *,
    config: str = DEFAULT_CONFIG,
    train_split: str = DEFAULT_TRAIN_SPLIT,
    eval_split: str = DEFAULT_EVAL_SPLIT,
):
    """Addestra il modello di token classification sui parametri."""

    dataset_dir = Path(dataset_dir)

    print(f"[INFO] Carico split di training '{train_split}' da {dataset_dir} ...")
    train_samples = load_samples(dataset_dir, config=config, split=train_split)
    X_train, y_train = prepare_token_dataset(train_samples)

    print(f"[INFO] Numero frasi train: {len(X_train)}")

    crf = build_token_classifier()
    print("[INFO] Avvio training CRF ...")
    crf.fit(X_train, y_train)
    print("[INFO] Training completato.")

    print(f"[INFO] Carico split di valutazione '{eval_split}' ...")
    eval_samples = load_samples(dataset_dir, config=config, split=eval_split)
    X_eval, y_eval = prepare_token_dataset(eval_samples)

    print(f"[INFO] Numero frasi eval: {len(X_eval)}")

    y_pred = crf.predict(X_eval)

    # flatten per classification_report
    true_flat: List[str] = []
    pred_flat: List[str] = []
    for y_true_sent, y_pred_sent in zip(y_eval, y_pred):
        true_flat.extend(y_true_sent)
        pred_flat.extend(y_pred_sent)

    report = ("\n=== REPORT COMPLETO (con O) ===\n"  
    + classification_report(
            true_flat,
            pred_flat,
            zero_division=0,
        )
    )

    # report senza O
    true_wo = [t for t in true_flat if t != "O"]
    pred_wo = [p for t, p in zip(true_flat, pred_flat) if t != "O"]

    if true_wo:
            report += ("\n=== REPORT SLOT (senza O) ===\n"
            + classification_report(
                true_wo,
                pred_wo,
                zero_division=0,
            )
        )
    else:
        report += ("\nNessuna etichetta diversa da 'O' trovata nello split di valutazione.")

    # bundle per salvataggio
    labels = sorted({lbl for sent in y_train for lbl in sent})
    bundle: Dict[str, object] = {
        "crf": crf,
        "labels": labels,
        "config": {
            "dataset_dir": str(dataset_dir),
            "config": config,
            "train_split": train_split,
            "eval_split": eval_split,
        },
    }

    return bundle, report


# ---------------------------------------------------------------------------
# salvataggio
# ---------------------------------------------------------------------------
def save_token_model(
    bundle: Dict[str, object],
    output_dir: str | Path,
    *,
    filename: str = TOKEN_MODEL_FILENAME,
) -> Path:
    """Salva su disco il modello addestrato per il riconoscimento dei parametri."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / filename
    joblib.dump(bundle, file_path)
    print(f"[INFO] Modello CRF salvato in: {file_path}")
    return file_path


__all__ = [
    "TOKEN_MODEL_FILENAME",
    "build_token_classifier",
    "prepare_token_dataset",
    "train_token_model",
    "save_token_model",
    "tokenize",
    "extract_token_features",
]
