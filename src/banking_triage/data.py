"""Dataset preparation for BANKING77.

Produces three frozen, stratified splits plus a label table. The official test
split is never touched again after this step: model selection uses CV inside
`train`, and thresholds are tuned on `val`. That separation is what makes the
final test numbers believable.

Usage:
    python -m banking_triage.data [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import settings
from .taxonomy import display_name

DATASET_NAME = "PolyAI/banking77"
SEED = 42
VAL_FRACTION = 0.2  # of the official train split


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)


def _digest(frame: pd.DataFrame, column: str) -> str:
    hasher = hashlib.sha256()
    for value in frame[column].tolist():
        hasher.update(str(value).encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()[:16]


def _to_frame(split: Any) -> pd.DataFrame:
    frame = split.to_pandas()
    frame["text"] = _normalize_text(frame["text"])
    frame = frame.rename(columns={"label": "label_id"})
    frame = frame[frame["text"].str.len() > 0]
    return frame[["text", "label_id"]].reset_index(drop=True)


def _overlap_detail(train_part: pd.DataFrame, other: pd.DataFrame) -> dict[str, Any]:
    """Count shared messages, and whether any of them disagree on the label."""
    shared = set(train_part["text"]) & set(other["text"])
    conflicting = 0
    for text in shared:
        train_labels = set(train_part.loc[train_part["text"] == text, "label_id"].tolist())
        other_labels = set(other.loc[other["text"] == text, "label_id"].tolist())
        if train_labels != other_labels:
            conflicting += 1
    return {"count": len(shared), "conflicting_labels": conflicting}


def _labels_from(names: list[str]) -> list[dict[str, Any]]:
    return [
        {"id": idx, "name": name, "display": display_name(name)} for idx, name in enumerate(names)
    ]


def prepare(force: bool = False) -> dict[str, Any]:
    """Materialise train/val/test parquet files and return a summary dict."""
    settings.ensure_dirs()
    processed = settings.processed_dir
    splits = {name: processed / f"{name}.parquet" for name in ("train", "val", "test")}
    summary_path = processed / "summary.json"

    if not force and all(p.exists() for p in splits.values()) and summary_path.exists():
        return read_summary()

    from datasets import load_dataset  # imported late: only needed for a refresh

    dataset = load_dataset(DATASET_NAME)
    label_names: list[str] = dataset["train"].features["label"].names

    official_train = _to_frame(dataset["train"])
    official_test = _to_frame(dataset["test"])

    train_part, val_part = train_test_split(
        official_train,
        test_size=VAL_FRACTION,
        random_state=SEED,
        shuffle=True,
        stratify=official_train["label_id"],
    )

    train_part = train_part.reset_index(drop=True)
    val_part = val_part.reset_index(drop=True)

    # The official test split must stay byte-identical to BANKING77 so metrics
    # remain comparable, but our val split is ours to construct: drop any val
    # message that also appears in train so val is a real held-out signal.
    train_texts = set(train_part["text"])
    leakage_mask = val_part["text"].isin(train_texts)
    dropped_from_val = int(leakage_mask.sum())
    val_part = val_part[~leakage_mask].reset_index(drop=True)

    # Cohort checks: duplicated messages across splits leak signal and inflate
    # accuracy. Record the overlap instead of assuming there is none.
    digests = {
        name: _digest(frame, "text")
        for name, frame in (("train", train_part), ("val", val_part), ("test", official_test))
    }
    overlap_train_test = _overlap_detail(train_part, official_test)
    overlap_train_val = _overlap_detail(train_part, val_part)

    for name, frame in (("train", train_part), ("val", val_part), ("test", official_test)):
        frame.to_parquet(splits[name], index=False)

    labels = _labels_from(label_names)
    settings.labels_path.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")

    summary: dict[str, Any] = {
        "dataset": DATASET_NAME,
        "prepared_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": SEED,
        "val_fraction_of_official_train": VAL_FRACTION,
        "source_rows": {
            "official_train": int(len(official_train)),
            "official_test": int(len(official_test)),
        },
        "splits": {
            "train": int(len(train_part)),
            "val": int(len(val_part)),
            "test": int(len(official_test)),
        },
        "n_intents": len(label_names),
        "digests": digests,
        "duplicate_texts": {
            "train_test": overlap_train_test,
            "train_val": overlap_train_val,
            "within_train": int(train_part["text"].duplicated().sum()),
            "within_test": int(official_test["text"].duplicated().sum()),
        },
        "val_rows_dropped_for_leakage": dropped_from_val,
        "notes": [
            "Official test split is preserved verbatim so metrics stay comparable "
            "with published BANKING77 baselines.",
            "Same-label duplicates across train/test are an upstream dataset "
            "property (0.16% of test); `duplicate_texts.train_test.conflicting_labels` "
            "records whether any of them disagree.",
        ],
        "class_balance": {
            "train_min_per_class": int(train_part["label_id"].value_counts().min()),
            "train_max_per_class": int(train_part["label_id"].value_counts().max()),
            "test_min_per_class": int(official_test["label_id"].value_counts().min()),
            "test_max_per_class": int(official_test["label_id"].value_counts().max()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def read_summary() -> dict[str, Any]:
    return json.loads((settings.processed_dir / "summary.json").read_text(encoding="utf-8"))


def load_splits() -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_parquet(settings.processed_dir / f"{name}.parquet")
        for name in ("train", "val", "test")
    }


def load_labels() -> list[dict[str, Any]]:
    return json.loads(settings.labels_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild even if files exist")
    args = parser.parse_args()

    summary = prepare(force=args.force)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
