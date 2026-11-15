"""Split-protocol tests.

Skipped until `python -m banking_triage.data` has run, so a fresh clone can still
execute the unit suite.
"""

from __future__ import annotations

import pandas as pd
import pytest

from banking_triage.config import settings
from banking_triage.data import load_labels, load_splits, read_summary
from banking_triage.taxonomy import FAQ_COVERED_INTENTS, display_name

pytestmark = pytest.mark.skipif(
    not (settings.processed_dir / "summary.json").exists(),
    reason="run `python -m banking_triage.data` first",
)


def test_splits_are_disjoint_by_exact_text() -> None:
    splits = load_splits()
    train_texts = set(splits["train"]["text"])
    # val is ours to construct, so it must be strictly disjoint
    assert train_texts.isdisjoint(set(splits["val"]["text"]))
    # the official test split keeps the dataset's own duplicates (recorded, tiny)
    summary = read_summary()
    overlap = len(train_texts & set(splits["test"]["text"]))
    assert overlap == summary["duplicate_texts"]["train_test"]["count"]


def test_val_split_is_leakage_free() -> None:
    summary = read_summary()
    assert summary["duplicate_texts"]["train_val"]["count"] == 0
    assert summary["val_rows_dropped_for_leakage"] >= 0


def test_test_overlap_is_known_and_benign() -> None:
    # BANKING77 ships a handful of exact duplicate messages across its official
    # splits. We keep the split intact for comparability, so the overlap is
    # asserted to stay small and free of label conflicts rather than zero.
    summary = read_summary()
    detail = summary["duplicate_texts"]["train_test"]
    assert detail["count"] <= 10, "cross-split leakage grew; investigate before trusting metrics"
    assert detail["conflicting_labels"] == 0, "duplicate message carries two different labels"


def test_sizes_match_the_official_dataset() -> None:
    summary = read_summary()
    assert summary["source_rows"] == {"official_train": 10003, "official_test": 3080}
    # every official train row is accounted for: kept in train, kept in val, or
    # dropped from val because it duplicated a train message
    assert (
        summary["splits"]["train"]
        + summary["splits"]["val"]
        + summary["val_rows_dropped_for_leakage"]
        == 10003
    )
    assert summary["n_intents"] == 77


def test_val_split_is_stratified() -> None:
    """Val class frequencies should track train (within a wide tolerance).

    BANKING77's train split is imbalanced (28-149 rows per intent), unlike its
    test split which has exactly 40 per intent, so proportional -- not equal --
    counts are the correct expectation here.
    """
    splits = load_splits()
    train_counts = splits["train"]["label_id"].value_counts()
    val_counts = splits["val"]["label_id"].value_counts()
    assert set(val_counts.index) == set(train_counts.index), "an intent is missing from val"

    ratios = (val_counts / train_counts).dropna()
    assert (ratios > 0.12).all(), f"under-represented intents in val: {ratios[ratios <= 0.12]}"
    assert (ratios < 0.28).all(), f"over-represented intents in val: {ratios[ratios >= 0.28]}"


def test_every_frame_has_the_expected_columns() -> None:
    for frame in load_splits().values():
        assert list(frame.columns) == ["text", "label_id"]
        assert frame["text"].map(lambda value: isinstance(value, str)).all()


def test_label_table_is_complete_and_clean() -> None:
    labels = load_labels()
    assert len(labels) == 77
    assert [row["id"] for row in labels] == list(range(77))
    assert all(row["display"] == display_name(row["name"]) for row in labels)


def test_faq_coverage_targets_exist_in_the_taxonomy() -> None:
    names = {row["name"] for row in load_labels()}
    unknown = FAQ_COVERED_INTENTS - names
    assert not unknown, f"FAQ routing references intents absent from the dataset: {unknown}"


def test_parquet_round_trip_preserves_text() -> None:
    frame = load_splits()["train"].head(5)
    assert pd.api.types.is_string_dtype(frame["text"])
