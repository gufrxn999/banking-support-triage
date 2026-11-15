"""Calibration and threshold-selection tests.

These use synthetic probabilities so they run fast and do not depend on the
trained artifact.
"""

from __future__ import annotations

import numpy as np
import pytest

from banking_triage.calibrate import (
    MIN_ROUTED_FOR_STABILITY,
    TARGET_PRECISION,
    choose_threshold,
    expected_calibration_error,
    probability_diagnostics,
    reliability_bins,
    threshold_grid,
)


def test_ece_is_zero_when_confidence_matches_accuracy() -> None:
    confidences = np.full(200, 0.5)
    correct = np.array([True, False] * 100)  # exactly 50% correct at 50% confidence
    assert expected_calibration_error(confidences, correct) == pytest.approx(0.0, abs=1e-9)


def test_ece_penalises_overconfidence() -> None:
    confidences = np.full(200, 0.9)
    correct = np.array([True, False] * 100)  # 50% correct at 90% confidence
    assert expected_calibration_error(confidences, correct) == pytest.approx(0.4, abs=1e-9)


def test_ece_is_weighted_by_bin_population() -> None:
    # 9 fully-correct predictions at 1.0 confidence (error 0) and one
    # 0.1-confidence error: ECE should be dominated by the well-calibrated bin.
    confidences = np.array([1.0] * 9 + [0.1])
    correct = np.array([True] * 9 + [False])
    value = expected_calibration_error(confidences, correct)
    assert 0 < value < 0.1


def test_reliability_bins_account_for_every_prediction() -> None:
    confidences = np.linspace(0.01, 0.99, 250)
    correct = confidences > 0.5
    bins = reliability_bins(confidences, correct)
    assert sum(row["count"] for row in bins) == 250
    assert len(bins) == 10


def _one_hot(index: int, n: int = 5) -> np.ndarray:
    row = np.zeros(n)
    row[index] = 1.0
    return row


def test_probability_diagnostics_reports_all_three_scores() -> None:
    proba = np.vstack([_one_hot(0), _one_hot(1), _one_hot(0)])
    y_true = np.array([0, 1, 2])
    diagnostics = probability_diagnostics(proba, y_true)
    # values are rounded to 4 dp by the reporter
    assert diagnostics["accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    assert diagnostics["ece"] == pytest.approx(1 / 3, abs=1e-4)
    assert diagnostics["brier"] > 0
    assert diagnostics["nll"] > 0


def test_threshold_grid_covers_both_policies() -> None:
    proba = np.vstack([_one_hot(0), _one_hot(1), _one_hot(2)])
    y_true = np.array([0, 1, 1])
    rows = threshold_grid(proba, y_true, ["a", "b", "c"])
    policies = {row["policy"] for row in rows}
    assert policies == {"with_safety_rules", "confidence_only"}
    assert all(0 <= row["coverage"] <= 1 for row in rows)


def test_choose_threshold_respects_the_precision_target() -> None:
    rows = [
        {
            "policy": "with_safety_rules",
            "tau": 0.3,
            "margin": 0.0,
            "routed": 100,
            "coverage": 0.9,
            "precision": 0.80,
        },
        {
            "policy": "with_safety_rules",
            "tau": 0.6,
            "margin": 0.1,
            "routed": 60,
            "coverage": 0.55,
            "precision": 0.97,
        },
        {
            "policy": "with_safety_rules",
            "tau": 0.9,
            "margin": 0.2,
            "routed": 10,
            "coverage": 0.09,
            "precision": 1.00,
        },
    ]
    chosen = choose_threshold(rows, "with_safety_rules")
    assert chosen["tau"] == 0.6
    assert chosen["precision"] >= TARGET_PRECISION


def test_choose_threshold_prefers_higher_coverage_at_equal_precision() -> None:
    rows = [
        {
            "policy": "with_safety_rules",
            "tau": 0.5,
            "margin": 0.0,
            "routed": MIN_ROUTED_FOR_STABILITY,
            "coverage": 0.40,
            "precision": 0.96,
        },
        {
            "policy": "with_safety_rules",
            "tau": 0.7,
            "margin": 0.1,
            "routed": 200,
            "coverage": 0.70,
            "precision": 0.96,
        },
    ]
    assert choose_threshold(rows, "with_safety_rules")["tau"] == 0.7


def test_choose_threshold_fails_loudly_when_target_is_unreachable() -> None:
    rows = [
        {
            "policy": "with_safety_rules",
            "tau": 0.9,
            "margin": 0.2,
            "routed": 100,
            "coverage": 0.1,
            "precision": 0.70,
        },
    ]
    with pytest.raises(SystemExit, match="no threshold"):
        choose_threshold(rows, "with_safety_rules")
