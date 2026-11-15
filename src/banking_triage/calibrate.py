"""Calibration and auto-route threshold selection.

M1 gave us an accuracy number. This step turns it into a *decision policy*: at
what confidence do we answer without a human, and what precision does that buy?

Protocol, kept strict so the headline can be trusted:
  1. Fit an uncalibrated and a sigmoid-calibrated model on the train split.
  2. Compare calibration on the val split (ECE / Brier / NLL). No test contact.
  3. Sweep confidence and margin thresholds on val, targeting the precision we
     are willing to promise on auto-answered traffic.
  4. Refit the winner on train+val with the chosen thresholds frozen.
  5. Score the official test split once.

Usage:
    python -m banking_triage.calibrate
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline

from .config import settings
from .data import load_labels, load_splits
from .taxonomy import NEVER_AUTO_ANSWER_INTENTS
from .train import build_pipeline

TARGET_PRECISION = 0.95  # promised precision on auto-answered traffic
MIN_ROUTED_FOR_STABILITY = 40
N_BINS = 10


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS
) -> float:
    """Mean |accuracy - confidence| across equal-width confidence bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.digitize(confidences, edges[1:-1], right=True)
    ece = 0.0
    for index in range(n_bins):
        mask = bin_index == index
        if not mask.any():
            continue
        weight = mask.mean()
        ece += weight * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def reliability_bins(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS
) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.digitize(confidences, edges[1:-1], right=True)
    rows: list[dict[str, Any]] = []
    for index in range(n_bins):
        mask = bin_index == index
        rows.append(
            {
                "bin": f"{edges[index]:.1f}-{edges[index + 1]:.1f}",
                "count": int(mask.sum()),
                "mean_confidence": round(float(confidences[mask].mean()), 4)
                if mask.any()
                else None,
                "accuracy": round(float(correct[mask].mean()), 4) if mask.any() else None,
            }
        )
    return rows


def probability_diagnostics(proba: np.ndarray, y_true: np.ndarray) -> dict[str, Any]:
    confidence = proba.max(axis=1)
    predicted = proba.argmax(axis=1)
    correct = predicted == y_true
    return {
        "ece": round(expected_calibration_error(confidence, correct), 4),
        "brier": round(
            float(np.mean(np.sum((proba - np.eye(proba.shape[1])[y_true]) ** 2, axis=1))), 4
        ),
        "nll": round(float(log_loss(y_true, proba, labels=np.arange(proba.shape[1]))), 4),
        "mean_confidence": round(float(confidence.mean()), 4),
        "accuracy": round(float(correct.mean()), 4),
        "bins": reliability_bins(confidence, correct),
    }


def threshold_grid(proba: np.ndarray, y_true: np.ndarray, label_names: list[str]) -> list[dict]:
    """Sweep (confidence, margin) and record coverage/precision of auto-routing."""
    confidence = proba.max(axis=1)
    predicted = proba.argmax(axis=1)
    order = np.argsort(-proba, axis=1)
    runner_up = proba[np.arange(len(y_true)), order[:, 1]]
    margin = confidence - runner_up
    correct = predicted == y_true
    blocked = np.array([label_names[i] in NEVER_AUTO_ANSWER_INTENTS for i in predicted])

    rows: list[dict[str, Any]] = []
    for tau in np.round(np.arange(0.20, 0.96, 0.02), 2):
        for delta in (0.0, 0.05, 0.10, 0.20):
            for policy in ("with_safety_rules", "confidence_only"):
                selected = (confidence >= tau) & (margin >= delta)
                if policy == "with_safety_rules":
                    selected &= ~blocked
                routed = int(selected.sum())
                rows.append(
                    {
                        "tau": float(tau),
                        "margin": float(delta),
                        "policy": policy,
                        "routed": routed,
                        "coverage": round(routed / len(y_true), 4),
                        "precision": round(float(correct[selected].mean()), 4) if routed else None,
                    }
                )
    return rows


def choose_threshold(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row["policy"] == policy
        and row["precision"] is not None
        and row["precision"] >= TARGET_PRECISION
        and row["routed"] >= MIN_ROUTED_FOR_STABILITY
    ]
    if not eligible:
        raise SystemExit(
            f"no threshold reaches {TARGET_PRECISION:.2f} precision with "
            f">= {MIN_ROUTED_FOR_STABILITY} routed messages; relax the target"
        )
    # highest coverage wins; ties broken by the tighter (safer) threshold
    return max(eligible, key=lambda row: (row["coverage"], row["tau"], row["margin"]))


def _fit(method: str | None, config: str, c_value: float, x, y) -> Pipeline:
    base = build_pipeline(config, c_value)
    if method is None:
        return base.fit(x, y)
    calibrated = CalibratedClassifierCV(base, method=method, cv=3)
    return calibrated.fit(x, y)


def _selected_from_m1() -> dict[str, Any]:
    path = settings.reports_dir / "m1_classifier.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run `python -m banking_triage.train` first")
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    return report["selected"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="sigmoid", choices=["sigmoid", "isotonic"])
    parser.add_argument("--no-artifact", action="store_true")
    args = parser.parse_args()

    settings.ensure_dirs()
    selected = _selected_from_m1()
    config, c_value = selected["config"], float(selected["C"])
    print(f"using M1 selection: {config} C={c_value}", flush=True)

    splits = load_splits()
    labels = load_labels()
    label_names = [row["name"] for row in labels]

    x_train = splits["train"]["text"].to_numpy()
    y_train = splits["train"]["label_id"].to_numpy()
    x_val = splits["val"]["text"].to_numpy()
    y_val = splits["val"]["label_id"].to_numpy()

    print("fitting uncalibrated + calibrated models on train...", flush=True)
    raw = _fit(None, config, c_value, x_train, y_train)
    calibrated = _fit(args.method, config, c_value, x_train, y_train)

    comparison: dict[str, Any] = {}
    for name, model in (("raw", raw), ("calibrated", calibrated)):
        proba = model.predict_proba(x_val)
        comparison[name] = probability_diagnostics(proba, y_val)
        print(
            f"  {name:<10} ece={comparison[name]['ece']:.4f} "
            f"brier={comparison[name]['brier']:.4f} nll={comparison[name]['nll']:.4f} "
            f"mean_conf={comparison[name]['mean_confidence']:.3f}",
            flush=True,
        )

    winner = "calibrated" if comparison["calibrated"]["nll"] <= comparison["raw"]["nll"] else "raw"
    chosen_model = calibrated if winner == "calibrated" else raw
    print(f"winner by val NLL: {winner}", flush=True)

    proba_val = chosen_model.predict_proba(x_val)
    rows = threshold_grid(proba_val, y_val, label_names)
    policy_best = choose_threshold(rows, "with_safety_rules")
    naive_best = choose_threshold(rows, "confidence_only")
    print(
        f"chosen policy: tau={policy_best['tau']} margin={policy_best['margin']} "
        f"-> val coverage={policy_best['coverage']:.3f} precision={policy_best['precision']:.3f}",
        flush=True,
    )
    print(
        f"  (confidence-only would allow coverage={naive_best['coverage']:.3f} "
        f"at the same precision; the gap is the cost of the never-auto-answer rules)",
        flush=True,
    )

    print("refitting winner on train+val with thresholds frozen...", flush=True)
    x_all = np.concatenate([x_train, x_val])
    y_all = np.concatenate([y_train, y_val])
    final_model = _fit(
        args.method if winner == "calibrated" else None, config, c_value, x_all, y_all
    )

    thresholds = {
        "auto_route_threshold": float(policy_best["tau"]),
        "min_margin": float(policy_best["margin"]),
        "escalate_threshold": float(min(0.40, policy_best["tau"])),
        "target_precision": TARGET_PRECISION,
        "calibrated": winner == "calibrated",
    }

    x_test = splits["test"]["text"].to_numpy()
    y_test = splits["test"]["label_id"].to_numpy()
    proba_test = final_model.predict_proba(x_test)
    test_diag = probability_diagnostics(proba_test, y_test)

    confidence = proba_test.max(axis=1)
    predicted = proba_test.argmax(axis=1)
    order = np.argsort(-proba_test, axis=1)
    margin = confidence - proba_test[np.arange(len(y_test)), order[:, 1]]
    blocked = np.array([label_names[i] in NEVER_AUTO_ANSWER_INTENTS for i in predicted])
    routed_mask = (
        (confidence >= thresholds["auto_route_threshold"])
        & (margin >= thresholds["min_margin"])
        & ~blocked
    )
    routed = int(routed_mask.sum())
    deflection = {
        "routed": routed,
        "n": int(len(y_test)),
        "coverage": round(routed / len(y_test), 4),
        "precision": round(float((predicted[routed_mask] == y_test[routed_mask]).mean()), 4),
    }

    from sklearn.metrics import accuracy_score, f1_score

    test_scores = {
        "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
        "macro_f1": round(float(f1_score(y_test, predicted, average="macro", zero_division=0)), 4),
    }

    trained_at = datetime.now(UTC)
    model_version = f"v2-{trained_at:%Y%m%d-%H%M}"
    metadata = {
        "model_version": model_version,
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "trained_on": "train+val",
        "n_train": int(len(y_all)),
        "n_intents": len(label_names),
        "vectorizer": config,
        "hyperparameters": {"C": c_value, "max_iter": 2000, "solver": "lbfgs"},
        "calibration": {"method": args.method if winner == "calibrated" else None, "cv": 3},
        "metrics": {
            "test_accuracy": test_scores["accuracy"],
            "test_macro_f1": test_scores["macro_f1"],
            "test_ece": test_diag["ece"],
            "test_routed_precision": deflection["precision"],
            "test_routed_coverage": deflection["coverage"],
        },
        "thresholds": thresholds,
        "selection_basis": "val_nll_then_val_threshold_sweep",
        "target_precision": TARGET_PRECISION,
    }

    if not args.no_artifact:
        joblib.dump(
            {"pipeline": final_model, "label_names": label_names, "metadata": metadata},
            settings.model_path,
            compress=3,
        )

    report = {
        "generated_at": trained_at.isoformat(timespec="seconds"),
        "protocol": {
            "calibration_comparison": "val split",
            "threshold_selection": "val split, target precision "
            f"{TARGET_PRECISION:.2f}, minimum {MIN_ROUTED_FOR_STABILITY} routed",
            "headline": "official test split, scored once, thresholds frozen",
        },
        "selected_from_m1": selected,
        "calibration_comparison": comparison,
        "winner": winner,
        "thresholds": thresholds,
        "val_at_chosen_threshold": policy_best,
        "confidence_only_comparison": naive_best,
        "deflection_on_test": deflection,
        "test_scores": test_scores,
        "test_diagnostics": test_diag,
        "sweep": rows,
        "metadata": metadata,
    }
    path = settings.reports_dir / "m2_calibration.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n--- official test (thresholds frozen from val) ---")
    print(
        f"accuracy={test_scores['accuracy']:.4f} macroF1={test_scores['macro_f1']:.4f} "
        f"ece={test_diag['ece']:.4f}"
    )
    print(
        f"auto-routed {deflection['routed']}/{deflection['n']} "
        f"({deflection['coverage']:.1%}) at precision {deflection['precision']:.4f}"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
