"""Train the intent classifier: TF-IDF + multinomial logistic regression.

Protocol (deliberately strict, because this is the part people skimp on):
  1. Model selection: stratified k-fold CV *inside* the train split only.
  2. Validation metrics: a single fit on train, scored on the untouched val split.
  3. Headline metrics: the same fit, scored once on the official test split.

The artifact is written with its own metadata so `/model` can report exactly
what is serving traffic.

Usage:
    python -m banking_triage.train
    python -m banking_triage.train --configs word_bigram --cs 1,2,4 --folds 3
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import FeatureUnion, Pipeline

from .config import settings
from .data import load_labels, load_splits

SEED = 42

# BANKING77 contains 5 same-label duplicate messages across its official
# train/test splits. Anything beyond this is a red flag, not dataset noise.
MAX_TOLERATED_TEST_OVERLAP = 10

VECTORIZERS: dict[str, Any] = {
    "word_unigram": lambda: TfidfVectorizer(
        ngram_range=(1, 1), min_df=1, sublinear_tf=True, strip_accents="unicode"
    ),
    "word_bigram": lambda: TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode"
    ),
    "word_char": lambda: FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True
                ),
            ),
        ]
    ),
}


def build_pipeline(config: str, c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("features", VECTORIZERS[config]()),
            (
                "clf",
                LogisticRegression(
                    C=c_value,
                    max_iter=2000,
                    tol=1e-4,
                    solver="lbfgs",
                    random_state=SEED,
                ),
            ),
        ]
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _top_k_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    k = min(k, proba.shape[1])
    top_k = np.argpartition(-proba, k - 1, axis=1)[:, :k]
    return float(np.mean([y in row for y, row in zip(y_true, top_k, strict=True)]))


def _per_intent(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]
) -> list[dict[str, Any]]:
    scores = f1_score(
        y_true, y_pred, labels=np.arange(len(label_names)), average=None, zero_division=0
    )
    support = np.bincount(y_true, minlength=len(label_names))
    rows = [
        {
            "id": idx,
            "intent": name,
            "f1": round(float(score), 4),
            "support": int(count),
        }
        for idx, (name, score, count) in enumerate(zip(label_names, scores, support, strict=True))
    ]
    return sorted(rows, key=lambda row: row["f1"])


def _confusions(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str], top: int = 15
) -> list[dict[str, Any]]:
    counts = Counter((int(t), int(p)) for t, p in zip(y_true, y_pred, strict=True) if t != p)
    return [
        {"actual": label_names[a], "predicted": label_names[b], "count": n}
        for (a, b), n in counts.most_common(top)
    ]


def assert_no_leakage(splits: dict[str, Any], allow_overlap: bool = False) -> None:
    """Refuse to train if leakage is larger than the documented upstream noise.

    Two different cases, deliberately handled differently:
      * train <-> val: our own construction, so it must be exactly zero.
      * train <-> test: BANKING77 ships a handful of same-label duplicates across
        its official splits. We keep the split verbatim for comparability, so a
        small documented overlap is tolerated and reported, not fatal.
    """
    train_texts = set(splits["train"]["text"])
    val_overlap = len(train_texts & set(splits["val"]["text"]))
    test_overlap = len(train_texts & set(splits["test"]["text"]))
    n_test = len(splits["test"])

    if val_overlap:
        message = f"val leakage: {val_overlap} messages also appear in train"
        if not allow_overlap:
            raise SystemExit(f"{message}. Re-run `python -m banking_triage.data --force`.")
        print(f"WARNING: {message} (continuing: --allow-overlap)", flush=True)

    if test_overlap:
        share = 100 * test_overlap / n_test
        print(
            f"NOTE: {test_overlap}/{n_test} test messages ({share:.2f}%) also appear in "
            "train -- upstream BANKING77 duplicates, same label, documented in "
            "data/processed/summary.json",
            flush=True,
        )
        if test_overlap > MAX_TOLERATED_TEST_OVERLAP and not allow_overlap:
            raise SystemExit(
                f"test overlap {test_overlap} exceeds the tolerated upstream maximum "
                f"({MAX_TOLERATED_TEST_OVERLAP}); investigate before reporting metrics."
            )


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def run_cv(
    configs: list[str], cs: list[float], folds: int, x_train, y_train
) -> list[dict[str, Any]]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    rows: list[dict[str, Any]] = []

    for config in configs:
        for c_value in cs:
            started = time.perf_counter()
            scores = cross_validate(
                build_pipeline(config, c_value),
                x_train,
                y_train,
                cv=splitter,
                scoring=["accuracy", "f1_macro"],
                n_jobs=-1,
                error_score="raise",
            )
            elapsed = time.perf_counter() - started
            row = {
                "config": config,
                "C": c_value,
                "folds": folds,
                "cv_accuracy_mean": round(float(scores["test_accuracy"].mean()), 4),
                "cv_accuracy_std": round(float(scores["test_accuracy"].std()), 4),
                "cv_macro_f1_mean": round(float(scores["test_f1_macro"].mean()), 4),
                "cv_macro_f1_std": round(float(scores["test_f1_macro"].std()), 4),
                "seconds": round(elapsed, 1),
            }
            rows.append(row)
            print(
                f"  {config:<13} C={c_value:<5} acc={row['cv_accuracy_mean']:.4f}"
                f"+-{row['cv_accuracy_std']:.4f}  macroF1={row['cv_macro_f1_mean']:.4f}"
                f"  ({elapsed:.0f}s)",
                flush=True,
            )
    return rows


def train_best(
    config: str,
    c_value: float,
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    label_names: list[str],
) -> dict[str, Any]:
    started = time.perf_counter()
    pipeline = build_pipeline(config, c_value)
    pipeline.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started

    fitted_features = pipeline.named_steps["features"]
    if isinstance(fitted_features, TfidfVectorizer):
        n_features = len(fitted_features.vocabulary_)
    else:
        n_features = sum(len(part.vocabulary_) for _, part in fitted_features.transformer_list)

    report: dict[str, Any] = {
        "config": config,
        "C": c_value,
        "n_features": int(n_features),
        "fit_seconds": round(fit_seconds, 1),
    }
    for split_name, x_split, y_split in (
        ("val", x_val, y_val),
        ("test", x_test, y_test),
    ):
        proba = pipeline.predict_proba(x_split)
        pred = proba.argmax(axis=1)
        report[split_name] = {
            **_metrics(y_split, pred),
            "top3_accuracy": round(_top_k_accuracy(y_split, proba, 3), 4),
            "n": int(len(y_split)),
            "per_intent": _per_intent(y_split, pred, label_names),
            "confusions": _confusions(y_split, pred, label_names) if split_name == "test" else [],
        }
    return {"pipeline": pipeline, "report": report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default="word_unigram,word_bigram")
    parser.add_argument("--cs", default="1,2,4,8")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--no-cv", action="store_true", help="skip CV, reuse saved selection")
    parser.add_argument("--no-artifact", action="store_true")
    parser.add_argument("--allow-overlap", action="store_true")
    args = parser.parse_args()

    settings.ensure_dirs()
    splits = load_splits()
    assert_no_leakage(splits, allow_overlap=args.allow_overlap)
    labels = load_labels()
    label_names = [row["name"] for row in labels]
    print(
        f"train={len(splits['train'])} val={len(splits['val'])} "
        f"test={len(splits['test'])} intents={len(label_names)}",
        flush=True,
    )

    x_train = splits["train"]["text"].to_numpy()
    y_train = splits["train"]["label_id"].to_numpy()
    x_val = splits["val"]["text"].to_numpy()
    y_val = splits["val"]["label_id"].to_numpy()
    x_test = splits["test"]["text"].to_numpy()
    y_test = splits["test"]["label_id"].to_numpy()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in configs if c not in VECTORIZERS]
    if unknown:
        raise SystemExit(f"unknown config(s): {unknown}; choose from {list(VECTORIZERS)}")
    cs = [float(c) for c in args.cs.split(",") if c.strip()]

    cv_rows: list[dict[str, Any]] = []
    if args.no_cv:
        best = {"config": configs[0], "C": cs[0]}
        print(f"skipping CV, using {best}", flush=True)
    else:
        print("cross-validation on train split:", flush=True)
        cv_rows = run_cv(configs, cs, args.folds, x_train, y_train)
        best = max(cv_rows, key=lambda row: (row["cv_macro_f1_mean"], row["cv_accuracy_mean"]))
        print(
            f"\nselected: {best['config']} C={best['C']} "
            f"(cv macroF1 {best['cv_macro_f1_mean']:.4f})",
            flush=True,
        )

    print("fitting final model on train split...", flush=True)
    trained = train_best(
        best["config"], best["C"], x_train, y_train, x_val, y_val, x_test, y_test, label_names
    )
    report = trained["report"]

    trained_at = datetime.now(UTC)
    model_version = f"v1-{trained_at:%Y%m%d-%H%M}"
    metadata = {
        "model_version": model_version,
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "trained_on": "train",
        "n_train": int(len(y_train)),
        "n_intents": len(label_names),
        "vectorizer": best["config"],
        "hyperparameters": {"C": best["C"], "max_iter": 2000, "solver": "lbfgs"},
        "metrics": {
            "val_accuracy": report["val"]["accuracy"],
            "val_macro_f1": report["val"]["macro_f1"],
            "test_accuracy": report["test"]["accuracy"],
            "test_macro_f1": report["test"]["macro_f1"],
            "test_top3_accuracy": report["test"]["top3_accuracy"],
        },
        "thresholds": {
            "auto_route_threshold": settings.auto_route_threshold,
            "escalate_threshold": settings.escalate_threshold,
            "min_margin": settings.min_margin,
            "calibrated": False,
        },
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "git_sha": _git_sha(),
        "seed": SEED,
        "selection_basis": "cv_macro_f1_on_train",
    }

    if not args.no_artifact:
        payload = {
            "pipeline": trained["pipeline"],
            "label_names": label_names,
            "metadata": metadata,
        }
        joblib.dump(payload, settings.model_path, compress=3)
        size_mb = settings.model_path.stat().st_size / (1024 * 1024)
        metadata["artifact_mb"] = round(size_mb, 2)

    full_report = {
        "generated_at": trained_at.isoformat(timespec="seconds"),
        "protocol": {
            "model_selection": "stratified k-fold CV within train split",
            "thresholds": "tuned on val split (see m2_calibration.json)",
            "headline": "official test split, scored once",
        },
        "cv": cv_rows,
        "selected": {
            "config": best["config"],
            "C": best["C"],
            "basis": "cv_macro_f1_mean",
        },
        "final": report,
        "metadata": metadata,
    }
    report_path = settings.reports_dir / "m1_classifier.json"
    report_path.write_text(json.dumps(full_report, indent=2) + "\n", encoding="utf-8")

    print("\n--- validation ---")
    print(
        f"acc={report['val']['accuracy']:.4f}  macroF1={report['val']['macro_f1']:.4f}  "
        f"top3={report['val']['top3_accuracy']:.4f}"
    )
    print("--- official test (scored once) ---")
    print(
        f"acc={report['test']['accuracy']:.4f}  macroF1={report['test']['macro_f1']:.4f}  "
        f"top3={report['test']['top3_accuracy']:.4f}"
    )
    weakest = ", ".join(
        f"{row['intent']}({row['f1']:.2f})" for row in report["test"]["per_intent"][:5]
    )
    print(f"\nweakest intents on test: {weakest}")
    print("top confusions on test:")
    for row in report["test"]["confusions"][:8]:
        print(f"  {row['actual']} -> {row['predicted']}  ({row['count']})")
    print(f"\nwrote {report_path}")
    if not args.no_artifact:
        print(f"wrote {settings.model_path} ({metadata.get('artifact_mb', 0)} MB)")


if __name__ == "__main__":
    main()
