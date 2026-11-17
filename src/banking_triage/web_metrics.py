"""Generate the showcase page's evidence block from the committed reports.

The page must not hand-copy numbers: every figure it displays comes from
`reports/m2_calibration.json` and `reports/m6_evaluation.json`, with the FAQ
count taken from the corpus itself.
Run `python -m banking_triage.web_metrics` after re-running the evaluation, or
`--check` to fail when `web/metrics.json` is stale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import PACKAGE_ROOT
from .config import settings as default_settings


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _faq_entries(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def build_metrics(
    reports_dir: Path | None = None,
    faq_path: Path | None = None,
) -> dict[str, Any]:
    """Build the JSON payload the page fetches from `/static/metrics.json`."""
    reports_dir = reports_dir or default_settings.reports_dir
    faq_path = faq_path or default_settings.faq_path

    m2 = _read_json(reports_dir / "m2_calibration.json")
    m6 = _read_json(reports_dir / "m6_evaluation.json")

    model = m6["model"]
    split = m6["test_split"]
    probes = m6["probes"]

    return {
        "generated_at": m6["generated_at"],
        "model": {
            "version": model["model_version"],
            "test_accuracy": m2["test_scores"]["accuracy"],
            "test_macro_f1": m2["test_scores"]["macro_f1"],
            "thresholds": {
                "confidence": m2["thresholds"]["auto_route_threshold"],
                "margin": m2["thresholds"]["min_margin"],
                "escalate": m2["thresholds"].get("escalate_threshold"),
            },
        },
        "workflow": {
            "n": split["n"],
            "auto_answered": split["auto_answered"],
            "coverage": split["coverage"],
            "precision": split["precision"],
            "correct": split["auto_answer_correct"],
            "correct_rate": split["correct_auto_rate"],
            "clarified": split["decisions"]["clarify"],
            "escalated": split["decisions"]["escalate"],
            "no_copy": split["reasons"]["no_curated_answer"],
            "protected": split["reasons"]["protected_intent"],
            "latency": {
                "p50": split["latency_ms"]["p50"],
                "p95": split["latency_ms"]["p95"],
                "p99": split["latency_ms"]["p99"],
            },
        },
        "classifier_policy": {
            "routed": split["classifier_policy"]["routed"],
            "coverage": split["classifier_policy"]["coverage"],
            "precision": split["classifier_policy"]["precision"],
        },
        "probes": {
            "passed": probes["passed"],
            "total": probes["n_probes"],
            "turns": probes["n_turns"],
            "route_accuracy": probes["route_accuracy"],
            "tool_expected": probes["tool_calls"]["expected"],
            "tool_correct": probes["tool_calls"]["correct"],
            "tool_precision": probes["tool_calls"]["precision"],
            "tool_recall": probes["tool_calls"]["recall"],
        },
        "safety": {
            "predicted_never_auto": split["never_auto"]["predicted_and_auto_answered"],
            "gold_never_auto": split["never_auto"]["true_label_and_auto_answered"],
        },
        "content": {
            "faq_intents": _faq_entries(faq_path),
            "total_intents": model["n_intents"],
        },
    }


def render(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=default_settings.reports_dir)
    parser.add_argument("--faq", type=Path, default=default_settings.faq_path)
    parser.add_argument("--out", type=Path, default=PACKAGE_ROOT / "web" / "metrics.json")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the output file differs from a fresh render",
    )
    args = parser.parse_args()

    payload = build_metrics(args.reports_dir, args.faq)
    rendered = render(payload)

    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        if current != rendered:
            print(f"stale showcase metrics: {args.out}", file=sys.stderr)
            raise SystemExit(1)
        print(f"showcase metrics are current: {args.out}")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
