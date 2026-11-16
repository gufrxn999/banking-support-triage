"""End-to-end evaluation of the triage workflow.

Two measurements, deliberately separate:

1. **Probe set** (`data/probes.jsonl`). Hand-written messages that encode policy
   expectations, including cases BANKING77 cannot express: slot-filled conversion
   requests, fee questions that mention a pair, fraud reports that mention a pair,
   an unsupported currency, and multi-turn clarifications. This is where routing
   and tool-call behaviour are measured.
2. **Official test split**. The 3,080 BANKING77 test messages, scored once
   through the same graph. This is where auto-answer coverage, precision, and
   latency come from.

The exchange tool is stubbed in both, with a fixed rate and date. The question
this harness answers is "did the policy route and answer correctly?", and a live
ECB rate would make the report neither reproducible nor comparable. The tool's
own tests cover the HTTP contract; M5 covers a live lookup.

A note on the headline. M2 measured a *classifier policy*: confident messages
were counted as routed even when no curated answer existed. The workflow is
stricter -- it only answers intents that have reviewed copy -- so its coverage is
lower by construction. Both numbers are reported (`test_split.classifier_policy`
reproduces M2 on the same corpus), and the difference is the cost of not
improvising.

Usage:
    python -m banking_triage.evaluate
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .classifier import IntentClassifier
from .config import settings as default_settings
from .faq import FAQStore
from .graph import TriageGraph
from .schemas import ExchangeRate, Route, TriageState
from .taxonomy import NEVER_AUTO_ANSWER_INTENTS
from .tools.exchange import UnsupportedCurrencyError

# Keep failures readable in a report; the counts above them are exhaustive.
MAX_SAMPLES = 25


class ProbeValidationError(ValueError):
    """The probe file is malformed or ambiguous."""


class ToolExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base: str = Field(pattern=r"^[A-Z]{3}$")
    quote: str = Field(pattern=r"^[A-Z]{3}$")
    amount: float = Field(gt=0)


class ProbeExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: Route
    reason: str | None = None
    intent: str | None = None
    # Required in the file (enforced by the loader) so "no tool call expected"
    # is stated, never inferred from a missing key.
    tool: ToolExpectation | None = None
    error_contains: str | None = None


class ProbeTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    expect: ProbeExpectation


class Probe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    text: str = Field(min_length=1)
    expect: ProbeExpectation
    follow_up: list[ProbeTurn] = Field(default_factory=list)
    why: str = ""


def load_probes(path: Path | None = None) -> list[Probe]:
    """Load and validate the probe set. Fails loudly on a typo."""
    source = path or default_settings.probes_path
    if not source.exists():
        raise ProbeValidationError(f"missing probe set at {source}")

    probes: list[Probe] = []
    seen: set[str] = set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{source}:{line_number}"
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProbeValidationError(f"{where}: invalid JSON ({exc})") from exc
        if not isinstance(payload, dict):
            raise ProbeValidationError(f"{where}: each line must be a JSON object")

        for label, expectation in [
            ("expect", payload.get("expect")),
            *[
                (f"follow_up[{index}].expect", turn.get("expect"))
                for index, turn in enumerate(payload.get("follow_up") or [])
                if isinstance(turn, dict)
            ],
        ]:
            if isinstance(expectation, dict) and "tool" not in expectation:
                raise ProbeValidationError(
                    f'{where}: {label} must state "tool": null or a pair; '
                    "silence is not an expectation"
                )

        try:
            probe = Probe.model_validate(payload)
        except ValidationError as exc:
            raise ProbeValidationError(f"{where}: {exc}") from exc
        if probe.id in seen:
            raise ProbeValidationError(f"{where}: duplicate probe id {probe.id!r}")
        seen.add(probe.id)
        probes.append(probe)

    if not probes:
        raise ProbeValidationError(f"{source}: no probes found")
    return probes


class StubExchange:
    """Deterministic stand-in for Frankfurter.

    It rejects TRY, mirroring the live API: the ECB does not publish a reference
    rate for it, so the unsupported-pair path is measured rather than assumed.
    """

    RATE = 0.9
    AS_OF = date(2026, 1, 2)
    UNSUPPORTED = frozenset({"TRY"})

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def clear(self) -> None:
        self.calls.clear()

    def latest(self, base: str, quote: str) -> ExchangeRate:
        self.calls.append((base, quote))
        if base in self.UNSUPPORTED or quote in self.UNSUPPORTED:
            raise UnsupportedCurrencyError(base, quote)
        return ExchangeRate(
            base=base,
            quote=quote,
            rate=self.RATE,
            as_of=self.AS_OF,
            source_url=f"https://api.frankfurter.dev/v1/latest?base={base}&symbols={quote}",
        )


def _reason_text(expectation: ProbeExpectation) -> str:
    return (
        "no tool call"
        if expectation.tool is None
        else (f"{expectation.tool.base}->{expectation.tool.quote} for {expectation.tool.amount:g}")
    )


def _compare_turn(
    expectation: ProbeExpectation,
    state: TriageState,
    calls: Sequence[tuple[str, str]],
) -> list[str]:
    """Return every way one turn disagreeed with its expectation."""
    problems: list[str] = []

    if state.route != expectation.route:
        problems.append(f"route {state.route!r} != {expectation.route!r}")
    if expectation.reason is not None and state.route_reason != expectation.reason:
        problems.append(f"reason {state.route_reason!r} != {expectation.reason!r}")

    predicted = state.prediction.intent if state.prediction else None
    if expectation.intent is not None and predicted != expectation.intent:
        problems.append(f"intent {predicted!r} != {expectation.intent!r}")

    expected_calls = (
        [] if expectation.tool is None else [(expectation.tool.base, expectation.tool.quote)]
    )
    if expectation.tool is None:
        if calls:
            problems.append(f"unexpected tool call(s) {list(calls)}")
    elif list(calls) != expected_calls:
        problems.append(
            f"tool calls {list(calls)} != {expected_calls} (expected {_reason_text(expectation)})"
        )
    if expectation.tool is not None:
        query = state.currency_query
        if query is None:
            problems.append("no currency query to check the amount against")
        elif not math.isclose(query.amount, expectation.tool.amount, rel_tol=1e-9):
            problems.append(f"amount {query.amount:g} != {expectation.tool.amount:g}")

    if expectation.error_contains and not any(
        expectation.error_contains in error for error in state.errors
    ):
        problems.append(
            f"no error containing {expectation.error_contains!r}; errors={state.errors}"
        )
    return problems


def _observed_turn(
    turn: int, text: str, state: TriageState, calls: Sequence[tuple[str, str]]
) -> dict:
    prediction = state.prediction
    return {
        "turn": turn,
        "text": text,
        "route": state.route,
        "reason": state.route_reason,
        "decision": state.decision,
        "intent": prediction.intent if prediction else None,
        "confidence": round(prediction.confidence, 4) if prediction else None,
        "amount": state.currency_query.amount if state.currency_query else None,
        "tool_calls": [list(call) for call in calls],
        "errors": list(state.errors),
    }


def run_probes(
    graph: TriageGraph, probes: Iterable[Probe], exchange: StubExchange
) -> dict[str, Any]:
    """Run every probe, including follow-up turns, and score the expectations."""
    details: list[dict[str, Any]] = []

    turns_total = route_checks = route_matches = intent_checks = intent_matches = 0
    expected_calls = actual_calls = correct_calls = 0
    unexpected: list[dict[str, Any]] = []
    missed: list[dict[str, Any]] = []

    for probe in probes:
        turns = [
            (probe.text, probe.expect),
            *[(turn.text, turn.expect) for turn in probe.follow_up],
        ]
        session_id = f"probe-{probe.id}" if probe.follow_up else None
        failures: list[str] = []
        observed: list[dict[str, Any]] = []

        for index, (text, expectation) in enumerate(turns, start=1):
            exchange.clear()
            state = graph.invoke(text, session_id=session_id)
            calls = list(exchange.calls)

            turns_total += 1
            route_checks += 1
            route_matches += int(state.route == expectation.route)
            if expectation.intent is not None:
                intent_checks += 1
                predicted = state.prediction.intent if state.prediction else None
                intent_matches += int(predicted == expectation.intent)

            wants_tool = expectation.tool is not None
            used_tool = bool(calls)
            expected_calls += int(wants_tool)
            actual_calls += int(used_tool)

            problems = _compare_turn(expectation, state, calls)
            if wants_tool and calls == [(expectation.tool.base, expectation.tool.quote)]:
                query = state.currency_query
                if query is not None and math.isclose(
                    query.amount, expectation.tool.amount, rel_tol=1e-9
                ):
                    correct_calls += 1
            if wants_tool and not used_tool:
                missed.append(
                    {"probe": probe.id, "turn": index, "expected": _reason_text(expectation)}
                )
            if not wants_tool and used_tool:
                unexpected.append({"probe": probe.id, "turn": index, "calls": calls})

            failures.extend(f"turn {index}: {problem}" for problem in problems)
            observed.append(_observed_turn(index, text, state, calls))

        details.append(
            {
                "id": probe.id,
                "passed": not failures,
                "failures": failures,
                "why": probe.why,
                "turns": observed,
            }
        )

    failed = [detail for detail in details if not detail["passed"]]
    return {
        "n_probes": len(details),
        "n_turns": turns_total,
        "passed": len(details) - len(failed),
        "failed": len(failed),
        "route_accuracy": round(route_matches / route_checks, 4) if route_checks else None,
        "intent_accuracy": round(intent_matches / intent_checks, 4) if intent_checks else None,
        "tool_calls": {
            "expected": expected_calls,
            "actual": actual_calls,
            "correct": correct_calls,
            "precision": round(correct_calls / actual_calls, 4) if actual_calls else None,
            "recall": round(correct_calls / expected_calls, 4) if expected_calls else None,
        },
        "unexpected_calls": unexpected,
        "missed_calls": missed,
        "details": details,
    }


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    p50, p95, p99 = np.percentile(np.asarray(values), [50, 95, 99])
    return {
        "p50": round(float(p50), 2),
        "p95": round(float(p95), 2),
        "p99": round(float(p99), 2),
        "mean": round(float(np.mean(values)), 2),
        "max": round(float(np.max(values)), 2),
    }


def evaluate_split(
    graph: TriageGraph,
    frame: pd.DataFrame,
    labels: list[dict[str, Any]],
    limit: int = MAX_SAMPLES,
) -> dict[str, Any]:
    """Route every test message through the real workflow and score the decisions."""
    names = {int(row["id"]): str(row["name"]) for row in labels}
    thresholds = graph.thresholds()

    routes: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    false_pairs: Counter[str] = Counter()
    error_messages: Counter[str] = Counter()
    latencies: list[float] = []

    auto_answered = auto_correct = degraded = 0
    classifier_policy_routed = classifier_policy_correct = 0
    predicted_never_answered = true_never_answered = 0
    explicit_pair_answers = 0

    false_auto_answers: list[dict[str, Any]] = []
    true_never_samples: list[dict[str, Any]] = []

    for text, label_id in zip(frame["text"], frame["label_id"], strict=True):
        true_intent = names[int(label_id)]
        started = perf_counter()
        state = graph.invoke(str(text))
        latencies.append((perf_counter() - started) * 1000)

        predicted = state.prediction.intent if state.prediction else None
        confidence = state.prediction.confidence if state.prediction else 0.0
        margin = state.prediction.margin if state.prediction else 0.0
        correct = predicted == true_intent

        routes[str(state.route)] += 1
        decisions[str(state.decision)] += 1
        reasons[str(state.route_reason)] += 1
        for error in state.errors:
            error_messages[error] += 1
        degraded += int(bool(state.errors))

        # The M2 policy, recomputed on the same corpus as a consistency check:
        # confident + wide margin + not a never-auto intent, regardless of
        # whether a curated answer exists.
        if (
            confidence >= thresholds["auto_route_threshold"]
            and margin >= thresholds["min_margin"]
            and predicted not in NEVER_AUTO_ANSWER_INTENTS
        ):
            classifier_policy_routed += 1
            classifier_policy_correct += int(correct)

        if state.decision == "auto_answer":
            auto_answered += 1
            auto_correct += int(correct)
            explicit_pair_answers += int(state.route_reason == "explicit_currency_pair")
            predicted_never_answered += int(predicted in NEVER_AUTO_ANSWER_INTENTS)
            true_never_answered += int(true_intent in NEVER_AUTO_ANSWER_INTENTS)

            if true_intent in NEVER_AUTO_ANSWER_INTENTS and len(true_never_samples) < limit:
                true_never_samples.append(
                    {
                        "text": str(text),
                        "true": true_intent,
                        "predicted": predicted,
                        "route_reason": state.route_reason,
                    }
                )
            if not correct and len(false_auto_answers) < limit:
                false_auto_answers.append(
                    {
                        "text": str(text),
                        "true": true_intent,
                        "predicted": predicted,
                        "confidence": round(confidence, 4),
                        "route": state.route,
                        "route_reason": state.route_reason,
                        "answer": state.answer,
                    }
                )
            if not correct:
                false_pairs[f"{true_intent} -> {predicted}"] += 1

    n = len(frame)
    return {
        "n": n,
        "coverage": round(auto_answered / n, 4),
        "precision": round(auto_correct / auto_answered, 4) if auto_answered else None,
        "correct_auto_rate": round(auto_correct / n, 4),
        "auto_answered": auto_answered,
        "auto_answer_correct": auto_correct,
        "precision_definition": (
            "share of auto-answered messages whose predicted intent matches the gold "
            "label; an exchange quote counts as correct only when the gold intent is "
            "exchange_rate"
        ),
        "decisions": dict(sorted(decisions.items())),
        "routes": dict(sorted(routes.items())),
        "reasons": dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0]))),
        "classifier_policy": {
            "routed": classifier_policy_routed,
            "coverage": round(classifier_policy_routed / n, 4),
            "precision": (
                round(classifier_policy_correct / classifier_policy_routed, 4)
                if classifier_policy_routed
                else None
            ),
            "note": (
                "M2 definition, recomputed here: confident and wide-margin and not a "
                "never-auto intent, whether or not curated copy exists. M2 reported "
                "2220/3080 at 0.9622; the workflow answers fewer because uncovered "
                "intents escalate instead of improvising."
            ),
        },
        "never_auto": {
            "predicted_and_auto_answered": predicted_never_answered,
            "true_label_and_auto_answered": true_never_answered,
            "safety_note": (
                "The first count must be zero by construction; the second is the "
                "residual risk from misclassifying a protected intent as a safe one."
            ),
            "samples": true_never_samples,
        },
        "explicit_pair_auto_answers": explicit_pair_answers,
        "degraded": {
            "responses_with_errors": degraded,
            "error_messages": dict(error_messages.most_common(limit)),
        },
        "latency_ms": _percentiles(latencies),
        "false_auto_pairs": dict(false_pairs.most_common(10)),
        "false_auto_answers": false_auto_answers,
    }


def _model_block(classifier: IntentClassifier, graph: TriageGraph) -> dict[str, Any]:
    metadata = classifier.metadata
    return {
        "model_version": classifier.model_version,
        "trained_at": metadata.get("trained_at"),
        "trained_on": metadata.get("trained_on"),
        "n_intents": len(classifier.label_names),
        "calibrated": (metadata.get("thresholds") or {}).get("calibrated"),
        "thresholds": graph.thresholds(),
        "recorded_test_metrics": metadata.get("metrics", {}),
    }


def build_report(
    classifier: IntentClassifier,
    graph: TriageGraph,
    probes: Sequence[Probe],
    probe_results: dict[str, Any],
    split_results: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": _model_block(classifier, graph),
        "protocol": {
            "probes": (
                f"{len(probes)} hand-written policy cases from data/probes.jsonl, "
                "including multi-turn clarifications; exchange tool stubbed"
            ),
            "test_split": (
                "official BANKING77 test split, scored once through the full graph; "
                "thresholds and copy frozen from earlier milestones"
            ),
            "tool_stub": (
                f"1 base = {StubExchange.RATE:.4f} quote as of "
                f"{StubExchange.AS_OF.isoformat()}; TRY rejected like the live API"
            ),
            "latency": (
                "in-process graph invocation, single thread, exchange stubbed; "
                "excludes network time (measured separately in M5)"
            ),
        },
        "probes": probe_results,
        "test_split": split_results,
    }


def _write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _print_summary(report: dict[str, Any], report_path: Path) -> None:
    probes = report["probes"]
    calls = probes["tool_calls"]
    print(
        f"probes: {probes['passed']}/{probes['n_probes']} passed  "
        f"routes {probes['route_accuracy']:.1%}  "
        f"tool precision {calls['precision']} recall {calls['recall']}"
    )
    for detail in probes["details"]:
        if not detail["passed"]:
            print(f"  FAIL {detail['id']}: {'; '.join(detail['failures'])}")

    split = report["test_split"]
    if split is not None:
        print(
            f"test split: auto-answered {split['auto_answered']}/{split['n']} "
            f"({split['coverage']:.1%}) at precision {_ratio(split['precision'])} "
            f"-> {split['correct_auto_rate']:.1%} of traffic answered correctly"
        )
        policy = split["classifier_policy"]
        print(
            f"  M2-equivalent classifier policy: {policy['routed']}/{split['n']} "
            f"({policy['coverage']:.1%}) at {_ratio(policy['precision'])}"
        )
        latency = split["latency_ms"]
        print(
            f"  never-auto: {split['never_auto']['predicted_and_auto_answered']} predicted, "
            f"{split['never_auto']['true_label_and_auto_answered']} true-label  "
            f"latency p50/p95/p99 {latency['p50']}/{latency['p95']}/{latency['p99']} ms"
        )
    print(f"wrote {report_path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate routing, tool calls, and latency.")
    parser.add_argument("--probes", type=Path, default=None, help="probe JSONL path")
    parser.add_argument("--report", type=Path, default=None, help="output JSON path")
    parser.add_argument(
        "--skip-split", action="store_true", help="probes only; do not score the test split"
    )
    args = parser.parse_args(argv)

    settings = default_settings
    settings.ensure_dirs()
    report_path = args.report or settings.reports_dir / "m6_evaluation.json"

    try:
        classifier = IntentClassifier.load(settings=settings)
    except Exception as exc:  # noqa: BLE001 - actionable message beats a traceback
        print(f"cannot evaluate: {exc}")
        return 2

    faq = FAQStore.load(settings.faq_path)
    exchange = StubExchange()
    graph = TriageGraph(classifier, faq, exchange, settings)
    try:
        try:
            probes = load_probes(args.probes or settings.probes_path)
        except ProbeValidationError as exc:
            print(f"invalid probe set: {exc}")
            return 2
        probe_results = run_probes(graph, probes, exchange)
        split_results = None
        if not args.skip_split:
            from .data import load_labels, load_splits

            split_results = evaluate_split(graph, load_splits()["test"], load_labels())
    finally:
        graph.close()

    report = build_report(classifier, graph, probes, probe_results, split_results)
    _write(report_path, report)
    _print_summary(report, report_path)
    return 0 if probe_results["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
