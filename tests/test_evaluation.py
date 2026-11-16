"""Tests for the evaluation harness.

The harness is the project's falsifiability claim, so it is tested like any other
component: the probe file is validated, mismatches are detected, tool-call
precision/recall is counted rather than asserted, and the split scorer's
arithmetic is pinned with a fake graph.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from banking_triage.config import settings
from banking_triage.evaluate import (
    ProbeValidationError,
    StubExchange,
    _percentiles,
    build_report,
    evaluate_split,
    load_probes,
    run_probes,
)
from banking_triage.schemas import (
    CurrencyQuery,
    IntentScore,
    Prediction,
    Source,
    TriageState,
)


def state(
    text: str,
    *,
    route: str = "faq",
    decision: str = "auto_answer",
    intent: str = "change_pin",
    confidence: float = 0.90,
    reason: str | None = "classifier",
    query: CurrencyQuery | None = None,
    errors: list[str] | None = None,
) -> TriageState:
    return TriageState(
        text=text,
        prediction=Prediction(
            intent=intent,
            confidence=confidence,
            alternatives=[IntentScore(intent="card_arrival", score=0.02)],
        ),
        route=route,  # type: ignore[arg-type]
        decision=decision,  # type: ignore[arg-type]
        route_reason=reason,
        currency_query=query,
        answer="an answer",
        sources=[Source(kind="faq", ref=intent)],
        errors=errors or [],
    )


class FakeGraph:
    """Minimal graph double: the evaluator only calls invoke() and thresholds()."""

    def __init__(self, factory) -> None:
        self.factory = factory
        self.sessions: list[str | None] = []

    def invoke(self, text: str, session_id: str | None = None) -> TriageState:
        self.sessions.append(session_id)
        return self.factory(text)

    def thresholds(self) -> dict[str, float]:
        return {
            "auto_route_threshold": 0.54,
            "min_margin": 0.10,
            "escalate_threshold": 0.40,
        }


def write_probes(path: Path, *rows: dict) -> Path:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


# --- probe loading -------------------------------------------------------


def test_shipped_probe_set_is_valid() -> None:
    probes = load_probes(settings.probes_path)

    assert len(probes) >= 20
    assert len({probe.id for probe in probes}) == len(probes)
    assert {probe.expect.route for probe in probes} >= {
        "faq",
        "exchange_rate",
        "clarify",
        "escalate",
    }
    assert sum(1 for probe in probes if probe.expect.tool) >= 6
    assert sum(1 for probe in probes if probe.follow_up) >= 3
    assert all(probe.why for probe in probes), "every probe must explain what policy it pins"


def test_probe_must_declare_its_tool_expectation(tmp_path: Path) -> None:
    path = write_probes(
        tmp_path / "probes.jsonl",
        {"id": "no-tool-key", "text": "hello", "expect": {"route": "faq"}},
    )

    with pytest.raises(ProbeValidationError, match="tool"):
        load_probes(path)


def test_duplicate_probe_ids_are_rejected(tmp_path: Path) -> None:
    row = {"id": "same", "text": "hello", "expect": {"route": "faq", "tool": None}}
    path = write_probes(tmp_path / "probes.jsonl", row, row)

    with pytest.raises(ProbeValidationError, match="duplicate probe id"):
        load_probes(path)


def test_unknown_route_is_rejected(tmp_path: Path) -> None:
    path = write_probes(
        tmp_path / "probes.jsonl",
        {"id": "bad-route", "text": "hello", "expect": {"route": "answer", "tool": None}},
    )

    with pytest.raises(ProbeValidationError):
        load_probes(path)


def test_unknown_expectation_field_is_rejected(tmp_path: Path) -> None:
    path = write_probes(
        tmp_path / "probes.jsonl",
        {"id": "typo", "text": "hello", "expect": {"route": "faq", "tool": None, "intnet": "x"}},
    )

    with pytest.raises(ProbeValidationError):
        load_probes(path)


def test_invalid_tool_pair_is_rejected(tmp_path: Path) -> None:
    path = write_probes(
        tmp_path / "probes.jsonl",
        {
            "id": "bad-pair",
            "text": "convert 5 USD to EUR",
            "expect": {
                "route": "exchange_rate",
                "tool": {"base": "usd", "quote": "EUR", "amount": 5},
            },
        },
    )

    with pytest.raises(ProbeValidationError):
        load_probes(path)


# --- probe scoring -------------------------------------------------------


def probe_row(probe_id: str, **expect) -> dict:
    return {
        "id": probe_id,
        "text": f"text for {probe_id}",
        "expect": {"route": "faq", "tool": None, **expect},
        "why": "test fixture",
    }


def test_matching_probe_passes(tmp_path: Path) -> None:
    path = write_probes(tmp_path / "probes.jsonl", probe_row("pins", intent="change_pin"))
    graph = FakeGraph(lambda text: state(text))
    exchange = StubExchange()

    result = run_probes(graph, load_probes(path), exchange)

    assert result["passed"] == 1
    assert result["failed"] == 0
    assert result["route_accuracy"] == 1.0
    assert result["intent_accuracy"] == 1.0
    assert result["tool_calls"] == {
        "expected": 0,
        "actual": 0,
        "correct": 0,
        "precision": None,
        "recall": None,
    }


def test_route_mismatch_fails_the_probe(tmp_path: Path) -> None:
    path = write_probes(
        tmp_path / "probes.jsonl", probe_row("should-clarify", route="clarify", reason="ambiguous")
    )
    graph = FakeGraph(lambda text: state(text))

    result = run_probes(graph, load_probes(path), StubExchange())

    assert result["failed"] == 1
    failures = result["details"][0]["failures"]
    assert any("route 'faq' != 'clarify'" in failure for failure in failures)
    assert any("reason 'classifier' != 'ambiguous'" in failure for failure in failures)


def test_unexpected_tool_call_reduces_precision(tmp_path: Path) -> None:
    path = write_probes(
        tmp_path / "probes.jsonl",
        probe_row("must-not-quote", route="clarify"),
    )
    exchange = StubExchange()

    def factory(text: str) -> TriageState:
        exchange.latest("USD", "EUR")
        return state(text, route="clarify", decision="clarify", reason="ambiguous")

    result = run_probes(FakeGraph(factory), load_probes(path), exchange)

    assert result["tool_calls"]["actual"] == 1
    assert result["tool_calls"]["expected"] == 0
    assert result["tool_calls"]["precision"] == 0.0
    assert result["unexpected_calls"] == [
        {"probe": "must-not-quote", "turn": 1, "calls": [("USD", "EUR")]}
    ]
    assert any("unexpected" in failure for failure in result["details"][0]["failures"])


def test_missed_tool_call_reduces_recall(tmp_path: Path) -> None:
    path = write_probes(
        tmp_path / "probes.jsonl",
        probe_row(
            "must-quote",
            route="exchange_rate",
            reason="explicit_currency_pair",
            intent=None,
            tool={"base": "USD", "quote": "EUR", "amount": 500},
        ),
    )
    graph = FakeGraph(
        lambda text: state(
            text,
            route="exchange_rate",
            reason="explicit_currency_pair",
            query=CurrencyQuery(base="USD", quote="EUR", amount=500),
        )
    )

    result = run_probes(graph, load_probes(path), StubExchange())

    assert result["tool_calls"]["expected"] == 1
    assert result["tool_calls"]["actual"] == 0
    assert result["tool_calls"]["recall"] == 0.0
    assert result["missed_calls"][0]["probe"] == "must-quote"


def test_correct_tool_call_counts_for_both_precision_and_recall(tmp_path: Path) -> None:
    path = write_probes(
        tmp_path / "probes.jsonl",
        probe_row(
            "quote",
            route="exchange_rate",
            reason="explicit_currency_pair",
            intent=None,
            tool={"base": "USD", "quote": "EUR", "amount": 500},
        ),
    )
    exchange = StubExchange()

    def factory(text: str) -> TriageState:
        exchange.latest("USD", "EUR")
        return state(
            text,
            route="exchange_rate",
            reason="explicit_currency_pair",
            query=CurrencyQuery(base="USD", quote="EUR", amount=500),
        )

    result = run_probes(FakeGraph(factory), load_probes(path), exchange)

    assert result["passed"] == 1
    assert result["tool_calls"] == {
        "expected": 1,
        "actual": 1,
        "correct": 1,
        "precision": 1.0,
        "recall": 1.0,
    }


def test_amount_mismatch_fails_even_when_the_pair_is_right(tmp_path: Path) -> None:
    path = write_probes(
        tmp_path / "probes.jsonl",
        probe_row(
            "wrong-amount",
            route="exchange_rate",
            reason="explicit_currency_pair",
            intent=None,
            tool={"base": "USD", "quote": "EUR", "amount": 500},
        ),
    )
    exchange = StubExchange()

    def factory(text: str) -> TriageState:
        exchange.latest("USD", "EUR")
        return state(
            text,
            route="exchange_rate",
            reason="explicit_currency_pair",
            query=CurrencyQuery(base="USD", quote="EUR", amount=250),
        )

    result = run_probes(FakeGraph(factory), load_probes(path), exchange)

    assert result["failed"] == 1
    assert any("amount 250" in failure for failure in result["details"][0]["failures"])


def test_follow_up_turns_share_a_session(tmp_path: Path) -> None:
    path = write_probes(
        tmp_path / "probes.jsonl",
        {
            "id": "session",
            "text": "what is the rate?",
            "expect": {"route": "clarify", "reason": "currency_pair_missing", "tool": None},
            "follow_up": [
                {
                    "text": "500 USD to EUR",
                    "expect": {
                        "route": "exchange_rate",
                        "reason": "follow_up_currency_pair",
                        "tool": {"base": "USD", "quote": "EUR", "amount": 500},
                    },
                }
            ],
            "why": "test fixture",
        },
    )
    exchange = StubExchange()

    def factory(text: str) -> TriageState:
        if text == "what is the rate?":
            return state(text, route="clarify", decision="clarify", reason="currency_pair_missing")
        exchange.latest("USD", "EUR")
        return state(
            text,
            route="exchange_rate",
            reason="follow_up_currency_pair",
            query=CurrencyQuery(base="USD", quote="EUR", amount=500),
        )

    graph = FakeGraph(factory)
    result = run_probes(graph, load_probes(path), exchange)

    assert result["passed"] == 1
    assert result["n_turns"] == 2
    assert graph.sessions == ["probe-session", "probe-session"]


def test_single_turn_probes_do_not_create_sessions(tmp_path: Path) -> None:
    path = write_probes(tmp_path / "probes.jsonl", probe_row("anon"))
    graph = FakeGraph(lambda text: state(text))

    run_probes(graph, load_probes(path), StubExchange())

    assert graph.sessions == [None]


# --- split scoring -------------------------------------------------------


def labels() -> list[dict]:
    return [
        {"id": 0, "name": "change_pin"},
        {"id": 1, "name": "card_arrival"},
        {"id": 2, "name": "lost_or_stolen_card"},
    ]


def test_evaluate_split_counts_coverage_and_precision() -> None:
    frame = pd.DataFrame(
        {
            "text": ["a", "b", "c", "d"],
            "label_id": [0, 1, 0, 2],
        }
    )

    def factory(text: str) -> TriageState:
        states = {
            "a": state("a", intent="change_pin"),
            "b": state("b", intent="change_pin"),  # wrong but auto-answered
            "c": state(
                "c", route="clarify", decision="clarify", intent="change_pin", reason="ambiguous"
            ),
            "d": state(
                "d",
                route="escalate",
                decision="escalate",
                intent="lost_or_stolen_card",
                reason="protected_intent",
            ),
        }
        return states[text]

    result = evaluate_split(FakeGraph(factory), frame, labels())

    assert result["n"] == 4
    assert result["auto_answered"] == 2
    assert result["auto_answer_correct"] == 1
    assert result["coverage"] == 0.5
    assert result["precision"] == 0.5
    assert result["correct_auto_rate"] == 0.25
    assert result["decisions"] == {"auto_answer": 2, "clarify": 1, "escalate": 1}
    assert result["never_auto"]["predicted_and_auto_answered"] == 0
    assert result["never_auto"]["true_label_and_auto_answered"] == 0
    assert set(result["latency_ms"]) == {"p50", "p95", "p99", "mean", "max"}


def test_evaluate_split_flags_true_never_auto_answers() -> None:
    # A protected label answered because the classifier confused it with a safe
    # neighbour: the residual risk the report must show, not hide.
    frame = pd.DataFrame({"text": ["d"], "label_id": [2]})

    result = evaluate_split(
        FakeGraph(lambda text: state(text, intent="card_arrival")), frame, labels()
    )

    assert result["never_auto"]["predicted_and_auto_answered"] == 0
    assert result["never_auto"]["true_label_and_auto_answered"] == 1
    assert result["never_auto"]["samples"] == [
        {
            "text": "d",
            "true": "lost_or_stolen_card",
            "predicted": "card_arrival",
            "route_reason": "classifier",
        }
    ]


def test_evaluate_split_records_degraded_responses() -> None:
    frame = pd.DataFrame({"text": ["a"], "label_id": [0]})
    graph = FakeGraph(lambda text: state(text, errors=["exchange: USD->TRY is not available"]))

    result = evaluate_split(graph, frame, labels())

    assert result["degraded"]["responses_with_errors"] == 1
    assert result["degraded"]["error_messages"] == {"exchange: USD->TRY is not available": 1}


def test_percentiles_are_sorted_and_rounded() -> None:
    result = _percentiles([1.0, 2.0, 3.0, 4.0, 5.0])

    assert result["p50"] == 3.0
    assert result["p95"] == pytest.approx(4.8)
    assert result["p99"] == pytest.approx(4.96)
    assert result["max"] == 5.0
    assert _percentiles([]) == {}


def test_report_is_json_serialisable(tmp_path: Path) -> None:
    path = write_probes(tmp_path / "probes.jsonl", probe_row("one"))
    probes = load_probes(path)
    graph = FakeGraph(lambda text: state(text))
    probe_results = run_probes(graph, probes, StubExchange())

    report = build_report(
        classifier=type("C", (), {"metadata": {}, "model_version": "v-test", "label_names": []})(),
        graph=graph,  # type: ignore[arg-type]
        probes=probes,
        probe_results=probe_results,
        split_results=None,
    )

    assert json.loads(json.dumps(report))["probes"]["passed"] == 1
    assert report["test_split"] is None


# --- the real thing ------------------------------------------------------


def test_shipped_probes_pass_against_the_real_artifact() -> None:
    if not settings.model_path.exists():
        pytest.skip("run `python -m banking_triage.train` first")

    from banking_triage.classifier import IntentClassifier
    from banking_triage.faq import FAQStore
    from banking_triage.graph import TriageGraph

    exchange = StubExchange()
    graph = TriageGraph(IntentClassifier.load(), FAQStore.load(), exchange)
    try:
        result = run_probes(graph, load_probes(), exchange)
    finally:
        graph.close()

    assert result["failed"] == 0, [detail for detail in result["details"] if not detail["passed"]]
    assert result["tool_calls"]["precision"] == 1.0
