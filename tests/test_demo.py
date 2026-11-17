"""Tests for the demo's pure rendering helpers.

`main()` belongs to Streamlit; the mapping from a `TriageState` to what the page
shows is ordinary code and is tested here without importing Streamlit.
"""

from __future__ import annotations

import pytest

from banking_triage.demo import EXAMPLES, ROUTE_LABELS, route_label, turn_view
from banking_triage.schemas import (
    CurrencyQuery,
    IntentScore,
    Prediction,
    Source,
    TriageState,
)


def state(**overrides) -> TriageState:
    base = {
        "text": "convert 250 EUR to CHF",
        "prediction": Prediction(
            intent="fiat_currency_support",
            confidence=0.9112,
            alternatives=[IntentScore(intent="exchange_via_app", score=0.025)],
        ),
        "route": "exchange_rate",
        "decision": "auto_answer",
        "route_reason": "explicit_currency_pair",
        "currency_query": CurrencyQuery(base="EUR", quote="CHF", amount=250),
        "answer": "250.00 EUR = 231.10 CHF at the ECB reference rate for 2026-11-16.",
        "sources": [
            Source(
                kind="exchange_rate",
                ref="https://api.frankfurter.dev/v1/latest?base=EUR&symbols=CHF",
                detail="ECB reference rate",
            )
        ],
    }
    base.update(overrides)
    return TriageState.model_validate(base)


def test_turn_view_maps_the_state_onto_the_page() -> None:
    view = turn_view(state())

    assert view.answer.startswith("250.00 EUR = 231.10 CHF")
    assert view.route == "exchange_rate"
    assert view.route_label == "Live ECB rate"
    assert view.intent == "fiat_currency_support"
    assert view.confidence == pytest.approx(0.911)
    assert view.route_reason == "explicit_currency_pair"
    assert view.needs_human is False
    assert view.sources == (
        ("exchange_rate", "https://api.frankfurter.dev/v1/latest?base=EUR&symbols=CHF"),
    )
    assert view.errors == ()


def test_headline_is_a_readable_trace() -> None:
    view = turn_view(state())

    assert view.headline == "Live ECB rate · fiat_currency_support · confidence 0.91"


def test_handoff_and_errors_are_visible() -> None:
    view = turn_view(
        state(
            route="escalate",
            decision="escalate",
            route_reason="protected_intent",
            needs_human=True,
            errors=["faq: no entry for 'lost_or_stolen_card'"],
        )
    )

    assert view.route_label == "Human handoff"
    assert "human follow-up" in view.headline
    assert view.errors == ("faq: no entry for 'lost_or_stolen_card'",)


def test_turn_view_tolerates_a_missing_prediction() -> None:
    # Defensive: the graph always sets one, but the demo must not crash if the
    # state shape ever changes.
    view = turn_view(state(prediction=None, route="clarify", answer="Could you rephrase?"))

    assert view.intent is None
    assert view.confidence is None
    assert view.headline == "Clarification"


def test_unknown_route_does_not_invent_a_label() -> None:
    assert route_label("something_new") == "Unknown route"
    assert route_label(None) == "Unknown route"


def test_examples_are_unique_and_cover_the_interesting_routes() -> None:
    assert len(EXAMPLES) == len(set(EXAMPLES))
    assert all(example.strip() for example in EXAMPLES)
    assert any("EUR" in example for example in EXAMPLES), "needs a quote example"
    assert any("rate" in example.lower() for example in EXAMPLES), "needs a clarification"
    assert any("lost" in example.lower() for example in EXAMPLES), "needs a protected intent"


def test_every_route_has_a_label() -> None:
    assert ROUTE_LABELS.keys() == {"faq", "exchange_rate", "clarify", "escalate"}
