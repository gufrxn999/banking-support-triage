"""Contract tests: the shapes the API and graph agree on."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from banking_triage.config import MAX_TEXT_CHARS
from banking_triage.schemas import (
    ExchangeRate,
    IntentScore,
    Prediction,
    TriageRequest,
    TriageState,
)


def test_request_rejects_blank_text() -> None:
    with pytest.raises(ValidationError):
        TriageRequest(text="")


def test_request_rejects_overlong_text() -> None:
    with pytest.raises(ValidationError):
        TriageRequest(text="a" * (MAX_TEXT_CHARS + 1))


def test_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        TriageRequest(text="hi", unexpected="nope")


def test_prediction_margin_uses_runner_up() -> None:
    prediction = Prediction(
        intent="card_arrival",
        confidence=0.82,
        alternatives=[IntentScore(intent="card_not_working", score=0.72)],
    )
    assert prediction.margin == pytest.approx(0.10)


def test_prediction_without_alternatives_has_full_margin() -> None:
    prediction = Prediction(intent="card_arrival", confidence=0.6)
    assert prediction.margin == pytest.approx(0.6)


def test_confidence_is_bounded() -> None:
    with pytest.raises(ValidationError):
        Prediction(intent="x", confidence=1.4)


def test_exchange_rate_conversion_rounds_to_cents() -> None:
    rate = ExchangeRate(
        base="USD",
        quote="EUR",
        rate=0.90481,
        as_of=date(2026, 11, 13),
        source_url="https://api.frankfurter.dev/v1/latest?base=USD&symbols=EUR",
        amount=500,
    )
    assert rate.converted() == pytest.approx(452.41)


def test_exchange_rate_without_amount_has_no_conversion() -> None:
    rate = ExchangeRate(
        base="USD",
        quote="EUR",
        rate=0.9,
        as_of=date(2026, 11, 13),
        source_url="https://example.test",
    )
    assert rate.converted() is None


def test_state_defaults_to_empty_evidence() -> None:
    state = TriageState(text="my card has not arrived")
    assert state.route is None
    assert state.needs_human is False
    assert state.errors == []
