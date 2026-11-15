"""Frankfurter client tests.

All HTTP is mocked with `httpx.MockTransport`, so the suite is offline-safe and
deterministic -- a requirement for the CI story and for the evaluation harness.
"""

from __future__ import annotations

from datetime import date

import httpx
import pytest

from banking_triage.tools.exchange import (
    ExchangeRateUnavailable,
    FrankfurterClient,
    UnsupportedCurrencyError,
)

LATEST_OK = {
    "amount": 1.0,
    "base": "USD",
    "date": "2026-11-13",
    "rates": {"EUR": 0.90481},
}


def _client(handler) -> FrankfurterClient:
    return FrankfurterClient(
        client=httpx.Client(
            base_url="https://api.frankfurter.dev/v1",
            transport=httpx.MockTransport(handler),
        ),
        max_retries=2,
    )


def test_latest_returns_a_dated_rate() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["base"] == "USD"
        assert request.url.params["symbols"] == "EUR"
        return httpx.Response(200, json=LATEST_OK)

    with _client(handler) as client:
        rate = client.latest("usd", "eur")

    assert (rate.base, rate.quote) == ("USD", "EUR")
    assert rate.rate == pytest.approx(0.90481)
    assert rate.as_of == date(2026, 11, 13)
    assert "/latest" in rate.source_url


def test_conversion_uses_the_dated_rate() -> None:
    with _client(lambda request: httpx.Response(200, json=LATEST_OK)) as client:
        rate = client.latest("USD", "EUR")
    assert rate.converted() is None  # no amount stated
    rate.amount = 500
    assert rate.converted() == pytest.approx(452.41)


def test_unsupported_pair_raises_recoverable_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "not found"})

    with _client(handler) as client, pytest.raises(UnsupportedCurrencyError):
        client.latest("USD", "XYZ")


def test_missing_quote_in_rates_is_an_unsupported_pair() -> None:
    payload = {"amount": 1.0, "base": "USD", "date": "2026-11-13", "rates": {}}
    with _client(lambda request: httpx.Response(200, json=payload)) as client, pytest.raises(
        UnsupportedCurrencyError
    ):
        client.latest("USD", "EUR")


def test_transient_failure_is_retried_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, json=LATEST_OK)

    with _client(handler) as client:
        rate = client.latest("USD", "EUR")

    assert calls["n"] == 3
    assert rate.rate == pytest.approx(0.90481)


def test_exhausted_retries_surface_the_upstream_status() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503)

    with _client(handler) as client, pytest.raises(ExchangeRateUnavailable, match="503"):
        client.latest("USD", "EUR")
    assert calls["n"] == 3  # initial attempt + 2 retries


def test_timeouts_are_retried_then_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("boom")

    with _client(handler) as client, pytest.raises(ExchangeRateUnavailable):
        client.latest("USD", "EUR")


def test_same_currency_is_rejected_before_any_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("should not be called")

    with _client(handler) as client, pytest.raises(ValueError, match="differ"):
        client.latest("EUR", "EUR")


def test_invalid_code_is_rejected() -> None:
    with _client(lambda request: httpx.Response(200, json=LATEST_OK)) as client, pytest.raises(
        ValueError, match="ISO-4217"
    ):
        client.latest("US", "EUR")


def test_currency_list_is_uppercased() -> None:
    payload = {"USD": "United States Dollar", "EUR": "Euro"}
    with _client(lambda request: httpx.Response(200, json=payload)) as client:
        currencies = client.currencies()
    assert currencies["USD"] == "United States Dollar"
    assert set(currencies) == {"USD", "EUR"}


def test_empty_currency_list_is_an_upstream_failure() -> None:
    with _client(lambda request: httpx.Response(200, json={})) as client, pytest.raises(
        ExchangeRateUnavailable
    ):
        client.currencies()
