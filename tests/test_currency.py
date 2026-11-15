"""Currency and amount extraction tests.

Extraction is the weakest link in the exchange-rate path, so it is tested
separately from the HTTP call. Every case here is a real phrasing pattern.
"""

from __future__ import annotations

import pytest

from banking_triage.currency import build_query, describe_mentions, extract_mentions


def test_symbol_amount_before_currency_word() -> None:
    query = build_query("how much is $500 in euros?")
    assert query is not None
    assert (query.base, query.quote) == ("USD", "EUR")
    assert query.amount == 500


def test_currency_word_after_amount() -> None:
    query = build_query("how much is 500 dollars in euros")
    assert query is not None
    assert (query.base, query.quote) == ("USD", "EUR")
    assert query.amount == 500


def test_lowercase_iso_codes() -> None:
    query = build_query("what is the rate for gbp to jpy")
    assert query is not None
    assert (query.base, query.quote) == ("GBP", "JPY")


def test_uppercase_iso_codes() -> None:
    query = build_query("convert 250 EUR into CHF")
    assert query is not None
    assert (query.base, query.quote) == ("EUR", "CHF")
    assert query.amount == 250


def test_amount_with_thousands_separator() -> None:
    query = build_query("1,250.50 dollars in euros please")
    assert query is not None
    assert query.amount == pytest.approx(1250.50)


def test_spoken_currency_names() -> None:
    query = build_query("how many quid is 200 bucks")
    assert query is not None
    assert (query.base, query.quote) == ("USD", "GBP")


def test_single_currency_is_not_enough_to_answer() -> None:
    # Must ask a clarifying question rather than invent a pair.
    assert build_query("what is the euro rate today?") is None


def test_text_without_currency_returns_nothing() -> None:
    assert build_query("my card payment was declined yesterday") is None


def test_english_word_try_is_not_turkish_lira() -> None:
    # "try" appears constantly in support text; treating it as TRY would produce
    # a confidently wrong currency pair.
    assert "TRY" not in describe_mentions("I will try to transfer my balance")
    assert build_query("I try to send euros") is None


def test_three_currencies_picks_the_first_two() -> None:
    query = build_query("convert 100 usd to eur and then to gbp")
    assert query is not None
    assert (query.base, query.quote) == ("USD", "EUR")


def test_duplicate_currency_mentions_do_not_form_a_pair() -> None:
    assert build_query("send 100 euros, all in euros") is None


def test_multiword_currency_is_not_double_counted() -> None:
    mentions = extract_mentions("500 swiss francs into euros")
    assert [m.currency for m in mentions] == ["CHF", "EUR"]


def test_default_amount_is_one_when_unstated() -> None:
    query = build_query("usd to eur")
    assert query is not None
    assert query.amount == 1.0


def test_mentions_keep_source_order() -> None:
    mentions = extract_mentions("eur to usd")
    assert [m.currency for m in mentions] == ["EUR", "USD"]


def test_empty_text_yields_no_mentions() -> None:
    assert extract_mentions("") == []
