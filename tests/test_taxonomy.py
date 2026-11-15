"""Routing policy tests.

These are cheap assertions with real value: they fail loudly if someone
accidentally makes a fraud dispute auto-answerable, or teaches the graph to
answer a "wrong exchange rate" complaint with a live quote.
"""

from __future__ import annotations

from banking_triage.taxonomy import (
    CURRENCY_DISPUTE_INTENTS,
    CURRENCY_RELATED_INTENTS,
    FAQ_COVERED_INTENTS,
    LOOKUP_INTENTS,
    NEVER_AUTO_ANSWER_INTENTS,
    display_name,
)


def test_lookup_intents_are_currency_related() -> None:
    assert LOOKUP_INTENTS <= CURRENCY_RELATED_INTENTS


def test_lookup_and_dispute_intents_do_not_overlap() -> None:
    assert LOOKUP_INTENTS.isdisjoint(CURRENCY_DISPUTE_INTENTS)


def test_no_lookup_intent_is_escalation_only() -> None:
    assert LOOKUP_INTENTS.isdisjoint(NEVER_AUTO_ANSWER_INTENTS)


def test_all_currency_intents_have_faq_coverage() -> None:
    assert CURRENCY_RELATED_INTENTS <= FAQ_COVERED_INTENTS


def test_disputes_are_never_auto_answered() -> None:
    assert CURRENCY_DISPUTE_INTENTS <= NEVER_AUTO_ANSWER_INTENTS


def test_escalation_only_intents_may_lack_faq_coverage() -> None:
    # A handoff is an acceptable answer for these; inventing policy is not.
    uncovered = NEVER_AUTO_ANSWER_INTENTS - FAQ_COVERED_INTENTS
    assert isinstance(uncovered, frozenset)


def test_display_name_cleans_known_dirty_labels() -> None:
    assert display_name("Refund_not_showing_up") == "Refund not showing up"
    assert display_name("reverted_card_payment?") == "Reverted card payment"
    assert display_name("card_arrival") == "Card arrival"
