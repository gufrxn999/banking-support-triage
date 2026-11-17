"""Graph routing, degradation, and multi-turn tests.

Every dependency is stubbed, so the suite is offline and deterministic. The
point of these tests is the decision boundary: which route a prediction earns,
and what the graph does when a tool or the content layer fails.
"""

from __future__ import annotations

from datetime import date

import pytest

from banking_triage.faq import FAQStore
from banking_triage.graph import TriageGraph
from banking_triage.schemas import ExchangeRate, IntentScore, Prediction
from banking_triage.tools.exchange import ExchangeRateUnavailable, UnsupportedCurrencyError


def prediction(intent: str, confidence: float, *alternatives: tuple[str, float]) -> Prediction:
    return Prediction(
        intent=intent,
        confidence=confidence,
        alternatives=[IntentScore(intent=name, score=score) for name, score in alternatives],
    )


class StubClassifier:
    """Returns a scripted prediction per text, and records what it was asked."""

    metadata: dict = {"thresholds": {}}

    def __init__(self, script, metadata: dict | None = None) -> None:
        self.script = script
        self.calls: list[str] = []
        if metadata is not None:
            self.metadata = metadata

    def predict(self, text: str, top_k: int = 3) -> Prediction:
        self.calls.append(text)
        result = self.script(text)
        if isinstance(result, Prediction):
            return result
        return prediction(result, 0.95)


class StubExchange:
    def __init__(self, rate: ExchangeRate | None = None, error: Exception | None = None) -> None:
        self.rate = rate
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def latest(self, base: str, quote: str) -> ExchangeRate:
        self.calls.append((base, quote))
        if self.error is not None:
            raise self.error
        if self.rate is not None:
            return self.rate
        # Echo the requested pair, like the real client does.
        return ExchangeRate(
            base=base,
            quote=quote,
            rate=0.90481,
            as_of=date(2026, 11, 13),
            source_url=f"https://api.frankfurter.dev/v1/latest?base={base}&symbols={quote}",
        )


@pytest.fixture(scope="module")
def faq() -> FAQStore:
    return FAQStore.load()


def build(script, faq: FAQStore, exchange=None, metadata: dict | None = None) -> TriageGraph:
    classifier = StubClassifier(script, metadata=metadata)
    return TriageGraph(classifier, faq, exchange or StubExchange())


# --- route coverage -------------------------------------------------------


def test_confident_covered_intent_is_answered_from_the_faq(faq: FAQStore) -> None:
    graph = build(lambda text: prediction("card_not_working", 0.91, ("card_arrival", 0.02)), faq)
    state = graph.invoke("my card keeps getting declined")

    assert state.route == "faq"
    assert state.decision == "auto_answer"
    assert state.needs_human is False
    assert state.sources[0].kind == "faq"
    assert state.sources[0].ref == "card_not_working"
    entry = faq.get("card_not_working")
    assert entry is not None
    assert state.answer in entry.answers
    assert state.steps == ["triage", "faq", "compose"]


def test_exchange_intent_with_a_pair_calls_the_tool(faq: FAQStore) -> None:
    exchange = StubExchange()
    graph = build(
        lambda text: prediction("exchange_rate", 0.96, ("exchange_charge", 0.01)), faq, exchange
    )

    state = graph.invoke("how much is 500 dollars in euros?")

    assert exchange.calls == [("USD", "EUR")]
    assert state.route == "exchange_rate"
    assert state.decision == "auto_answer"
    assert state.route_reason == "explicit_currency_pair"
    assert "500.00 USD = 452.41 EUR" in state.answer
    assert "2026-11-13" in state.answer
    assert "today" not in state.answer.lower()
    assert state.sources[0].kind == "exchange_rate"
    assert state.sources[0].ref == "https://api.frankfurter.dev/v1/latest?base=USD&symbols=EUR"


def test_explicit_pair_overrides_a_confidently_wrong_classifier(faq: FAQStore) -> None:
    # The real model puts "convert 250 EUR to CHF" on `fiat_currency_support` at
    # 0.911: it has never seen a slot-filled conversion request. The pair wins.
    exchange = StubExchange()
    graph = build(lambda text: prediction("fiat_currency_support", 0.91), faq, exchange)

    state = graph.invoke("convert 250 EUR to CHF")

    assert exchange.calls == [("EUR", "CHF")]
    assert state.route == "exchange_rate"
    assert state.route_reason == "explicit_currency_pair"


def test_complaint_with_a_pair_is_not_turned_into_a_quote(faq: FAQStore) -> None:
    exchange = StubExchange()
    graph = build(lambda text: prediction("card_payment_wrong_exchange_rate", 0.78), faq, exchange)

    state = graph.invoke("I was charged the wrong rate on a 500 USD to EUR payment")

    assert exchange.calls == []
    assert state.route == "escalate"
    assert state.route_reason == "protected_intent"


def test_a_weak_protected_prediction_does_not_block_an_explicit_quote(faq: FAQStore) -> None:
    # Regression: a smaller C put this text on `cash_withdrawal_not_recognised`
    # at 0.056. A weak guess must not turn a slot-filled conversion request into
    # a clarification; the pair and amount are the request.
    exchange = StubExchange()
    graph = build(
        lambda text: prediction(
            "cash_withdrawal_not_recognised", 0.056, ("fiat_currency_support", 0.052)
        ),
        faq,
        exchange,
    )

    state = graph.invoke("how much is 500 dollars in euros?")

    assert exchange.calls == [("USD", "EUR")]
    assert state.route == "exchange_rate"
    assert state.route_reason == "explicit_currency_pair"


def test_a_confident_protected_prediction_still_blocks_an_explicit_pair(faq: FAQStore) -> None:
    # The converse guard: if the classifier is sure this is fraud or a dispute,
    # a pair in the text does not turn the turn into a rate quote.
    exchange = StubExchange()
    graph = build(lambda text: prediction("compromised_card", 0.78), faq, exchange)

    state = graph.invoke("convert 500 USD to EUR")

    assert exchange.calls == []
    assert state.route == "escalate"
    assert state.route_reason == "protected_intent"


def test_app_how_to_question_with_a_pair_keeps_its_faq_route(faq: FAQStore) -> None:
    exchange = StubExchange()
    graph = build(lambda text: prediction("exchange_via_app", 0.60), faq, exchange)

    state = graph.invoke("I want to exchange USD to EUR in the app")

    assert exchange.calls == []
    assert state.route == "faq"
    assert state.route_reason == "classifier"


def test_low_confidence_asks_instead_of_answering(faq: FAQStore) -> None:
    graph = build(
        lambda text: prediction(
            "card_arrival", 0.31, ("card_not_working", 0.28), ("card_delivery_estimate", 0.2)
        ),
        faq,
    )
    state = graph.invoke("something is wrong with the card thing")

    assert state.route == "clarify"
    assert state.decision == "clarify"
    assert state.route_reason == "low_confidence"
    assert state.needs_human is False
    assert state.sources[0].kind == "disambiguation"
    # Offers the plausible readings and asserts none of them as the answer.
    assert "Card not working" in state.answer
    assert "Card delivery estimate" in state.answer


def test_clarifying_question_omits_noise_candidates(faq: FAQStore) -> None:
    graph = build(
        lambda text: prediction(
            "card_arrival", 0.45, ("card_not_working", 0.05), ("card_pin_blocked", 0.01)
        ),
        faq,
    )
    state = graph.invoke("where is my card")

    assert "Card arrival" in state.answer or "card" in state.answer.lower()
    assert "Card pin blocked" not in state.answer


def test_fraud_intent_escalates_even_with_high_confidence(faq: FAQStore) -> None:
    graph = build(lambda text: prediction("lost_or_stolen_card", 0.99), faq)
    state = graph.invoke("I lost my card, block it now")

    assert state.route == "escalate"
    assert state.decision == "escalate"
    assert state.needs_human is True
    assert state.sources[0].kind == "handoff"


def test_low_confidence_fraud_intent_still_asks_first(faq: FAQStore) -> None:
    # Below the escalate threshold we do not even claim the intent; asking is safe.
    graph = build(lambda text: prediction("lost_or_stolen_card", 0.20), faq)
    state = graph.invoke("hmm")

    assert state.route == "clarify"
    assert state.needs_human is False


def test_confident_intent_without_curated_copy_hands_off(faq: FAQStore) -> None:
    graph = build(lambda text: prediction("apple_pay_or_google_pay", 0.93), faq)
    state = graph.invoke("can I use apple pay")

    assert state.route == "escalate"
    assert state.needs_human is True
    assert "human agent" in state.answer
    assert state.sources[0].detail == "no curated answer"


def test_artifact_thresholds_override_settings(faq: FAQStore) -> None:
    # 0.90 would auto-route under the shipped defaults; this artifact demands 0.99.
    metadata = {
        "thresholds": {"auto_route_threshold": 0.99, "min_margin": 0.0, "escalate_threshold": 0.1}
    }
    graph = build(lambda text: prediction("card_not_working", 0.90), faq, metadata=metadata)

    assert graph.invoke("my card is declined").route == "clarify"


def test_thin_margin_blocks_auto_routing(faq: FAQStore) -> None:
    graph = build(lambda text: prediction("top_up_failed", 0.90, ("top_up_reverted", 0.85)), faq)
    state = graph.invoke("my top up did not arrive")

    assert state.route == "clarify"


def test_whitespace_only_input_does_not_reach_the_classifier(faq: FAQStore) -> None:
    graph = build(lambda text: prediction("card_not_working", 0.95), faq)
    state = graph.invoke("   ")

    assert state.route == "clarify"
    assert state.answer
    assert graph.classifier.calls == []


# --- tool degradation ----------------------------------------------------


def test_unsupported_pair_asks_for_another_instead_of_failing(faq: FAQStore) -> None:
    # TRY is a valid ISO code the ECB reference set does not publish.
    exchange = StubExchange(error=UnsupportedCurrencyError("USD", "TRY"))
    graph = build(lambda text: prediction("exchange_rate", 0.97), faq, exchange)

    state = graph.invoke("500 USD to TRY")

    assert state.route == "clarify"
    assert state.needs_human is False
    assert state.pending_intent == "exchange_rate"
    assert state.errors and "exchange" in state.errors[0]
    assert state.answer  # a usable reply, not an exception


def test_rate_service_outage_hands_off_and_says_so(faq: FAQStore) -> None:
    exchange = StubExchange(error=ExchangeRateUnavailable("upstream returned 503"))
    graph = build(lambda text: prediction("exchange_rate", 0.97), faq, exchange)

    state = graph.invoke("500 USD to EUR")

    assert state.route == "escalate"
    assert state.needs_human is True
    assert state.sources[0].detail == "rate service unavailable"
    assert "won't guess" in state.answer
    assert state.errors


def test_missing_faq_entry_degrades_to_handoff(faq: FAQStore) -> None:
    # Simulates the corpus drifting from taxonomy.py after the route was chosen.
    graph = TriageGraph(
        StubClassifier(lambda text: prediction("card_not_working", 0.95)),
        FAQStore({}),
        StubExchange(),
    )
    state = graph.invoke("my card is declined")

    assert state.route == "escalate"
    assert state.decision == "escalate"
    assert state.needs_human is True
    assert any("faq: no entry" in error for error in state.errors)


def test_close_releases_a_client_that_owns_connections(faq: FAQStore) -> None:
    class ClosableExchange(StubExchange):
        closed = False

        def close(self) -> None:
            self.closed = True

    exchange = ClosableExchange()
    graph = build(lambda text: prediction("exchange_rate", 0.96), faq, exchange)
    graph.close()

    assert exchange.closed is True


# --- multi-turn clarification -------------------------------------------


def test_exchange_intent_without_pair_asks_a_question(faq: FAQStore) -> None:
    exchange = StubExchange()
    graph = build(lambda text: prediction("exchange_rate", 0.96), faq, exchange)

    state = graph.invoke("what is the exchange rate?", session_id="clarify-1")

    assert state.route == "clarify"
    assert state.pending_intent == "exchange_rate"
    assert state.route_reason == "currency_pair_missing"
    assert "two currencies" in state.answer
    assert exchange.calls == []


def test_follow_up_turn_resumes_without_reclassifying(faq: FAQStore) -> None:
    exchange = StubExchange()
    classifier_calls: list[str] = []

    def script(text: str) -> Prediction:
        classifier_calls.append(text)
        return prediction("exchange_rate", 0.96)

    graph = build(script, faq, exchange)
    first = graph.invoke("what is the exchange rate?", session_id="resume-1")
    second = graph.invoke("500 USD to EUR", session_id="resume-1")

    assert first.route == "clarify"
    assert second.route == "exchange_rate"
    assert second.route_reason == "follow_up_currency_pair"
    assert second.pending_intent is None
    assert second.turn == 2
    assert "452.41 EUR" in second.answer
    # The currency reply is data for the pending question, not a new utterance.
    assert classifier_calls == ["what is the exchange rate?"]


def test_one_currency_reply_keeps_asking_until_a_pair_is_given(faq: FAQStore) -> None:
    graph = build(lambda text: prediction("exchange_rate", 0.96), faq, StubExchange())

    graph.invoke("I need a rate", session_id="three-1")
    middle = graph.invoke("EUR", session_id="three-1")
    final = graph.invoke("250 GBP to EUR", session_id="three-1")

    assert middle.route == "clarify"
    assert middle.pending_intent == "exchange_rate"
    assert "I found EUR" in middle.answer
    assert final.route == "exchange_rate"
    assert "250.00 GBP" in final.answer


def test_topic_change_after_clarification_is_reclassified(faq: FAQStore) -> None:
    def script(text: str) -> Prediction:
        if "exchange" in text.lower():
            return prediction("exchange_rate", 0.96)
        return prediction("card_not_working", 0.93)

    graph = build(script, faq, StubExchange())
    graph.invoke("what is the exchange rate?", session_id="switch-1")
    state = graph.invoke("forget it, my card is declining everywhere", session_id="switch-1")

    assert state.route == "faq"
    assert state.pending_intent is None
    assert state.prediction.intent == "card_not_working"


def test_terse_reply_is_classified_with_the_previous_message(faq: FAQStore) -> None:
    classifier_calls: list[str] = []

    def script(text: str) -> Prediction:
        classifier_calls.append(text)
        if "virtual" in text.lower():
            return prediction("virtual_card_not_working", 0.95)
        return prediction("card_not_working", 0.42, ("virtual_card_not_working", 0.30))

    graph = build(script, faq, StubExchange())
    first = graph.invoke("my card is not working", session_id="context-1")
    second = graph.invoke("the virtual one", session_id="context-1")

    assert first.route == "clarify"
    assert second.route == "faq"
    assert classifier_calls[-1] == "my card is not working the virtual one"


def test_sessions_do_not_share_state(faq: FAQStore) -> None:
    graph = build(lambda text: prediction("exchange_rate", 0.96), faq, StubExchange())

    graph.invoke("what is the exchange rate?", session_id="session-a")
    other = graph.invoke("what is the exchange rate?", session_id="session-b")

    assert other.turn == 1
    assert other.pending_intent == "exchange_rate"


def test_anonymous_turns_are_standalone(faq: FAQStore) -> None:
    graph = build(lambda text: prediction("exchange_rate", 0.96), faq, StubExchange())

    first = graph.invoke("what is the exchange rate?")
    second = graph.invoke("what is the exchange rate?")

    assert first.turn == second.turn == 1
    assert second.route == "clarify"
