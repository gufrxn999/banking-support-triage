"""LangGraph workflow: classify, decide, act, compose.

One job per node, and every node returns a partial state update so the graph
accumulates evidence instead of overwriting it. The routing decision itself is
made in `_triage` and is the only place that interprets thresholds -- the
downstream nodes execute a decision, they do not second-guess it.

Failure policy is the same everywhere: a tool or content failure degrades into a
visible answer (clarify or hand off) and appends to `state.errors`. Nothing in
this module raises at request time except a missing classifier, which is a
deployment error.

Multi-turn behaviour uses LangGraph's checkpointer. A session's state is keyed by
`session_id`; a second turn can therefore see that we asked for a currency pair
and skip re-classifying an answer like "500 USD to EUR".
"""

from __future__ import annotations

from typing import NamedTuple, Protocol
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .classifier import IntentClassifier
from .config import Settings
from .config import settings as default_settings
from .currency import build_query, describe_mentions, explicit_conversion_request
from .faq import FAQStore
from .schemas import (
    CurrencyQuery,
    Decision,
    ExchangeRate,
    Prediction,
    Route,
    Source,
    TriageState,
)
from .taxonomy import (
    CURRENCY_DISPUTE_INTENTS,
    FAQ_COVERED_INTENTS,
    LOOKUP_INTENTS,
    NEVER_AUTO_ANSWER_INTENTS,
    display_name,
)
from .tools.exchange import (
    ExchangeRateUnavailable,
    FrankfurterClient,
    UnsupportedCurrencyError,
)

# Alternatives below this score are noise, not a plausible reading of the
# message, so a clarifying question does not offer them.
CLARIFY_OPTION_MIN_SCORE = 0.10

# A reply this short is treated as an answer to the previous clarifying
# question. Anything longer is a new message and is classified on its own.
MAX_CONTEXT_REPLY_WORDS = 6

GENERIC_ESCALATION = (
    "I don't have a vetted answer for this one, so I've passed it to a human "
    "agent who can look at the details."
)


class ExchangeRateTool(Protocol):
    """The slice of the Frankfurter client the graph depends on."""

    def latest(self, base: str, quote: str) -> ExchangeRate: ...


class _Choice(NamedTuple):
    """The router's whole decision, including why it was made."""

    route: Route
    decision: Decision
    reason: str
    pending: str | None = None
    query: CurrencyQuery | None = None


def _default_checkpointer() -> InMemorySaver:
    """In-memory checkpointer that is allowed to persist our state models.

    LangGraph's msgpack serde requires types to be registered explicitly before
    it will deserialize them. Naming them here keeps future strict mode working
    without allowing arbitrary classes through.
    """
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("banking_triage.schemas", "CurrencyQuery"),
            ("banking_triage.schemas", "ExchangeRate"),
            ("banking_triage.schemas", "IntentScore"),
            ("banking_triage.schemas", "Prediction"),
            ("banking_triage.schemas", "Source"),
        ]
    )
    return InMemorySaver(serde=serde)


class TriageGraph:
    """Compiled triage workflow with a per-session checkpointer."""

    def __init__(
        self,
        classifier: IntentClassifier,
        faq: FAQStore,
        exchange: ExchangeRateTool,
        settings: Settings | None = None,
        checkpointer: InMemorySaver | None = None,
    ) -> None:
        self.classifier = classifier
        self.faq = faq
        self.exchange = exchange
        self.settings = settings or default_settings
        self._checkpointer = checkpointer or _default_checkpointer()
        self._graph = self._build()

    # --- construction ---------------------------------------------------
    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> TriageGraph:
        """Load every dependency from disk/network config. Used by the API."""
        settings = settings or default_settings

        return cls(
            classifier=IntentClassifier.load(settings=settings),
            faq=FAQStore.load(settings.faq_path),
            exchange=FrankfurterClient(),
            settings=settings,
        )

    def _build(self) -> CompiledStateGraph:
        builder = StateGraph(TriageState)
        builder.add_node("triage", self._triage)
        builder.add_node("clarify", self._clarify)
        builder.add_node("exchange", self._exchange)
        builder.add_node("faq", self._faq)
        builder.add_node("escalate", self._escalate)
        builder.add_node("compose", self._compose)

        builder.set_entry_point("triage")
        builder.add_conditional_edges(
            "triage",
            _route_of,
            {
                "clarify": "clarify",
                "exchange_rate": "exchange",
                "faq": "faq",
                "escalate": "escalate",
            },
        )
        for node in ("clarify", "exchange", "faq", "escalate"):
            builder.add_edge(node, "compose")
        builder.add_edge("compose", END)
        return builder.compile(checkpointer=self._checkpointer)

    # --- public entry point ---------------------------------------------
    def close(self) -> None:
        """Release the exchange client if it owns a connection pool."""
        close = getattr(self.exchange, "close", None)
        if callable(close):
            close()

    def invoke(self, text: str, session_id: str | None = None) -> TriageState:
        """Run one turn.

        Without a `session_id` the turn is treated as standalone and its
        checkpoint is discarded afterwards, so anonymous traffic cannot
        accumulate in memory.
        """
        thread_id = session_id or f"anon-{uuid4().hex}"
        config = {"configurable": {"thread_id": thread_id}}
        try:
            result = self._graph.invoke({"text": text}, config)
        finally:
            if session_id is None:
                self._checkpointer.delete_thread(thread_id)
        return result if isinstance(result, TriageState) else TriageState.model_validate(result)

    # --- nodes ----------------------------------------------------------
    def _triage(self, state: TriageState) -> dict:
        """Classify, then commit to exactly one route."""
        steps = [*state.steps, "triage"]
        reset = {
            "turn": state.turn + 1,
            "last_user_text": state.text,
            "pending_intent": None,
            "prediction": None,
            "route": None,
            "decision": None,
            "route_reason": None,
            "currency_query": None,
            "exchange_rate": None,
            "answer": "",
            "sources": [],
            "needs_human": False,
            "errors": [],
        }

        # Whitespace-only input is not a classification problem; do not let the
        # vectorizer raise a 500 at the edge.
        if not state.text.strip():
            return {
                **reset,
                "route": "clarify",
                "decision": "clarify",
                "route_reason": "empty_text",
                "answer": "Could you describe what you need help with?",
                "sources": [Source(kind="disambiguation", ref="empty", detail="no usable text")],
                "steps": steps,
            }

        resumed = self._resume_exchange(state)
        if resumed is not None:
            return {**reset, **resumed, "steps": steps}

        prediction = self.classifier.predict(
            self._classification_text(state), top_k=self.settings.top_k
        )
        choice = self._decide(prediction, state.text, self._thresholds())

        return {
            **reset,
            "prediction": prediction,
            "route": choice.route,
            "decision": choice.decision,
            "route_reason": choice.reason,
            "pending_intent": choice.pending,
            "currency_query": choice.query,
            "steps": steps,
        }

    def _resume_exchange(self, state: TriageState) -> dict | None:
        """Handle a reply to "which two currencies?" without re-classifying it.

        A classifier would read "500 USD to EUR" as an unrelated intent, so the
        pending flag is what makes the second turn work. If the reply changes
        topic, return None and let the normal path classify it.
        """
        if state.pending_intent not in LOOKUP_INTENTS:
            return None

        text = state.text
        query = build_query(text)
        if query is not None:
            return {
                "prediction": state.prediction,
                "route": "exchange_rate",
                "decision": "auto_answer",
                "route_reason": "follow_up_currency_pair",
                "currency_query": query,
            }

        mentions = describe_mentions(text)
        looks_like_an_answer = bool(mentions) or len(text.split()) <= MAX_CONTEXT_REPLY_WORDS
        if looks_like_an_answer:
            # Still missing one side of the pair: ask again, keeping the intent
            # pending so the next reply is read as currency information too.
            return {
                "prediction": state.prediction,
                "route": "clarify",
                "decision": "clarify",
                "route_reason": "currency_pair_missing",
                "pending_intent": state.pending_intent,
            }
        return None

    def _classification_text(self, state: TriageState) -> str:
        """Give a terse reply to a clarifying question the context it lacks."""
        prior = state.last_user_text
        if state.pending_intent and prior and len(state.text.split()) <= MAX_CONTEXT_REPLY_WORDS:
            return f"{prior} {state.text}"
        return state.text

    def _decide(
        self,
        prediction: Prediction,
        text: str,
        thresholds: dict[str, float],
    ) -> _Choice:
        """The decision policy. Mirrors the M2 measurement, plus one explicit rule.

        The extra rule: an explicit conversion request ("convert 250 EUR to CHF")
        goes to the tool even when the classifier is confident about a different
        intent, because BANKING77 has no slot-filled rate questions in it. The
        cue lists in `currency.py` keep complaints and app-how-to questions out.
        """
        explicit = explicit_conversion_request(text)
        if (
            explicit is not None
            and prediction.intent not in NEVER_AUTO_ANSWER_INTENTS
            and prediction.intent not in CURRENCY_DISPUTE_INTENTS
        ):
            return _Choice("exchange_rate", "auto_answer", "explicit_currency_pair", None, explicit)

        confident = (
            prediction.confidence >= thresholds["auto_route_threshold"]
            and prediction.margin >= thresholds["min_margin"]
        )

        # We do not know what the message is about: ask, never assert.
        if prediction.confidence < thresholds["escalate_threshold"]:
            return _Choice("clarify", "clarify", "low_confidence", prediction.intent)

        # Fraud and dispute reports always reach a person, however confident.
        if prediction.intent in NEVER_AUTO_ANSWER_INTENTS:
            return _Choice("escalate", "escalate", "protected_intent")

        if prediction.intent in LOOKUP_INTENTS:
            query = build_query(text)
            if query is None:
                return _Choice("clarify", "clarify", "currency_pair_missing", prediction.intent)
            return _Choice("exchange_rate", "auto_answer", "classifier", None, query)

        if confident and prediction.intent in FAQ_COVERED_INTENTS:
            return _Choice("faq", "auto_answer", "classifier")

        if confident:
            # No curated answer exists for this intent; do not improvise one.
            return _Choice("escalate", "escalate", "no_curated_answer")

        # Reasonably confident but not enough to answer (thin margin, or below
        # the auto-route threshold): surface the candidates and let the user pick.
        return _Choice("clarify", "clarify", "ambiguous", prediction.intent)

    def _clarify(self, state: TriageState) -> dict:
        if state.pending_intent in LOOKUP_INTENTS:
            answer = self._currency_clarification(state.text)
            ref, detail = "exchange_rate", "currency pair incomplete"
        elif state.prediction is not None:
            answer = self._intent_clarification(state.prediction)
            ref, detail = state.prediction.intent, "no answer asserted"
        else:  # pragma: no cover - _triage always sets a prediction first
            answer = "Could you rephrase that with a little more detail?"
            ref, detail = "unknown", "no prediction available"

        return {
            "answer": answer,
            "sources": [Source(kind="disambiguation", ref=ref, detail=detail)],
            "steps": [*state.steps, "clarify"],
        }

    def _currency_clarification(self, text: str) -> str:
        found = describe_mentions(text)
        known = ", ".join(dict.fromkeys(found))
        if known:
            return (
                f"I found {known}, but I need two currencies to quote a rate. "
                "Which currency should I convert it to, and how much?"
            )
        return (
            "I can look up an ECB reference rate. Which two currencies, and how much? "
            "For example: 500 USD to EUR."
        )

    def _intent_clarification(self, prediction: Prediction) -> str:
        entry = self.faq.get(prediction.intent)
        if entry and entry.clarifying_question:
            prompt = f"I want to make sure I answer the right thing. {entry.clarifying_question}"
        else:
            prompt = "I'm not confident I've understood this one."

        options = [
            display_name(alt.intent)
            for alt in prediction.alternatives
            if alt.score >= CLARIFY_OPTION_MIN_SCORE
        ]
        if options:
            listed = " or ".join(options)
            prompt = f"{prompt} If not, is it about {listed}?"
        return prompt

    def _faq(self, state: TriageState) -> dict:
        intent = state.prediction.intent if state.prediction else ""
        result = self.faq.answer(intent, state.text)
        if result is None:
            # Coverage drifted from taxonomy.py; degrade rather than invent copy.
            return {
                "route": "escalate",
                "decision": "escalate",
                "route_reason": "no_curated_answer",
                "answer": GENERIC_ESCALATION,
                "needs_human": True,
                "sources": [
                    Source(kind="handoff", ref=intent or "unknown", detail="no curated answer")
                ],
                "errors": [*state.errors, f"faq: no entry for {intent!r}"],
                "steps": [*state.steps, "faq"],
            }

        answer, entry = result
        return {
            "answer": answer,
            "sources": [Source(kind="faq", ref=entry.intent, detail="curated answer")],
            "steps": [*state.steps, "faq"],
        }

    def _exchange(self, state: TriageState) -> dict:
        query = state.currency_query
        steps = [*state.steps, "exchange"]
        if query is None:  # pragma: no cover - routing guarantees a query
            return {
                "route": "clarify",
                "decision": "clarify",
                "route_reason": "currency_pair_missing",
                "pending_intent": "exchange_rate",
                "answer": self._currency_clarification(state.text),
                "errors": [*state.errors, "exchange: missing currency query"],
                "steps": steps,
            }

        try:
            rate = self.exchange.latest(query.base, query.quote)
        except UnsupportedCurrencyError as exc:
            message = (
                f"I can't get a reference rate for {query.base} to {query.quote}. "
                "Try a different pair, or ask a human agent about a rate you were charged."
            )
            return {
                "route": "clarify",
                "decision": "clarify",
                "route_reason": "unsupported_pair",
                "pending_intent": "exchange_rate",
                "answer": message,
                "sources": [
                    Source(
                        kind="exchange_rate",
                        ref=f"{query.base}->{query.quote}",
                        detail="pair not served",
                    )
                ],
                "errors": [*state.errors, f"exchange: {exc}"],
                "steps": steps,
            }
        except ExchangeRateUnavailable as exc:
            return {
                "route": "escalate",
                "decision": "escalate",
                "route_reason": "tool_unavailable",
                "answer": (
                    "The exchange-rate service is unavailable right now, so I won't guess "
                    "a rate. I've passed this to a human agent."
                ),
                "needs_human": True,
                "sources": [
                    Source(kind="handoff", ref="frankfurter", detail="rate service unavailable")
                ],
                "errors": [*state.errors, f"exchange: {exc}"],
                "steps": steps,
            }

        rate = rate.model_copy(update={"amount": query.amount})
        converted = rate.converted()
        answer = (
            f"{query.amount:,.2f} {rate.base} = {converted:,.2f} {rate.quote} "
            f"at the ECB reference rate for {rate.as_of.isoformat()} "
            f"(1 {rate.base} = {rate.rate:.4f} {rate.quote}). "
            "Reference rates are published on business days."
        )
        return {
            "exchange_rate": rate,
            "answer": answer,
            "sources": [
                Source(
                    kind="exchange_rate",
                    ref=rate.source_url,
                    detail=f"ECB reference rate as of {rate.as_of.isoformat()}",
                )
            ],
            "steps": steps,
        }

    def _escalate(self, state: TriageState) -> dict:
        intent = state.prediction.intent if state.prediction else ""
        result = self.faq.answer(intent, state.text)
        if result is not None:
            guidance, entry = result
            answer = f"{guidance}\n\nThis has been flagged for a human agent."
            source = Source(kind="handoff", ref=entry.intent, detail="curated guidance + handoff")
        else:
            answer = GENERIC_ESCALATION
            source = Source(kind="handoff", ref=intent or "unknown", detail="no curated answer")

        return {
            "answer": answer,
            "needs_human": True,
            "sources": [source],
            "steps": [*state.steps, "escalate"],
        }

    def _compose(self, state: TriageState) -> dict:
        """Final invariant check: never return an empty or unrouted response."""
        steps = [*state.steps, "compose"]
        if not state.answer.strip():
            return {
                "route": "escalate",
                "decision": "escalate",
                "route_reason": "fallback",
                "answer": GENERIC_ESCALATION,
                "needs_human": True,
                "errors": [*state.errors, "compose: empty answer"],
                "steps": steps,
            }
        if state.route is None or state.decision is None:  # pragma: no cover - defensive
            return {
                "route": "escalate",
                "decision": "escalate",
                "route_reason": "fallback",
                "needs_human": True,
                "errors": [*state.errors, "compose: missing route or decision"],
                "steps": steps,
            }
        return {"steps": steps}

    # --- helpers --------------------------------------------------------
    def _thresholds(self) -> dict[str, float]:
        """Artifact thresholds win over defaults, so serving matches M2."""
        raw = self.classifier.metadata.get("thresholds") or {}
        defaults = {
            "auto_route_threshold": self.settings.auto_route_threshold,
            "min_margin": self.settings.min_margin,
            "escalate_threshold": self.settings.escalate_threshold,
        }
        resolved: dict[str, float] = {}
        for key, fallback in defaults.items():
            try:
                resolved[key] = float(raw.get(key, fallback))
            except (TypeError, ValueError):
                resolved[key] = fallback
        return resolved


def _route_of(state: TriageState) -> str:
    return state.route or "escalate"
