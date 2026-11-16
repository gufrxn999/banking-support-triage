"""Shared data contracts.

This module is the single source of truth for shapes that cross a boundary:
HTTP request/response, the LangGraph state, and the evaluation harness. If the
API and the graph disagree about a field, they disagree here first.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import MAX_TEXT_CHARS

Route = Literal["faq", "exchange_rate", "clarify", "escalate"]
Decision = Literal["auto_answer", "clarify", "escalate"]

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


class IntentScore(BaseModel):
    """One candidate intent and its calibrated probability."""

    intent: str
    score: Confidence


class Prediction(BaseModel):
    """Classifier output. Never claims more certainty than it has."""

    intent: str
    confidence: Confidence
    alternatives: list[IntentScore] = Field(default_factory=list)

    @property
    def margin(self) -> float:
        """Top-1 minus top-2. Large margins are safer to auto-route."""
        if not self.alternatives:
            return self.confidence
        return self.confidence - self.alternatives[0].score


class TokenAmount(BaseModel):
    """A currency amount as stated by the user, before it is validated."""

    currency: str = Field(min_length=1, max_length=3)
    amount: float | None = Field(default=None, gt=0)


class CurrencyQuery(BaseModel):
    """A resolved exchange-rate question ready to send to the tool."""

    base: str = Field(min_length=3, max_length=3)
    quote: str = Field(min_length=3, max_length=3)
    amount: float = Field(default=1.0, gt=0)


class ExchangeRate(BaseModel):
    """A rate observed from the Frankfurter API (ECB reference data)."""

    base: str
    quote: str
    rate: float = Field(gt=0)
    as_of: date
    source_url: str
    amount: float | None = None

    def converted(self) -> float | None:
        if self.amount is None:
            return None
        return round(self.amount * self.rate, 2)


SourceKind = Literal["faq", "exchange_rate", "handoff", "disambiguation"]


class Source(BaseModel):
    """Provenance for an answer, so a reviewer can check it was not invented."""

    kind: SourceKind
    ref: str
    detail: str | None = None


class TriageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    session_id: str | None = Field(default=None, max_length=128)


class TriageResponse(BaseModel):
    request_id: str
    intent: str
    confidence: Confidence
    alternatives: list[IntentScore] = Field(default_factory=list)
    route: Route
    decision: Decision
    route_reason: str | None = None
    answer: str
    sources: list[Source] = Field(default_factory=list)
    needs_human: bool
    model_version: str
    latency_ms: float


class ModelInfo(BaseModel):
    model_version: str
    trained_at: datetime
    trained_on: str
    n_intents: int
    vectorizer: str
    calibrated: bool | None = None
    hyperparameters: dict[str, float | str | int]
    metrics: dict[str, float]
    thresholds: dict[str, float]


class IntentEntry(BaseModel):
    id: int
    name: str
    display: str
    faq_covered: bool
    auto_answerable: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None


class ProblemDetail(BaseModel):
    """RFC 9457 problem document, used for every non-2xx response."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
    request_id: str | None = None


class TriageState(BaseModel):
    """LangGraph state. Accumulates evidence rather than overwriting it."""

    model_config = ConfigDict(extra="forbid")

    text: str
    session_id: str | None = None

    # Multi-turn bookkeeping. `pending_intent` marks a clarification the caller
    # is answering; the checkpointer restores it on the next turn of the same
    # session. `last_user_text` gives a terse reply ("the virtual one") the
    # context it needs to be classified.
    turn: int = 0
    pending_intent: str | None = None
    last_user_text: str | None = None

    prediction: Prediction | None = None
    route: Route | None = None
    decision: Decision | None = None
    # Why the router chose this route; a rule-based override and a confident
    # classifier must not look identical in an audit.
    route_reason: str | None = None

    currency_query: CurrencyQuery | None = None
    exchange_rate: ExchangeRate | None = None

    answer: str = ""
    sources: list[Source] = Field(default_factory=list)
    needs_human: bool = False

    # Populated when a node degrades instead of raising; surfaced in the
    # response so failures are visible rather than silently swallowed.
    errors: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
