"""Curated FAQ retrieval.

Deliberately not embeddings and not an LLM: this is a small, reviewed corpus
keyed by predicted intent. The same question always produces the same answer,
which is what makes the evaluation meaningful and the answers auditable.

Every entry carries `review_status`. The shipped copy is `draft`: it explains
what to check and what happens next without asserting bank-specific timelines,
fees, or limits. Replacing it with an institution's approved wording is a
content change, not a code change.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import settings as default_settings
from .taxonomy import FAQ_COVERED_INTENTS

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "my",
        "not",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "why",
        "will",
        "with",
        "you",
        "your",
    }
)


class FAQValidationError(ValueError):
    """The FAQ corpus does not match the routing taxonomy."""


@dataclass(frozen=True)
class FAQEntry:
    intent: str
    answers: tuple[str, ...]
    clarifying_question: str | None = None
    review_status: str = "draft"

    @property
    def primary_answer(self) -> str:
        return self.answers[0]


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 2
    }


class FAQStore:
    def __init__(self, entries: dict[str, FAQEntry]) -> None:
        self._entries = entries

    @classmethod
    def load(cls, path: Path | None = None) -> FAQStore:
        source = path or default_settings.faq_path
        if not source.exists():
            raise FAQValidationError(f"missing FAQ corpus at {source}")

        entries: dict[str, FAQEntry] = {}
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FAQValidationError(f"{source}:{line_number}: invalid JSON ({exc})") from exc

            intent = payload.get("intent")
            answers = payload.get("answers") or []
            if not intent or not isinstance(answers, list) or not answers:
                raise FAQValidationError(
                    f"{source}:{line_number}: entry needs an intent and at least one answer"
                )
            if intent in entries:
                raise FAQValidationError(f"{source}:{line_number}: duplicate intent {intent!r}")

            entries[intent] = FAQEntry(
                intent=intent,
                answers=tuple(str(answer).strip() for answer in answers),
                clarifying_question=payload.get("clarifying_question"),
                review_status=str(payload.get("review_status", "draft")),
            )

        missing = FAQ_COVERED_INTENTS - entries.keys()
        extra = entries.keys() - FAQ_COVERED_INTENTS
        if missing or extra:
            raise FAQValidationError(
                "FAQ corpus and routing taxonomy disagree: "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
        return cls(entries)

    def __len__(self) -> int:
        return len(self._entries)

    def intents(self) -> frozenset[str]:
        return frozenset(self._entries)

    def get(self, intent: str) -> FAQEntry | None:
        return self._entries.get(intent)

    def answer(self, intent: str, text: str = "") -> tuple[str, FAQEntry] | None:
        """Pick the best-matching variant for the query, deterministically.

        Variant choice is lexical overlap, ties broken by original order, so the
        same input always yields the same output -- a requirement for honest
        latency and routing evaluation.
        """
        entry = self._entries.get(intent)
        if entry is None:
            return None
        if len(entry.answers) == 1 or not text:
            return entry.primary_answer, entry

        query = _tokens(text)
        best_index, best_score = 0, -1.0
        for index, answer in enumerate(entry.answers):
            candidate = _tokens(answer)
            union = query | candidate
            score = len(query & candidate) / len(union) if union else 0.0
            if score > best_score:
                best_index, best_score = index, score
        return entry.answers[best_index], entry

    def draft_entries(self) -> list[FAQEntry]:
        return [entry for entry in self._entries.values() if entry.review_status == "draft"]
