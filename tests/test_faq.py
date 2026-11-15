"""FAQ corpus and retrieval tests.

The corpus is checked against the routing taxonomy on load, so these tests also
guard against the two drifting apart as intents are added.
"""

from __future__ import annotations

import json

import pytest

from banking_triage.config import settings
from banking_triage.faq import FAQEntry, FAQStore, FAQValidationError
from banking_triage.taxonomy import CURRENCY_RELATED_INTENTS, FAQ_COVERED_INTENTS


@pytest.fixture(scope="module")
def store() -> FAQStore:
    return FAQStore.load()


def test_corpus_covers_exactly_the_routing_taxonomy(store: FAQStore) -> None:
    assert store.intents() == FAQ_COVERED_INTENTS


def test_every_entry_is_flagged_for_policy_review(store: FAQStore) -> None:
    drafts = store.draft_entries()
    assert len(drafts) == len(store), (
        "shipped FAQ copy is a neutral draft; mark entries reviewed only after "
        "replacing them with institution-approved wording"
    )


def test_every_entry_has_a_clarifying_question(store: FAQStore) -> None:
    missing = [entry.intent for entry in store.draft_entries() if not entry.clarifying_question]
    assert not missing


def test_currency_intents_all_have_answers(store: FAQStore) -> None:
    for intent in CURRENCY_RELATED_INTENTS:
        assert store.get(intent) is not None


def test_unknown_intent_returns_none(store: FAQStore) -> None:
    assert store.get("not_a_real_intent") is None
    assert store.answer("not_a_real_intent", "hello") is None


def test_retrieval_is_deterministic(store: FAQStore) -> None:
    text = "my replacement card has not arrived and tracking is stuck"
    first = store.answer("card_arrival", text)
    second = store.answer("card_arrival", text)
    assert first is not None and second is not None
    assert first[0] == second[0]


def test_variant_choice_follows_the_query() -> None:
    entry = FAQEntry(
        intent="demo",
        answers=(
            "Freeze your card in the app immediately.",
            "Card delivery takes a few days depending on your country.",
        ),
    )
    store = FAQStore({"demo": entry})
    delivery, _ = store.answer("demo", "how long does delivery take?")  # type: ignore[misc]
    freeze, _ = store.answer("demo", "someone stole my card")  # type: ignore[misc]
    assert "delivery" in delivery.lower()
    assert "freeze" in freeze.lower()


def test_empty_query_falls_back_to_the_primary_answer(store: FAQStore) -> None:
    answer, entry = store.answer("change_pin", "")  # type: ignore[misc]
    assert answer == entry.primary_answer


def _write_corpus(tmp_path, entries: list[dict]) -> None:
    path = tmp_path / "faq.jsonl"
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")
    return path


def test_corpus_missing_an_intent_is_rejected(tmp_path) -> None:
    entries = [{"intent": "card_arrival", "answers": ["x"]}]
    path = _write_corpus(tmp_path, entries)
    with pytest.raises(FAQValidationError, match="missing"):
        FAQStore.load(path)


def test_duplicate_intent_is_rejected(tmp_path) -> None:
    entries = [{"intent": intent, "answers": ["x"]} for intent in sorted(FAQ_COVERED_INTENTS)]
    entries.append({"intent": "card_arrival", "answers": ["duplicate"]})
    path = _write_corpus(tmp_path, entries)
    with pytest.raises(FAQValidationError, match="duplicate"):
        FAQStore.load(path)


def test_entry_without_answers_is_rejected(tmp_path) -> None:
    entries = [{"intent": intent, "answers": ["x"]} for intent in sorted(FAQ_COVERED_INTENTS)]
    entries[0]["answers"] = []
    path = _write_corpus(tmp_path, entries)
    with pytest.raises(FAQValidationError, match="at least one answer"):
        FAQStore.load(path)


def test_invalid_json_is_rejected(tmp_path) -> None:
    path = tmp_path / "faq.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(FAQValidationError, match="invalid JSON"):
        FAQStore.load(path)


def test_missing_corpus_is_reported_clearly(tmp_path) -> None:
    with pytest.raises(FAQValidationError, match="missing FAQ corpus"):
        FAQStore.load(tmp_path / "absent.jsonl")


def test_shipped_corpus_path_matches_config() -> None:
    assert settings.faq_path.name == "faq.jsonl"
