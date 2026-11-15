"""Inference tests against the trained artifact.

Skipped when no artifact exists. These assert behaviour we rely on downstream,
not exact probabilities, so they survive a retrain.
"""

from __future__ import annotations

import pytest

from banking_triage.classifier import IntentClassifier, ModelNotTrainedError
from banking_triage.config import settings


@pytest.fixture(scope="module")
def model() -> IntentClassifier:
    if not settings.model_path.exists():
        pytest.skip("run `python -m banking_triage.train` first")
    return IntentClassifier.load()


def test_artifact_metadata_is_complete(model: IntentClassifier) -> None:
    assert model.n_intents == 77
    assert model.metadata["model_version"]
    assert "test_macro_f1" in model.metadata["metrics"]
    assert model.metadata["trained_on"] == "train"


def test_predictions_are_probability_distributions(model: IntentClassifier) -> None:
    proba = model.predict_proba("my card never arrived")
    assert proba.shape == (77,)
    assert proba.sum() == pytest.approx(1.0, abs=1e-6)
    assert (proba >= 0).all()


def test_predict_returns_sorted_alternatives(model: IntentClassifier) -> None:
    prediction = model.predict("I want to change my PIN", top_k=3)
    assert 0 < prediction.confidence <= 1
    scores = [prediction.confidence, *(alt.score for alt in prediction.alternatives)]
    assert scores == sorted(scores, reverse=True)


def test_known_ambiguous_pair_stays_inside_the_pair(model: IntentClassifier) -> None:
    # `card_arrival` and `card_delivery_estimate` overlap by design in
    # BANKING77: the same sentence can legitimately mean either. Asserting one
    # specific label would be testing the dataset's ambiguity, not the model.
    # What matters is that it does not wander off into an unrelated intent.
    prediction = model.predict("how long until my new card arrives?")
    assert prediction.intent in {"card_arrival", "card_delivery_estimate"}


def test_unambiguous_card_loss_query_lands_on_the_right_intent(model: IntentClassifier) -> None:
    prediction = model.predict("I lost my card, please block it before someone uses it")
    assert prediction.intent == "lost_or_stolen_card"


def test_clear_pin_change_query_lands_on_the_right_intent(model: IntentClassifier) -> None:
    prediction = model.predict("I would like to change my pin number please")
    assert prediction.intent == "change_pin"


def test_blank_input_is_rejected(model: IntentClassifier) -> None:
    with pytest.raises(ValueError):
        model.predict("   ")


def test_missing_artifact_raises_actionable_error(tmp_path) -> None:
    with pytest.raises(ModelNotTrainedError, match="train"):
        IntentClassifier.load(path=tmp_path / "nope.joblib")
