"""HTTP contract tests.

Services are injected, so no artifact, FAQ file, or network is touched. The
tests check the adapter, not the routing policy (that is `test_graph.py`).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from banking_triage.api import Services, create_app
from banking_triage.classifier import IntentClassifier
from banking_triage.config import settings
from banking_triage.schemas import IntentScore, Prediction, Source, TriageState

METADATA = {
    "model_version": "v2-test",
    "trained_at": "2026-11-15T10:30:00+00:00",
    "trained_on": "train+val",
    "vectorizer": "tfidf_word_unigram",
    "hyperparameters": {"C": 16, "max_iter": 2000, "solver": "lbfgs"},
    "metrics": {"test_macro_f1": 0.8992},
    "thresholds": {
        "auto_route_threshold": 0.54,
        "min_margin": 0.10,
        "escalate_threshold": 0.40,
        "calibrated": False,
    },
}


class StubClassifier:
    label_names = ["card_arrival", "lost_or_stolen_card", "reverted_card_payment?"]
    metadata = METADATA

    @property
    def model_version(self) -> str:
        return str(self.metadata["model_version"])


class StubGraph:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.closed = False

    def invoke(self, text: str, session_id: str | None = None) -> TriageState:
        self.calls.append((text, session_id))
        return TriageState(
            text=text,
            session_id=session_id,
            turn=1,
            prediction=Prediction(
                intent="card_arrival",
                confidence=0.91,
                alternatives=[IntentScore(intent="card_delivery_estimate", score=0.04)],
            ),
            route="faq",
            decision="auto_answer",
            route_reason="classifier",
            answer="Card orders can be tracked in the app under Cards.",
            sources=[Source(kind="faq", ref="card_arrival", detail="curated answer")],
            needs_human=False,
            steps=["triage", "faq", "compose"],
        )

    def close(self) -> None:
        self.closed = True


class RaisingGraph:
    def invoke(self, text: str, session_id: str | None = None) -> TriageState:
        raise RuntimeError("boom: secret internals")

    def close(self) -> None:  # pragma: no cover - only called on autoload shutdown
        pass


@pytest.fixture
def graph() -> StubGraph:
    return StubGraph()


@pytest.fixture
def client(graph: StubGraph) -> TestClient:
    services = Services(graph=graph, classifier=StubClassifier())
    with TestClient(create_app(services=services), raise_server_exceptions=False) as test_client:
        yield test_client


def test_healthz_reports_ok_and_model_version(client: TestClient) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": "v2-test",
    }


def test_triage_maps_state_onto_the_response(client: TestClient, graph: StubGraph) -> None:
    response = client.post("/triage", json={"text": "where is my card", "session_id": "sess-1"})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "card_arrival"
    assert body["confidence"] == pytest.approx(0.91)
    assert body["route"] == "faq"
    assert body["decision"] == "auto_answer"
    assert body["route_reason"] == "classifier"
    assert body["needs_human"] is False
    assert body["sources"][0]["kind"] == "faq"
    assert body["model_version"] == "v2-test"
    assert body["latency_ms"] >= 0
    assert graph.calls == [("where is my card", "sess-1")]


def test_request_id_is_echoed(client: TestClient) -> None:
    response = client.post(
        "/triage",
        json={"text": "where is my card"},
        headers={"X-Request-ID": "req-123"},
    )

    assert response.status_code == 200
    assert response.json()["request_id"] == "req-123"
    assert response.headers["x-request-id"] == "req-123"


def test_blank_text_is_a_problem_document(client: TestClient) -> None:
    response = client.post("/triage", json={"text": ""})

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["title"] == "Request validation failed"
    assert body["status"] == 422
    assert "text" in body["detail"]
    assert body["request_id"]
    assert response.headers["x-request-id"] == body["request_id"]


def test_unknown_field_is_rejected(client: TestClient) -> None:
    response = client.post("/triage", json={"text": "hello", "typo": True})

    assert response.status_code == 422
    assert "typo" in response.json()["detail"]


def test_unknown_route_is_a_problem_document(client: TestClient) -> None:
    response = client.get("/no-such-route")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["title"] == "Not found"
    assert body["status"] == 404


def test_model_info_exposes_the_artifact_metadata(client: TestClient) -> None:
    response = client.get("/model")

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "v2-test"
    assert body["n_intents"] == 3
    assert body["trained_on"] == "train+val"
    assert body["calibrated"] is False
    assert body["thresholds"] == {
        "auto_route_threshold": 0.54,
        "min_margin": 0.10,
        "escalate_threshold": 0.40,
    }


def test_intents_expose_taxonomy_flags(client: TestClient) -> None:
    response = client.get("/intents")

    assert response.status_code == 200
    entries = {entry["name"]: entry for entry in response.json()}
    assert entries["card_arrival"]["faq_covered"] is True
    assert entries["card_arrival"]["auto_answerable"] is True
    assert entries["lost_or_stolen_card"]["auto_answerable"] is False
    # Malformed dataset labels are still presentable.
    assert entries["reverted_card_payment?"]["display"] == "Reverted card payment"


def test_degraded_app_serves_503_without_crashing() -> None:
    services = Services(load_error="no model artifact at artifacts/model.joblib")
    with TestClient(create_app(services=services), raise_server_exceptions=False) as client:
        health = client.get("/healthz")
        assert health.status_code == 503
        assert health.json()["status"] == "degraded"
        assert health.json()["model_loaded"] is False

        triage = client.post("/triage", json={"text": "hello"})
        assert triage.status_code == 503
        assert "no model artifact" in triage.json()["detail"]

        assert client.get("/model").status_code == 503
        assert client.get("/intents").status_code == 503


def test_unhandled_error_is_a_problem_without_internals() -> None:
    services = Services(graph=RaisingGraph(), classifier=StubClassifier())
    with TestClient(create_app(services=services), raise_server_exceptions=False) as client:
        response = client.post("/triage", json={"text": "hello"})

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["title"] == "Internal server error"
    assert body["detail"] is None
    assert "boom" not in response.text
    assert "secret" not in response.text


def test_injected_services_are_not_closed_by_the_app(graph: StubGraph) -> None:
    with TestClient(create_app(services=Services(graph=graph, classifier=StubClassifier()))):
        pass

    # The app did not create the graph, so it must not close someone else's client.
    assert graph.closed is False


def test_real_artifact_model_endpoint_keeps_thresholds_numeric() -> None:
    if not settings.model_path.exists():
        pytest.skip("run `python -m banking_triage.train` first")

    classifier = IntentClassifier.load()
    with TestClient(create_app(services=Services(classifier=classifier))) as client:
        response = client.get("/model")

    assert response.status_code == 200
    body = response.json()
    assert body["n_intents"] == 77
    assert body["calibrated"] is not None
    assert all(isinstance(value, float) for value in body["thresholds"].values())
