"""FastAPI surface.

The API is deliberately a thin adapter over `TriageGraph`: it validates the
request, runs one turn, and maps the resulting state onto the response contract.
No routing or policy decisions live here.

Operational choices worth stating:

- `/triage` is a sync endpoint. The classifier is CPU-bound and the exchange
  client is synchronous, so FastAPI's threadpool is the right place for them;
  declaring it async and blocking the event loop would be worse under load.
- A missing model does not stop the process from starting. `/healthz` reports
  `degraded` and `/triage` returns 503, which is what a deploy probe needs.
- Every non-2xx response is an RFC 9457 problem document, including the ones
  FastAPI generates for validation errors, and every response carries
  `X-Request-ID`.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .classifier import IntentClassifier
from .config import Settings
from .config import settings as default_settings
from .faq import FAQStore
from .graph import TriageGraph
from .schemas import (
    HealthResponse,
    IntentEntry,
    ModelInfo,
    ProblemDetail,
    TriageRequest,
    TriageResponse,
)
from .taxonomy import (
    FAQ_COVERED_INTENTS,
    NEVER_AUTO_ANSWER_INTENTS,
    display_name,
)

logger = logging.getLogger(__name__)

PROBLEM_MEDIA_TYPE = "application/problem+json"

# The showcase page ships inside the package so one container serves both the
# API and the site. Static files are mounted, not read per request.
WEB_DIR = Path(__file__).resolve().parent / "web"


@dataclass
class Services:
    """Everything the routes need. Injectable so tests load no artifacts."""

    graph: TriageGraph | None = None
    classifier: IntentClassifier | None = None
    faq: FAQStore | None = None
    load_error: str | None = None


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


def get_graph(services: ServicesDep) -> TriageGraph:
    if services.graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"triage unavailable: {services.load_error or 'model not loaded'}",
        )
    return services.graph


GraphDep = Annotated[TriageGraph, Depends(get_graph)]


def get_classifier(services: ServicesDep) -> IntentClassifier:
    if services.classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"model unavailable: {services.load_error or 'model not loaded'}",
        )
    return services.classifier


ClassifierDep = Annotated[IntentClassifier, Depends(get_classifier)]


def _request_id(request: Request) -> str:
    existing = getattr(request.state, "request_id", None)
    if existing:
        return str(existing)
    return request.headers.get("x-request-id") or uuid4().hex


def _problem(
    request: Request,
    status_code: int,
    title: str,
    detail: str | None = None,
) -> JSONResponse:
    request_id = _request_id(request)
    body = ProblemDetail(
        title=title,
        status=status_code,
        detail=detail,
        request_id=request_id,
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(),
        media_type=PROBLEM_MEDIA_TYPE,
        headers={"X-Request-ID": request_id},
    )


def create_app(
    settings: Settings | None = None,
    services: Services | None = None,
) -> FastAPI:
    """Build the app.

    Pass `services` to skip disk/network loading; omit it to load the real model,
    FAQ corpus, and exchange client at startup.
    """
    settings = settings or default_settings
    owns_graph = services is None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if services is not None:
            app.state.services = services
        else:
            try:
                graph = TriageGraph.from_settings(settings)
                app.state.services = Services(
                    graph=graph,
                    classifier=graph.classifier,
                    faq=graph.faq,
                )
            except Exception as exc:  # noqa: BLE001 - startup must not crash the probe
                logger.exception("triage services failed to load")
                app.state.services = Services(load_error=str(exc))

        try:
            yield
        finally:
            graph = app.state.services.graph
            if owns_graph and graph is not None:
                graph.close()

    app = FastAPI(
        title="Banking Support Triage",
        description=(
            "Intent classification, policy routing, curated answers, dated ECB "
            "reference rates, and clarification turns over BANKING77."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next):
        request.state.request_id = request.headers.get("x-request-id") or uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    # --- dependencies ---------------------------------------------------

    # --- error handling -------------------------------------------------
    @app.exception_handler(RequestValidationError)
    async def validation_problem(request: Request, exc: RequestValidationError) -> JSONResponse:
        parts = []
        for error in exc.errors():
            location = ".".join(str(item) for item in error.get("loc", ()))
            parts.append(f"{location}: {error.get('msg', 'invalid value')}")
        return _problem(
            request,
            422,
            "Request validation failed",
            "; ".join(parts) or None,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_problem(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _problem(request, exc.status_code, _title_for(exc.status_code), str(exc.detail))

    @app.exception_handler(Exception)
    async def unhandled_problem(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error while processing %s", request.url.path)
        # The exception text stays in the log; clients get no internals.
        return _problem(
            request,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Internal server error",
        )

    # --- routes ---------------------------------------------------------
    @app.get(
        "/healthz",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse, "description": "Dependencies unavailable"}},
        tags=["ops"],
    )
    def healthz(services: ServicesDep) -> JSONResponse:
        healthy = services.graph is not None
        body = HealthResponse(
            status="ok" if healthy else "degraded",
            model_loaded=services.classifier is not None,
            model_version=(
                services.classifier.model_version if services.classifier is not None else None
            ),
        )
        return JSONResponse(
            status_code=200 if healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
            content=body.model_dump(),
        )

    @app.post("/triage", response_model=TriageResponse, tags=["triage"])
    def triage(
        payload: TriageRequest,
        request: Request,
        graph: GraphDep,
        classifier: ClassifierDep,
    ) -> TriageResponse:
        started = perf_counter()
        state = graph.invoke(payload.text, session_id=payload.session_id)
        latency_ms = round((perf_counter() - started) * 1000, 2)

        if state.prediction is None or state.route is None or state.decision is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="triage produced an incomplete state",
            )

        return TriageResponse(
            request_id=_request_id(request),
            intent=state.prediction.intent,
            confidence=state.prediction.confidence,
            alternatives=state.prediction.alternatives,
            route=state.route,
            decision=state.decision,
            route_reason=state.route_reason,
            answer=state.answer,
            sources=state.sources,
            needs_human=state.needs_human,
            model_version=classifier.model_version,
            latency_ms=latency_ms,
            currency_query=state.currency_query,
            steps=state.steps,
            errors=state.errors,
        )

    @app.get("/model", response_model=ModelInfo, tags=["model"])
    def model_info(classifier: ClassifierDep) -> ModelInfo:
        metadata = classifier.metadata
        try:
            trained_at = datetime.fromisoformat(str(metadata["trained_at"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="model metadata is incomplete",
            ) from exc

        return ModelInfo(
            model_version=classifier.model_version,
            trained_at=trained_at,
            trained_on=str(metadata.get("trained_on", "unknown")),
            n_intents=len(classifier.label_names),
            vectorizer=str(metadata.get("vectorizer", "unknown")),
            calibrated=_calibrated(metadata),
            hyperparameters=metadata.get("hyperparameters", {}),
            metrics=metadata.get("metrics", {}),
            thresholds=_numeric_thresholds(metadata.get("thresholds", {})),
        )

    @app.get("/intents", response_model=list[IntentEntry], tags=["model"])
    def intents(classifier: ClassifierDep) -> list[IntentEntry]:
        return [
            IntentEntry(
                id=index,
                name=name,
                display=display_name(name),
                faq_covered=name in FAQ_COVERED_INTENTS,
                auto_answerable=name not in NEVER_AUTO_ANSWER_INTENTS,
            )
            for index, name in enumerate(classifier.label_names)
        ]

    # --- showcase page --------------------------------------------------
    # Mounted last so /docs, /healthz, and the API routes keep precedence.
    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

    return app


def _title_for(status_code: int) -> str:
    return {
        404: "Not found",
        405: "Method not allowed",
        422: "Request validation failed",
        503: "Service unavailable",
    }.get(status_code, "Request failed")


def _numeric_thresholds(raw: object) -> dict[str, float]:
    """`thresholds` also carries flags like `calibrated`; this endpoint is numeric only."""
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in raw.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


def _calibrated(metadata: dict) -> bool | None:
    thresholds = metadata.get("thresholds")
    if isinstance(thresholds, dict) and isinstance(thresholds.get("calibrated"), bool):
        return bool(thresholds["calibrated"])
    calibration = metadata.get("calibration")
    if isinstance(calibration, dict) and "method" in calibration:
        return calibration["method"] is not None
    return None


app = create_app()
