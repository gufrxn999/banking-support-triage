"""Classifier loading and inference.

Wraps the Joblib artifact so callers never touch a raw sklearn Pipeline. This is
the only place that knows how to turn text into a `Prediction`, which keeps the
API and the LangGraph nodes consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from .config import Settings
from .config import settings as default_settings
from .schemas import IntentScore, Prediction
from .taxonomy import display_name


class ModelNotTrainedError(RuntimeError):
    """Raised when no artifact exists yet."""


@dataclass
class IntentClassifier:
    pipeline: Pipeline
    label_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    @property
    def n_intents(self) -> int:
        return len(self.label_names)

    @classmethod
    def load(cls, path: Path | None = None, settings: Settings | None = None) -> IntentClassifier:
        settings = settings or default_settings
        model_path = path or settings.model_path
        if not model_path.exists():
            raise ModelNotTrainedError(
                f"no model artifact at {model_path}; run `python -m banking_triage.train`"
            )
        payload = joblib.load(model_path)
        return cls(
            pipeline=payload["pipeline"],
            label_names=list(payload["label_names"]),
            metadata=dict(payload.get("metadata", {})),
        )

    def predict(self, text: str, top_k: int = 3) -> Prediction:
        proba = self.predict_proba(text)
        order = np.argsort(-proba)[: max(1, top_k)]
        best = int(order[0])
        return Prediction(
            intent=self.label_names[best],
            confidence=float(proba[best]),
            alternatives=[
                IntentScore(intent=self.label_names[int(idx)], score=float(proba[int(idx)]))
                for idx in order[1:]
            ],
        )

    def predict_proba(self, text: str) -> np.ndarray:
        cleaned = (text or "").strip()
        if not cleaned:
            raise ValueError("text must contain at least one non-whitespace character")
        cleaned = cleaned[: default_settings.max_text_chars]
        return self.pipeline.predict_proba([cleaned])[0]

    def display(self, intent: str) -> str:
        return display_name(intent)
