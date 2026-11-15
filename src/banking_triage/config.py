"""Runtime configuration.

Everything tunable lives here so the API, the graph, and the evaluation harness
cannot drift apart. Values are env-overridable for deployment, but every knob
has a working default so a fresh clone runs without setup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

# Hard input limit. Kept here so the Pydantic schema and the API agree.
MAX_TEXT_CHARS = 2000


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name) or default).expanduser().resolve()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else float(raw)


@dataclass(frozen=True)
class Settings:
    """Immutable settings snapshot. Pass a custom instance in tests."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = _env_path("TRIAGE_DATA_DIR", PROJECT_ROOT / "data")
    artifacts_dir: Path = _env_path("TRIAGE_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")
    reports_dir: Path = _env_path("TRIAGE_REPORTS_DIR", PROJECT_ROOT / "reports")

    # --- model -----------------------------------------------------------
    model_filename: str = "model.joblib"
    top_k: int = 3

    # --- triage thresholds -------------------------------------------------
    # Below `escalate_threshold` we never assert an intent; we ask or hand off.
    # Values below were measured in M2 (reports/m2_calibration.json), on the
    # val split, targeting >=0.95 precision on auto-answered traffic.
    # The model artifact carries its own copy; that one wins at load time.
    escalate_threshold: float = 0.40

    # At or above `auto_route_threshold` we answer without a human, but only
    # if the top-1/top-2 margin is at least `min_margin`.
    auto_route_threshold: float = 0.54
    min_margin: float = 0.10

    # --- exchange-rate tool ----------------------------------------------
    exchange_api_base: str = os.getenv("FRANKFURTER_BASE_URL", "https://api.frankfurter.dev/v1")
    exchange_timeout_s: float = _env_float("FRANKFURTER_TIMEOUT_S", 5.0)
    exchange_max_retries: int = 2

    # --- http ------------------------------------------------------------
    request_timeout_s: float = 15.0
    max_text_chars: int = MAX_TEXT_CHARS

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / self.model_filename

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def labels_path(self) -> Path:
        return self.processed_dir / "labels.json"

    @property
    def faq_path(self) -> Path:
        return self.data_dir / "faq.jsonl"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.processed_dir, self.artifacts_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
