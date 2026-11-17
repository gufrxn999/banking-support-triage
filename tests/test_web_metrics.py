"""The page's evidence block must stay in sync with the committed reports."""

from __future__ import annotations

from banking_triage.config import PACKAGE_ROOT
from banking_triage.web_metrics import build_metrics, render


def test_showcase_metrics_match_the_committed_reports() -> None:
    committed = (PACKAGE_ROOT / "web" / "metrics.json").read_text(encoding="utf-8")

    assert committed == render(build_metrics())


def test_showcase_metrics_keep_the_headline_figures() -> None:
    payload = build_metrics()

    assert payload["workflow"]["n"] == 3080
    assert payload["workflow"]["auto_answered"] == 1147
    assert payload["probes"]["passed"] == payload["probes"]["total"] == 24
    assert payload["content"] == {"faq_intents": 42, "total_intents": 77}
