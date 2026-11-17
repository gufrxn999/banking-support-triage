"""Streamlit demo for the triage workflow.

The demo is a thin shell over `TriageGraph`: the same artifact, FAQ corpus, routing
policy, and session store as the API. It exists to make the behaviour visible -- the
intent, the confidence, why the route was chosen, and where the answer came from --
not to add a second implementation.

Streamlit is an optional dependency (`pip install -e ".[demo]"`). Only `main()`
imports it, so the pure helpers below stay importable and testable in the base
environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from .config import settings
from .graph import TriageGraph
from .schemas import TriageState

EXAMPLES = (
    "I would like to change my pin number please",
    "convert 250 EUR to CHF",
    "what's the exchange rate?",
    "I was charged the wrong rate on a 500 USD to EUR payment",
    "I lost my card, please block it before someone uses it",
    "can I use apple pay",
)

ROUTE_LABELS = {
    "faq": "Curated FAQ",
    "exchange_rate": "Live ECB rate",
    "clarify": "Clarification",
    "escalate": "Human handoff",
}

ROUTE_CAPTIONS = {
    "faq": "answered from reviewed copy",
    "exchange_rate": "answered with a dated reference rate",
    "clarify": "asked a question instead of guessing",
    "escalate": "handed to a human",
}


@dataclass(frozen=True)
class TurnView:
    """One rendered assistant turn. Pure data, so it is testable without Streamlit."""

    answer: str
    route: str
    route_label: str
    intent: str | None
    confidence: float | None
    route_reason: str | None
    needs_human: bool
    sources: tuple[tuple[str, str], ...]
    errors: tuple[str, ...]

    @property
    def headline(self) -> str:
        """The one-line trace under the answer."""
        parts = [self.route_label]
        if self.intent:
            parts.append(self.intent)
        if self.confidence is not None:
            parts.append(f"confidence {self.confidence:.2f}")
        if self.needs_human:
            parts.append("human follow-up")
        return " · ".join(parts)


def route_label(route: str | None) -> str:
    return ROUTE_LABELS.get(route or "", "Unknown route")


def turn_view(state: TriageState) -> TurnView:
    prediction = state.prediction
    return TurnView(
        answer=state.answer,
        route=state.route or "escalate",
        route_label=route_label(state.route),
        intent=prediction.intent if prediction else None,
        confidence=round(prediction.confidence, 3) if prediction else None,
        route_reason=state.route_reason,
        needs_human=state.needs_human,
        sources=tuple((source.kind, source.ref) for source in state.sources),
        errors=tuple(state.errors),
    )


def main() -> None:  # pragma: no cover - exercised by Streamlit, not pytest
    import streamlit as st

    st.set_page_config(
        page_title="Banking Support Triage",
        page_icon="🏦",
        layout="centered",
    )

    @st.cache_resource(show_spinner="Loading the classifier and FAQ corpus…")
    def load_graph() -> TriageGraph:
        return TriageGraph.from_settings(settings)

    try:
        graph = load_graph()
    except Exception as exc:  # noqa: BLE001 - the demo must explain, not traceback
        st.error(f"The triage services could not load: {exc}")
        st.code(
            "python -m banking_triage.data\n"
            "python -m banking_triage.train\n"
            "python -m banking_triage.calibrate",
            language="powershell",
        )
        st.stop()

    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader("Serving model")
        metadata = graph.classifier.metadata
        st.caption(f"`{graph.classifier.model_version}`")
        thresholds = graph.thresholds()
        st.write(
            f"{len(graph.classifier.label_names)} intents · {len(graph.faq)} with curated copy"
        )
        st.caption(
            f"auto-answer at confidence ≥ {thresholds['auto_route_threshold']:.2f} "
            f"and margin ≥ {thresholds['min_margin']:.2f}"
        )
        st.caption(f"trained on {metadata.get('trained_on', 'unknown')}")

        if st.button("New conversation", use_container_width=True):
            st.session_state.session_id = uuid4().hex
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.subheader("Try one")
        for example in EXAMPLES:
            if st.button(example, use_container_width=True):
                st.session_state.pending_prompt = example

        st.divider()
        st.caption(
            "Reference implementation over public BANKING77 data. FAQ entries are "
            "drafts awaiting policy review; rates are ECB reference rates, not live "
            "quotes."
        )

    st.title("Banking support triage")
    st.caption(
        "One message in, one decision out: answer, quote, clarify, or hand off. "
        "The trace shows why."
    )

    def render_trace(view: TurnView) -> None:
        st.caption(f"{view.headline} — {ROUTE_CAPTIONS.get(view.route, '')}")
        with st.expander("Why this answer"):
            if view.route_reason:
                st.markdown(f"**Route reason** · `{view.route_reason}`")
            if view.sources:
                st.markdown("**Sources**")
                for kind, ref in view.sources:
                    st.markdown(f"- `{kind}` · {ref}")
            else:
                st.caption("No external source: this turn asserts nothing.")
            if view.errors:
                st.warning(
                    "Degraded during this turn:\n\n"
                    + "\n".join(f"- {error}" for error in view.errors)
                )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("view") is not None:
                render_trace(message["view"])

    prompt = st.chat_input("Describe what you need help with")
    if st.session_state.get("pending_prompt"):
        prompt = st.session_state.pop("pending_prompt")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Routing…"):
                state = graph.invoke(prompt, session_id=st.session_state.session_id)
            view = turn_view(state)
            st.markdown(view.answer)
            render_trace(view)

        st.session_state.messages.append(
            {"role": "assistant", "content": view.answer, "view": view}
        )


if __name__ == "__main__":
    main()
