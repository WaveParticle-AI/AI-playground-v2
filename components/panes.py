"""Reusable Streamlit panes used inside the comparison columns."""

from __future__ import annotations

import json

import streamlit as st

from playground.rag import Chunk


def reasoning_pane(reasoning: str) -> None:
    with st.expander("Reasoning / thinking", expanded=False):
        if reasoning:
            st.markdown(reasoning)
        else:
            st.caption("No reasoning content returned by this model.")


def rag_pane(chunks: list[Chunk]) -> None:
    with st.expander("Retrieved RAG chunks", expanded=False):
        if not chunks:
            st.caption("No chunks retrieved.")
            return
        for i, c in enumerate(chunks, 1):
            st.markdown(
                f"**[{i}] {c.source_file} — {c.heading}**  \nscore: `{c.score:.3f}`"
            )
            st.markdown(c.text)
            st.divider()


def assembled_prompt_pane(messages: list[dict]) -> None:
    with st.expander("Assembled prompt", expanded=False):
        st.code(json.dumps(messages, indent=2, ensure_ascii=False), language="json")


def mood_panel(mood: dict | None) -> None:
    st.subheader("Mood classification")
    if not mood:
        st.caption("Mood classification is off (or not yet computed).")
        return
    cols = st.columns([1, 1, 3])
    cols[0].metric("Label", mood.get("label", "neutral"))
    cols[1].metric("Confidence", f"{float(mood.get('confidence', 0.0)):.2f}")
    cols[2].markdown(f"**Signals:** {mood.get('signals', '') or '—'}")
    rider = mood.get("rider") or ""
    if rider:
        st.markdown(f"**Rider injected:**\n\n> {rider}")
