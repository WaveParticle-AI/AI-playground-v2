"""Side-by-side N-column comparison row across selected models."""

from __future__ import annotations

import streamlit as st

from components.panes import assembled_prompt_pane, rag_pane, reasoning_pane
from playground.providers.base import ProviderResult
from playground.rag import Chunk


THINKING_MODE_BADGES = {
    "default": "default",
    "inner_os": "role-immersion",
    "no_inner_os": "pure-analysis",
}


def render_columns(
    *,
    selections: list[tuple[str, str]],
    results: list[ProviderResult],
    thinking_mode: str,
    rag_chunks: list[Chunk],
    assembled_messages: list[dict],
    rerun_callback,
) -> None:
    if not selections:
        st.info("Pick at least one model in the sidebar.")
        return

    cols = st.columns(len(selections))
    for i, ((pname, model), result) in enumerate(zip(selections, results)):
        with cols[i]:
            st.markdown(f"### `{pname}:{model}`")
            badge = THINKING_MODE_BADGES.get(thinking_mode, thinking_mode)
            meta_cols = st.columns(2)
            meta_cols[0].caption(f"thinking-mode: `{badge}`")
            meta_cols[1].caption(f"latency: `{result.latency_ms} ms`")

            usage = result.token_usage or {}
            st.caption(
                f"tokens — prompt: {usage.get('prompt', 0)}, "
                f"completion: {usage.get('completion', 0)}, "
                f"total: {usage.get('total', 0)}"
            )

            if result.error:
                st.error(result.error)
            else:
                st.markdown(result.text or "_(empty reply)_")

            reasoning_pane(result.reasoning or "")
            rag_pane(rag_chunks)
            assembled_prompt_pane(assembled_messages)

            if st.button("Re-run this column", key=f"rerun_{i}_{pname}_{model}"):
                rerun_callback(i)
