"""Streamlit entrypoint for the Waveparticle Playground.

Compares character-driven LLM replies across selected provider+model pairs,
with full visibility into RAG retrievals, mood classification, the assembled
prompt, and per-model reasoning content.
"""

from __future__ import annotations

import streamlit as st

from components.comparison import render_columns
from components.panes import mood_panel
from components.sidebar import render as render_sidebar
from playground import mood as mood_mod
from playground.prompt_assembly import assemble
from playground.rag import retrieve
from playground.runner import ProviderSelection, run_parallel


st.set_page_config(page_title="Waveparticle Playground", layout="wide")


def _ensure_session_defaults(n_columns: int) -> None:
    if "history" not in st.session_state or not isinstance(
        st.session_state["history"], list
    ):
        st.session_state["history"] = []
    while len(st.session_state["history"]) < n_columns:
        st.session_state["history"].append([])
    if "last_user_message" not in st.session_state:
        st.session_state["last_user_message"] = ""


def _build_selections(state, key_lookup: dict[str, str]) -> list[ProviderSelection]:
    return [
        ProviderSelection(
            provider_name=p,
            model=m,
            api_key=key_lookup.get(p, ""),
        )
        for p, m in state.selected_models
    ]


def _per_column_assemble(
    *,
    state,
    user_message: str,
    column_index: int,
    rag_chunks,
    mood_dict,
):
    history = []
    if state.mode == "conversation" and column_index < len(st.session_state["history"]):
        history = list(st.session_state["history"][column_index])
    return assemble(
        character=state.character,
        user_message=user_message,
        history=history,
        rag_chunks=rag_chunks if state.rag_enabled else None,
        mood=mood_dict if state.mood_enabled else None,
        thinking_mode=state.thinking_mode,
    )


def _run_all(state, user_message: str):
    if not state.character:
        return None, [], []

    rag_chunks = []
    if state.rag_enabled:
        rag_chunks = retrieve(user_message, state.character.id, k=state.rag_top_k)

    mood_dict = None
    if state.mood_enabled and state.classifier_provider and state.classifier_model:
        key = state.api_keys.get(state.classifier_provider, "")
        mood_dict = mood_mod.classify(
            user_message,
            provider_name=state.classifier_provider,
            model=state.classifier_model,
            api_key=key,
        )

    selections = _build_selections(state, state.api_keys)
    assembled_per_col: list[list[dict]] = []
    for i, _ in enumerate(state.selected_models):
        msgs = _per_column_assemble(
            state=state,
            user_message=user_message,
            column_index=i,
            rag_chunks=rag_chunks,
            mood_dict=mood_dict,
        )
        assembled_per_col.append(msgs)

    results = []
    if assembled_per_col:
        # Each column may have a slightly different message list (history per
        # column in conversation mode). Run them sequentially through a small
        # threadpool by zipping selections with their messages via run_parallel
        # one at a time when histories diverge; in single-shot mode they're
        # identical, so we can fan out together.
        if state.mode == "single" or len({tuple((m["role"], m["content"]) for m in msgs) for msgs in assembled_per_col}) == 1:
            results = run_parallel(
                selections,
                assembled_per_col[0],
                state.temperature,
                state.max_tokens,
            )
        else:
            from playground.runner import _run_one
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=max(1, len(selections))) as ex:
                futs = [
                    ex.submit(
                        _run_one, sel, msgs, state.temperature, state.max_tokens
                    )
                    for sel, msgs in zip(selections, assembled_per_col)
                ]
                results = [f.result() for f in futs]

    return mood_dict, rag_chunks, list(zip(assembled_per_col, results))


def main() -> None:
    state = render_sidebar()

    st.title("Waveparticle Playground")

    if state.character is None:
        st.warning(
            "No characters found. Add a folder under `characters/` with "
            "`persona.md` and `voice_reminder.md`."
        )
        return

    n_cols = max(1, len(state.selected_models))
    _ensure_session_defaults(n_cols)

    if state.mode == "conversation":
        if len(st.session_state["history"]) != n_cols:
            st.session_state["history"] = [[] for _ in range(n_cols)]
        cols_h = st.columns(n_cols) if n_cols else None
        if cols_h is not None:
            for i, (pname, model) in enumerate(state.selected_models):
                with cols_h[i]:
                    with st.expander(f"History — `{pname}:{model}`", expanded=False):
                        h = st.session_state["history"][i]
                        if not h:
                            st.caption("(empty)")
                        for turn in h:
                            st.markdown(f"**{turn['role']}**: {turn['content']}")
        if st.button("Clear conversation history"):
            st.session_state["history"] = [[] for _ in range(n_cols)]
            st.rerun()

    user_message = st.text_area(
        "Your message",
        height=140,
        placeholder="Speak to the character. They'll receive a fully assembled prompt.",
    )

    send = st.button("Send", type="primary")

    if "last_run" not in st.session_state:
        st.session_state["last_run"] = None

    if send and user_message.strip():
        if not state.selected_models:
            st.error("Pick at least one model in the sidebar.")
            return
        with st.spinner("Calling models in parallel..."):
            mood_dict, rag_chunks, paired = _run_all(state, user_message)
        st.session_state["last_run"] = {
            "user_message": user_message,
            "mood": mood_dict,
            "rag_chunks": rag_chunks,
            "paired": paired,
            "selections": state.selected_models,
            "thinking_mode": state.thinking_mode,
        }
        if state.mode == "conversation":
            for i, (_, result) in enumerate(paired):
                if i >= len(st.session_state["history"]):
                    st.session_state["history"].append([])
                st.session_state["history"][i].append(
                    {"role": "user", "content": user_message}
                )
                if not result.error and result.text:
                    st.session_state["history"][i].append(
                        {"role": "assistant", "content": result.text}
                    )

    last = st.session_state.get("last_run")
    if last:
        def _rerun_one(column_index: int) -> None:
            sel_p, sel_m = last["selections"][column_index]
            sel = ProviderSelection(
                provider_name=sel_p, model=sel_m, api_key=state.api_keys.get(sel_p, "")
            )
            msgs = last["paired"][column_index][0]
            with st.spinner(f"Re-running `{sel_p}:{sel_m}`..."):
                results = run_parallel(
                    [sel], msgs, state.temperature, state.max_tokens
                )
            new_paired = list(last["paired"])
            new_paired[column_index] = (msgs, results[0])
            last["paired"] = new_paired
            st.session_state["last_run"] = last
            st.rerun()

        results_only = [r for _, r in last["paired"]]
        assembled_for_first = (
            last["paired"][0][0] if last["paired"] else []
        )
        render_columns(
            selections=last["selections"],
            results=results_only,
            thinking_mode=last["thinking_mode"],
            rag_chunks=last["rag_chunks"],
            assembled_messages=assembled_for_first,
            rerun_callback=_rerun_one,
        )
        st.divider()
        mood_panel(last["mood"])


if __name__ == "__main__":
    main()
