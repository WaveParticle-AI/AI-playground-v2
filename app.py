"""Streamlit entrypoint for the Waveparticle Playground.

Compares character-driven LLM replies across selected provider+model pairs,
with full visibility into RAG retrievals, mood classification, the assembled
prompt, and per-model reasoning content.
"""

from __future__ import annotations

import streamlit as st

from components.comparison import THINKING_MODE_BADGES
from components.panes import (
    assembled_prompt_pane,
    mood_panel,
    rag_pane,
    reasoning_pane,
)
from components.sidebar import render as render_sidebar
from playground import mood as mood_mod
from playground.prompt_assembly import assemble
from playground.rag import retrieve
from playground.runner import ProviderSelection, run_parallel
from playground.story import generate_tasks


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
    if column_index < len(st.session_state["history"]):
        history = list(st.session_state["history"][column_index])
    character = state.character
    if state.persona_edited is not None or state.voice_edited is not None:
        from dataclasses import replace
        character = replace(
            character,
            persona=state.persona_edited if state.persona_edited is not None else character.persona,
            voice_reminder=state.voice_edited if state.voice_edited is not None else character.voice_reminder,
        )
    return assemble(
        character=character,
        user_message=user_message,
        history=history,
        rag_chunks=rag_chunks if state.rag_enabled else None,
        mood=mood_dict if state.mood_enabled else None,
        thinking_mode=state.thinking_mode,
        story_arc=state.story_arc,
        story_state=state.story_state,
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
        if len({tuple((m["role"], m["content"]) for m in msgs) for msgs in assembled_per_col}) == 1:
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


def _render_story_setup(state) -> None:
    arc = state.story_arc
    char = state.character
    st.subheader(f"Begin a project with {char.name}")
    st.markdown(
        f"**{char.name}** is already inside his own work — *{arc.story_goal}*. "
        f"Name your project below; the two of you will move in parallel from "
        f"there."
    )

    goal = st.text_input(
        "Your project goal",
        key="setup_goal",
        placeholder="e.g. Review key algorithm concepts",
    )

    classifier_key = state.api_keys.get(state.classifier_provider or "", "")
    can_generate = bool(
        state.classifier_provider and state.classifier_model and classifier_key
    )
    gen_help = (
        f"Uses your mood-classifier provider "
        f"(`{state.classifier_provider}:{state.classifier_model}`) to draft "
        f"a 5-step plan you can edit."
        if can_generate else
        "Enable the mood classifier in the sidebar (and set its API key) to "
        "use AI-generated tasks."
    )

    btn_cols = st.columns([1, 1, 4])
    if btn_cols[0].button(
        "Generate tasks ✨",
        disabled=not can_generate or not goal.strip(),
        help=gen_help,
        key="setup_generate_tasks",
    ):
        with st.spinner(f"Drafting tasks for “{goal.strip()}”..."):
            tasks = generate_tasks(
                goal=goal.strip(),
                character_name=char.name,
                provider_name=state.classifier_provider,
                model=state.classifier_model,
                api_key=classifier_key,
            )
        if tasks:
            st.session_state["setup_tasks_raw"] = "\n".join(tasks)
            st.rerun()
        else:
            st.warning(
                "Task generation returned nothing — check the classifier "
                "provider/key, or type tasks yourself."
            )

    tasks_raw = st.text_area(
        "Your tasks (one per line — leave blank to let them emerge through "
        "the chat)",
        key="setup_tasks_raw",
        placeholder="step 1\nstep 2\nstep 3",
        height=160,
    )

    if btn_cols[1].button("Begin", type="primary", disabled=not goal.strip()):
        ss = state.story_state
        ss.user_goal = goal.strip()
        ss.user_tasks = [
            line.strip() for line in tasks_raw.splitlines() if line.strip()
        ]
        ss.user_task_idx = 0
        ss.char_task_idx = 0
        st.session_state[f"story_state_{char.id}"] = ss
        n_cols = max(1, len(state.selected_models))
        st.session_state["history"] = [[] for _ in range(n_cols)]
        st.session_state["last_run"] = None
        st.rerun()


def _render_arc_panel(state, n_cols: int) -> None:
    arc = state.story_arc
    ss = state.story_state
    char_name = state.character.name

    with st.container(border=True):
        head_cols = st.columns([4, 1, 1])
        head_cols[0].markdown(f"**Project**: {ss.user_goal}")
        if head_cols[1].button("Reset chat", help="Clear conversation, keep goal"):
            st.session_state["history"] = [[] for _ in range(n_cols)]
            st.session_state["last_run"] = None
            st.rerun()
        if head_cols[2].button("Reset goal", help="Start a new project"):
            ss.user_goal = ""
            ss.user_tasks = []
            ss.user_task_idx = 0
            ss.char_task_idx = 0
            st.session_state[f"story_state_{state.character.id}"] = ss
            st.session_state["history"] = [[] for _ in range(n_cols)]
            st.session_state["last_run"] = None
            st.rerun()

        arc_cols = st.columns(2)
        with arc_cols[0]:
            st.caption("Your tasks")
            if ss.user_tasks:
                for i, t in enumerate(ss.user_tasks):
                    marker = "▶" if i == ss.user_task_idx else "—"
                    st.markdown(f"{marker} `{i}` {t}")
                btns = st.columns(2)
                if btns[0].button("◀ Prev", key="user_prev_task"):
                    ss.user_task_idx = max(0, ss.user_task_idx - 1)
                    st.session_state[f"story_state_{state.character.id}"] = ss
                    st.rerun()
                if btns[1].button("Next ▶", key="user_next_task"):
                    ss.user_task_idx = min(
                        len(ss.user_tasks) - 1, ss.user_task_idx + 1
                    )
                    st.session_state[f"story_state_{state.character.id}"] = ss
                    st.rerun()
            else:
                st.caption("(no explicit task list — letting the chat shape it)")

        with arc_cols[1]:
            st.caption(f"{char_name}'s arc — {arc.story_goal}")
            for i, t in enumerate(arc.tasks):
                marker = "▶" if i == ss.char_task_idx else "—"
                st.markdown(f"{marker} `{i}` {t}")
            btns = st.columns(2)
            if btns[0].button("◀ Prev", key="char_prev_task"):
                ss.char_task_idx = max(0, ss.char_task_idx - 1)
                st.session_state[f"story_state_{state.character.id}"] = ss
                st.rerun()
            if btns[1].button("Next ▶", key="char_next_task"):
                ss.char_task_idx = min(arc.n_tasks - 1, ss.char_task_idx + 1)
                st.session_state[f"story_state_{state.character.id}"] = ss
                st.rerun()


def _render_story_chat(state, n_cols: int) -> None:
    _render_arc_panel(state, n_cols)

    if not state.selected_models:
        st.error("Pick at least one model in the sidebar to begin.")
        return

    ss = state.story_state
    last = st.session_state.get("last_run") or {}
    last_paired = last.get("paired") or []
    last_results = [r for _, r in last_paired] if last_paired else []
    last_assembled = [a for a, _ in last_paired] if last_paired else []
    last_rag = last.get("rag_chunks") or []
    story_caption = (
        f"story: char-task `{ss.char_task_idx}` · user-task `{ss.user_task_idx}`"
    )

    cols = st.columns(n_cols)
    for i, ((pname, model), col) in enumerate(zip(state.selected_models, cols)):
        with col:
            st.markdown(f"### `{pname}:{model}`")
            st.caption(story_caption)
            history = (
                st.session_state["history"][i]
                if i < len(st.session_state["history"]) else []
            )
            if not history:
                st.info(f"Speak to {state.character.name} below to begin.")
            for turn in history:
                role = turn["role"]
                avatar = "🧪" if role == "assistant" else None
                with st.chat_message(role, avatar=avatar):
                    st.markdown(turn["content"])

            r = last_results[i] if i < len(last_results) else None
            if r:
                if r.error:
                    st.error(r.error)
                badge = THINKING_MODE_BADGES.get(
                    state.thinking_mode, state.thinking_mode
                )
                meta = st.columns(2)
                meta[0].caption(f"thinking-mode: `{badge}`")
                meta[1].caption(f"latency: `{r.latency_ms} ms`")
                usage = r.token_usage or {}
                st.caption(
                    f"tokens — prompt: {usage.get('prompt', 0)}, "
                    f"completion: {usage.get('completion', 0)}, "
                    f"total: {usage.get('total', 0)}"
                )
                reasoning_pane(r.reasoning or "", expanded=False)
                rag_pane(last_rag)
                if i < len(last_assembled):
                    assembled_prompt_pane(last_assembled[i])

    user_msg = st.chat_input(
        f"Talk to {state.character.name} about “{ss.user_goal}”."
    )
    if user_msg and user_msg.strip():
        with st.spinner("Models running in parallel..."):
            mood_dict, rag_chunks, paired = _run_all(state, user_msg.strip())
        st.session_state["last_run"] = {
            "user_message": user_msg.strip(),
            "mood": mood_dict,
            "rag_chunks": rag_chunks,
            "paired": paired,
            "selections": state.selected_models,
            "thinking_mode": state.thinking_mode,
            "story_caption": story_caption,
        }
        for i, (_, result) in enumerate(paired):
            while i >= len(st.session_state["history"]):
                st.session_state["history"].append([])
            st.session_state["history"][i].append(
                {"role": "user", "content": user_msg.strip()}
            )
            if not result.error and result.text:
                st.session_state["history"][i].append(
                    {"role": "assistant", "content": result.text}
                )
        st.rerun()

    if last and last.get("mood"):
        st.divider()
        mood_panel(last["mood"])


def main() -> None:
    state = render_sidebar()

    st.title("Waveparticle Playground")

    if state.character is None:
        st.warning(
            "No characters found. Add a folder under `characters/` with "
            "`persona.md` and `voice_reminder.md`."
        )
        return

    if state.story_arc is None or state.story_state is None:
        st.warning(
            f"`{state.character.id}` has no `story.json` (or it's empty). "
            f"Add one under `characters/{state.character.id}/` with a "
            f"`story_goal` and a non-empty `tasks` list."
        )
        return

    n_cols = max(1, len(state.selected_models))
    _ensure_session_defaults(n_cols)

    if not state.story_state.user_goal:
        _render_story_setup(state)
        return

    _render_story_chat(state, n_cols)


if __name__ == "__main__":
    main()
