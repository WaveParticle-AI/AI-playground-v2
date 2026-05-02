"""Sidebar UI: API keys (BYOK), character pick, model multiselect, toggles."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from playground.characters import Character, list_characters
from playground.providers import PROVIDERS
from playground.story import StoryArc, StoryState, load_arc


PROVIDER_ORDER = ["openai", "deepseek", "gemini", "qwen"]
MAX_MODELS = 4


@dataclass
class SidebarState:
    api_keys: dict[str, str]
    character: Character | None
    selected_models: list[tuple[str, str]]  # (provider_name, model)
    thinking_mode: str  # "default" | "inner_os" | "no_inner_os"
    temperature: float
    max_tokens: int
    rag_enabled: bool
    mood_enabled: bool
    rag_top_k: int
    classifier_provider: str | None
    classifier_model: str | None
    story_arc: StoryArc | None
    story_state: StoryState | None
    persona_edited: str | None = None
    voice_edited: str | None = None


def _all_model_options() -> list[str]:
    out: list[str] = []
    for pname in PROVIDER_ORDER:
        provider = PROVIDERS[pname]
        for m in provider.available_models:
            out.append(f"{pname}:{m}")
    return out


def _parse_selection(label: str) -> tuple[str, str]:
    pname, model = label.split(":", 1)
    return pname, model


def render() -> SidebarState:
    st.sidebar.title("Waveparticle Playground")

    st.sidebar.header("API Keys")
    api_keys: dict[str, str] = {}
    for pname in PROVIDER_ORDER:
        key_state = f"{pname}_api_key"
        api_keys[pname] = st.sidebar.text_input(
            f"{pname.capitalize()} API key",
            type="password",
            key=key_state,
        )
    st.sidebar.caption(
        "Keys stay in this browser session. Never sent anywhere except the "
        "provider's API."
    )

    st.sidebar.header("Character")
    chars = list_characters()
    character: Character | None = None
    if chars:
        labels = [c.name for c in chars]
        default_idx = 0
        for i, c in enumerate(chars):
            if c.id == "oppenheimer":
                default_idx = i
                break
        picked = st.sidebar.selectbox("Character", labels, index=default_idx)
        character = next(c for c in chars if c.name == picked)
    else:
        st.sidebar.info("No characters found under `characters/`.")

    persona_edited = None
    voice_edited = None
    if character is not None:
        persona_key = f"persona_{character.id}"
        voice_key = f"voice_{character.id}"
        if persona_key not in st.session_state:
            st.session_state[persona_key] = character.persona
        if voice_key not in st.session_state:
            st.session_state[voice_key] = character.voice_reminder
        with st.sidebar.expander("Edit character prompt", expanded=False):
            st.session_state[persona_key] = st.text_area(
                "Persona",
                value=st.session_state[persona_key],
                height=200,
                key=f"persona_editor_{character.id}",
            )
            st.session_state[voice_key] = st.text_area(
                "Voice reminder",
                value=st.session_state[voice_key],
                height=100,
                key=f"voice_editor_{character.id}",
            )
            cols = st.columns(2)
            if cols[0].button("Reset", key=f"reset_prompt_{character.id}"):
                st.session_state[persona_key] = character.persona
                st.session_state[voice_key] = character.voice_reminder
                st.rerun()
            if cols[1].button("Save", key=f"save_prompt_{character.id}"):
                from pathlib import Path
                char_dir = Path("characters") / character.id
                char_dir.mkdir(parents=True, exist_ok=True)
                (char_dir / "persona.md").write_text(st.session_state[persona_key], encoding="utf-8")
                (char_dir / "voice_reminder.md").write_text(st.session_state[voice_key], encoding="utf-8")
                st.success("Saved to files.")
        persona_edited = st.session_state[persona_key]
        voice_edited = st.session_state[voice_key]

    st.sidebar.header("Models to compare")
    options = _all_model_options()
    default_models = (
        ["deepseek:deepseek-reasoner"] if "deepseek:deepseek-reasoner" in options else options[:1]
    )
    picked_models = st.sidebar.multiselect(
        "Models",
        options=options,
        default=default_models,
        max_selections=MAX_MODELS,
        help=f"Up to {MAX_MODELS} models compared side-by-side.",
    )
    selected_models = [_parse_selection(s) for s in picked_models]

    st.sidebar.header("Thinking mode")
    mode_label = st.sidebar.radio(
        "Thinking mode",
        ["Default", "Role Immersion", "Pure Analysis"],
        index=0,
        label_visibility="collapsed",
    )
    thinking_mode = {
        "Default": "default",
        "Role Immersion": "inner_os",
        "Pure Analysis": "no_inner_os",
    }[mode_label]

    st.sidebar.header("Generation")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.8, 0.05)
    max_tokens = st.sidebar.slider("Max tokens", 128, 4096, 512, 64)

    st.sidebar.header("Pipeline toggles")
    rag_enabled = st.sidebar.toggle("RAG", value=True)
    mood_enabled = st.sidebar.toggle("Mood classification", value=True)
    rag_top_k = st.sidebar.slider("RAG top-k", 1, 10, 4, 1)

    classifier_provider: str | None = None
    classifier_model: str | None = None
    if mood_enabled:
        candidates = [
            (p, api_keys[p]) for p in PROVIDER_ORDER if api_keys[p]
        ]
        if candidates:
            default_classifier = candidates[0][0]
            classifier_provider = st.sidebar.selectbox(
                "Mood classifier provider",
                [p for p, _ in candidates],
                index=0,
                help="Reuses one of your BYOK keys for the mood call.",
            )
            cmodel_default = PROVIDERS[classifier_provider].available_models[0]
            classifier_model = st.sidebar.selectbox(
                "Mood classifier model",
                PROVIDERS[classifier_provider].available_models,
                index=PROVIDERS[classifier_provider].available_models.index(cmodel_default),
            )
        else:
            st.sidebar.info("Add at least one API key to enable mood classification.")

    story_arc: StoryArc | None = None
    story_state: StoryState | None = None
    if character is not None:
        story_arc = load_arc(character.id)
        if story_arc is not None and story_arc.tasks:
            ss_key = f"story_state_{character.id}"
            if ss_key not in st.session_state:
                st.session_state[ss_key] = StoryState()
            story_state = st.session_state[ss_key]

    return SidebarState(
        api_keys=api_keys,
        character=character,
        selected_models=selected_models,
        thinking_mode=thinking_mode,
        temperature=temperature,
        max_tokens=max_tokens,
        rag_enabled=rag_enabled,
        mood_enabled=mood_enabled,
        rag_top_k=rag_top_k,
        classifier_provider=classifier_provider,
        classifier_model=classifier_model,
        story_arc=story_arc,
        story_state=story_state,
        persona_edited=persona_edited,
        voice_edited=voice_edited,
    )
