"""Assemble the message list passed to the provider.

Order:
  1. system: persona
  2. system: RAG chunks (if enabled and non-empty)
  3. system: mood block (if enabled)
  4. assistant/user history (oldest first)
  5. user: current user message
  6. system: voice reminder (last instruction wins)

After assembly, the thinking-mode marker is appended to the first user message
via `inject_marker`.
"""

from playground.characters import Character
from playground.rag import Chunk, format_chunks_for_prompt
from playground.thinking_modes import inject_marker


def _mood_block(mood: dict) -> str:
    label = mood.get("label", "neutral")
    conf = float(mood.get("confidence", 0.0))
    signals = mood.get("signals", "")
    rider = mood.get("rider", "")
    return (
        f"USER STATE — read mood: {label} (confidence {conf:.2f}). "
        f"Signals: {signals}\n{rider}"
    )


def assemble(
    *,
    character: Character,
    user_message: str,
    history: list[dict],
    rag_chunks: list[Chunk] | None,
    mood: dict | None,
    thinking_mode: str,
) -> list[dict]:
    messages: list[dict] = []

    messages.append({"role": "system", "content": character.persona})

    if rag_chunks:
        rag_text = format_chunks_for_prompt(rag_chunks)
        messages.append(
            {
                "role": "system",
                "content": "Background knowledge you may draw on:\n\n" + rag_text,
            }
        )

    if mood:
        messages.append({"role": "system", "content": _mood_block(mood)})

    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": user_message})

    messages.append({"role": "system", "content": character.voice_reminder})

    return inject_marker(messages, thinking_mode)
