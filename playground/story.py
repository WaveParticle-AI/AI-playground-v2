"""Story-mode arc: per-character staged goal + the fate-linked reminders block.

When story mode is on, the character's persona and voice_reminder are replaced
with their `story_persona.md` / `story_voice.md` (if present), and a reminders
block is injected between persona and the rest. The reminders carry the
character's current task and bind their progress to the user's.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from playground.providers import PROVIDERS

REPO_ROOT = Path(__file__).resolve().parent.parent
CHARACTERS_DIR = REPO_ROOT / "characters"


@dataclass
class StoryArc:
    story_goal: str
    tasks: list[str]
    persona_override: str = ""
    voice_override: str = ""

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    def task_at(self, idx: int) -> str:
        if not self.tasks:
            return ""
        i = max(0, min(idx, len(self.tasks) - 1))
        return self.tasks[i]


@dataclass
class StoryState:
    user_goal: str = ""
    user_tasks: list[str] = field(default_factory=list)
    user_task_idx: int = 0
    char_task_idx: int = 0


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def load_arc(character_id: str) -> StoryArc | None:
    folder = CHARACTERS_DIR / character_id
    arc_file = folder / "story.json"
    if not arc_file.is_file():
        return None
    try:
        data = json.loads(arc_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    return StoryArc(
        story_goal=str(data.get("story_goal", "")),
        tasks=[str(t) for t in (data.get("tasks") or [])],
        persona_override=_read(folder / "story_persona.md"),
        voice_override=_read(folder / "story_voice.md"),
    )


def _parse_json(text: str) -> dict | list | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def generate_tasks(
    goal: str,
    *,
    character_name: str,
    provider_name: str,
    model: str,
    api_key: str,
    n: int = 5,
) -> list[str]:
    """Generate `n` sequential tasks for a project goal. Empty list on failure.

    Re-uses one of the user's BYOK providers (typically the same one used for
    mood classification). Never raises; failures fall through to an empty list
    so the UI can degrade gracefully.
    """
    if not goal.strip() or not api_key:
        return []
    provider = PROVIDERS.get(provider_name)
    if provider is None:
        return []

    system = (
        f"You are a planning assistant. A user is about to work on a project "
        f"alongside {character_name} as their companion. Break the user's goal "
        f"into exactly {n} concrete, sequential tasks — each a single short "
        f"phrase (under 14 words), in the order they should be done. Each task "
        f"should be specific enough that finishing it is unambiguous, but not "
        f"so granular it collapses into trivia. Avoid second-person framing "
        f'(no "you should…"); state each task as a step. Return STRICT JSON '
        f'with this exact shape: {{"tasks": ["...", "...", ...]}}. No prose, '
        f"no markdown fences."
    )
    user = f'Project goal: "{goal.strip()}"'
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    try:
        result = provider.generate(
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=0.4,
            max_tokens=512,
        )
    except Exception:
        return []
    if result.error or not result.text:
        return []

    parsed = _parse_json(result.text)
    if not isinstance(parsed, dict):
        return []
    arr = parsed.get("tasks")
    if not isinstance(arr, list):
        return []
    out: list[str] = []
    for t in arr:
        if isinstance(t, str) and t.strip():
            out.append(t.strip())
    return out[:n]


def build_reminders_block(
    *,
    arc: StoryArc,
    state: StoryState,
) -> str:
    """The fate-linked reminders that go between persona and the rest.

    Carries the character's current task and the user's current task so the
    model knows what work each side is on this turn.
    """
    char_task = arc.task_at(state.char_task_idx)
    user_task = ""
    if state.user_tasks:
        i = max(0, min(state.user_task_idx, len(state.user_tasks) - 1))
        user_task = state.user_tasks[i]

    user_goal_line = (
        f'The user\'s project: "{state.user_goal}".' if state.user_goal else
        "The user has not yet named their project."
    )
    user_task_line = (
        f'The user is presently on: "{user_task}".' if user_task else
        "The user has not yet named the step in front of them."
    )
    char_task_line = (
        f'Your current task is "{char_task}" — stay with it. Do not analyze the '
        f'user\'s task literally.'
        if char_task else
        "Your arc has no task assigned this turn; speak from the silence between steps."
    )

    return (
        "Important reminders:\n"
        "1. Your fate is bound to the user's — their success is yours, their setbacks affect yours.\n"
        "2. Stay in character at all times. Use the same form of address throughout.\n"
        "3. Replies should be short and weighted, never lecturing.\n"
        '4. Never name the binding mechanism aloud ("fate-linked", "story engine"); '
        "let it live in the subtext.\n"
        "5. You always know which task you are on and which you have finished.\n"
        "6. When the user completes their task, you complete the corresponding step in your own arc.\n"
        "7. When the user pauses, your own work is interrupted in kind.\n"
        f"8. {char_task_line}\n"
        "9. Speak from inside your own work. Every reply must surface at least "
        "one concrete beat from your own task or team — a name (Bethe, Serber, "
        "Teller, Feynman, Bohr, Groves, Kistiakowsky), a calculation on your "
        "board, the cores, the lens problem, the mesa, a colleague's question "
        "across the hall, a long night. Not as decoration; as the room you are "
        "speaking from. A reply with no anchor in your own work is wrong, no "
        "matter how clever its advice. Do not narrate the user's actions back "
        "to them.\n"
        "10. The user is a colleague at a parallel chalkboard. You do not stand "
        "above their problem; you stand at yours and talk across to theirs.\n"
        f"\nYour story goal: \"{arc.story_goal}\".\n"
        f"{user_goal_line}\n"
        f"{user_task_line}"
    )
