"""Character loader: each subfolder of `characters/` is one character."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHARACTERS_DIR = REPO_ROOT / "characters"


@dataclass
class Character:
    id: str
    name: str
    persona: str
    voice_reminder: str


_NAME_LINE = re.compile(r"^\s*#\s*Name:\s*(.+?)\s*$", re.MULTILINE)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _name_from_persona(persona: str, fallback_id: str) -> str:
    m = _NAME_LINE.search(persona)
    if m:
        return m.group(1).strip()
    return fallback_id.replace("_", " ").replace("-", " ").title()


def list_characters() -> list[Character]:
    if not CHARACTERS_DIR.is_dir():
        return []
    out: list[Character] = []
    for d in sorted(CHARACTERS_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        out.append(load_character(d.name))
    return out


def load_character(character_id: str) -> Character:
    folder = CHARACTERS_DIR / character_id
    persona = _read(folder / "persona.md")
    voice = _read(folder / "voice_reminder.md")
    name = _name_from_persona(persona, character_id)
    return Character(
        id=character_id,
        name=name,
        persona=persona,
        voice_reminder=voice,
    )
