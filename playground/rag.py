
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


REPO_ROOT = Path(__file__).resolve().parent.parent
CHARACTERS_DIR = REPO_ROOT / "characters"
CACHE_DIR = REPO_ROOT / "data" / "cache"


@dataclass
class Chunk:
    source_file: str
    heading: str
    text: str
    score: float = 0.0


def _split_by_h2(md_text: str, source_file: str) -> list[Chunk]:
    """Split a markdown file by H2 headers; fall back to ~500-token windows."""
    h2 = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
    matches = list(h2.finditer(md_text))
    chunks: list[Chunk] = []

    if matches:
        for i, m in enumerate(matches):
            heading = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
            body = md_text[start:end].strip()
            if body:
                chunks.append(Chunk(source_file=source_file, heading=heading, text=body))
        return chunks

    words = md_text.split()
    window = 500
    for i in range(0, len(words), window):
        body = " ".join(words[i : i + window]).strip()
        if body:
            chunks.append(
                Chunk(source_file=source_file, heading="(no heading)", text=body)
            )
    return chunks


def _knowledge_dir(character_id: str) -> Path:
    return CHARACTERS_DIR / character_id / "knowledge"


def _load_chunks(character_id: str) -> list[Chunk]:
    kdir = _knowledge_dir(character_id)
    if not kdir.is_dir():
        return []
    chunks: list[Chunk] = []
    for path in sorted(kdir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks.extend(_split_by_h2(text, source_file=path.name))
    return chunks


def _cache_path(character_id: str) -> Path:
    return CACHE_DIR / f"{character_id}.pkl"


def _newest_source_mtime(character_id: str) -> float:
    kdir = _knowledge_dir(character_id)
    if not kdir.is_dir():
        return 0.0
    mtimes = [p.stat().st_mtime for p in kdir.glob("*.md")]
    return max(mtimes) if mtimes else 0.0


def _build_index(character_id: str) -> tuple[TfidfVectorizer, np.ndarray, list[Chunk]] | None:
    chunks = _load_chunks(character_id)
    if not chunks:
        return None
    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2)
    )
    matrix = vectorizer.fit_transform([c.text for c in chunks])
    return vectorizer, matrix, chunks


def _load_or_build(character_id: str):
    cache_file = _cache_path(character_id)
    src_mtime = _newest_source_mtime(character_id)
    if cache_file.exists() and src_mtime and cache_file.stat().st_mtime >= src_mtime:
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    built = _build_index(character_id)
    if built is None:
        return None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(built, f)
    except Exception:
        pass
    return built


def retrieve(query: str, character_id: str, k: int = 4) -> list[Chunk]:
    built = _load_or_build(character_id)
    if built is None or not query.strip():
        return []
    vectorizer, matrix, chunks = built
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, matrix)[0]
    top_idx = np.argsort(sims)[::-1][:k]
    out: list[Chunk] = []
    for i in top_idx:
        c = chunks[int(i)]
        out.append(
            Chunk(
                source_file=c.source_file,
                heading=c.heading,
                text=c.text,
                score=float(sims[i]),
            )
        )
    return out


def format_chunks_for_prompt(chunks: list[Chunk]) -> str:
    """Render chunks as a single string suitable for a system message."""
    if not chunks:
        return ""
    parts: list[str] = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[{i}] {c.source_file} — {c.heading} (score {c.score:.3f})\n{c.text}"
        )
    return "\n\n".join(parts)
