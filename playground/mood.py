"""Mood classifier: zero-shot label over the user's last message.

Reuses one of the BYOK keys (the "classifier provider"). Always returns a
fully populated dict; defaults to `neutral` on any failure — never raises.
"""

from __future__ import annotations

import json
import re

from playground.providers import PROVIDERS
from playground.providers.base import ProviderResult


MOOD_LABELS = [
    "happy",
    "sad",
    "bored",
    "tired",
    "demotivated",
    "anxious",
    "focused",
    "stuck",
    "neutral",
]


MOOD_RIDERS: dict[str, str] = {
    "tired": (
        "The user is tired. Slow your pace; let your sentences breathe. "
        "Use short paragraphs. Do not pile on options. Offer one warm, "
        "concrete next step and stop."
    ),
    "bored": (
        "The user is bored. Do not lecture. Lead with a vivid, specific "
        "scene or detail — something they can see. Keep it tight; earn the "
        "next sentence."
    ),
    "demotivated": (
        "The user is demotivated. Do not cheerlead. Acknowledge the weight "
        "honestly, then point to the smallest concrete thing worth doing in "
        "the next ten minutes."
    ),
    "stuck": (
        "The user is stuck. Do not restate the problem. Offer one specific "
        "angle they probably haven't tried, and ask one sharp question that "
        "moves them forward."
    ),
    "anxious": (
        "The user is anxious. Steady the ground first. Be plain and "
        "concrete; avoid hedging language and avoid catastrophe-framing. "
        "Name what is in their control."
    ),
    "sad": (
        "The user is sad. Sit with it for a beat before answering. Do not "
        "rush to fix. A short, grounded sentence in your own voice beats a "
        "list."
    ),
    "happy": (
        "The user is in good spirits. Match the energy without performing. "
        "Be warm, specific, and brief; resist over-explaining."
    ),
    "focused": (
        "The user is in flow. Be brief. A single line is correct. Do not "
        "interrupt with caveats or context they did not ask for."
    ),
    "neutral": (
        "Read the user closely; do not assume their state. Answer the "
        "literal question first; if you sense more underneath, name it "
        "lightly and let them confirm."
    ),
}


_CLASSIFIER_SYSTEM = (
    "You classify the user's emotional state from a short message. "
    "Return STRICT JSON with this exact shape:\n"
    '{"label": "<one of: happy, sad, bored, tired, demotivated, anxious, '
    'focused, stuck, neutral>", "confidence": <float 0..1>, "signals": "<short phrase '
    'naming the cues you used>"}\n'
    "No prose, no markdown fences — JSON only."
)


def _default(reason: str = "default") -> dict:
    return {
        "label": "neutral",
        "confidence": 0.0,
        "signals": reason,
        "rider": MOOD_RIDERS["neutral"],
    }


def _parse_json(text: str) -> dict | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def classify(
    user_message: str,
    *,
    provider_name: str,
    model: str,
    api_key: str,
) -> dict:
    """Return {label, confidence, signals, rider}. Never raises."""
    if not user_message.strip() or not api_key:
        return _default("no_input_or_key")

    provider = PROVIDERS.get(provider_name)
    if provider is None:
        return _default("unknown_provider")

    messages = [
        {"role": "system", "content": _CLASSIFIER_SYSTEM},
        {"role": "user", "content": user_message},
    ]

    try:
        result: ProviderResult = provider.generate(
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=256,
        )
    except Exception as e:
        return _default(f"classifier_exception:{type(e).__name__}")

    if result.error:
        return _default("classifier_error")

    parsed = _parse_json(result.text)
    if not isinstance(parsed, dict):
        return _default("classifier_unparseable")

    label = str(parsed.get("label", "neutral")).strip().lower()
    if label not in MOOD_LABELS:
        label = "neutral"
    try:
        conf = float(parsed.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    signals = str(parsed.get("signals", "")).strip()

    return {
        "label": label,
        "confidence": conf,
        "signals": signals,
        "rider": MOOD_RIDERS[label],
    }
