"""Thinking-mode marker strings and the injection helper.

The marker is appended to the FIRST user message in an assembled message list,
exactly once, never to a system message and never to subsequent turns.
"""


INNER_OS_MARKER = """

[Role immersion requirement] In your thinking process (inside the thinking tags), follow these rules:
1. Use the character's first-person inner monologue, wrapped in parentheses, e.g. "(thinking: ...)" or "(inner OS: ...)".
2. Describe the character's inner feelings in first person, e.g. "I think", "I feel", "I secretly...".
3. Stay immersed in the character; analyze the situation and plan your reply through inner monologue."""


NO_INNER_OS_MARKER = """

[Thinking-mode requirement] In your thinking process (inside the thinking tags), follow these rules:
1. Do NOT wrap inner monologue in parentheses; state analysis directly.
2. Do NOT use first-person character voice for inner activity; replace with analytical phrasing.
3. Focus thinking on plot/state analysis and reply planning, not in-character monologue."""


_MARKERS = {
    "default": "",
    "inner_os": INNER_OS_MARKER,
    "no_inner_os": NO_INNER_OS_MARKER,
}


def inject_marker(messages: list[dict], mode: str) -> list[dict]:
    """Append the chosen marker to the first user message in `messages`.

    Returns a new list. Mode is one of: "default", "inner_os", "no_inner_os".
    "default" is a no-op. Marker is appended only once and only on the first
    user-role turn — never on system messages, never on later user turns.
    """
    marker = _MARKERS.get(mode, "")
    if not marker:
        return [dict(m) for m in messages]

    out: list[dict] = []
    injected = False
    for m in messages:
        m2 = dict(m)
        if not injected and m2.get("role") == "user":
            m2["content"] = (m2.get("content") or "") + marker
            injected = True
        out.append(m2)
    return out
