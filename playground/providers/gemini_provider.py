
from __future__ import annotations

import time

from playground.providers.base import ProviderResult


def _empty_usage() -> dict:
    return {"prompt": 0, "completion": 0, "total": 0}


def _split_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Pull system messages into a single instruction string; keep the rest."""
    system_parts: list[str] = []
    rest: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            system_parts.append(m.get("content") or "")
        else:
            rest.append(m)
    return "\n\n".join(p for p in system_parts if p), rest


def _to_genai_contents(rest: list[dict]):
    """Convert chat messages to google-genai Content objects."""
    from google.genai import types

    contents = []
    for m in rest:
        role = "user" if m.get("role") == "user" else "model"
        text = m.get("content") or ""
        contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=text)])
        )
    return contents


class GeminiProvider:
    name = "gemini"
    supports_thinking = True
    available_models = ["gemini-2.5-flash", "gemini-2.5-pro"]

    def generate(
        self,
        api_key: str,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> ProviderResult:
        try:
            from google import genai
            from google.genai import types
        except Exception as e:
            return ProviderResult(
                text="", reasoning="", token_usage=_empty_usage(), latency_ms=0,
                error=f"google-genai SDK not installed: {e}",
            )

        client = genai.Client(api_key=api_key)
        system_text, rest = _split_messages(messages)
        contents = _to_genai_contents(rest)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_text or None,
            thinking_config=types.ThinkingConfig(
                thinking_budget=2048,
                include_thoughts=True,
            ),
        )

        t0 = time.perf_counter()
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            return ProviderResult(
                text="", reasoning="", token_usage=_empty_usage(),
                latency_ms=int((time.perf_counter() - t0) * 1000),
                error=f"{type(e).__name__}: {e}",
            )
        elapsed = int((time.perf_counter() - t0) * 1000)

        try:
            text_parts: list[str] = []
            thought_parts: list[str] = []
            candidates = getattr(resp, "candidates", None) or []
            if candidates:
                content = candidates[0].content
                for part in getattr(content, "parts", None) or []:
                    is_thought = bool(getattr(part, "thought", False))
                    txt = getattr(part, "text", None) or ""
                    if not txt:
                        continue
                    if is_thought:
                        thought_parts.append(txt)
                    else:
                        text_parts.append(txt)

            usage = _empty_usage()
            usage_obj = getattr(resp, "usage_metadata", None)
            if usage_obj is not None:
                prompt_t = getattr(usage_obj, "prompt_token_count", 0) or 0
                cand_t = getattr(usage_obj, "candidates_token_count", 0) or 0
                total_t = getattr(usage_obj, "total_token_count", 0) or (prompt_t + cand_t)
                usage = {"prompt": prompt_t, "completion": cand_t, "total": total_t}

            return ProviderResult(
                text="".join(text_parts).strip(),
                reasoning="".join(thought_parts).strip(),
                token_usage=usage,
                latency_ms=elapsed,
            )
        except Exception as e:
            return ProviderResult(
                text="", reasoning="", token_usage=_empty_usage(),
                latency_ms=elapsed,
                error=f"response_parse_error: {type(e).__name__}: {e}",
            )
