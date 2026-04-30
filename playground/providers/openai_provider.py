"""OpenAI provider adapter.

Supports both standard chat models (`gpt-4o`, `gpt-4o-mini`) and the
reasoning models (`o1-mini`, `o3-mini`). Reasoning models receive
`reasoning_effort="medium"` and have their reasoning summary surfaced when
the API returns one.
"""

from __future__ import annotations

import time

from playground.providers.base import ProviderResult


_REASONING_MODELS = {"o1-mini", "o3-mini"}


def _is_reasoning(model: str) -> bool:
    return model in _REASONING_MODELS or model.startswith("o1") or model.startswith("o3")


def _empty_usage() -> dict:
    return {"prompt": 0, "completion": 0, "total": 0}


def _extract_reasoning(message) -> str:
    """Best-effort: pick up a reasoning summary if the SDK returns one."""
    for attr in ("reasoning", "reasoning_content", "thinking"):
        val = getattr(message, attr, None)
        if isinstance(val, str) and val:
            return val
        if isinstance(val, dict):
            summary = val.get("summary") or val.get("content")
            if isinstance(summary, str) and summary:
                return summary
    return ""


class OpenAIProvider:
    name = "openai"
    supports_thinking = True
    available_models = ["gpt-4o", "gpt-4o-mini", "o1-mini", "o3-mini"]

    def generate(
        self,
        api_key: str,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> ProviderResult:
        try:
            from openai import OpenAI
        except Exception as e:
            return ProviderResult(
                text="", reasoning="", token_usage=_empty_usage(), latency_ms=0,
                error=f"openai SDK not installed: {e}",
            )

        client = OpenAI(api_key=api_key)
        kwargs: dict = {
            "model": model,
            "messages": messages,
        }
        if _is_reasoning(model):
            kwargs["reasoning_effort"] = "medium"
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["temperature"] = temperature
            kwargs["max_tokens"] = max_tokens

        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            return ProviderResult(
                text="", reasoning="", token_usage=_empty_usage(),
                latency_ms=int((time.perf_counter() - t0) * 1000),
                error=f"{type(e).__name__}: {e}",
            )
        elapsed = int((time.perf_counter() - t0) * 1000)

        try:
            choice = resp.choices[0]
            text = (choice.message.content or "").strip()
            reasoning = _extract_reasoning(choice.message)
            usage_obj = getattr(resp, "usage", None)
            usage = _empty_usage()
            if usage_obj is not None:
                usage = {
                    "prompt": getattr(usage_obj, "prompt_tokens", 0) or 0,
                    "completion": getattr(usage_obj, "completion_tokens", 0) or 0,
                    "total": getattr(usage_obj, "total_tokens", 0) or 0,
                }
            return ProviderResult(
                text=text,
                reasoning=reasoning,
                token_usage=usage,
                latency_ms=elapsed,
            )
        except Exception as e:
            return ProviderResult(
                text="", reasoning="", token_usage=_empty_usage(),
                latency_ms=elapsed,
                error=f"response_parse_error: {type(e).__name__}: {e}",
            )
