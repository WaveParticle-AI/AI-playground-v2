"""Qwen provider adapter via DashScope International (OpenAI-compatible).

Uses the OpenAI SDK with `base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"`.
For thinking-capable models, passes `extra_body={"enable_thinking": True}` and
captures `reasoning_content` from the response message.
"""

from __future__ import annotations

import time

from playground.providers.base import ProviderResult


THINKING_MODELS = {"qwen3-32b", "qwen3-235b-a22b", "qwq-32b"}


def _empty_usage() -> dict:
    return {"prompt": 0, "completion": 0, "total": 0}


class QwenProvider:
    name = "qwen"
    supports_thinking = True
    available_models = ["qwen3-32b", "qwen3-235b-a22b", "qwq-32b"]
    base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

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

        client = OpenAI(api_key=api_key, base_url=self.base_url)
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if model in THINKING_MODELS:
            kwargs["extra_body"] = {"enable_thinking": True}

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
            reasoning = getattr(choice.message, "reasoning_content", "") or ""
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
