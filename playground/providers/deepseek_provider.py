"""DeepSeek provider adapter — uses the OpenAI SDK with a custom base_url.

For `deepseek-reasoner`, captures `choice.message.reasoning_content` into
`ProviderResult.reasoning`. Only `deepseek-reasoner` exposes thinking content.
"""

from __future__ import annotations

import time

from playground.providers.base import ProviderResult


def _empty_usage() -> dict:
    return {"prompt": 0, "completion": 0, "total": 0}


class DeepSeekProvider:
    name = "deepseek"
    supports_thinking = True  # toggled per-model below
    available_models = ["deepseek-chat", "deepseek-reasoner"]
    base_url = "https://api.deepseek.com"

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

        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
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
            reasoning = ""
            if model == "deepseek-reasoner":
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
