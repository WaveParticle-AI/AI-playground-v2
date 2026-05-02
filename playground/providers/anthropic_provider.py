"""Anthropic (Claude) provider adapter.

Supports Claude models via the Anthropic SDK. Reasoning/chain-of-thought is
surfaced when returned by the API.
"""

from __future__ import annotations

import time

from playground.providers.base import ProviderResult


def _empty_usage() -> dict:
    return {"prompt": 0, "completion": 0, "total": 0}


def _extract_reasoning(response) -> str:
    """Best-effort extraction of reasoning/thinking from Anthropic response."""
    for attr in ("reasoning", "thinking", "reasoning_content"):
        val = getattr(response, attr, None)
        if isinstance(val, str) and val:
            return val
    content = getattr(response, "content", None)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("thinking", "reasoning"):
                return block.get("text", "")
    return ""


class AnthropicProvider:
    name = "anthropic"
    supports_thinking = True
    available_models = [
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
        "claude-3-opus-latest",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
    ]

    def generate(
        self,
        api_key: str,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> ProviderResult:
        try:
            from anthropic import Anthropic
        except Exception as e:
            return ProviderResult(
                text="", reasoning="", token_usage=_empty_usage(), latency_ms=0,
                error=f"anthropic SDK not installed: {e}",
            )

        client = Anthropic(api_key=api_key)

        # Convert messages to Anthropic format
        system_msg = ""
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg += msg["content"] + "\n"
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        kwargs = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
        }
        if system_msg.strip():
            kwargs["system"] = system_msg.strip()
        if temperature > 0:
            kwargs["temperature"] = temperature

        t0 = time.perf_counter()
        try:
            resp = client.messages.create(**kwargs)
        except Exception as e:
            return ProviderResult(
                text="", reasoning="", token_usage=_empty_usage(),
                latency_ms=int((time.perf_counter() - t0) * 1000),
                error=f"{type(e).__name__}: {e}",
            )
        elapsed = int((time.perf_counter() - t0) * 1000)

        try:
            text = ""
            reasoning = _extract_reasoning(resp)
            for block in resp.content:
                if hasattr(block, "type") and block.type == "text":
                    text = block.text
                    break
                elif isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    break

            usage_obj = getattr(resp, "usage", None)
            usage = _empty_usage()
            if usage_obj is not None:
                usage = {
                    "prompt": getattr(usage_obj, "input_tokens", 0) or 0,
                    "completion": getattr(usage_obj, "output_tokens", 0) or 0,
                    "total": (getattr(usage_obj, "input_tokens", 0) or 0)
                    + (getattr(usage_obj, "output_tokens", 0) or 0),
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
