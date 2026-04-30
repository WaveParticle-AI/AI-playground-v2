"""Parallel fan-out runner across selected provider+model combinations."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from playground.providers import PROVIDERS
from playground.providers.base import ProviderResult


@dataclass
class ProviderSelection:
    provider_name: str
    model: str
    api_key: str
    thinking_mode_already_applied: bool = True


def _run_one(
    sel: ProviderSelection,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> ProviderResult:
    provider = PROVIDERS.get(sel.provider_name)
    if provider is None:
        return ProviderResult(
            text="",
            reasoning="",
            token_usage={"prompt": 0, "completion": 0, "total": 0},
            latency_ms=0,
            error=f"Unknown provider: {sel.provider_name}",
        )
    if not sel.api_key:
        return ProviderResult(
            text="",
            reasoning="",
            token_usage={"prompt": 0, "completion": 0, "total": 0},
            latency_ms=0,
            error=f"No API key set for {sel.provider_name}",
        )
    try:
        return provider.generate(
            api_key=sel.api_key,
            model=sel.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        return ProviderResult(
            text="",
            reasoning="",
            token_usage={"prompt": 0, "completion": 0, "total": 0},
            latency_ms=0,
            error=f"{type(e).__name__}: {e}",
        )


def run_parallel(
    selected: list[ProviderSelection],
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> list[ProviderResult]:
    """Fan out to providers concurrently; return results in input order."""
    if not selected:
        return []
    with ThreadPoolExecutor(max_workers=max(1, len(selected))) as ex:
        futures = [
            ex.submit(_run_one, sel, messages, temperature, max_tokens)
            for sel in selected
        ]
        return [f.result() for f in futures]
