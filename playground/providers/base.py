from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ProviderResult:
    text: str
    reasoning: str
    token_usage: dict
    latency_ms: int
    error: str | None = None


class Provider(Protocol):
    name: str
    supports_thinking: bool
    available_models: list[str]

    def generate(
        self,
        api_key: str,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> ProviderResult: ...
