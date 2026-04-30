"""LLM provider adapters (OpenAI, DeepSeek, Gemini, Qwen) behind one Protocol."""

from playground.providers.base import Provider, ProviderResult
from playground.providers.openai_provider import OpenAIProvider
from playground.providers.deepseek_provider import DeepSeekProvider
from playground.providers.gemini_provider import GeminiProvider
from playground.providers.qwen_provider import QwenProvider

PROVIDERS: dict[str, Provider] = {
    "openai": OpenAIProvider(),
    "deepseek": DeepSeekProvider(),
    "gemini": GeminiProvider(),
    "qwen": QwenProvider(),
}

__all__ = [
    "Provider",
    "ProviderResult",
    "OpenAIProvider",
    "DeepSeekProvider",
    "GeminiProvider",
    "QwenProvider",
    "PROVIDERS",
]
