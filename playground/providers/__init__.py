"""LLM provider adapters (OpenAI, Anthropic, DeepSeek, Gemini, Qwen) behind one Protocol."""

from playground.providers.base import Provider, ProviderResult
from playground.providers.openai_provider import OpenAIProvider
from playground.providers.anthropic_provider import AnthropicProvider
from playground.providers.deepseek_provider import DeepSeekProvider
from playground.providers.gemini_provider import GeminiProvider
from playground.providers.qwen_provider import QwenProvider

PROVIDERS: dict[str, Provider] = {
    "openai": OpenAIProvider(),
    "anthropic": AnthropicProvider(),
    "deepseek": DeepSeekProvider(),
    "gemini": GeminiProvider(),
    "qwen": QwenProvider(),
}

__all__ = [
    "Provider",
    "ProviderResult",
    "OpenAIProvider",
    "AnthropicProvider",
    "DeepSeekProvider",
    "GeminiProvider",
    "QwenProvider",
    "PROVIDERS",
]
