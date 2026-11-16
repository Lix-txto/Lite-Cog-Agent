import pytest

from lite_cog_agent.llm.exceptions import LLMConfigurationError
from lite_cog_agent.llm.providers.deepseek import DeepSeekProvider


def test_provider_rejects_missing_api_key():
    with pytest.raises(LLMConfigurationError):
        DeepSeekProvider(
            api_key="",
            base_url="https://api.deepseek.com",
            model="deepseek-flash",
        )


def test_provider_rejects_empty_base_url():
    with pytest.raises(LLMConfigurationError):
        DeepSeekProvider(
            api_key="test-key",
            base_url="",
            model="deepseek-flash",
        )


def test_provider_rejects_empty_model():
    with pytest.raises(LLMConfigurationError):
        DeepSeekProvider(
            api_key="test-key",
            base_url="https://api.deepseek.com",
            model="",
        )


def test_provider_accepts_valid_configuration():
    provider = DeepSeekProvider(
        api_key="test-key",
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
    )

    assert provider.base_url == "https://api.deepseek.com"
    assert provider.model == "deepseek-flash"

    import asyncio

    asyncio.run(provider.close())
