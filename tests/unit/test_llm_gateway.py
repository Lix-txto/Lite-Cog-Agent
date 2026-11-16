import pytest

from lite_cog_agent.llm.gateway import LLMGateway
from lite_cog_agent.llm.providers.deepseek import DeepSeekProviderError
from lite_cog_agent.llm.schemas import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    TokenUsage,
)
from lite_cog_agent.llm.exceptions import (
    LLMConfigurationError,
    LLMGatewayError,
    LLMRequestError,
)


class FakeProvider:
    """A lightweight provider for Gateway unit tests."""

    def __init__(self) -> None:
        self.called = False
        self.closed = False
        self.last_request = None

    async def chat_completion(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        self.called = True
        self.last_request = request

        return LLMResponse(
            content="Mock response from FakeProvider.",
            usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
            model="fake-model",
        )

    async def close(self) -> None:
        self.closed = True


class FailingProvider:
    """A provider that simulates an API failure."""

    async def chat_completion(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        raise DeepSeekProviderError("Simulated provider failure.")

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_gateway_chat():
    provider = FakeProvider()
    gateway = LLMGateway(provider=provider)

    request = LLMRequest(
        messages=[
            LLMMessage(
                role="user",
                content="Analyze this dataset.",
            )
        ]
    )

    response = await gateway.chat(request)

    assert provider.called is True
    assert provider.last_request is request
    assert response.content == "Mock response from FakeProvider."
    assert response.usage.total_tokens == 30

    await gateway.close()

    assert provider.closed is True


@pytest.mark.asyncio
async def test_gateway_wraps_provider_error():
    gateway = LLMGateway(provider=FailingProvider())

    request = LLMRequest(
        messages=[
            LLMMessage(
                role="user",
                content="Test request.",
            )
        ]
    )

    with pytest.raises(LLMGatewayError):
        await gateway.chat(request)

    await gateway.close()


@pytest.mark.asyncio
async def test_gateway_rejects_empty_messages():
    provider = FakeProvider()
    gateway = LLMGateway(provider=provider)

    request = LLMRequest(messages=[])

    with pytest.raises(LLMRequestError):
        await gateway.chat(request)

    assert provider.called is False

    await gateway.close()


@pytest.mark.asyncio
async def test_gateway_rejects_request_after_close():
    provider = FakeProvider()
    gateway = LLMGateway(provider=provider)

    await gateway.close()

    request = LLMRequest(
        messages=[
            LLMMessage(role="user", content="Hello.")
        ]
    )

    with pytest.raises(LLMGatewayError):
        await gateway.chat(request)


@pytest.mark.asyncio
async def test_gateway_close_is_idempotent():
    provider = FakeProvider()
    gateway = LLMGateway(provider=provider)

    await gateway.close()
    await gateway.close()

    assert provider.closed is True


@pytest.mark.asyncio
async def test_gateway_async_context_manager():
    provider = FakeProvider()

    async with LLMGateway(provider=provider) as gateway:
        request = LLMRequest(
            messages=[
                LLMMessage(role="user", content="Hello.")
            ]
        )

        response = await gateway.chat(request)

        assert response.content == "Mock response from FakeProvider."

    assert provider.closed is True



@pytest.mark.asyncio
async def test_gateway_rejects_invalid_temperature():
    provider = FakeProvider()
    gateway = LLMGateway(provider=provider)

    request = LLMRequest(
        messages=[
            LLMMessage(role="user", content="Hello.")
        ],
        temperature=5.0,
    )

    with pytest.raises(LLMRequestError):
        await gateway.chat(request)

    assert provider.called is False

    await gateway.close()


@pytest.mark.asyncio
async def test_gateway_rejects_invalid_max_tokens():
    provider = FakeProvider()
    gateway = LLMGateway(provider=provider)

    request = LLMRequest(
        messages=[
            LLMMessage(role="user", content="Hello.")
        ],
        max_tokens=0,
    )

    with pytest.raises(LLMRequestError):
        await gateway.chat(request)

    assert provider.called is False

    await gateway.close()


@pytest.mark.asyncio
async def test_gateway_rejects_empty_model():
    provider = FakeProvider()
    gateway = LLMGateway(provider=provider)

    request = LLMRequest(
        messages=[
            LLMMessage(role="user", content="Hello.")
        ],
        model="   ",
    )

    with pytest.raises(LLMRequestError):
        await gateway.chat(request)

    assert provider.called is False

    await gateway.close()


@pytest.mark.asyncio
async def test_gateway_wraps_unexpected_provider_error():
    class UnexpectedErrorProvider:
        async def chat_completion(
            self,
            request: LLMRequest,
        ) -> LLMResponse:
            raise RuntimeError("Internal test error.")

        async def close(self) -> None:
            pass

    gateway = LLMGateway(provider=UnexpectedErrorProvider())

    request = LLMRequest(
        messages=[
            LLMMessage(role="user", content="Hello.")
        ]
    )

    with pytest.raises(LLMGatewayError) as exc_info:
        await gateway.chat(request)

    assert isinstance(exc_info.value.__cause__, RuntimeError)

    await gateway.close()
