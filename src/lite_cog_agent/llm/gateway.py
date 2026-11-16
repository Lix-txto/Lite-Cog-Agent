"""
LLM Gateway for Lite-Cog-Agent.

Provides a unified interface between the application layer
and concrete LLM provider implementations.
"""

from __future__ import annotations

import math
from typing import Protocol

from lite_cog_agent.config import settings
from lite_cog_agent.llm.exceptions import (
    LLMConfigurationError,
    LLMGatewayError,
    LLMProviderError,
    LLMRequestError,
)
from lite_cog_agent.llm.providers.deepseek import DeepSeekProvider
from lite_cog_agent.llm.schemas import LLMRequest, LLMResponse


class LLMProvider(Protocol):
    """Structural interface required by the LLM Gateway."""

    async def chat_completion(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """Execute a chat completion request."""
        ...

    async def close(self) -> None:
        """Release provider resources."""
        ...


class LLMGateway:
    """Unified asynchronous entry point for LLM interactions."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
    ) -> None:
        self._provider = (
            provider if provider is not None else self._create_provider()
        )
        self._closed = False

    @staticmethod
    def _create_provider() -> LLMProvider:
        """Create a provider based on application configuration."""

        provider_name = settings.LLM_PROVIDER.lower()

        if provider_name == "deepseek":
            try:
                return DeepSeekProvider()
            except LLMConfigurationError:
                raise
            except Exception as exc:
                raise LLMConfigurationError(
                    "Failed to initialize the DeepSeek provider."
                ) from exc

        raise LLMConfigurationError(
            f"Unsupported LLM provider: {provider_name}"
        )

    @staticmethod
    def _validate_request(request: LLMRequest) -> None:
        """Validate a unified LLM request before provider invocation."""

        if not isinstance(request, LLMRequest):
            raise LLMRequestError(
                "Request must be an LLMRequest instance."
            )

        if not request.messages:
            raise LLMRequestError(
                "LLM request must contain at least one message."
            )

        if request.model is not None and not request.model.strip():
            raise LLMRequestError(
                "Model name cannot be empty."
            )

        if request.temperature is not None:
            temperature = request.temperature

            if (
                isinstance(temperature, bool)
                or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature)
                or not 0 <= temperature <= 2
            ):
                raise LLMRequestError(
                    "Temperature must be a finite number between 0 and 2."
                )

        if request.max_tokens is not None:
            if (
                isinstance(request.max_tokens, bool)
                or not isinstance(request.max_tokens, int)
                or request.max_tokens <= 0
            ):
                raise LLMRequestError(
                    "max_tokens must be a positive integer."
                )

        for tool in request.tools:
            if not tool.name or not tool.name.strip():
                raise LLMRequestError(
                    "Tool name cannot be empty."
                )

            if not tool.description or not tool.description.strip():
                raise LLMRequestError(
                    f"Tool '{tool.name}' description cannot be empty."
                )

            if not isinstance(tool.parameters, dict):
                raise LLMRequestError(
                    f"Tool '{tool.name}' parameters must be a dictionary."
                )

    async def chat(self, request: LLMRequest) -> LLMResponse:
        """Send a validated request through the configured provider."""

        if self._closed:
            raise LLMGatewayError(
                "LLM Gateway has already been closed."
            )

        self._validate_request(request)

        try:
            return await self._provider.chat_completion(request)

        except LLMProviderError as exc:
            raise LLMGatewayError(
                f"LLM provider request failed: {exc}"
            ) from exc

        except Exception as exc:
            raise LLMGatewayError(
                "Unexpected error occurred while calling the LLM provider."
            ) from exc

    async def close(self) -> None:
        """Close the underlying provider and release resources."""

        if self._closed:
            return

        await self._provider.close()
        self._closed = True

    async def __aenter__(self) -> LLMGateway:
        """Enter asynchronous context manager."""

        if self._closed:
            raise LLMGatewayError(
                "Cannot reuse a closed LLM Gateway."
            )

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Close resources when leaving the context."""

        await self.close()
