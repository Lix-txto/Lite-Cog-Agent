"""
DeepSeek LLM provider implementation.

This module adapts the OpenAI-compatible DeepSeek API
to Lite-Cog-Agent's unified LLM schemas.
"""

from __future__ import annotations

import json
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAIError,
)

from lite_cog_agent.config import settings
from lite_cog_agent.llm.schemas import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)
from lite_cog_agent.llm.exceptions import (
    LLMConfigurationError,
    LLMProviderError,
)


class DeepSeekProviderError(LLMProviderError):
    """Raised when the DeepSeek provider cannot complete a request."""


class DeepSeekProvider:
    """Async provider adapter for the DeepSeek API."""


    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.api_key = (
            api_key if api_key is not None else settings.DEEPSEEK_API_KEY
        )

        self.base_url = (
            base_url if base_url is not None else settings.DEEPSEEK_BASE_URL
        )

        self.model = (
            model if model is not None else settings.DEEPSEEK_MODEL
        )

        if not self.api_key or not self.api_key.strip():
            if client is None:
                raise LLMConfigurationError(
                    "DeepSeek API Key is missing. "
                    "Configure DEEPSEEK_API_KEY in your .env file."
                )

        if not self.base_url or not self.base_url.strip():
            raise LLMConfigurationError(
                "DeepSeek Base URL cannot be empty."
            )

        if not self.model or not self.model.strip():
            raise LLMConfigurationError(
                "DeepSeek model name cannot be empty."
            )

        if client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key.strip(),
                base_url=self.base_url,
            )
        else:
            self._client = client


    @staticmethod
    def _convert_messages(
        messages: list[LLMMessage],
    ) -> list[dict[str, Any]]:
        """Convert unified messages into OpenAI-compatible messages."""

        converted: list[dict[str, Any]] = []

        for message in messages:
            item: dict[str, Any] = {
                "role": message.role,
            }

            if message.content is not None:
                item["content"] = message.content

            if message.name is not None:
                item["name"] = message.name

            if message.tool_call_id is not None:
                item["tool_call_id"] = message.tool_call_id

            if message.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(
                                tool_call.arguments,
                                ensure_ascii=False,
                            ),
                        },
                    }
                    for tool_call in message.tool_calls
                ]

            converted.append(item)

        return converted

    @staticmethod
    def _convert_tools(
        tools: list[ToolDefinition],
    ) -> list[dict[str, Any]]:
        """Convert unified tool definitions into API tool schemas."""

        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    @staticmethod
    def _convert_response(response: Any) -> LLMResponse:
        """Convert an SDK response into a unified LLM response."""

        if not response.choices:
            raise DeepSeekProviderError(
                "DeepSeek returned an empty choices list."
            )

        choice = response.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []

        for item in message.tool_calls or []:
            try:
                arguments = json.loads(item.function.arguments)
            except json.JSONDecodeError as exc:
                raise DeepSeekProviderError(
                    f"Invalid JSON arguments returned for tool "
                    f"'{item.function.name}'."
                ) from exc

            if not isinstance(arguments, dict):
                raise DeepSeekProviderError(
                    f"Tool arguments for '{item.function.name}' "
                    "must be a JSON object."
                )

            tool_calls.append(
                ToolCall(
                    id=item.id,
                    name=item.function.name,
                    arguments=arguments,
                )
            )

        usage = response.usage

        token_usage = TokenUsage(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage=token_usage,
            model=response.model,
        )

    async def chat_completion(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """Send an asynchronous chat completion request."""

        if not request.messages:
            raise ValueError("LLM request must contain at least one message.")

        model = request.model or self.model

        if not model:
            raise ValueError("DeepSeek model name is missing.")

        payload: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(request.messages),
        }

        if request.temperature is not None:
            payload["temperature"] = request.temperature

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        if request.tools:
            payload["tools"] = self._convert_tools(request.tools)

        try:
            response = await self._client.chat.completions.create(
                **payload
            )
        except APITimeoutError as exc:
            raise DeepSeekProviderError(
                "DeepSeek API request timed out."
            ) from exc

        except APIConnectionError as exc:
            raise DeepSeekProviderError(
                "Unable to connect to DeepSeek API. "
                "Check your network and Base URL."
            ) from exc

        except APIStatusError as exc:
            raise DeepSeekProviderError(
                f"DeepSeek API returned HTTP {exc.status_code}."
            ) from exc

        except OpenAIError as exc:
            raise DeepSeekProviderError(
                "DeepSeek SDK request failed."
            ) from exc

        return self._convert_response(response)

    async def close(self) -> None:
        """Close the underlying asynchronous HTTP client."""

        await self._client.close()
