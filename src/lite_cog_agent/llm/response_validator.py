"""
Validation utilities for unified LLM responses.
"""

from __future__ import annotations

from lite_cog_agent.llm.exceptions import LLMResponseError
from lite_cog_agent.llm.schemas import LLMResponse, TokenUsage


class LLMResponseValidator:
    """Validate the structure and consistency of LLM responses."""

    @staticmethod
    def validate_usage(usage: TokenUsage) -> None:
        """Validate token usage values."""

        if not isinstance(usage, TokenUsage):
            raise LLMResponseError(
                "Response usage must be a TokenUsage instance."
            )

        values = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

        for field_name, value in values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise LLMResponseError(
                    f"{field_name} must be a non-negative integer."
                )

        expected_total = (
            usage.prompt_tokens + usage.completion_tokens
        )

        if usage.total_tokens != expected_total:
            raise LLMResponseError(
                "Token usage inconsistency: total_tokens must equal "
                "prompt_tokens + completion_tokens."
            )

    @classmethod
    def validate(cls, response: LLMResponse) -> LLMResponse:
        """Validate an LLM response and return it if valid."""

        if not isinstance(response, LLMResponse):
            raise LLMResponseError(
                "Provider must return an LLMResponse instance."
            )

        has_content = (
            isinstance(response.content, str)
            and bool(response.content.strip())
        )

        has_tool_calls = bool(response.tool_calls)

        if not has_content and not has_tool_calls:
            raise LLMResponseError(
                "LLM response contains neither text content nor tool calls."
            )

        if response.content is not None and not isinstance(
            response.content, str
        ):
            raise LLMResponseError(
                "Response content must be a string or None."
            )

        if response.model is not None:
            if not isinstance(response.model, str) or not response.model.strip():
                raise LLMResponseError(
                    "Response model must be a non-empty string when provided."
                )

        if response.finish_reason is not None:
            if (
                not isinstance(response.finish_reason, str)
                or not response.finish_reason.strip()
            ):
                raise LLMResponseError(
                    "Finish reason must be a non-empty string when provided."
                )

        cls.validate_usage(response.usage)

        return response
