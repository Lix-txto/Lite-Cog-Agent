"""
In-memory token usage tracker for LLM requests.
"""

from __future__ import annotations

from dataclasses import dataclass

from lite_cog_agent.llm.response_validator import (
    LLMResponseValidator,
)
from lite_cog_agent.llm.schemas import TokenUsage


@dataclass(slots=True)
class UsageSummary:
    """Aggregated token usage statistics."""

    request_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class TokenUsageTracker:
    """Accumulate token usage across multiple LLM requests."""

    def __init__(self) -> None:
        self._summary = UsageSummary()

    def record(self, usage: TokenUsage) -> None:
        """Record token usage from one completed LLM request."""

        LLMResponseValidator.validate_usage(usage)

        self._summary.request_count += 1
        self._summary.prompt_tokens += usage.prompt_tokens
        self._summary.completion_tokens += usage.completion_tokens
        self._summary.total_tokens += usage.total_tokens

    def snapshot(self) -> UsageSummary:
        """Return a copy of the current usage summary."""

        return UsageSummary(
            request_count=self._summary.request_count,
            prompt_tokens=self._summary.prompt_tokens,
            completion_tokens=self._summary.completion_tokens,
            total_tokens=self._summary.total_tokens,
        )

    def reset(self) -> None:
        """Reset all accumulated usage statistics."""

        self._summary = UsageSummary()
