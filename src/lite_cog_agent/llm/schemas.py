"""
Unified data schemas for LLM requests and responses.

These schemas decouple the application layer from specific
LLM providers and SDK response formats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class LLMMessage:
    """A provider-independent chat message."""

    role: MessageRole
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

@dataclass(slots=True)
class ToolCall:
    """A tool call requested by an LLM."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolDefinition:
    """A tool definition exposed to an LLM."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(slots=True)
class TokenUsage:
    """Token usage statistics returned by an LLM provider."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class LLMRequest:
    """A unified request sent through the LLM Gateway."""

    messages: list[LLMMessage]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[ToolDefinition] = field(default_factory=list)


@dataclass(slots=True)
class LLMResponse:
    """A unified response returned by the LLM Gateway."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str | None = None
