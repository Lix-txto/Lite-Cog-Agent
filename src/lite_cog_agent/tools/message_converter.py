"""Convert tool execution results into LLM messages."""

from __future__ import annotations

import json
from typing import Any

from lite_cog_agent.llm.schemas import LLMMessage, ToolCall
from lite_cog_agent.tools.schemas import ToolResult


def _validate_tool_call(tool_call: ToolCall) -> None:
    """Validate the structure of a single tool call."""

    if not isinstance(tool_call.id, str) or not tool_call.id.strip():
        raise ValueError("ToolCall.id must be a non-empty string.")

    if not isinstance(tool_call.name, str) or not tool_call.name.strip():
        raise ValueError("ToolCall.name must be a non-empty string.")

    if not isinstance(tool_call.arguments, dict):
        raise TypeError("ToolCall.arguments must be a dictionary.")


def convert_tool_results_to_messages(
    tool_calls: list[ToolCall],
    tool_results: list[ToolResult],
) -> list[LLMMessage]:
    """Convert tool execution results into LLM tool messages.

    Each tool result is associated with its corresponding tool call
    by list position. Tool call IDs must be non-empty and unique
    within the current batch.
    """

    if not isinstance(tool_calls, list):
        raise TypeError("tool_calls must be a list.")

    if not isinstance(tool_results, list):
        raise TypeError("tool_results must be a list.")

    if len(tool_calls) != len(tool_results):
        raise ValueError(
            "tool_calls and tool_results must have the same length."
        )

    seen_ids: set[str] = set()

    for tool_call in tool_calls:
        if not isinstance(tool_call, ToolCall):
            raise TypeError(
                "Every item in tool_calls must be a ToolCall instance."
            )

        _validate_tool_call(tool_call)

        if tool_call.id in seen_ids:
            raise ValueError(
                f"Duplicate tool call ID detected: {tool_call.id!r}."
            )

        seen_ids.add(tool_call.id)

    messages: list[LLMMessage] = []

    for tool_call, tool_result in zip(
        tool_calls,
        tool_results,
        strict=True,
    ):
        if not isinstance(tool_result, ToolResult):
            raise TypeError(
                "Every item in tool_results must be a ToolResult instance."
            )

        result_payload: dict[str, Any] = {
            "success": tool_result.success,
            "output": tool_result.output,
            "error": tool_result.error,
            "metadata": tool_result.metadata,
        }

        content = json.dumps(
            result_payload,
            ensure_ascii=False,
            default=str,
        )

        messages.append(
            LLMMessage(
                role="tool",
                content=content,
                tool_call_id=tool_call.id,
            )
        )

    return messages
