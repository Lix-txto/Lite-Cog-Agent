"""Message history management for Agent conversations."""

from __future__ import annotations

from copy import deepcopy

from lite_cog_agent.llm.schemas import (
    LLMMessage,
    ToolCall,
)


class MessageHistoryError(ValueError):
    """Raised when a conversation message history is invalid."""


def _validate_tool_call(tool_call: ToolCall) -> None:
    """Validate a single Tool Call."""

    if not isinstance(tool_call, ToolCall):
        raise MessageHistoryError(
            "Every item in tool_calls must be a ToolCall instance."
        )

    if (
        not isinstance(tool_call.id, str)
        or not tool_call.id.strip()
    ):
        raise MessageHistoryError(
            "ToolCall.id must be a non-empty string."
        )

    if (
        not isinstance(tool_call.name, str)
        or not tool_call.name.strip()
    ):
        raise MessageHistoryError(
            "ToolCall.name must be a non-empty string."
        )

    if not isinstance(tool_call.arguments, dict):
        raise MessageHistoryError(
            "ToolCall.arguments must be a dictionary."
        )


def validate_message_history(
    messages: list[LLMMessage],
    *,
    allow_pending: bool = False,
) -> set[str]:
    """Validate message history and return historical Tool Call IDs.

    Args:
        messages: Conversation messages to validate.
        allow_pending: Whether unresolved Tool Calls may remain
            at the end of the history.

    Returns:
        All Tool Call IDs found in the history.

    Raises:
        MessageHistoryError: If the history violates the protocol.
    """

    if not isinstance(messages, list):
        raise MessageHistoryError(
            "Message history must be a list."
        )

    supported_roles = {
        "system",
        "user",
        "assistant",
        "tool",
    }

    seen_tool_call_ids: set[str] = set()
    pending_tool_call_ids: set[str] = set()

    for index, message in enumerate(messages):
        if not isinstance(message, LLMMessage):
            raise MessageHistoryError(
                f"Message at index {index} must be an LLMMessage instance."
            )

        if (
            not isinstance(message.role, str)
            or message.role not in supported_roles
        ):
            raise MessageHistoryError(
                f"Unsupported message role at index {index}: "
                f"{message.role!r}."
            )

        if not isinstance(message.tool_calls, list):
            raise MessageHistoryError(
                f"Message.tool_calls at index {index} must be a list."
            )

        if message.role != "assistant" and message.tool_calls:
            raise MessageHistoryError(
                "Only assistant messages can contain tool_calls. "
                f"Invalid message at index {index}."
            )

        if (
            message.role != "tool"
            and message.tool_call_id is not None
        ):
            raise MessageHistoryError(
                "Only tool messages can contain tool_call_id. "
                f"Invalid message at index {index}."
            )

        if (
            message.role != "tool"
            and pending_tool_call_ids
        ):
            raise MessageHistoryError(
                "Tool Call batch has unresolved results before "
                f"message at index {index}."
            )

        if message.role == "assistant":
            for tool_call in message.tool_calls:
                _validate_tool_call(tool_call)

                if tool_call.id in seen_tool_call_ids:
                    raise MessageHistoryError(
                        "Duplicate tool call ID in message history: "
                        f"{tool_call.id!r}."
                    )

                seen_tool_call_ids.add(tool_call.id)
                pending_tool_call_ids.add(tool_call.id)

        elif message.role == "tool":
            tool_call_id = message.tool_call_id

            if (
                not isinstance(tool_call_id, str)
                or not tool_call_id.strip()
            ):
                raise MessageHistoryError(
                    "Tool message must have a non-empty tool_call_id."
                )

            if tool_call_id not in pending_tool_call_ids:
                raise MessageHistoryError(
                    "Tool message references an unknown or already "
                    f"resolved Tool Call ID: {tool_call_id!r}."
                )

            pending_tool_call_ids.remove(tool_call_id)

    if pending_tool_call_ids and not allow_pending:
        unresolved_ids = sorted(pending_tool_call_ids)

        raise MessageHistoryError(
            "Message history contains unresolved Tool Calls: "
            f"{unresolved_ids!r}."
        )

    return seen_tool_call_ids


class MessageManager:
    """Manage an isolated conversation message history."""

    def __init__(
        self,
        messages: list[LLMMessage] | None = None,
    ) -> None:
        if messages is None:
            messages = []

        if not isinstance(messages, list):
            raise TypeError("messages must be a list.")

        copied_messages = deepcopy(messages)

        validate_message_history(copied_messages)

        self._messages = copied_messages

    def add_message(self, message: LLMMessage) -> None:
        """Append one message after validating the candidate history."""

        if not isinstance(message, LLMMessage):
            raise TypeError(
                "message must be an LLMMessage instance."
            )

        self.add_messages([message])

    def add_messages(
        self,
        messages: list[LLMMessage],
    ) -> None:
        """Append messages atomically after validating the candidate."""

        if not isinstance(messages, list):
            raise TypeError("messages must be a list.")

        candidate = deepcopy(self._messages)
        candidate.extend(deepcopy(messages))

        validate_message_history(
            candidate,
            allow_pending=True,
        )

        self._messages = candidate

    def add_system_message(self, content: str) -> None:
        """Append a system message."""

        self._add_text_message("system", content)

    def add_user_message(self, content: str) -> None:
        """Append a user message."""

        self._add_text_message("user", content)

    def add_assistant_message(
        self,
        content: str | None,
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        """Append an assistant message."""

        if content is not None and not isinstance(content, str):
            raise TypeError("content must be a string or None.")

        if tool_calls is None:
            tool_calls = []

        if not isinstance(tool_calls, list):
            raise TypeError("tool_calls must be a list.")

        self.add_message(
            LLMMessage(
                role="assistant",
                content=content,
                tool_calls=tool_calls,
            )
        )

    def add_tool_message(
        self,
        content: str | None,
        tool_call_id: str,
        name: str | None = None,
    ) -> None:
        """Append a tool result message."""

        if content is not None and not isinstance(content, str):
            raise TypeError("content must be a string or None.")

        self.add_message(
            LLMMessage(
                role="tool",
                content=content,
                name=name,
                tool_call_id=tool_call_id,
            )
        )

    def _add_text_message(
        self,
        role: str,
        content: str,
    ) -> None:
        if not isinstance(content, str):
            raise TypeError("content must be a string.")

        if not content.strip():
            raise ValueError("content cannot be empty.")

        self.add_message(
            LLMMessage(
                role=role,
                content=content,
            )
        )

    def get_messages(self) -> list[LLMMessage]:
        """Return a deep copy of the conversation history."""

        return deepcopy(self._messages)

    def validate(self) -> None:
        """Validate the complete conversation history."""

        validate_message_history(self._messages)

    def clear(self) -> None:
        """Clear all conversation messages."""

        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)
