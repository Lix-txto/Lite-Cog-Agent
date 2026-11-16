"""Multi-round tool-calling orchestration."""

from __future__ import annotations

from lite_cog_agent.llm.gateway import LLMGateway
from lite_cog_agent.llm.schemas import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    ToolDefinition,
    ToolCall,
)
from lite_cog_agent.tools.dispatcher import ToolDispatcher
from lite_cog_agent.tools.message_converter import (
    convert_tool_results_to_messages,
)
from lite_cog_agent.tools.registry import ToolRegistry

from lite_cog_agent.core.tool_calling_config import (
    ToolCallingConfig,
)


class ToolCallingError(Exception):
    """Base exception for tool-calling orchestration errors."""


class ToolCallingLimitError(ToolCallingError):
    """Raised when a tool-calling limit is exceeded."""


class ToolCallingProtocolError(ToolCallingError):
    """Raised when an LLM response violates the tool-calling protocol."""


class ToolCallingService:
    """Coordinate multi-round LLM tool calling."""

    def __init__(
        self,
        gateway: LLMGateway,
        registry: ToolRegistry,
        dispatcher: ToolDispatcher,
        config: ToolCallingConfig | None = None,
    ) -> None:
        self._gateway = gateway
        self._registry = registry
        self._dispatcher = dispatcher
        self._config = config or ToolCallingConfig()


    @staticmethod
    def _validate_tool_call(tool_call: ToolCall) -> None:
        """Validate a single Tool Call."""

        if not isinstance(tool_call, ToolCall):
            raise ToolCallingProtocolError(
                "Every item in tool_calls must be a ToolCall instance."
            )

        if (
            not isinstance(tool_call.id, str)
            or not tool_call.id.strip()
        ):
            raise ToolCallingProtocolError(
                "ToolCall.id must be a non-empty string."
            )

        if (
            not isinstance(tool_call.name, str)
            or not tool_call.name.strip()
        ):
            raise ToolCallingProtocolError(
                "ToolCall.name must be a non-empty string."
            )

        if not isinstance(tool_call.arguments, dict):
            raise ToolCallingProtocolError(
                "ToolCall.arguments must be a dictionary."
            )


    def _validate_response(
        self,
        response: LLMResponse,
        *,
        seen_tool_call_ids: set[str],
    ) -> None:
        """Validate an LLM response and its tool calls."""

        if not isinstance(response, LLMResponse):
            raise ToolCallingProtocolError(
                "Gateway response must be an LLMResponse instance."
            )

        if not isinstance(response.tool_calls, list):
            raise ToolCallingProtocolError(
                "LLMResponse.tool_calls must be a list."
            )

        current_round_ids: set[str] = set()

        for tool_call in response.tool_calls:
            self._validate_tool_call(tool_call)

            if tool_call.id in current_round_ids:
                raise ToolCallingProtocolError(
                    f"Duplicate tool call ID in current round: "
                    f"{tool_call.id!r}."
                )

            if tool_call.id in seen_tool_call_ids:
                raise ToolCallingProtocolError(
                    f"Duplicate tool call ID across rounds: "
                    f"{tool_call.id!r}."
                )

            current_round_ids.add(tool_call.id)

        seen_tool_call_ids.update(current_round_ids)


    @staticmethod
    def _validate_message_history(
        messages: list[LLMMessage],
    ) -> set[str]:
        """Validate message history and return historical Tool Call IDs."""

        if not isinstance(messages, list):
            raise ToolCallingProtocolError(
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
                raise ToolCallingProtocolError(
                    f"Message at index {index} must be an LLMMessage instance."
                )

            if message.role not in supported_roles:
                raise ToolCallingProtocolError(
                    f"Unsupported message role at index {index}: "
                    f"{message.role!r}."
                )

            if not isinstance(message.tool_calls, list):
                raise ToolCallingProtocolError(
                    f"Message.tool_calls at index {index} must be a list."
                )

            # Non-assistant messages cannot contain Tool Calls.
            if message.role != "assistant" and message.tool_calls:
                raise ToolCallingProtocolError(
                    f"Only assistant messages can contain tool_calls. "
                    f"Invalid message at index {index}."
                )

            # Only tool messages can carry tool_call_id.
            if (
                message.role != "tool"
                and message.tool_call_id is not None
            ):
                raise ToolCallingProtocolError(
                    f"Only tool messages can contain tool_call_id. "
                    f"Invalid message at index {index}."
                )

            # A pending Tool Call batch must be completed before
            # another non-tool message can appear.
            if (
                message.role != "tool"
                and pending_tool_call_ids
            ):
                raise ToolCallingProtocolError(
                    "Tool Call batch has unresolved results before "
                    f"message at index {index}."
                )

            if message.role == "assistant":
                for tool_call in message.tool_calls:
                    ToolCallingService._validate_tool_call(
                        tool_call
                    )

                    if tool_call.id in seen_tool_call_ids:
                        raise ToolCallingProtocolError(
                            f"Duplicate tool call ID in message history: "
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
                    raise ToolCallingProtocolError(
                        "Tool message must have a non-empty tool_call_id."
                    )

                if tool_call_id not in pending_tool_call_ids:
                    raise ToolCallingProtocolError(
                        f"Tool message references an unknown or already "
                        f"resolved Tool Call ID: {tool_call_id!r}."
                    )

                pending_tool_call_ids.remove(tool_call_id)

        if pending_tool_call_ids:
            unresolved_ids = sorted(pending_tool_call_ids)

            raise ToolCallingProtocolError(
                "Message history contains unresolved Tool Calls: "
                f"{unresolved_ids!r}."
            )

        return seen_tool_call_ids

    async def run(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """Run a multi-round tool-calling interaction.

        max_rounds limits the number of tool-execution rounds.
        One additional LLM request is allowed for final synthesis
        after the last tool-execution round.
        """

        if not isinstance(request, LLMRequest):
            raise TypeError("request must be an LLMRequest instance.")

        if not isinstance(request.messages, list):
            raise ToolCallingProtocolError(
                "LLM request messages must be a list."
            )

        if not request.messages:
            raise ValueError(
                "LLM request must contain at least one message."
            )

        # Copy the original message list.
        # The caller's request history must not be modified.
        messages = list(request.messages)

        # Validate the initial conversation history.
        # Historical Tool Call IDs must remain unique throughout this run.
        seen_tool_call_ids = self._validate_message_history(
            messages
        )
        # The registry is the source of truth for available tools.
        tool_definitions = self._registry.get_definitions()

        for round_index in range(self._config.max_rounds):
            llm_request = self._build_request(
                original_request=request,
                messages=messages,
                tools=tool_definitions,
            )

            response = await self._gateway.chat(llm_request)

            # Validate the complete response before processing it.
            self._validate_response(
                response,
                seen_tool_call_ids=seen_tool_call_ids,
            )

            # No tool calls means the model has completed its response.
            if not response.tool_calls:
                return response

            # Enforce the per-round tool-call limit before dispatch.
            if (
                len(response.tool_calls)
                > self._config.max_tool_calls_per_round
            ):
                raise ToolCallingLimitError(
                    "The number of tool calls in round "
                    f"{round_index + 1} exceeds the configured limit "
                    f"of {self._config.max_tool_calls_per_round}."
                )

            # Preserve the assistant message containing tool calls.
            assistant_message = LLMMessage(
                role="assistant",
                content=response.content,
                tool_calls=list(response.tool_calls),
            )

            messages.append(assistant_message)

            # Execute all tool calls in the current round.
            tool_results = await self._dispatcher.dispatch_many(
                response.tool_calls
            )

            # Convert execution results into LLM tool messages.
            tool_messages = convert_tool_results_to_messages(
                response.tool_calls,
                tool_results,
            )

            # Append tool messages in the original call order.
            messages.extend(tool_messages)

        # All configured tool-execution rounds have been consumed.
        # Make one final request without tools so the model can
        # synthesize the results instead of starting another round.
        final_request = self._build_request(
            original_request=request,
            messages=messages,
            tools=[],
        )

        final_response = await self._gateway.chat(final_request)

        # Final synthesis must also satisfy the response protocol.
        self._validate_response(
            final_response,
            seen_tool_call_ids=seen_tool_call_ids,
        )

        if final_response.tool_calls:
            raise ToolCallingProtocolError(
                "Final synthesis response must not contain tool calls."
            )

        return final_response

    @staticmethod
    def _build_request(
        original_request: LLMRequest,
        messages: list[LLMMessage],
        tools: list[ToolDefinition],
    ) -> LLMRequest:
        """Build a new request while preserving generation settings."""

        return LLMRequest(
            messages=messages,
            model=original_request.model,
            temperature=original_request.temperature,
            max_tokens=original_request.max_tokens,
            tools=tools,
        )
