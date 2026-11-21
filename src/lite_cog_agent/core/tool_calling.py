"""Multi-round tool-calling orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from lite_cog_agent.core.messages import (
    MessageHistoryError,
    validate_message_history,
)
from lite_cog_agent.core.tool_calling_config import (
    ToolCallingConfig,
)
from lite_cog_agent.llm.gateway import LLMGateway
from lite_cog_agent.llm.schemas import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    ToolCall,
    ToolDefinition,
)
from lite_cog_agent.tools.dispatcher import ToolDispatcher
from lite_cog_agent.tools.message_converter import (
    convert_tool_results_to_messages,
)
from lite_cog_agent.tools.registry import ToolRegistry
from lite_cog_agent.tools.schemas import ToolResult


class ToolCallingError(Exception):
    """Base exception for tool-calling orchestration errors."""


class ToolCallingLimitError(ToolCallingError):
    """Raised when a tool-calling limit is exceeded."""


class ToolCallingProtocolError(ToolCallingError):
    """Raised when an LLM response violates the tool-calling protocol."""


@dataclass(slots=True)
class ToolCallingExecution:
    """Result of a tool-calling execution with complete history."""

    response: LLMResponse
    messages: list[LLMMessage]


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
        """Validate message history using the shared protocol."""

        try:
            return validate_message_history(messages)

        except MessageHistoryError as exc:
            raise ToolCallingProtocolError(
                str(exc)
            ) from exc

    async def run(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """Run tool calling and return only the final LLM response.

        This method preserves the original public interface.
        """

        execution = await self._run(request)
        return execution.response

    async def run_with_history(
        self,
        request: LLMRequest,
        *,
        on_iteration: Callable[[], None] | None = None,
        before_llm_request: Callable[[], None] | None = None,
        before_tool_dispatch: Callable[[int], None] | None = None,
    ) -> ToolCallingExecution:
        """Run tool calling and return the final response with full history.

        Args:
            request: The original LLM request.
            on_iteration: Callback invoked after an LLM response is received.
            before_llm_request: Callback invoked before each LLM request.
            before_tool_dispatch: Callback invoked before each tool batch.

        Returns:
            ToolCallingExecution: The final response and complete history.

        Raises:
            TypeError: If a supplied callback is not callable.
        """

        callbacks = (
            ("on_iteration", on_iteration),
            ("before_llm_request", before_llm_request),
            ("before_tool_dispatch", before_tool_dispatch),
        )

        for name, callback in callbacks:
            if callback is not None and not callable(callback):
                raise TypeError(f"{name} must be callable.")

        return await self._run(
            request,
            on_iteration=on_iteration,
            before_llm_request=before_llm_request,
            before_tool_dispatch=before_tool_dispatch,
        )

    async def _run(
        self,
        request: LLMRequest,
        *,
        on_iteration: Callable[[], None] | None = None,
        before_llm_request: Callable[[], None] | None = None,
        before_tool_dispatch: Callable[[int], None] | None = None,
    ) -> ToolCallingExecution:
        """Execute a multi-round tool-calling interaction.

        max_rounds limits the number of tool-execution rounds.
        One additional LLM request is allowed for final synthesis
        after the last tool-execution round.

        The returned message list is independent from the caller's
        original list. Existing LLMMessage objects are not modified.
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

            # Check execution budget before sending the LLM request.
            if before_llm_request is not None:
                before_llm_request()

            response = await self._gateway.chat(llm_request)

            # Record one completed LLM inference request.
            if on_iteration is not None:
                on_iteration()

            # Validate the complete response before processing it.
            self._validate_response(
                response,
                seen_tool_call_ids=seen_tool_call_ids,
            )

            # No tool calls means the model has completed its response.
            if not response.tool_calls:
                messages.append(
                    LLMMessage(
                        role="assistant",
                        content=response.content,
                    )
                )

                return ToolCallingExecution(
                    response=response,
                    messages=messages,
                )

            # Reject the entire batch when it exceeds the per-round limit.
            # No tool in this batch is dispatched or charged to the tool budget.
            if (
                len(response.tool_calls)
                > self._config.max_tool_calls_per_round
            ):
                assistant_message = LLMMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=list(response.tool_calls),
                )
                messages.append(assistant_message)

                rejected_results = [
                    ToolResult.fail(
                        error=(
                            "Tool call batch rejected: "
                            f"the configured limit is "
                            f"{self._config.max_tool_calls_per_round}, "
                            f"but the model requested "
                            f"{len(response.tool_calls)} calls. "
                            "No tools in this batch were executed. "
                            "Please replan and issue fewer tool calls."
                        ),
                        metadata={
                            "reason": "max_tool_calls_per_round_exceeded",
                            "executed": False,
                        },
                    )
                    for _ in response.tool_calls
                ]

                messages.extend(
                    convert_tool_results_to_messages(
                        response.tool_calls,
                        rejected_results,
                    )
                )

                continue

            # Reserve the entire tool-call batch before dispatch.
            if before_tool_dispatch is not None:
                before_tool_dispatch(len(response.tool_calls))

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

        # Final synthesis also consumes one LLM iteration.
        if before_llm_request is not None:
            before_llm_request()

        final_response = await self._gateway.chat(final_request)

        # Final synthesis is also an LLM inference iteration.
        if on_iteration is not None:
            on_iteration()

        # Final synthesis must also satisfy the response protocol.
        self._validate_response(
            final_response,
            seen_tool_call_ids=seen_tool_call_ids,
        )

        if final_response.tool_calls:
            raise ToolCallingProtocolError(
                "Final synthesis response must not contain tool calls."
            )

        # Preserve the final assistant answer in the returned history.
        messages.append(
            LLMMessage(
                role="assistant",
                content=final_response.content,
            )
        )

        return ToolCallingExecution(
            response=final_response,
            messages=messages,
        )

    @staticmethod
    def _build_request(
        original_request: LLMRequest,
        messages: list[LLMMessage],
        tools: list[ToolDefinition],
    ) -> LLMRequest:
        """Build a new request while preserving generation settings."""

        return LLMRequest(
            messages=list(messages),
            model=original_request.model,
            temperature=original_request.temperature,
            max_tokens=original_request.max_tokens,
            tools=tools,
        )
