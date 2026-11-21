"""Agent request lifecycle orchestration."""

from __future__ import annotations

from copy import deepcopy

from lite_cog_agent.core.agent_budget import (
    AgentBudget,
    AgentBudgetExceededError,
)
from lite_cog_agent.core.agent_state import AgentState
from lite_cog_agent.core.messages import (
    MessageHistoryError,
    validate_message_history,
)
from lite_cog_agent.core.prompts import AGENT_SYSTEM_PROMPT
from lite_cog_agent.core.tool_calling import (
    ToolCallingExecution,
    ToolCallingService,
)
from lite_cog_agent.llm.schemas import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
)


class AgentLoopError(RuntimeError):
    """Raised when AgentLoop receives an invalid execution result."""


class AgentLoop:
    """Manage the lifecycle of one Agent request.

    AgentLoop prepares the conversation history, delegates all
    LLM and tool-calling operations to ToolCallingService, and
    commits the completed message history atomically.
    """

    def __init__(
        self,
        tool_calling_service: ToolCallingService,
        state: AgentState,
        system_prompt: str = AGENT_SYSTEM_PROMPT,
        budget: AgentBudget | None = None,
    ) -> None:
        if not callable(
            getattr(tool_calling_service, "run_with_history", None)
        ):
            raise TypeError(
                "tool_calling_service must provide run_with_history()."
            )

        if not isinstance(state, AgentState):
            raise TypeError(
                "state must be an AgentState instance."
            )

        if (
            not isinstance(system_prompt, str)
            or not system_prompt.strip()
        ):
            raise ValueError(
                "system_prompt must be a non-empty string."
            )

        self._tool_calling_service = tool_calling_service
        self._state = state
        self._system_prompt = system_prompt

        if budget is None:
            budget = AgentBudget()

        if not isinstance(budget, AgentBudget):
            raise TypeError("budget must be an AgentBudget instance.")

        self._budget = budget

    @property
    def session_id(self) -> str:
        """Return the current session identifier."""

        return self._state.session_id

    def get_messages(self) -> list[LLMMessage]:
        """Return a deep copy of the current conversation history."""

        return self._state.message_manager.get_messages()

    async def run(self, user_input: str) -> LLMResponse:
        """Execute one user request.

        The conversation history is committed only after the
        complete Tool Calling execution succeeds.

        Raises:
            TypeError: If user_input is not a string.
            ValueError: If user_input is empty.
            AgentStateError: If the state is already running.
            AgentLoopError: If the execution result is invalid.
        """

        self._validate_user_input(user_input)

        self._state.start()

        try:
            message_manager = self._state.message_manager

            # get_messages() already returns a deep copy.
            # All modifications below are isolated from persistent history.
            original_messages = message_manager.get_messages()

            working_messages = self._prepare_messages(
                original_messages,
                user_input,
            )

            expected_prefix = deepcopy(working_messages)

            request = LLMRequest(
                messages=working_messages,
            )

            execution = (
                await self._tool_calling_service.run_with_history(
                    request,
                    on_iteration=self._state.increment_iteration,
                    before_llm_request=self._check_iteration_budget,
                    before_tool_dispatch=self._reserve_tool_call_budget,
                )
            )

            new_messages = self._validate_execution(
                execution=execution,
                expected_prefix=expected_prefix,
                original_history_length=len(original_messages),
            )

            # Validate the complete candidate history before committing.
            candidate_history = (
                original_messages + new_messages
            )

            try:
                validate_message_history(candidate_history)
            except MessageHistoryError as exc:
                raise AgentLoopError(
                    "Execution produced an invalid message history."
                ) from exc

            # MessageManager.add_messages() performs an atomic append.
            message_manager.add_messages(new_messages)

            return execution.response

        finally:
            self._state.finish()

    def reset(self) -> None:
        """Reset the session while preserving its session identifier."""

        self._state.reset()

    def _check_iteration_budget(self) -> None:
        """Ensure another LLM request is within the iteration budget."""

        used = self._state.iteration_count
        limit = self._budget.max_iterations

        if used >= limit:
            raise AgentBudgetExceededError(
                resource="iterations",
                limit=limit,
                used=used,
            )


    def _reserve_tool_call_budget(self, count: int) -> None:
        """Reserve a complete tool-call batch within the budget."""

        used = self._state.tool_call_count
        limit = self._budget.max_tool_calls

        if used + count > limit:
            raise AgentBudgetExceededError(
                resource="tool_calls",
                limit=limit,
                used=used,
                requested=count,
            )

        self._state.increment_tool_calls(count)


    def _prepare_messages(
        self,
        messages: list[LLMMessage],
        user_input: str,
    ) -> list[LLMMessage]:
        """Prepare an isolated working history for one request."""

        working_messages = deepcopy(messages)

        if not working_messages:
            working_messages.append(
                LLMMessage(
                    role="system",
                    content=self._system_prompt,
                )
            )
        else:
            system_indices = [
                index
                for index, message in enumerate(working_messages)
                if message.role == "system"
            ]

            if (
                system_indices != [0]
                or working_messages[0].content != self._system_prompt
            ):
                raise AgentLoopError(
                    "Existing non-empty message history must contain "
                    "exactly one matching system prompt at index 0."
                )

        working_messages.append(
            LLMMessage(
                role="user",
                content=user_input,
            )
        )

        return working_messages

    def _validate_execution(
        self,
        *,
        execution: ToolCallingExecution,
        expected_prefix: list[LLMMessage],
        original_history_length: int,
    ) -> list[LLMMessage]:
        """Validate execution and extract messages added by this request."""

        if not isinstance(execution, ToolCallingExecution):
            raise AgentLoopError(
                "Tool Calling service must return a "
                "ToolCallingExecution instance."
            )

        if not isinstance(execution.response, LLMResponse):
            raise AgentLoopError(
                "Tool Calling execution response must be an "
                "LLMResponse instance."
            )

        if not isinstance(execution.messages, list):
            raise AgentLoopError(
                "Tool Calling execution messages must be a list."
            )

        returned_messages = execution.messages

        if len(returned_messages) <= len(expected_prefix):
            raise AgentLoopError(
                "Tool Calling execution did not append a final response."
            )

        if returned_messages[:len(expected_prefix)] != expected_prefix:
            raise AgentLoopError(
                "Tool Calling execution modified the original "
                "message history prefix."
            )

        try:
            validate_message_history(returned_messages)
        except MessageHistoryError as exc:
            raise AgentLoopError(
                "Tool Calling execution returned an invalid "
                "message history."
            ) from exc

        final_message = returned_messages[-1]

        if (
            final_message.role != "assistant"
            or final_message.tool_calls
            or final_message.content != execution.response.content
            or execution.response.tool_calls
        ):
            raise AgentLoopError(
                "Tool Calling execution did not return a valid "
                "final assistant response."
            )

        # On the first request, original_history_length is zero,
        # so the system prompt is committed together with the user
        # message and the execution results.
        #
        # On subsequent requests, the existing history is preserved
        # and only the newly appended messages are committed.
        return returned_messages[original_history_length:]


    @staticmethod
    def _validate_user_input(user_input: str) -> None:
        """Validate user input before starting the Agent state."""

        if not isinstance(user_input, str):
            raise TypeError("user_input must be a string.")

        if not user_input.strip():
            raise ValueError("user_input cannot be empty.")
