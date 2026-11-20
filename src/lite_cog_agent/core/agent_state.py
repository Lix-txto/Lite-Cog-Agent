"""Agent session state and lifecycle management."""

from __future__ import annotations

from uuid import uuid4

from lite_cog_agent.core.messages import MessageManager


class AgentStateError(RuntimeError):
    """Raised when an Agent state transition is invalid."""


class AgentState:
    """Manage the lifecycle and runtime state of one Agent session.

    AgentState owns one MessageManager as the single source of truth
    for conversation history.

    The state object does not execute LLM requests, dispatch tools,
    persist sessions, or enforce iteration limits.
    """

    __slots__ = (
        "_session_id",
        "_message_manager",
        "_iteration_count",
        "_is_running",
        "_tool_call_count",
    )

    def __init__(
        self,
        session_id: str | None = None,
        message_manager: MessageManager | None = None,
    ) -> None:
        if session_id is None:
            session_id = str(uuid4())

        if (
            not isinstance(session_id, str)
            or not session_id.strip()
        ):
            raise ValueError(
                "session_id must be a non-empty string."
            )

        if message_manager is None:
            message_manager = MessageManager()

        if not isinstance(message_manager, MessageManager):
            raise TypeError(
                "message_manager must be a MessageManager instance."
            )

        self._session_id = session_id
        self._message_manager = message_manager
        self._iteration_count = 0
        self._tool_call_count = 0
        self._is_running = False

    @property
    def session_id(self) -> str:
        """Return the unique session identifier."""

        return self._session_id

    @property
    def message_manager(self) -> MessageManager:
        """Return the session's message manager."""

        return self._message_manager

    @property
    def iteration_count(self) -> int:
        """Return the iteration count of the current request."""

        return self._iteration_count

    @property
    def tool_call_count(self) -> int:
        """Return the tool-call count of the current request."""

        return self._tool_call_count

    @property
    def is_running(self) -> bool:
        """Return whether the Agent is currently running."""

        return self._is_running

    def start(self) -> None:
        """Start a new request within this session.

        Raises:
            AgentStateError: If the session is already running.
        """

        if self._is_running:
            raise AgentStateError(
                "Agent state is already running."
            )

        self._iteration_count = 0
        self._tool_call_count = 0
        self._is_running = True

    def increment_iteration(self) -> int:
        """Increment and return the current request iteration count.

        Raises:
            AgentStateError: If the session is not running.
        """

        if not self._is_running:
            raise AgentStateError(
                "Cannot increment iteration when Agent is not running."
            )

        self._iteration_count += 1
        return self._iteration_count

    def increment_tool_calls(self, count: int) -> int:
        """Record tool calls reserved for dispatch.

        Raises:
            AgentStateError: If the state is not running.
            TypeError: If count is not a positive integer.
            ValueError: If count is not greater than zero.
        """

        if not self._is_running:
            raise AgentStateError(
                "Cannot increment tool calls when Agent is not running."
            )

        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("count must be an integer.")

        if count <= 0:
            raise ValueError("count must be greater than 0.")

        self._tool_call_count += count
        return self._tool_call_count

    def finish(self) -> None:
        """Finish the current request.

        This operation is idempotent so it can safely be called
        during cleanup, including from a finally block.
        """

        self._is_running = False

    def reset(self) -> None:
        """Clear session messages and reset the iteration count.

        The session identifier is intentionally preserved.

        Raises:
            AgentStateError: If the session is currently running.
        """

        if self._is_running:
            raise AgentStateError(
                "Cannot reset Agent state while it is running."
            )
        self._message_manager.clear()
        self._iteration_count = 0
        self._tool_call_count = 0
