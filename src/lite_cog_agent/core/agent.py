"""Public interface for the Lite-Cog-Agent."""

from __future__ import annotations

from lite_cog_agent.core.agent_loop import AgentLoop
from lite_cog_agent.llm.schemas import LLMResponse


class Agent:
    """Expose a simple interface for interacting with an Agent session."""

    def __init__(self, loop: AgentLoop) -> None:
        if not callable(getattr(loop, "run", None)):
            raise TypeError("loop must provide run().")

        if not callable(getattr(loop, "reset", None)):
            raise TypeError("loop must provide reset().")

        if not isinstance(
            getattr(loop, "session_id", None),
            str,
        ):
            raise TypeError("loop must expose a string session_id.")

        self._loop = loop

    @property
    def session_id(self) -> str:
        """Return the current session identifier."""

        return self._loop.session_id

    async def run(self, user_input: str) -> LLMResponse:
        """Submit a user request and return the final LLM response."""

        return await self._loop.run(user_input)

    def reset(self) -> None:
        """Reset the current session."""

        self._loop.reset()
