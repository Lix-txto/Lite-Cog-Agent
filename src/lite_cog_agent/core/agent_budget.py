"""Execution budget configuration for Agent."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentBudget:
    """Configure resource limits for one Agent request."""

    max_iterations: int = 20
    max_tool_calls: int = 30

    def __post_init__(self) -> None:
        """Validate Agent budget configuration."""

        if isinstance(self.max_iterations, bool) or not isinstance(
            self.max_iterations, int
        ):
            raise TypeError("max_iterations must be an integer.")

        if self.max_iterations <= 0:
            raise ValueError(
                "max_iterations must be greater than 0."
            )

        if isinstance(self.max_tool_calls, bool) or not isinstance(
            self.max_tool_calls, int
        ):
            raise TypeError("max_tool_calls must be an integer.")

        if self.max_tool_calls <= 0:
            raise ValueError(
                "max_tool_calls must be greater than 0."
            )


class AgentBudgetExceededError(RuntimeError):
    """Raised when an Agent execution budget is exhausted."""

    def __init__(
        self,
        *,
        resource: str,
        limit: int,
        used: int,
        requested: int = 1,
    ) -> None:
        self.resource = resource
        self.limit = limit
        self.used = used
        self.requested = requested

        super().__init__(
            f"Agent {resource} budget exceeded: "
            f"limit={limit}, used={used}, requested={requested}."
        )
