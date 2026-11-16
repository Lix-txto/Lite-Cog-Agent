"""Configuration for multi-round tool calling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCallingConfig:
    """Configuration for controlling tool-calling rounds."""

    max_rounds: int = 10
    max_tool_calls_per_round: int = 10

    def __post_init__(self) -> None:
        """Validate tool-calling configuration."""

        if isinstance(self.max_rounds, bool) or not isinstance(
            self.max_rounds, int
        ):
            raise TypeError("max_rounds must be an integer.")

        if self.max_rounds <= 0:
            raise ValueError("max_rounds must be greater than 0.")

        if isinstance(
            self.max_tool_calls_per_round, bool
        ) or not isinstance(self.max_tool_calls_per_round, int):
            raise TypeError(
                "max_tool_calls_per_round must be an integer."
            )

        if self.max_tool_calls_per_round <= 0:
            raise ValueError(
                "max_tool_calls_per_round must be greater than 0."
            )
