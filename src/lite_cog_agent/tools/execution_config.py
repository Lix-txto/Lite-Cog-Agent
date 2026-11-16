from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolExecutionConfig:
    """Configuration for tool execution control.

    Attributes:
        timeout_seconds: Maximum execution time for one tool call.
        max_concurrency: Maximum number of concurrent tool executions.
    """

    timeout_seconds: float = 30.0
    max_concurrency: int = 5

    def __post_init__(self) -> None:
        """Validate execution configuration."""

        # Validate timeout_seconds.
        if isinstance(self.timeout_seconds, bool) or not isinstance(
            self.timeout_seconds, (int, float)
        ):
            raise TypeError(
                "timeout_seconds must be an integer or float."
            )

        if not math.isfinite(self.timeout_seconds):
            raise ValueError(
                "timeout_seconds must be a finite number."
            )

        if self.timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than 0."
            )

        # Validate max_concurrency.
        if isinstance(self.max_concurrency, bool) or not isinstance(
            self.max_concurrency, int
        ):
            raise TypeError(
                "max_concurrency must be an integer."
            )

        if self.max_concurrency <= 0:
            raise ValueError(
                "max_concurrency must be greater than 0."
            )
