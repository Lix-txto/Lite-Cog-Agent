"""
Unified schemas for tool execution.

These schemas define the result contract shared by
all tools in Lite-Cog-Agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolResult:
    """
    Standardized result returned by a tool execution.

    Attributes:
        success: Whether the tool execution succeeded.
        output: The actual tool output.
        error: Error description when execution fails.
        metadata: Additional execution information.
    """

    success: bool
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the result state."""

        if not isinstance(self.success, bool):
            raise TypeError("ToolResult.success must be a boolean.")

        if self.success and self.error is not None:
            raise ValueError(
                "A successful ToolResult cannot contain an error."
            )

        if not self.success and not self.error:
            raise ValueError(
                "A failed ToolResult must contain an error message."
            )

        if not isinstance(self.metadata, dict):
            raise TypeError("ToolResult.metadata must be a dictionary.")

    @classmethod
    def ok(
        cls,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Create a successful tool result."""

        return cls(
            success=True,
            output=output,
            metadata=metadata or {},
        )

    @classmethod
    def fail(
        cls,
        error: str,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Create a failed tool result."""

        if not isinstance(error, str) or not error.strip():
            raise ValueError("Tool error message cannot be empty.")

        return cls(
            success=False,
            error=error,
            metadata=metadata or {},
        )
