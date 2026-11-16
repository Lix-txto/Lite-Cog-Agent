"""
Exception hierarchy for the tool infrastructure.
"""



class ToolError(Exception):
    """Base exception for tool-related errors."""


class ToolSchemaValidationError(ToolError):
    """Raised when a tool's JSON Schema is invalid."""


class ToolParameterValidationError(ToolError):
    """Raised when tool arguments do not match the JSON Schema."""


class ToolExecutionError(ToolError):
    """Raised when a tool fails during execution."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name


class ToolExecutionTimeoutError(ToolExecutionError):
    """Raised when a tool execution exceeds its timeout."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
        timeout_seconds: float,
    ) -> None:
        super().__init__(
            message,
            tool_name=tool_name,
        )
        self.timeout_seconds = timeout_seconds
