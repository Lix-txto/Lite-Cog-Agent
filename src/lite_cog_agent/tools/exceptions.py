"""
Exception hierarchy for the tool infrastructure.
"""


class ToolError(Exception):
    """Base exception for tool-related errors."""


class ToolSchemaValidationError(ToolError):
    """Raised when a tool's JSON Schema is invalid."""


class ToolParameterValidationError(ToolError):
    """Raised when tool arguments do not match the JSON Schema."""
