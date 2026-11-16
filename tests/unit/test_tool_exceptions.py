import pytest

from lite_cog_agent.tools.exceptions import (
    ToolError,
    ToolExecutionError,
    ToolExecutionTimeoutError,
    ToolParameterValidationError,
    ToolSchemaValidationError,
)


def test_tool_error_is_base_exception():
    error = ToolError("Tool error.")

    assert isinstance(error, Exception)
    assert str(error) == "Tool error."


def test_schema_validation_error_inherits_tool_error():
    error = ToolSchemaValidationError("Invalid schema.")

    assert isinstance(error, ToolError)
    assert str(error) == "Invalid schema."


def test_parameter_validation_error_inherits_tool_error():
    error = ToolParameterValidationError("Invalid arguments.")

    assert isinstance(error, ToolError)
    assert str(error) == "Invalid arguments."


def test_execution_error_inherits_tool_error():
    error = ToolExecutionError(
        "Tool execution failed.",
        tool_name="analyze_dataset",
    )

    assert isinstance(error, ToolError)
    assert error.tool_name == "analyze_dataset"
    assert str(error) == "Tool execution failed."


def test_execution_timeout_error_inherits_execution_error():
    error = ToolExecutionTimeoutError(
        "Tool execution timed out.",
        tool_name="analyze_dataset",
        timeout_seconds=30.0,
    )

    assert isinstance(error, ToolExecutionError)
    assert isinstance(error, ToolError)
    assert error.tool_name == "analyze_dataset"
    assert error.timeout_seconds == 30.0
    assert str(error) == "Tool execution timed out."


def test_execution_timeout_error_can_be_caught_as_execution_error():
    with pytest.raises(ToolExecutionError):
        raise ToolExecutionTimeoutError(
            "Tool execution timed out.",
            tool_name="analyze_dataset",
            timeout_seconds=30.0,
        )
