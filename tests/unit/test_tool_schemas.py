import pytest

from lite_cog_agent.tools.schemas import ToolResult


def test_create_successful_tool_result():
    result = ToolResult.ok(
        output={"rows": 100, "columns": 5}
    )

    assert result.success is True
    assert result.output == {"rows": 100, "columns": 5}
    assert result.error is None
    assert result.metadata == {}


def test_create_failed_tool_result():
    result = ToolResult.fail(
        error="Dataset file does not exist."
    )

    assert result.success is False
    assert result.output is None
    assert result.error == "Dataset file does not exist."


def test_create_tool_result_with_metadata():
    result = ToolResult.ok(
        output={"rows": 100},
        metadata={
            "source": "test.csv",
            "execution_time_ms": 12,
        },
    )

    assert result.metadata["source"] == "test.csv"
    assert result.metadata["execution_time_ms"] == 12


def test_successful_result_cannot_contain_error():
    with pytest.raises(ValueError):
        ToolResult(
            success=True,
            error="Unexpected error.",
        )


def test_failed_result_requires_error():
    with pytest.raises(ValueError):
        ToolResult(
            success=False,
        )


def test_success_must_be_boolean():
    with pytest.raises(TypeError):
        ToolResult(
            success="yes",
        )


def test_metadata_must_be_dictionary():
    with pytest.raises(TypeError):
        ToolResult(
            success=True,
            metadata=["invalid"],
        )


def test_fail_rejects_empty_error():
    with pytest.raises(ValueError):
        ToolResult.fail("")


def test_fail_rejects_whitespace_error():
    with pytest.raises(ValueError):
        ToolResult.fail("   ")
