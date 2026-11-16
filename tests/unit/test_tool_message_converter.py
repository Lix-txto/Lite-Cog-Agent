import json

import pytest

from lite_cog_agent.llm.schemas import LLMMessage, ToolCall
from lite_cog_agent.tools.message_converter import (
    convert_tool_results_to_messages,
)
from lite_cog_agent.tools.schemas import ToolResult


def create_tool_call(
    call_id: str = "call_001",
    name: str = "test_tool",
) -> ToolCall:
    return ToolCall(
        id=call_id,
        name=name,
        arguments={"value": 10},
    )


def test_convert_success_result_to_tool_message():
    tool_call = create_tool_call()

    tool_result = ToolResult(
        success=True,
        output={"rows": 100, "columns": 5},
        metadata={"execution_time_seconds": 0.15},
    )

    messages = convert_tool_results_to_messages(
        [tool_call],
        [tool_result],
    )

    assert len(messages) == 1

    message = messages[0]

    assert isinstance(message, LLMMessage)
    assert message.role == "tool"
    assert message.tool_call_id == "call_001"

    payload = json.loads(message.content)

    assert payload["success"] is True
    assert payload["output"] == {
        "rows": 100,
        "columns": 5,
    }
    assert payload["error"] is None
    assert payload["metadata"] == {
        "execution_time_seconds": 0.15,
    }


def test_convert_failed_result_to_tool_message():
    tool_call = create_tool_call()

    tool_result = ToolResult.fail(
        error="Dataset not found.",
        metadata={"error_type": "FileNotFoundError"},
    )

    messages = convert_tool_results_to_messages(
        [tool_call],
        [tool_result],
    )

    assert len(messages) == 1

    message = messages[0]

    assert message.role == "tool"
    assert message.tool_call_id == "call_001"

    payload = json.loads(message.content)

    assert payload["success"] is False
    assert payload["error"] == "Dataset not found."
    assert payload["metadata"]["error_type"] == (
        "FileNotFoundError"
    )


def test_convert_multiple_results_preserves_call_id_mapping():
    tool_calls = [
        create_tool_call("call_001", "tool_a"),
        create_tool_call("call_002", "tool_b"),
        create_tool_call("call_003", "tool_c"),
    ]

    tool_results = [
        ToolResult.ok(output={"value": 1}),
        ToolResult.ok(output={"value": 2}),
        ToolResult.ok(output={"value": 3}),
    ]

    messages = convert_tool_results_to_messages(
        tool_calls,
        tool_results,
    )

    assert len(messages) == 3

    assert [
        message.tool_call_id
        for message in messages
    ] == [
        "call_001",
        "call_002",
        "call_003",
    ]

    assert [
        json.loads(message.content)["output"]["value"]
        for message in messages
    ] == [1, 2, 3]


def test_convert_mixed_success_and_failure_results():
    tool_calls = [
        create_tool_call("call_001"),
        create_tool_call("call_002"),
    ]

    tool_results = [
        ToolResult.ok(output={"value": 100}),
        ToolResult.fail(error="Calculation failed."),
    ]

    messages = convert_tool_results_to_messages(
        tool_calls,
        tool_results,
    )

    assert len(messages) == 2

    first_payload = json.loads(messages[0].content)
    second_payload = json.loads(messages[1].content)

    assert first_payload["success"] is True
    assert second_payload["success"] is False

    assert messages[0].tool_call_id == "call_001"
    assert messages[1].tool_call_id == "call_002"


def test_convert_empty_lists_returns_empty_list():
    messages = convert_tool_results_to_messages([], [])

    assert messages == []


def test_convert_rejects_mismatched_list_lengths():
    tool_calls = [
        create_tool_call("call_001"),
        create_tool_call("call_002"),
    ]

    tool_results = [
        ToolResult.ok(output={"value": 1}),
    ]

    with pytest.raises(
        ValueError,
        match="must have the same length",
    ):
        convert_tool_results_to_messages(
            tool_calls,
            tool_results,
        )


def test_convert_rejects_non_list_tool_calls():
    with pytest.raises(
        TypeError,
        match="tool_calls must be a list",
    ):
        convert_tool_results_to_messages(
            None,
            [],
        )


def test_convert_rejects_non_list_tool_results():
    with pytest.raises(
        TypeError,
        match="tool_results must be a list",
    ):
        convert_tool_results_to_messages(
            [],
            None,
        )


def test_convert_rejects_invalid_tool_call_item():
    with pytest.raises(
        TypeError,
        match="Every item in tool_calls",
    ):
        convert_tool_results_to_messages(
            ["invalid"],
            [ToolResult.ok(output={"value": 1})],
        )


def test_convert_rejects_invalid_tool_result_item():
    with pytest.raises(
        TypeError,
        match="Every item in tool_results",
    ):
        convert_tool_results_to_messages(
            [create_tool_call()],
            ["invalid"],
        )


def test_convert_preserves_chinese_content():
    tool_call = create_tool_call()

    tool_result = ToolResult.fail(
        error="数据集文件不存在",
    )

    messages = convert_tool_results_to_messages(
        [tool_call],
        [tool_result],
    )

    assert "数据集文件不存在" in messages[0].content

    payload = json.loads(messages[0].content)

    assert payload["error"] == "数据集文件不存在"


def test_convert_stringifies_non_json_serializable_output():
    tool_call = create_tool_call()

    tool_result = ToolResult.ok(
        output={"path": object()},
    )

    messages = convert_tool_results_to_messages(
        [tool_call],
        [tool_result],
    )

    payload = json.loads(messages[0].content)

    assert isinstance(payload["output"]["path"], str)



def test_convert_rejects_empty_tool_call_id():
    tool_call = create_tool_call(call_id="")

    with pytest.raises(
        ValueError,
        match="ToolCall.id must be a non-empty string",
    ):
        convert_tool_results_to_messages(
            [tool_call],
            [ToolResult.ok(output={"value": 1})],
        )


def test_convert_rejects_whitespace_tool_call_id():
    tool_call = create_tool_call(call_id="   ")

    with pytest.raises(
        ValueError,
        match="ToolCall.id must be a non-empty string",
    ):
        convert_tool_results_to_messages(
            [tool_call],
            [ToolResult.ok(output={"value": 1})],
        )


def test_convert_rejects_duplicate_tool_call_ids():
    tool_calls = [
        create_tool_call(call_id="call_001"),
        create_tool_call(call_id="call_001"),
    ]

    tool_results = [
        ToolResult.ok(output={"value": 1}),
        ToolResult.ok(output={"value": 2}),
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate tool call ID",
    ):
        convert_tool_results_to_messages(
            tool_calls,
            tool_results,
        )


def test_convert_rejects_empty_tool_name():
    tool_call = create_tool_call(name="")

    with pytest.raises(
        ValueError,
        match="ToolCall.name must be a non-empty string",
    ):
        convert_tool_results_to_messages(
            [tool_call],
            [ToolResult.ok(output={"value": 1})],
        )


def test_convert_rejects_whitespace_tool_name():
    tool_call = create_tool_call(name="   ")

    with pytest.raises(
        ValueError,
        match="ToolCall.name must be a non-empty string",
    ):
        convert_tool_results_to_messages(
            [tool_call],
            [ToolResult.ok(output={"value": 1})],
        )


def test_convert_rejects_non_dictionary_arguments():
    tool_call = create_tool_call()
    tool_call.arguments = ["invalid"]

    with pytest.raises(
        TypeError,
        match="ToolCall.arguments must be a dictionary",
    ):
        convert_tool_results_to_messages(
            [tool_call],
            [ToolResult.ok(output={"value": 1})],
        )
