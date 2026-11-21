import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from lite_cog_agent.core.tool_calling import (
    ToolCallingExecution,
    ToolCallingProtocolError,
    ToolCallingService,
)
from lite_cog_agent.core.tool_calling_config import (
    ToolCallingConfig,
)
from lite_cog_agent.llm.schemas import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    ToolCall,
    ToolDefinition,
)
from lite_cog_agent.tools.schemas import ToolResult


def make_tool_call(
    call_id: str = "call_001",
    name: str = "inspect_file",
) -> ToolCall:
    return ToolCall(
        id=call_id,
        name=name,
        arguments={"file_path": "sales.xlsx"},
    )


def make_tool_result(value: str = "success") -> ToolResult:
    return ToolResult(
        success=True,
        output={"result": value},
    )


def make_tool_definition() -> ToolDefinition:
    return ToolDefinition(
        name="inspect_file",
        description="Inspect a data file.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
            },
            "required": ["file_path"],
            "additionalProperties": False,
        },
    )


def make_request() -> LLMRequest:
    return LLMRequest(
        messages=[
            LLMMessage(
                role="user",
                content="Analyze sales.xlsx.",
            )
        ],
        model="deepseek-chat",
        temperature=0.2,
        max_tokens=2048,
    )


def make_service(
    responses: list[LLMResponse],
    *,
    config: ToolCallingConfig | None = None,
    tool_definitions: list[ToolDefinition] | None = None,
    tool_results: list[ToolResult] | None = None,
) -> tuple[ToolCallingService, AsyncMock, MagicMock, AsyncMock]:
    gateway = MagicMock()
    gateway.chat = AsyncMock(side_effect=responses)

    registry = MagicMock()
    registry.get_definitions.return_value = (
        tool_definitions
        if tool_definitions is not None
        else [make_tool_definition()]
    )

    dispatcher = MagicMock()
    dispatcher.dispatch_many = AsyncMock(
        side_effect=lambda calls: (
            tool_results
            if tool_results is not None
            else [make_tool_result() for _ in calls]
        )
    )

    service = ToolCallingService(
        gateway=gateway,
        registry=registry,
        dispatcher=dispatcher,
        config=config,
    )

    return service, gateway.chat, registry, dispatcher


@pytest.mark.asyncio
async def test_run_returns_normal_text_without_tool_calls() -> None:
    response = LLMResponse(
        content="The analysis is complete.",
    )

    service, chat, registry, dispatcher = make_service(
        [response]
    )

    result = await service.run(make_request())

    assert result is response
    assert chat.await_count == 1
    registry.get_definitions.assert_called_once()
    dispatcher.dispatch_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_executes_single_tool_call_then_returns_final_answer() -> None:
    tool_call = make_tool_call()

    tool_response = LLMResponse(
        content="I will inspect the file.",
        tool_calls=[tool_call],
    )

    final_response = LLMResponse(
        content="The file contains sales data.",
    )

    service, chat, _, dispatcher = make_service(
        [tool_response, final_response]
    )

    result = await service.run(make_request())

    assert result is final_response
    assert chat.await_count == 2

    dispatcher.dispatch_many.assert_awaited_once_with(
        [tool_call]
    )

    first_request = chat.await_args_list[0].args[0]
    second_request = chat.await_args_list[1].args[0]

    assert len(first_request.tools) == 1

    # The second request is still within the configured tool-calling
    # rounds, so tool definitions must remain available.
    assert second_request.tools == first_request.tools


@pytest.mark.asyncio
async def test_run_supports_multiple_tool_calling_rounds() -> None:
    first_call = make_tool_call(
        call_id="call_001",
        name="inspect_file",
    )

    second_call = make_tool_call(
        call_id="call_002",
        name="analyze_data",
    )

    first_response = LLMResponse(
        content="First inspect the file.",
        tool_calls=[first_call],
    )

    second_response = LLMResponse(
        content="Now analyze the data.",
        tool_calls=[second_call],
    )

    final_response = LLMResponse(
        content="Analysis completed.",
    )

    service, chat, _, dispatcher = make_service(
        [
            first_response,
            second_response,
            final_response,
        ],
        config=ToolCallingConfig(max_rounds=2),
    )

    result = await service.run(make_request())

    assert result is final_response
    assert chat.await_count == 3
    assert dispatcher.dispatch_many.await_count == 2

    final_request = chat.await_args_list[2].args[0]

    assert [message.role for message in final_request.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]

    assert final_request.messages[2].tool_call_id == "call_001"
    assert final_request.messages[4].tool_call_id == "call_002"
    assert final_request.tools == []


@pytest.mark.asyncio
async def test_run_preserves_multiple_tool_call_ids() -> None:
    first_call = make_tool_call(
        call_id="call_001",
    )

    second_call = make_tool_call(
        call_id="call_002",
    )

    tool_response = LLMResponse(
        content="Run both tools.",
        tool_calls=[
            first_call,
            second_call,
        ],
    )

    final_response = LLMResponse(
        content="Both tools completed.",
    )

    service, chat, _, dispatcher = make_service(
        [tool_response, final_response]
    )

    result = await service.run(make_request())

    assert result is final_response

    dispatcher.dispatch_many.assert_awaited_once_with(
        [first_call, second_call]
    )

    final_request = chat.await_args_list[1].args[0]

    assistant_message = final_request.messages[1]
    first_tool_message = final_request.messages[2]
    second_tool_message = final_request.messages[3]

    assert assistant_message.role == "assistant"
    assert [
        call.id for call in assistant_message.tool_calls
    ] == ["call_001", "call_002"]

    assert first_tool_message.tool_call_id == "call_001"
    assert second_tool_message.tool_call_id == "call_002"


@pytest.mark.asyncio
async def test_run_performs_final_synthesis_after_max_rounds() -> None:
    first_call = make_tool_call(
        call_id="call_001",
    )

    second_call = make_tool_call(
        call_id="call_002",
    )

    responses = [
        LLMResponse(tool_calls=[first_call]),
        LLMResponse(tool_calls=[second_call]),
        LLMResponse(content="Final synthesis."),
    ]

    service, chat, _, dispatcher = make_service(
        responses,
        config=ToolCallingConfig(max_rounds=2),
    )

    result = await service.run(make_request())

    assert result.content == "Final synthesis."
    assert chat.await_count == 3
    assert dispatcher.dispatch_many.await_count == 2

    final_request = chat.await_args_list[-1].args[0]

    assert final_request.tools == []
    assert final_request.messages[-1].role == "tool"


@pytest.mark.asyncio
async def test_run_rejects_oversized_batch_and_replans() -> None:
    rejected_calls = [
        make_tool_call(call_id="call_001"),
        make_tool_call(call_id="call_002"),
        make_tool_call(call_id="call_003"),
    ]

    retry_call = make_tool_call(
        call_id="call_004",
        name="inspect_file",
    )

    service, chat, _, dispatcher = make_service(
        [
            LLMResponse(
                content="I will inspect the file and analyze it.",
                tool_calls=rejected_calls,
            ),
            LLMResponse(
                content="I will retry with one tool call.",
                tool_calls=[retry_call],
            ),
            LLMResponse(
                content="Analysis completed.",
            ),
        ],
        config=ToolCallingConfig(
            max_rounds=2,
            max_tool_calls_per_round=2,
        ),
    )

    before_dispatch = MagicMock()

    result = await service.run_with_history(
        make_request(),
        before_tool_dispatch=before_dispatch,
    )

    assert result.response.content == "Analysis completed."
    assert chat.await_count == 3

    # The oversized batch must not reach the dispatcher.
    dispatcher.dispatch_many.assert_awaited_once_with(
        [retry_call]
    )

    # Only the valid retry consumes tool execution budget.
    before_dispatch.assert_called_once_with(1)

    # The second LLM request must contain the rejected batch
    # and one failed Tool Message for every rejected call.
    retry_request = chat.await_args_list[1].args[0]

    rejected_assistant = retry_request.messages[1]
    rejected_messages = retry_request.messages[2:5]

    assert rejected_assistant.role == "assistant"
    assert [
        call.id for call in rejected_assistant.tool_calls
    ] == [
        "call_001",
        "call_002",
        "call_003",
    ]

    assert len(rejected_messages) == 3

    for tool_call, tool_message in zip(
        rejected_calls,
        rejected_messages,
        strict=True,
    ):
        assert tool_message.role == "tool"
        assert tool_message.tool_call_id == tool_call.id

        payload = json.loads(tool_message.content)

        assert payload["success"] is False
        assert payload["metadata"]["reason"] == (
            "max_tool_calls_per_round_exceeded"
        )
        assert payload["metadata"]["executed"] is False
        assert "No tools in this batch were executed" in (
            payload["error"]
        )

    # Final synthesis must include the successful retry result.
    final_request = chat.await_args_list[2].args[0]

    assert final_request.tools == []
    assert final_request.messages[-1].role == "tool"
    assert final_request.messages[-1].tool_call_id == "call_004"


@pytest.mark.asyncio
async def test_run_bounds_repeated_oversized_batches() -> None:
    first_batch = [
        make_tool_call(call_id="call_001"),
        make_tool_call(call_id="call_002"),
    ]

    second_batch = [
        make_tool_call(call_id="call_003"),
        make_tool_call(call_id="call_004"),
    ]

    service, chat, _, dispatcher = make_service(
        [
            LLMResponse(tool_calls=first_batch),
            LLMResponse(tool_calls=second_batch),
            LLMResponse(content="Unable to complete the operation."),
        ],
        config=ToolCallingConfig(
            max_rounds=2,
            max_tool_calls_per_round=1,
        ),
    )

    result = await service.run(make_request())

    assert result.content == "Unable to complete the operation."

    # Two bounded tool-calling rounds and one final synthesis.
    assert chat.await_count == 3

    # Neither oversized batch may execute.
    dispatcher.dispatch_many.assert_not_awaited()

    final_request = chat.await_args_list[-1].args[0]

    assert final_request.tools == []

    # Both rejected batches must remain in conversation history.
    assert [
        message.tool_call_id
        for message in final_request.messages
        if message.role == "tool"
    ] == [
        "call_001",
        "call_002",
        "call_003",
        "call_004",
    ]


@pytest.mark.asyncio
async def test_run_does_not_modify_original_request_messages() -> None:
    tool_call = make_tool_call()

    service, _, _, _ = make_service(
        [
            LLMResponse(tool_calls=[tool_call]),
            LLMResponse(content="Completed."),
        ]
    )

    request = make_request()
    original_messages = list(request.messages)

    await service.run(request)

    assert request.messages == original_messages
    assert len(request.messages) == 1


@pytest.mark.asyncio
async def test_run_rejects_tool_calls_in_final_synthesis() -> None:
    tool_call = make_tool_call()

    service, _, _, _ = make_service(
        [
            LLMResponse(tool_calls=[tool_call]),
            LLMResponse(tool_calls=[make_tool_call("call_002")]),
        ],
        config=ToolCallingConfig(max_rounds=1),
    )

    with pytest.raises(
        ToolCallingProtocolError,
        match="Final synthesis response must not contain tool calls",
    ):
        await service.run(make_request())


@pytest.mark.asyncio
async def test_run_rejects_invalid_request_type() -> None:
    service, _, _, _ = make_service([])

    with pytest.raises(
        TypeError,
        match="request must be an LLMRequest",
    ):
        await service.run("invalid")


@pytest.mark.asyncio
async def test_run_rejects_request_without_messages() -> None:
    service, _, _, _ = make_service([])

    request = LLMRequest(messages=[])

    with pytest.raises(
        ValueError,
        match="at least one message",
    ):
        await service.run(request)


@pytest.mark.asyncio
async def test_run_rejects_invalid_gateway_response_type() -> None:
    service, _, _, dispatcher = make_service(
        ["invalid response"]
    )

    with pytest.raises(
        ToolCallingProtocolError,
        match="Gateway response must be an LLMResponse",
    ):
        await service.run(make_request())

    dispatcher.dispatch_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_rejects_non_list_tool_calls() -> None:
    response = LLMResponse()
    response.tool_calls = None

    service, _, _, dispatcher = make_service(
        [response]
    )

    with pytest.raises(
        ToolCallingProtocolError,
        match="tool_calls must be a list",
    ):
        await service.run(make_request())

    dispatcher.dispatch_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_rejects_invalid_tool_call_item() -> None:
    response = LLMResponse(
        tool_calls=["invalid"],
    )

    service, _, _, dispatcher = make_service(
        [response]
    )

    with pytest.raises(
        ToolCallingProtocolError,
        match="Every item in tool_calls must be a ToolCall",
    ):
        await service.run(make_request())

    dispatcher.dispatch_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_rejects_empty_tool_call_id() -> None:
    response = LLMResponse(
        tool_calls=[
            make_tool_call(call_id=""),
        ]
    )

    service, _, _, dispatcher = make_service(
        [response]
    )

    with pytest.raises(
        ToolCallingProtocolError,
        match="ToolCall.id must be a non-empty string",
    ):
        await service.run(make_request())

    dispatcher.dispatch_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_rejects_empty_tool_call_name() -> None:
    response = LLMResponse(
        tool_calls=[
            make_tool_call(name=""),
        ]
    )

    service, _, _, dispatcher = make_service(
        [response]
    )

    with pytest.raises(
        ToolCallingProtocolError,
        match="ToolCall.name must be a non-empty string",
    ):
        await service.run(make_request())

    dispatcher.dispatch_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_rejects_invalid_tool_call_arguments() -> None:
    tool_call = make_tool_call()
    tool_call.arguments = ["invalid"]

    response = LLMResponse(
        tool_calls=[tool_call],
    )

    service, _, _, dispatcher = make_service(
        [response]
    )

    with pytest.raises(
        ToolCallingProtocolError,
        match="ToolCall.arguments must be a dictionary",
    ):
        await service.run(make_request())

    dispatcher.dispatch_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_rejects_duplicate_tool_call_ids_in_same_round() -> None:
    response = LLMResponse(
        tool_calls=[
            make_tool_call(call_id="call_001"),
            make_tool_call(call_id="call_001"),
        ]
    )

    service, _, _, dispatcher = make_service(
        [response]
    )

    with pytest.raises(
        ToolCallingProtocolError,
        match="Duplicate tool call ID in current round",
    ):
        await service.run(make_request())

    dispatcher.dispatch_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_rejects_duplicate_tool_call_ids_across_rounds() -> None:
    first_response = LLMResponse(
        tool_calls=[
            make_tool_call(call_id="call_001"),
        ]
    )

    second_response = LLMResponse(
        tool_calls=[
            make_tool_call(call_id="call_001"),
        ]
    )

    service, chat, _, dispatcher = make_service(
        [
            first_response,
            second_response,
        ],
        config=ToolCallingConfig(max_rounds=2),
    )

    with pytest.raises(
        ToolCallingProtocolError,
        match="Duplicate tool call ID across rounds",
    ):
        await service.run(make_request())

    assert chat.await_count == 2

    # The first round is valid and executes once.
    # The second round must be rejected before dispatch.
    assert dispatcher.dispatch_many.await_count == 1


@pytest.mark.asyncio
async def test_run_rejects_invalid_final_synthesis_response_type() -> None:
    first_response = LLMResponse(
        tool_calls=[
            make_tool_call(call_id="call_001"),
        ]
    )

    service, _, _, dispatcher = make_service(
        [
            first_response,
            "invalid final response",
        ],
        config=ToolCallingConfig(max_rounds=1),
    )

    with pytest.raises(
        ToolCallingProtocolError,
        match="Gateway response must be an LLMResponse",
    ):
        await service.run(make_request())

    assert dispatcher.dispatch_many.await_count == 1


@pytest.mark.asyncio
async def test_run_rejects_invalid_message_history_item() -> None:
    request = make_request()
    request.messages = ["invalid message"]

    service, chat, _, dispatcher = make_service(
        [LLMResponse(content="Completed.")]
    )

    with pytest.raises(
        ToolCallingProtocolError,
        match="must be an LLMMessage instance",
    ):
        await service.run(request)

    chat.assert_not_awaited()
    dispatcher.dispatch_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_rejects_unsupported_message_role() -> None:
    request = LLMRequest(
        messages=[
            LLMMessage(
                role="invalid",
                content="Invalid role.",
            )
        ]
    )

    service, chat, _, dispatcher = make_service(
        [LLMResponse(content="Completed.")]
    )

    with pytest.raises(
        ToolCallingProtocolError,
        match="Unsupported message role",
    ):
        await service.run(request)

    chat.assert_not_awaited()
    dispatcher.dispatch_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_rejects_non_list_message_tool_calls() -> None:
    message = LLMMessage(
        role="assistant",
        content="Invalid history.",
    )
    message.tool_calls = None

    request = LLMRequest(messages=[message])

    service, chat, _, _ = make_service(
        [LLMResponse(content="Completed.")]
    )

    with pytest.raises(
        ToolCallingProtocolError,
        match="Message.tool_calls.*must be a list",
    ):
        await service.run(request)

    chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_rejects_tool_message_without_call_id() -> None:
    request = LLMRequest(
        messages=[
            LLMMessage(
                role="tool",
                content='{"success": true}',
            )
        ]
    )

    service, chat, _, _ = make_service(
        [LLMResponse(content="Completed.")]
    )

    with pytest.raises(
        ToolCallingProtocolError,
        match="non-empty tool_call_id",
    ):
        await service.run(request)

    chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_rejects_orphan_tool_message() -> None:
    request = LLMRequest(
        messages=[
            LLMMessage(
                role="tool",
                content='{"success": true}',
                tool_call_id="call_999",
            )
        ]
    )

    service, chat, _, _ = make_service(
        [LLMResponse(content="Completed.")]
    )

    with pytest.raises(
        ToolCallingProtocolError,
        match="unknown or already resolved",
    ):
        await service.run(request)

    chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_rejects_mismatched_tool_message_id() -> None:
    request = LLMRequest(
        messages=[
            LLMMessage(
                role="assistant",
                tool_calls=[
                    make_tool_call(call_id="call_001"),
                ],
            ),
            LLMMessage(
                role="tool",
                content='{"success": true}',
                tool_call_id="call_002",
            ),
        ]
    )

    service, chat, _, _ = make_service(
        [LLMResponse(content="Completed.")]
    )

    with pytest.raises(
        ToolCallingProtocolError,
        match="unknown or already resolved",
    ):
        await service.run(request)

    chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_rejects_unresolved_tool_calls_in_history() -> None:
    request = LLMRequest(
        messages=[
            LLMMessage(
                role="assistant",
                tool_calls=[
                    make_tool_call(call_id="call_001"),
                ],
            )
        ]
    )

    service, chat, _, _ = make_service(
        [LLMResponse(content="Completed.")]
    )

    with pytest.raises(
        ToolCallingProtocolError,
        match="unresolved Tool Calls",
    ):
        await service.run(request)

    chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_rejects_message_before_tool_results_are_completed() -> None:
    request = LLMRequest(
        messages=[
            LLMMessage(
                role="assistant",
                tool_calls=[
                    make_tool_call(call_id="call_001"),
                ],
            ),
            LLMMessage(
                role="user",
                content="Continue analysis.",
            ),
        ]
    )

    service, chat, _, _ = make_service(
        [LLMResponse(content="Completed.")]
    )

    with pytest.raises(
        ToolCallingProtocolError,
        match="unresolved results",
    ):
        await service.run(request)

    chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_rejects_duplicate_tool_call_ids_in_history() -> None:
    request = LLMRequest(
        messages=[
            LLMMessage(
                role="assistant",
                tool_calls=[
                    make_tool_call(call_id="call_001"),
                ],
            ),
            LLMMessage(
                role="tool",
                content='{"success": true}',
                tool_call_id="call_001",
            ),
            LLMMessage(
                role="assistant",
                tool_calls=[
                    make_tool_call(call_id="call_001"),
                ],
            ),
            LLMMessage(
                role="tool",
                content='{"success": true}',
                tool_call_id="call_001",
            ),
        ]
    )

    service, chat, _, _ = make_service(
        [LLMResponse(content="Completed.")]
    )

    with pytest.raises(
        ToolCallingProtocolError,
        match="Duplicate tool call ID in message history",
    ):
        await service.run(request)

    chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_rejects_tool_call_id_used_in_history() -> None:
    request = LLMRequest(
        messages=[
            LLMMessage(
                role="assistant",
                tool_calls=[
                    make_tool_call(call_id="call_001"),
                ],
            ),
            LLMMessage(
                role="tool",
                content='{"success": true}',
                tool_call_id="call_001",
            ),
            LLMMessage(
                role="user",
                content="Analyze another file.",
            ),
        ]
    )

    response = LLMResponse(
        tool_calls=[
            make_tool_call(call_id="call_001"),
        ]
    )

    service, chat, _, dispatcher = make_service(
        [response]
    )

    with pytest.raises(
        ToolCallingProtocolError,
        match="Duplicate tool call ID across rounds",
    ):
        await service.run(request)

    assert chat.await_count == 1
    dispatcher.dispatch_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_accepts_valid_multi_tool_message_history() -> None:
    request = LLMRequest(
        messages=[
            LLMMessage(
                role="user",
                content="Analyze sales.xlsx.",
            ),
            LLMMessage(
                role="assistant",
                content="I will inspect two aspects.",
                tool_calls=[
                    make_tool_call(call_id="call_001"),
                    make_tool_call(call_id="call_002"),
                ],
            ),
            LLMMessage(
                role="tool",
                content='{"success": true}',
                tool_call_id="call_002",
            ),
            LLMMessage(
                role="tool",
                content='{"success": true}',
                tool_call_id="call_001",
            ),
            LLMMessage(
                role="assistant",
                content="Both tools completed.",
            ),
        ]
    )

    response = LLMResponse(
        content="The analysis is complete.",
    )

    service, chat, _, dispatcher = make_service(
        [response]
    )

    result = await service.run(request)

    assert result is response
    assert chat.await_count == 1
    dispatcher.dispatch_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_with_history_returns_final_assistant_message() -> None:
    """A normal response is included in the returned conversation history."""

    response = LLMResponse(
        content="The analysis is complete.",
    )

    service, chat, _, dispatcher = make_service([response])
    request = make_request()

    execution = await service.run_with_history(request)

    assert isinstance(execution, ToolCallingExecution)
    assert execution.response is response

    assert len(execution.messages) == 2
    assert execution.messages[0] is request.messages[0]

    final_message = execution.messages[-1]
    assert final_message.role == "assistant"
    assert final_message.content == "The analysis is complete."
    assert final_message.tool_calls == []

    chat.assert_awaited_once()
    dispatcher.dispatch_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_with_history_preserves_tool_call_and_result_messages() -> None:
    """Tool calls and their results are retained in execution history."""

    tool_call = make_tool_call()

    final_response = LLMResponse(
        content="The file inspection is complete.",
    )

    service, chat, _, dispatcher = make_service(
        [
            LLMResponse(
                content="I will inspect the file.",
                tool_calls=[tool_call],
            ),
            final_response,
        ]
    )

    request = make_request()

    execution = await service.run_with_history(request)

    assert execution.response is final_response

    assert [message.role for message in execution.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]

    assistant_tool_message = execution.messages[1]

    assert assistant_tool_message.content == "I will inspect the file."
    assert len(assistant_tool_message.tool_calls) == 1
    assert assistant_tool_message.tool_calls[0].id == "call_001"

    tool_message = execution.messages[2]

    assert tool_message.tool_call_id == "call_001"
    assert tool_message.content is not None

    final_message = execution.messages[3]

    assert final_message.role == "assistant"
    assert final_message.content == "The file inspection is complete."

    assert chat.await_count == 2
    dispatcher.dispatch_many.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_with_history_preserves_final_synthesis_message() -> None:
    """Final synthesis is included after the maximum tool rounds."""

    final_response = LLMResponse(
        content="The final result is 42.",
    )

    service, chat, _, dispatcher = make_service(
        [
            LLMResponse(
                tool_calls=[
                    make_tool_call(),
                ]
            ),
            final_response,
        ],
        config=ToolCallingConfig(max_rounds=1),
    )

    execution = await service.run_with_history(make_request())

    assert execution.response is final_response

    assert [message.role for message in execution.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]

    assert execution.messages[-1].content == "The final result is 42."

    # The final synthesis request must not expose tools.
    final_request = chat.await_args_list[-1].args[0]
    assert final_request.tools == []

    dispatcher.dispatch_many.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_with_history_does_not_modify_original_request_messages() -> None:
    """Returning execution history does not mutate the original request."""

    request = make_request()
    original_messages = list(request.messages)

    service, _, _, _ = make_service(
        [
            LLMResponse(content="Completed."),
        ]
    )

    execution = await service.run_with_history(request)

    assert request.messages == original_messages
    assert request.messages is not execution.messages

    assert len(request.messages) == 1
    assert len(execution.messages) == 2

    # Existing message objects are shared but never modified.
    assert execution.messages[0] is request.messages[0]


@pytest.mark.asyncio
async def test_run_with_history_invokes_iteration_callback_once():
    response = LLMResponse(
        content="Analysis completed.",
    )

    service, chat, _, _ = make_service([response])

    callback = MagicMock()

    execution = await service.run_with_history(
        make_request(),
        on_iteration=callback,
    )

    assert execution.response is response
    assert chat.await_count == 1
    assert callback.call_count == 1


@pytest.mark.asyncio
async def test_run_with_history_counts_each_llm_response():
    first_call = make_tool_call(
        call_id="call_001",
    )

    second_call = make_tool_call(
        call_id="call_002",
    )

    responses = [
        LLMResponse(
            tool_calls=[first_call],
        ),
        LLMResponse(
            tool_calls=[second_call],
        ),
        LLMResponse(
            content="Final synthesis.",
        ),
    ]

    service, chat, _, dispatcher = make_service(
        responses,
        config=ToolCallingConfig(max_rounds=2),
    )

    callback = MagicMock()

    execution = await service.run_with_history(
        make_request(),
        on_iteration=callback,
    )

    assert execution.response.content == "Final synthesis."
    assert chat.await_count == 3
    assert dispatcher.dispatch_many.await_count == 2
    assert callback.call_count == 3


@pytest.mark.asyncio
async def test_run_with_history_does_not_invoke_callback_when_llm_fails():
    service, chat, _, _ = make_service([])

    chat.side_effect = RuntimeError(
        "LLM provider unavailable."
    )

    callback = MagicMock()

    with pytest.raises(
        RuntimeError,
        match="LLM provider unavailable",
    ):
        await service.run_with_history(
            make_request(),
            on_iteration=callback,
        )

    callback.assert_not_called()


@pytest.mark.asyncio
async def test_run_with_history_supports_missing_iteration_callback():
    response = LLMResponse(
        content="Analysis completed.",
    )

    service, _, _, _ = make_service([response])

    execution = await service.run_with_history(
        make_request()
    )

    assert execution.response is response



@pytest.mark.asyncio
async def test_before_llm_request_callback_blocks_gateway_call():
    """Budget callback must run before the LLM Gateway."""

    service, chat, _, dispatcher = make_service(
        [
            LLMResponse(content="Analysis completed."),
        ]
    )

    callback = MagicMock(
        side_effect=RuntimeError("Iteration budget exceeded.")
    )

    with pytest.raises(
        RuntimeError,
        match="Iteration budget exceeded",
    ):
        await service.run_with_history(
            make_request(),
            before_llm_request=callback,
        )

    callback.assert_called_once()
    chat.assert_not_awaited()
    dispatcher.dispatch_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_before_tool_dispatch_callback_blocks_dispatcher():
    """An exhausted tool budget must prevent the entire batch."""

    tool_calls = [
        make_tool_call(call_id="call_001"),
        make_tool_call(call_id="call_002"),
    ]

    response = LLMResponse(
        content="Run analysis tools.",
        tool_calls=tool_calls,
    )

    service, chat, _, dispatcher = make_service(
        [response]
    )

    callback = MagicMock(
        side_effect=RuntimeError("Tool-call budget exceeded.")
    )

    with pytest.raises(
        RuntimeError,
        match="Tool-call budget exceeded",
    ):
        await service.run_with_history(
            make_request(),
            before_tool_dispatch=callback,
        )

    callback.assert_called_once_with(2)
    chat.assert_awaited_once()
    dispatcher.dispatch_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_before_tool_dispatch_callback_allows_dispatch():
    """A successful budget reservation must allow dispatch."""

    tool_call = make_tool_call()

    responses = [
        LLMResponse(
            content="Run the analysis tool.",
            tool_calls=[tool_call],
        ),
        LLMResponse(
            content="Analysis completed.",
        ),
    ]

    service, chat, _, dispatcher = make_service(
        responses
    )

    callback = MagicMock()

    execution = await service.run_with_history(
        make_request(),
        before_tool_dispatch=callback,
    )

    assert execution.response.content == "Analysis completed."

    callback.assert_called_once_with(1)
    assert chat.await_count == 2

    dispatcher.dispatch_many.assert_awaited_once_with(
        [tool_call]
    )


@pytest.mark.asyncio
async def test_before_llm_request_callback_blocks_final_synthesis():
    """Final synthesis must also consume the LLM iteration budget."""

    tool_call = make_tool_call()

    responses = [
        LLMResponse(
            content="Run the analysis tool.",
            tool_calls=[tool_call],
        ),
        LLMResponse(
            content="Final synthesis.",
        ),
    ]

    service, chat, _, dispatcher = make_service(
        responses,
        config=ToolCallingConfig(max_rounds=1),
    )

    callback_count = 0

    def before_llm_request():
        nonlocal callback_count

        callback_count += 1

        if callback_count > 1:
            raise RuntimeError("Iteration budget exceeded.")

    with pytest.raises(
        RuntimeError,
        match="Iteration budget exceeded",
    ):
        await service.run_with_history(
            make_request(),
            before_llm_request=before_llm_request,
        )

    assert callback_count == 2

    # The first LLM request was allowed.
    # The second request (final synthesis) was blocked.
    chat.assert_awaited_once()

    # The first tool batch was dispatched successfully.
    dispatcher.dispatch_many.assert_awaited_once_with(
        [tool_call]
    )
