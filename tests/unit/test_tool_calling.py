from unittest.mock import AsyncMock, MagicMock

import pytest

from lite_cog_agent.core.tool_calling import (
    ToolCallingError,
    ToolCallingLimitError,
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
async def test_run_rejects_too_many_tool_calls_in_one_round() -> None:
    tool_calls = [
        make_tool_call(call_id="call_001"),
        make_tool_call(call_id="call_002"),
        make_tool_call(call_id="call_003"),
    ]

    response = LLMResponse(
        tool_calls=tool_calls,
    )

    service, chat, _, dispatcher = make_service(
        [response],
        config=ToolCallingConfig(
            max_rounds=2,
            max_tool_calls_per_round=2,
        ),
    )

    with pytest.raises(
        ToolCallingLimitError,
        match="exceeds the configured limit",
    ):
        await service.run(make_request())

    assert chat.await_count == 1
    dispatcher.dispatch_many.assert_not_awaited()


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