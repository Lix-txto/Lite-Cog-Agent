from __future__ import annotations

import json
from typing import Any

import pytest

from lite_cog_agent.core.tool_calling import ToolCallingLimitError
from lite_cog_agent.llm.gateway import LLMGateway
from lite_cog_agent.llm.schemas import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    ToolCall,
)
from lite_cog_agent.tools.base import BaseTool
from lite_cog_agent.tools.factory import (
    ToolRuntime,
    create_tool_runtime,
)
from lite_cog_agent.tools.schemas import ToolResult


class FakeProvider:
    """Deterministic provider for Tool Runtime integration tests."""

    def __init__(
        self,
        responses: list[LLMResponse],
    ) -> None:
        self._responses = list(responses)
        self.requests: list[LLMRequest] = []
        self.closed = False

    async def chat_completion(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        if self.closed:
            raise RuntimeError("Provider is closed.")

        self.requests.append(request)

        if not self._responses:
            raise AssertionError("No fake LLM response remains.")

        return self._responses.pop(0)

    async def close(self) -> None:
        self.closed = True


class EchoTool(BaseTool):
    """A deterministic tool that returns its input arguments."""

    def __init__(self) -> None:
        self.executed_arguments: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "test.echo"

    @property
    def description(self) -> str:
        return "Return the supplied value."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                },
            },
            "required": ["value"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolResult:
        self.executed_arguments.append(dict(arguments))

        return ToolResult.ok(
            output={
                "echoed": arguments["value"],
            },
            metadata={
                "source": "echo_tool",
            },
        )


class FailingTool(EchoTool):
    """A tool that raises an exception during execution."""

    @property
    def name(self) -> str:
        return "test.failing"

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolResult:
        raise RuntimeError("Simulated tool execution failure.")


def make_tool_call(
    *,
    call_id: str = "call_001",
    name: str = "test.echo",
    value: str = "hello",
) -> ToolCall:
    return ToolCall(
        id=call_id,
        name=name,
        arguments={
            "value": value,
        },
    )


def make_request() -> LLMRequest:
    return LLMRequest(
        messages=[
            LLMMessage(
                role="user",
                content="Echo the value hello.",
            )
        ],
        model="deepseek-chat",
        temperature=0.2,
        max_tokens=1024,
    )


def make_runtime(
    responses: list[LLMResponse],
    tools: list[BaseTool],
) -> tuple[ToolRuntime, FakeProvider, LLMGateway]:
    provider = FakeProvider(responses)
    gateway = LLMGateway(provider=provider)

    runtime = create_tool_runtime(
        gateway,
        tools,
    )

    return runtime, provider, gateway


@pytest.mark.asyncio
async def test_runtime_executes_complete_tool_calling_workflow() -> None:
    """The real runtime should execute a tool and return a final answer."""

    tool = EchoTool()

    tool_response = LLMResponse(
        content="I will execute the echo tool.",
        tool_calls=[
            make_tool_call(),
        ],
    )

    final_response = LLMResponse(
        content="The echoed value is hello.",
    )

    runtime, provider, gateway = make_runtime(
        [
            tool_response,
            final_response,
        ],
        [tool],
    )

    try:
        result = await runtime.tool_calling_service.run(
            make_request(),
        )

        assert result is final_response

        # One tool execution and two LLM requests.
        assert len(provider.requests) == 2
        assert tool.executed_arguments == [
            {"value": "hello"},
        ]

        # The first request exposes the registered tool definition.
        first_request = provider.requests[0]

        assert len(first_request.tools) == 1
        assert first_request.tools[0].name == "test.echo"

        # The second request contains the assistant tool call
        # and the corresponding tool result.
        second_request = provider.requests[1]

        assert second_request.messages[-2].role == "assistant"
        assert second_request.messages[-1].role == "tool"
        assert second_request.messages[-1].tool_call_id == "call_001"

        assert runtime.dispatcher._registry is runtime.registry

    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_runtime_passes_tool_arguments_correctly() -> None:
    """Tool Call arguments should reach the registered tool unchanged."""

    tool = EchoTool()

    runtime, provider, gateway = make_runtime(
        [
            LLMResponse(
                tool_calls=[
                    make_tool_call(value="sales-data"),
                ],
            ),
            LLMResponse(
                content="The tool returned sales-data.",
            ),
        ],
        [tool],
    )

    try:
        await runtime.tool_calling_service.run(make_request())

        assert tool.executed_arguments == [
            {"value": "sales-data"},
        ]

        assert len(provider.requests) == 2

    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_runtime_converts_tool_result_into_llm_message() -> None:
    """ToolResult should be serialized into a correlated tool message."""

    tool = EchoTool()

    runtime, provider, gateway = make_runtime(
        [
            LLMResponse(
                tool_calls=[
                    make_tool_call(value="integration-test"),
                ],
            ),
            LLMResponse(
                content="Tool result received.",
            ),
        ],
        [tool],
    )

    try:
        await runtime.tool_calling_service.run(make_request())

        tool_message = provider.requests[1].messages[-1]

        assert tool_message.role == "tool"
        assert tool_message.tool_call_id == "call_001"

        payload = json.loads(tool_message.content)

        assert payload["success"] is True
        assert payload["output"] == {
            "echoed": "integration-test",
        }

        # Dispatcher adds execution metadata to the tool result.
        assert payload["metadata"]["tool_name"] == "test.echo"
        assert payload["metadata"]["status"] == "success"

    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_runtime_handles_unregistered_tool() -> None:
    """An unknown tool should become a failed ToolResult."""

    tool = EchoTool()

    runtime, provider, gateway = make_runtime(
        [
            LLMResponse(
                tool_calls=[
                    make_tool_call(
                        name="test.unknown",
                    ),
                ],
            ),
            LLMResponse(
                content="The requested tool was unavailable.",
            ),
        ],
        [tool],
    )

    try:
        result = await runtime.tool_calling_service.run(
            make_request(),
        )

        assert result.content == "The requested tool was unavailable."

        assert tool.executed_arguments == []

        tool_message = provider.requests[1].messages[-1]
        payload = json.loads(tool_message.content)

        assert payload["success"] is False
        assert "Tool not found" in payload["error"]
        assert payload["metadata"]["error_type"] == "ToolNotFoundError"

    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_runtime_handles_tool_execution_exception() -> None:
    """A tool exception should be isolated by the dispatcher."""

    tool = FailingTool()

    runtime, provider, gateway = make_runtime(
        [
            LLMResponse(
                tool_calls=[
                    make_tool_call(
                        name="test.failing",
                    ),
                ],
            ),
            LLMResponse(
                content="The tool failed, but the workflow continued.",
            ),
        ],
        [tool],
    )

    try:
        result = await runtime.tool_calling_service.run(
            make_request(),
        )

        assert result.content == (
            "The tool failed, but the workflow continued."
        )

        assert len(provider.requests) == 2

        tool_message = provider.requests[1].messages[-1]
        payload = json.loads(tool_message.content)

        assert payload["success"] is False
        assert payload["error"] == "Simulated tool execution failure."
        assert payload["metadata"]["error_type"] == "RuntimeError"

    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_runtime_supports_multiple_sequential_tool_rounds() -> None:
    """The runtime should preserve results across multiple tool rounds."""

    tool = EchoTool()

    first_call = make_tool_call(
        call_id="call_001",
        value="first",
    )

    second_call = make_tool_call(
        call_id="call_002",
        value="second",
    )

    runtime, provider, gateway = make_runtime(
        [
            LLMResponse(
                tool_calls=[first_call],
            ),
            LLMResponse(
                tool_calls=[second_call],
            ),
            LLMResponse(
                content="Both tool calls completed.",
            ),
        ],
        [tool],
    )

    try:
        result = await runtime.tool_calling_service.run(
            make_request(),
        )

        assert result.content == "Both tool calls completed."

        assert tool.executed_arguments == [
            {"value": "first"},
            {"value": "second"},
        ]

        assert len(provider.requests) == 3

        final_request = provider.requests[2]

        assert [
            message.role
            for message in final_request.messages
        ] == [
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
        ]

        assert final_request.messages[2].tool_call_id == "call_001"
        assert final_request.messages[4].tool_call_id == "call_002"

    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_runtime_rejects_multiple_tool_calls_in_one_round() -> None:
    """The Agent runtime default limit should reject a multi-call batch."""

    tool = EchoTool()

    runtime, provider, gateway = make_runtime(
        [
            LLMResponse(
                tool_calls=[
                    make_tool_call(
                        call_id="call_001",
                        value="first",
                    ),
                    make_tool_call(
                        call_id="call_002",
                        value="second",
                    ),
                ],
            ),
        ],
        [tool],
    )

    try:
        with pytest.raises(ToolCallingLimitError):
            await runtime.tool_calling_service.run(
                make_request(),
            )

        assert len(provider.requests) == 1

        # The complete batch is rejected before dispatch.
        assert tool.executed_arguments == []

    finally:
        await gateway.close()
