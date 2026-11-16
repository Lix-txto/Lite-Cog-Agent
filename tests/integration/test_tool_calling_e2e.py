"""End-to-end integration tests for the Tool Calling pipeline."""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import pytest

from lite_cog_agent.core.tool_calling import (
    ToolCallingLimitError,
    ToolCallingService,
)
from lite_cog_agent.core.tool_calling_config import ToolCallingConfig
from lite_cog_agent.llm.gateway import LLMGateway
from lite_cog_agent.llm.schemas import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    TokenUsage,
    ToolCall,
)
from lite_cog_agent.tools.base import BaseTool
from lite_cog_agent.tools.dispatcher import ToolDispatcher
from lite_cog_agent.tools.execution_config import ToolExecutionConfig
from lite_cog_agent.tools.registry import ToolRegistry
from lite_cog_agent.tools.schemas import ToolResult


class FakeProvider:
    """Scripted provider that avoids real external API calls."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = deque(responses)
        self.requests: list[LLMRequest] = []
        self.closed = False

    async def chat_completion(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        # Store an independent snapshot of the request.
        # ToolCallingService may continue appending messages
        # to its internal conversation history after this call.
        request_snapshot = LLMRequest(
            messages=[
                LLMMessage(
                    role=message.role,
                    content=message.content,
                    name=message.name,
                    tool_call_id=message.tool_call_id,
                    tool_calls=list(message.tool_calls),
                )
                for message in request.messages
            ],
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tools=list(request.tools),
        )

        self.requests.append(request_snapshot)

        if not self._responses:
            raise AssertionError(
                "FakeProvider has no scripted response."
            )

        return self._responses.popleft()

    async def close(self) -> None:
        self.closed = True


class AddTool(BaseTool):
    name = "add"
    description = "Add two numbers."
    parameters = {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult.ok(
            output={"result": arguments["a"] + arguments["b"]}
        )


class MultiplyTool(BaseTool):
    name = "multiply"
    description = "Multiply two numbers."
    parameters = {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult.ok(
            output={"result": arguments["a"] * arguments["b"]}
        )


class FailingTool(BaseTool):
    name = "fail"
    description = "A tool that returns an expected failure."
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult.fail("Intentional integration-test failure.")


def make_response(
    *,
    content: str | None = None,
    tool_calls: list[ToolCall] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> LLMResponse:
    """Build a valid response with internally consistent usage."""

    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason="tool_calls" if tool_calls else "stop",
        usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        model="fake-model",
    )


def make_service(
    responses: list[LLMResponse],
    *,
    tools: list[BaseTool] | None = None,
    config: ToolCallingConfig | None = None,
) -> tuple[ToolCallingService, FakeProvider, LLMGateway]:
    """Build the real application pipeline with a fake provider."""

    provider = FakeProvider(responses)
    gateway = LLMGateway(provider=provider)

    registry = ToolRegistry()
    for tool in tools or [AddTool(), MultiplyTool(), FailingTool()]:
        registry.register(tool)

    dispatcher = ToolDispatcher(
        registry=registry,
        config=ToolExecutionConfig(
            timeout_seconds=2.0,
            max_concurrency=3,
        ),
    )

    service = ToolCallingService(
        gateway=gateway,
        registry=registry,
        dispatcher=dispatcher,
        config=config,
    )

    return service, provider, gateway


def make_request() -> LLMRequest:
    return LLMRequest(
        messages=[
            LLMMessage(
                role="user",
                content="Calculate the requested values.",
            )
        ],
        model="fake-model",
        temperature=0,
        max_tokens=512,
    )


@pytest.mark.asyncio
async def test_single_tool_call_end_to_end() -> None:
    """One tool call is executed and synthesized into a final answer."""

    service, provider, gateway = make_service(
        [
            make_response(
                tool_calls=[
                    ToolCall(
                        id="call_add_1",
                        name="add",
                        arguments={"a": 2, "b": 3},
                    )
                ]
            ),
            make_response(content="The result is 5."),
        ]
    )

    result = await service.run(make_request())

    assert result.content == "The result is 5."
    assert len(provider.requests) == 2

    first_request = provider.requests[0]
    assert len(first_request.tools) == 3
    assert first_request.messages[-1].role == "user"

    final_request = provider.requests[1]

    # The next LLM request still allows tool calling.
    assert len(final_request.tools) == 3

    # The assistant message must contain the original tool call.
    assistant_message = final_request.messages[-2]
    assert assistant_message.role == "assistant"
    assert len(assistant_message.tool_calls) == 1
    assert assistant_message.tool_calls[0].id == "call_add_1"

    # The tool result must be included in the conversation.
    tool_message = final_request.messages[-1]
    assert tool_message.role == "tool"
    assert tool_message.tool_call_id == "call_add_1"

    # ToolResult is serialized as a JSON string.
    tool_result = json.loads(tool_message.content)

    assert tool_result["success"] is True
    assert tool_result["output"]["result"] == 5
    assert tool_result["error"] is None
    assert tool_result["metadata"]["tool_name"] == "add"

    # Verify accumulated token usage.
    usage = gateway.usage_summary
    assert usage.request_count == 2
    assert usage.prompt_tokens == 20
    assert usage.completion_tokens == 10
    assert usage.total_tokens == 30


@pytest.mark.asyncio
async def test_multiple_tool_calls_end_to_end() -> None:
    """Multiple tools execute in one round and preserve result order."""

    service, provider, gateway = make_service(
        [
            make_response(
                tool_calls=[
                    ToolCall(
                        id="call_add",
                        name="add",
                        arguments={"a": 2, "b": 3},
                    ),
                    ToolCall(
                        id="call_multiply",
                        name="multiply",
                        arguments={"a": 4, "b": 5},
                    ),
                ]
            ),
            make_response(content="Addition is 5; multiplication is 20."),
        ]
    )

    result = await service.run(make_request())

    assert result.content == "Addition is 5; multiplication is 20."
    assert len(provider.requests) == 2

    tool_messages = [
        message
        for message in provider.requests[1].messages
        if message.role == "tool"
    ]

    assert len(tool_messages) == 2
    assert [message.tool_call_id for message in tool_messages] == [
        "call_add",
        "call_multiply",
    ]

    assert "5" in str(tool_messages[0].content)
    assert "20" in str(tool_messages[1].content)
    assert gateway.usage_summary.request_count == 2


@pytest.mark.asyncio
async def test_multi_round_tool_calling_end_to_end() -> None:
    """The model can request tools across multiple execution rounds."""

    service, provider, gateway = make_service(
        [
            make_response(
                tool_calls=[
                    ToolCall(
                        id="round_1_add",
                        name="add",
                        arguments={"a": 2, "b": 3},
                    )
                ]
            ),
            make_response(
                tool_calls=[
                    ToolCall(
                        id="round_2_multiply",
                        name="multiply",
                        arguments={"a": 5, "b": 4},
                    )
                ]
            ),
            make_response(content="The final result is 20."),
        ],
        config=ToolCallingConfig(max_rounds=2),
    )

    result = await service.run(make_request())

    assert result.content == "The final result is 20."
    assert len(provider.requests) == 3

    second_request_messages = provider.requests[1].messages
    assert second_request_messages[-1].role == "tool"
    assert second_request_messages[-1].tool_call_id == "round_1_add"

    final_messages = provider.requests[2].messages
    assert final_messages[-2].role == "assistant"
    assert final_messages[-1].role == "tool"
    assert final_messages[-1].tool_call_id == "round_2_multiply"

    assert gateway.usage_summary.request_count == 3


@pytest.mark.asyncio
async def test_tool_failure_is_returned_to_model_and_can_be_synthesized() -> None:
    """An expected tool failure does not prevent final synthesis."""

    service, provider, gateway = make_service(
        [
            make_response(
                tool_calls=[
                    ToolCall(
                        id="call_fail",
                        name="fail",
                        arguments={},
                    )
                ]
            ),
            make_response(content="The operation failed as expected."),
        ]
    )

    result = await service.run(make_request())

    assert result.content == "The operation failed as expected."

    tool_message = provider.requests[1].messages[-1]
    assert tool_message.role == "tool"
    assert tool_message.tool_call_id == "call_fail"
    assert "Intentional integration-test failure." in str(
        tool_message.content
    )

    assert gateway.usage_summary.request_count == 2


@pytest.mark.asyncio
async def test_tool_call_limit_rejects_oversized_batch() -> None:
    """A batch exceeding the configured limit is rejected before dispatch."""

    service, provider, gateway = make_service(
        [
            make_response(
                tool_calls=[
                    ToolCall(
                        id="limit_add",
                        name="add",
                        arguments={"a": 1, "b": 2},
                    ),
                    ToolCall(
                        id="limit_multiply",
                        name="multiply",
                        arguments={"a": 2, "b": 3},
                    ),
                ]
            )
        ],
        config=ToolCallingConfig(
            max_rounds=2,
            max_tool_calls_per_round=1,
        ),
    )

    with pytest.raises(ToolCallingLimitError):
        await service.run(make_request())

    assert len(provider.requests) == 1
    assert gateway.usage_summary.request_count == 1


@pytest.mark.asyncio
async def test_max_rounds_allows_one_final_synthesis_request() -> None:
    """One execution round is followed by one synthesis request."""

    service, provider, gateway = make_service(
        [
            make_response(
                tool_calls=[
                    ToolCall(
                        id="last_round_add",
                        name="add",
                        arguments={"a": 10, "b": 5},
                    )
                ]
            ),
            make_response(content="The result is 15."),
        ],
        config=ToolCallingConfig(max_rounds=1),
    )

    result = await service.run(make_request())

    assert result.content == "The result is 15."
    assert len(provider.requests) == 2
    assert provider.requests[0].tools
    assert provider.requests[1].tools == []
    assert gateway.usage_summary.request_count == 2
