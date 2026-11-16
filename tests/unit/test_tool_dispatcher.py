import asyncio

import pytest

from lite_cog_agent.llm.schemas import ToolCall
from lite_cog_agent.tools.base import BaseTool
from lite_cog_agent.tools.dispatcher import ToolDispatcher
from lite_cog_agent.tools.registry import ToolRegistry
from lite_cog_agent.tools.schemas import ToolResult


class FakeTool(BaseTool):
    """A fake tool for dispatcher testing."""

    name = "fake_tool"
    description = "A fake tool for testing."
    parameters = {
        "type": "object",
        "properties": {
            "value": {"type": "integer"},
        },
        "required": ["value"],
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult.ok({"result": arguments["value"] * 2})


class FakeFailingTool(BaseTool):
    name = "failing_tool"
    description = "A tool that returns a failure."
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult.fail("Simulated tool failure.")


def create_dispatcher() -> ToolDispatcher:
    registry = ToolRegistry()
    registry.register(FakeTool())
    registry.register(FakeFailingTool())

    return ToolDispatcher(registry)


def test_dispatch_success():
    dispatcher = create_dispatcher()

    tool_call = ToolCall(
        id="call_001",
        name="fake_tool",
        arguments={"value": 10},
    )

    result = asyncio.run(dispatcher.dispatch(tool_call))

    assert result.success is True
    assert result.output == {"result": 20}


def test_dispatch_tool_not_found():
    dispatcher = create_dispatcher()

    tool_call = ToolCall(
        id="call_002",
        name="unknown_tool",
        arguments={},
    )

    result = asyncio.run(dispatcher.dispatch(tool_call))

    assert result.success is False
    assert "Tool not found" in result.error


def test_dispatch_invalid_arguments():
    dispatcher = create_dispatcher()

    tool_call = ToolCall(
        id="call_003",
        name="fake_tool",
        arguments={"value": "not-an-integer"},
    )

    result = asyncio.run(dispatcher.dispatch(tool_call))

    assert result.success is False
    assert "Invalid tool arguments" in result.error


def test_dispatch_missing_required_argument():
    dispatcher = create_dispatcher()

    tool_call = ToolCall(
        id="call_004",
        name="fake_tool",
        arguments={},
    )

    result = asyncio.run(dispatcher.dispatch(tool_call))

    assert result.success is False
    assert "value" in result.error


def test_dispatch_tool_execution_failure():
    dispatcher = create_dispatcher()

    tool_call = ToolCall(
        id="call_005",
        name="failing_tool",
        arguments={},
    )

    result = asyncio.run(dispatcher.dispatch(tool_call))

    assert result.success is False
    assert result.error == "Simulated tool failure."


def test_dispatch_invalid_tool_call():
    dispatcher = create_dispatcher()

    result = asyncio.run(dispatcher.dispatch("not-a-tool-call"))

    assert result.success is False
    assert "expected a ToolCall" in result.error
