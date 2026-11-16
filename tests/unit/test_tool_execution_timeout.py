import asyncio

import pytest

from lite_cog_agent.llm.schemas import ToolCall
from lite_cog_agent.tools.base import BaseTool
from lite_cog_agent.tools.dispatcher import ToolDispatcher
from lite_cog_agent.tools.exceptions import (
    ToolExecutionTimeoutError,
)
from lite_cog_agent.tools.execution_config import ToolExecutionConfig
from lite_cog_agent.tools.registry import ToolRegistry
from lite_cog_agent.tools.schemas import ToolResult


class SlowTool(BaseTool):
    """A tool that takes time to execute."""

    name = "slow_tool"
    description = "A slow tool for timeout testing."
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self.cancelled = False

    async def execute(self, arguments: dict) -> ToolResult:
        try:
            await asyncio.sleep(self.delay_seconds)
            return ToolResult.ok({"status": "completed"})
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def create_dispatcher(
    tool: BaseTool,
    timeout_seconds: float,
) -> ToolDispatcher:
    registry = ToolRegistry()
    registry.register(tool)

    config = ToolExecutionConfig(
        timeout_seconds=timeout_seconds,
    )

    return ToolDispatcher(
        registry=registry,
        config=config,
    )


def create_tool_call() -> ToolCall:
    return ToolCall(
        id="timeout_test_001",
        name="slow_tool",
        arguments={},
    )


def test_dispatch_raises_timeout_error():
    tool = SlowTool(delay_seconds=1.0)

    dispatcher = create_dispatcher(
        tool=tool,
        timeout_seconds=0.05,
    )

    with pytest.raises(ToolExecutionTimeoutError) as exc_info:
        asyncio.run(
            dispatcher.dispatch(create_tool_call())
        )

    assert exc_info.value.tool_name == "slow_tool"
    assert exc_info.value.timeout_seconds == 0.05


def test_dispatch_cancels_timed_out_tool():
    tool = SlowTool(delay_seconds=1.0)

    dispatcher = create_dispatcher(
        tool=tool,
        timeout_seconds=0.05,
    )

    with pytest.raises(ToolExecutionTimeoutError):
        asyncio.run(
            dispatcher.dispatch(create_tool_call())
        )

    assert tool.cancelled is True


def test_dispatch_completes_before_timeout():
    tool = SlowTool(delay_seconds=0.01)

    dispatcher = create_dispatcher(
        tool=tool,
        timeout_seconds=1.0,
    )

    result = asyncio.run(
        dispatcher.dispatch(create_tool_call())
    )

    assert result.success is True
    assert result.output == {"status": "completed"}
    assert tool.cancelled is False
