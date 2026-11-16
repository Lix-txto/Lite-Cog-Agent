import asyncio

from lite_cog_agent.llm.schemas import ToolCall
from lite_cog_agent.tools.base import BaseTool
from lite_cog_agent.tools.dispatcher import ToolDispatcher

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


def test_dispatch_returns_failure_on_timeout():
    tool = SlowTool(delay_seconds=1.0)
    dispatcher = create_dispatcher(
        tool=tool,
        timeout_seconds=0.05,
    )

    result = asyncio.run(dispatcher.dispatch(create_tool_call()))

    assert result.success is False
    assert "timed out" in result.error.lower()
    assert result.metadata["tool_name"] == "slow_tool"
    assert result.metadata["status"] == "failed"
    assert result.metadata["error_type"] == ("ToolExecutionTimeoutError")
    assert result.metadata["execution_time_seconds"] >= 0.05


def test_dispatch_cancels_timed_out_tool():
    tool = SlowTool(delay_seconds=1.0)
    dispatcher = create_dispatcher(
        tool=tool,
        timeout_seconds=0.05,
    )

    result = asyncio.run(dispatcher.dispatch(create_tool_call()))

    assert result.success is False
    assert tool.cancelled is True


def test_dispatch_completes_before_timeout():
    tool = SlowTool(delay_seconds=0.01)
    dispatcher = create_dispatcher(
        tool=tool,
        timeout_seconds=1.0,
    )

    result = asyncio.run(dispatcher.dispatch(create_tool_call()))

    assert result.success is True
    assert result.output == {"status": "completed"}
    assert result.metadata["tool_name"] == "slow_tool"
    assert result.metadata["status"] == "success"
    assert result.metadata["error_type"] is None
    assert result.metadata["execution_time_seconds"] >= 0
    assert result.metadata["queue_wait_seconds"] >= 0
    assert tool.cancelled is False
