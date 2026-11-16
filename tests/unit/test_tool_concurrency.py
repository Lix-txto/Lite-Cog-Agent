import asyncio

from lite_cog_agent.llm.schemas import ToolCall
from lite_cog_agent.tools.base import BaseTool
from lite_cog_agent.tools.dispatcher import ToolDispatcher

from lite_cog_agent.tools.execution_config import ToolExecutionConfig
from lite_cog_agent.tools.registry import ToolRegistry
from lite_cog_agent.tools.schemas import ToolResult


class TrackingTool(BaseTool):
    """Tool that tracks concurrent executions."""

    name = "tracking_tool"
    description = "A tool for testing concurrency control."
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self, delay_seconds: float = 0.1) -> None:
        self.delay_seconds = delay_seconds
        self.active_count = 0
        self.max_active_count = 0
        self.execution_count = 0

    async def execute(self, arguments: dict) -> ToolResult:
        self.active_count += 1
        self.execution_count += 1

        self.max_active_count = max(
            self.max_active_count,
            self.active_count,
        )

        try:
            await asyncio.sleep(self.delay_seconds)

            return ToolResult.ok({"status": "completed"})
        finally:
            self.active_count -= 1


def create_dispatcher(
    tool: BaseTool,
    *,
    timeout_seconds: float = 2.0,
    max_concurrency: int = 2,
) -> ToolDispatcher:
    registry = ToolRegistry()
    registry.register(tool)

    config = ToolExecutionConfig(
        timeout_seconds=timeout_seconds,
        max_concurrency=max_concurrency,
    )

    return ToolDispatcher(
        registry=registry,
        config=config,
    )


def create_tool_call(
    call_id: str,
) -> ToolCall:
    return ToolCall(
        id=call_id,
        name="tracking_tool",
        arguments={},
    )


def test_dispatch_respects_max_concurrency():
    async def run_test():
        tool = TrackingTool(delay_seconds=0.1)

        dispatcher = create_dispatcher(
            tool,
            max_concurrency=2,
        )

        tool_calls = [create_tool_call(f"call_{index}") for index in range(5)]

        results = await asyncio.gather(
            *[dispatcher.dispatch(tool_call) for tool_call in tool_calls]
        )

        assert len(results) == 5
        assert all(result.success for result in results)

        assert tool.execution_count == 5
        assert tool.max_active_count == 2
        assert tool.active_count == 0

    asyncio.run(run_test())


def test_dispatch_releases_semaphore_after_execution():
    async def run_test():
        tool = TrackingTool(delay_seconds=0.01)

        dispatcher = create_dispatcher(
            tool,
            max_concurrency=1,
        )

        first_result = await dispatcher.dispatch(create_tool_call("first_call"))

        second_result = await dispatcher.dispatch(create_tool_call("second_call"))

        assert first_result.success is True
        assert second_result.success is True

        assert tool.execution_count == 2
        assert tool.active_count == 0

    asyncio.run(run_test())


def test_dispatch_releases_semaphore_after_timeout():
    async def run_test():
        tool = TrackingTool(delay_seconds=0.2)

        dispatcher = create_dispatcher(
            tool,
            timeout_seconds=0.05,
            max_concurrency=1,
        )

        first_result = await dispatcher.dispatch(create_tool_call("timeout_call"))

        assert first_result.success is False
        assert first_result.metadata["error_type"] == ("ToolExecutionTimeoutError")
        assert tool.active_count == 0

        # Replace the delay so the next call can complete.
        tool.delay_seconds = 0.01

        second_result = await dispatcher.dispatch(create_tool_call("next_call"))

        assert second_result.success is True
        assert tool.active_count == 0

    asyncio.run(run_test())
