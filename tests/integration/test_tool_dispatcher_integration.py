import asyncio

from lite_cog_agent.llm.schemas import ToolCall
from lite_cog_agent.tools.base import BaseTool
from lite_cog_agent.tools.dispatcher import ToolDispatcher
from lite_cog_agent.tools.execution_config import ToolExecutionConfig
from lite_cog_agent.tools.registry import ToolRegistry
from lite_cog_agent.tools.schemas import ToolResult


class IntegrationTool(BaseTool):
    name = "integration_tool"
    description = "A tool for dispatcher integration testing."

    parameters = {
        "type": "object",
        "properties": {
            "delay": {
                "type": "number",
                "minimum": 0,
            },
            "should_fail": {
                "type": "boolean",
            },
        },
        "required": ["delay", "should_fail"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.active_count = 0
        self.max_active_count = 0

    async def execute(self, arguments: dict) -> ToolResult:
        self.active_count += 1
        self.max_active_count = max(
            self.max_active_count,
            self.active_count,
        )

        try:
            await asyncio.sleep(arguments["delay"])

            if arguments["should_fail"]:
                raise RuntimeError("Integration test failure")

            return ToolResult(
                success=True,
                output={"message": "completed"},
                metadata={"source": "integration_test"},
            )
        finally:
            self.active_count -= 1


def create_integration_dispatcher():
    registry = ToolRegistry()
    tool = IntegrationTool()
    registry.register(tool)

    config = ToolExecutionConfig(
        timeout_seconds=1.0,
        max_concurrency=2,
    )

    dispatcher = ToolDispatcher(
        registry=registry,
        config=config,
    )

    return dispatcher, tool


def create_call(
    call_id: str,
    *,
    delay: float = 0.05,
    should_fail: bool = False,
) -> ToolCall:
    return ToolCall(
        id=call_id,
        name="integration_tool",
        arguments={
            "delay": delay,
            "should_fail": should_fail,
        },
    )


def test_dispatcher_integrates_success_failure_and_metadata():
    async def run_test():
        dispatcher, tool = create_integration_dispatcher()

        calls = [
            create_call("success_1"),
            create_call("failure_1", should_fail=True),
            create_call("success_2"),
        ]

        results = await asyncio.gather(*[dispatcher.dispatch(call) for call in calls])

        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True

        assert results[1].metadata["error_type"] == "RuntimeError"

        for result in results:
            assert result.metadata["tool_name"] == "integration_tool"
            assert result.metadata["status"] in {
                "success",
                "failed",
            }
            assert result.metadata["execution_time_seconds"] >= 0
            assert result.metadata["queue_wait_seconds"] >= 0

        assert tool.active_count == 0
        assert tool.max_active_count <= 2

        assert results[0].metadata["source"] == "integration_test"

    asyncio.run(run_test())


def test_dispatcher_concurrent_calls_do_not_leak_slots():
    async def run_test():
        dispatcher, tool = create_integration_dispatcher()

        calls = [create_call(f"concurrent_{index}") for index in range(6)]

        results = await asyncio.gather(*[dispatcher.dispatch(call) for call in calls])

        assert all(result.success for result in results)
        assert tool.max_active_count == 2
        assert tool.active_count == 0

    asyncio.run(run_test())
