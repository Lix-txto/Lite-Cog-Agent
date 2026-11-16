import asyncio

import pytest

from lite_cog_agent.llm.schemas import ToolCall
from lite_cog_agent.tools.base import BaseTool
from lite_cog_agent.tools.dispatcher import ToolDispatcher
from lite_cog_agent.tools.execution_config import ToolExecutionConfig
from lite_cog_agent.tools.registry import ToolRegistry
from lite_cog_agent.tools.schemas import ToolResult


class BatchTestTool(BaseTool):
    name = "batch_test_tool"
    description = "A tool for batch dispatch testing."

    parameters = {
        "type": "object",
        "properties": {
            "value": {
                "type": "integer",
            },
            "delay": {
                "type": "number",
                "minimum": 0,
            },
        },
        "required": ["value", "delay"],
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict) -> ToolResult:
        await asyncio.sleep(arguments["delay"])

        return ToolResult(
            success=True,
            output={
                "value": arguments["value"],
            },
        )


def create_dispatcher(
    *,
    max_concurrency: int = 2,
) -> ToolDispatcher:
    registry = ToolRegistry()
    registry.register(BatchTestTool())

    config = ToolExecutionConfig(
        timeout_seconds=2.0,
        max_concurrency=max_concurrency,
    )

    return ToolDispatcher(
        registry=registry,
        config=config,
    )


def create_call(
    call_id: str,
    value: int,
    delay: float = 0.05,
) -> ToolCall:
    return ToolCall(
        id=call_id,
        name="batch_test_tool",
        arguments={
            "value": value,
            "delay": delay,
        },
    )


def test_dispatch_many_returns_empty_list_for_no_calls():
    async def run_test():
        dispatcher = create_dispatcher()

        results = await dispatcher.dispatch_many([])

        assert results == []

    asyncio.run(run_test())


def test_dispatch_many_preserves_result_order():
    async def run_test():
        dispatcher = create_dispatcher()

        tool_calls = [
            create_call("call_1", 1, delay=0.15),
            create_call("call_2", 2, delay=0.05),
            create_call("call_3", 3, delay=0.01),
        ]

        results = await dispatcher.dispatch_many(tool_calls)

        assert len(results) == 3
        assert all(result.success for result in results)

        assert [
            result.output["value"]
            for result in results
        ] == [1, 2, 3]

    asyncio.run(run_test())


def test_dispatch_many_respects_concurrency_limit():
    async def run_test():
        registry = ToolRegistry()

        class ConcurrencyTestTool(BatchTestTool):
            name = "concurrency_test_tool"

            def __init__(self):
                self.active_count = 0
                self.max_active_count = 0

            async def execute(self, arguments: dict) -> ToolResult:
                self.active_count += 1
                self.max_active_count = max(
                    self.max_active_count,
                    self.active_count,
                )

                try:
                    await asyncio.sleep(0.05)

                    return ToolResult(
                        success=True,
                        output={"value": arguments["value"]},
                    )
                finally:
                    self.active_count -= 1

        tool = ConcurrencyTestTool()
        registry.register(tool)

        dispatcher = ToolDispatcher(
            registry=registry,
            config=ToolExecutionConfig(
                timeout_seconds=2.0,
                max_concurrency=2,
            ),
        )

        tool_calls = [
            ToolCall(
                id=f"call_{index}",
                name="concurrency_test_tool",
                arguments={
                    "value": index,
                    "delay": 0.05,
                },
            )
            for index in range(5)
        ]

        results = await dispatcher.dispatch_many(tool_calls)

        assert all(result.success for result in results)
        assert tool.max_active_count == 2
        assert tool.active_count == 0

    asyncio.run(run_test())


def test_dispatch_many_returns_individual_failures():
    async def run_test():
        dispatcher = create_dispatcher()

        tool_calls = [
            create_call("valid_call", 10),
            ToolCall(
                id="invalid_call",
                name="batch_test_tool",
                arguments={
                    "value": "not_an_integer",
                    "delay": 0.05,
                },
            ),
            create_call("another_valid_call", 30),
        ]

        results = await dispatcher.dispatch_many(tool_calls)

        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True

        assert results[1].metadata["error_type"] == (
            "ToolParameterValidationError"
        )

    asyncio.run(run_test())


def test_dispatch_many_rejects_non_list_input():
    async def run_test():
        dispatcher = create_dispatcher()

        with pytest.raises(TypeError, match="tool_calls must be a list"):
            await dispatcher.dispatch_many(None)

    asyncio.run(run_test())