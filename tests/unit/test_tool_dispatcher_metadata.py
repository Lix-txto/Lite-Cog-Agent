import asyncio

from lite_cog_agent.llm.schemas import ToolCall
from lite_cog_agent.tools.base import BaseTool
from lite_cog_agent.tools.dispatcher import ToolDispatcher
from lite_cog_agent.tools.registry import ToolRegistry
from lite_cog_agent.tools.schemas import ToolResult


class SuccessfulTool(BaseTool):
    name = "successful_tool"
    description = "A successful test tool."
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult.ok(
            {"message": "success"},
            metadata={"source": "test_tool"},
        )


class FailingTool(BaseTool):
    name = "failing_tool"
    description = "A tool that raises an exception."
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict) -> ToolResult:
        raise RuntimeError("Unexpected tool failure")


class ReportedFailureTool(BaseTool):
    name = "reported_failure_tool"
    description = "A tool that returns a failed result."
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult.fail("Business operation failed")


class BlockingTool(BaseTool):
    name = "blocking_tool"
    description = "A tool that waits until cancelled."
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict) -> ToolResult:
        await asyncio.Event().wait()
        return ToolResult.ok({"status": "completed"})


def create_dispatcher(tool: BaseTool) -> ToolDispatcher:
    registry = ToolRegistry()
    registry.register(tool)

    return ToolDispatcher(registry=registry)


def create_tool_call(name: str) -> ToolCall:
    return ToolCall(
        id=f"metadata_{name}",
        name=name,
        arguments={},
    )


def test_dispatch_adds_execution_metadata():
    tool = SuccessfulTool()
    dispatcher = create_dispatcher(tool)

    result = asyncio.run(dispatcher.dispatch(create_tool_call("successful_tool")))

    assert result.success is True
    assert result.output == {"message": "success"}

    assert result.metadata["tool_name"] == "successful_tool"
    assert result.metadata["status"] == "success"
    assert result.metadata["execution_time_seconds"] >= 0
    assert result.metadata["queue_wait_seconds"] >= 0
    assert result.metadata["error_type"] is None


def test_dispatch_preserves_tool_metadata():
    tool = SuccessfulTool()
    dispatcher = create_dispatcher(tool)

    result = asyncio.run(dispatcher.dispatch(create_tool_call("successful_tool")))

    assert result.metadata["source"] == "test_tool"
    assert result.metadata["tool_name"] == "successful_tool"


def test_dispatch_converts_unexpected_exception():
    tool = FailingTool()
    dispatcher = create_dispatcher(tool)

    result = asyncio.run(dispatcher.dispatch(create_tool_call("failing_tool")))

    assert result.success is False
    assert result.error == "Unexpected tool failure"

    assert result.metadata["tool_name"] == "failing_tool"
    assert result.metadata["status"] == "failed"
    assert result.metadata["error_type"] == "RuntimeError"
    assert result.metadata["execution_time_seconds"] >= 0


def test_dispatch_preserves_tool_reported_failure():
    tool = ReportedFailureTool()
    dispatcher = create_dispatcher(tool)

    result = asyncio.run(dispatcher.dispatch(create_tool_call("reported_failure_tool")))

    assert result.success is False
    assert result.error == "Business operation failed"
    assert result.metadata["status"] == "failed"
    assert result.metadata["error_type"] == "ToolReportedFailure"


def test_dispatch_adds_metadata_to_missing_tool():
    registry = ToolRegistry()
    dispatcher = ToolDispatcher(registry=registry)

    result = asyncio.run(dispatcher.dispatch(create_tool_call("missing_tool")))

    assert result.success is False
    assert result.metadata["tool_name"] == "missing_tool"
    assert result.metadata["status"] == "failed"
    assert result.metadata["error_type"] == "ToolNotFoundError"


def test_dispatch_propagates_task_cancellation():
    async def run_test():
        tool = BlockingTool()
        dispatcher = create_dispatcher(tool)

        task = asyncio.create_task(
            dispatcher.dispatch(create_tool_call("blocking_tool"))
        )

        await asyncio.sleep(0.01)
        task.cancel()

        try:
            await task
            assert False, "Expected CancelledError"
        except asyncio.CancelledError:
            pass

        # A new call should still be able to acquire the semaphore.
        next_task = asyncio.create_task(
            dispatcher.dispatch(create_tool_call("blocking_tool"))
        )

        await asyncio.sleep(0.01)

        assert not next_task.done()

        next_task.cancel()

        try:
            await next_task
            assert False, "Expected CancelledError"
        except asyncio.CancelledError:
            pass

    asyncio.run(run_test())
