import asyncio
import json
from types import SimpleNamespace

from lite_cog_agent import cli
from lite_cog_agent.llm.schemas import (
    LLMMessage,
    LLMResponse,
    TokenUsage,
    ToolCall,
)
from lite_cog_agent.loading.config import DatasetLoadingConfig


def make_response(content: str = "测试回答") -> LLMResponse:
    return LLMResponse(
        content=content,
        model="deepseek-flash",
        finish_reason="stop",
        usage=TokenUsage(10, 20, 30),
    )


def test_print_help(capsys):
    cli.print_help()

    output = capsys.readouterr().out

    assert "help" in output
    assert "status" in output
    assert "usage" in output
    assert "reset" in output
    assert "clear" in output
    assert "exit" in output
    assert "Agent" in output


def test_print_status(capsys):
    gateway = SimpleNamespace(
        usage_summary=SimpleNamespace(
            request_count=2,
        )
    )

    agent = SimpleNamespace(
        session_id="test-session",
    )

    context = SimpleNamespace(
        dataset_ids=("dataset-1", "dataset-2"),
    )

    cli.print_status(
        gateway,
        connected=True,
        agent=agent,
        context=context,
    )

    output = capsys.readouterr().out

    assert "Lite-Cog-Agent" in output
    assert "Connected" in output
    assert "test-session" in output
    assert "Loaded Datasets: 2" in output
    assert "LLM Requests: 2" in output


def test_print_status_without_connection(capsys):
    gateway = SimpleNamespace(
        usage_summary=SimpleNamespace(
            request_count=0,
        )
    )

    cli.print_status(
        gateway,
        connected=False,
    )

    output = capsys.readouterr().out

    assert "Not connected yet" in output


def test_print_usage(capsys):
    gateway = SimpleNamespace(
        usage_summary=SimpleNamespace(
            request_count=3,
            prompt_tokens=300,
            completion_tokens=150,
            total_tokens=450,
        )
    )

    cli.print_usage(gateway)

    output = capsys.readouterr().out

    assert "Requests           : 3" in output
    assert "Prompt Tokens      : 300" in output
    assert "Completion Tokens  : 150" in output
    assert "Total Tokens       : 450" in output


def test_print_response(capsys):
    cli.print_response(make_response())

    output = capsys.readouterr().out

    assert "Assistant:" in output
    assert "测试回答" in output
    assert "deepseek-flash" in output
    assert "Total Tokens" in output



def make_tool_message(
    tool_call_id: str,
    *,
    success: bool,
    output=None,
    error=None,
    metadata=None,
) -> LLMMessage:
    """Build a serialized Tool Message for CLI trace tests."""

    return LLMMessage(
        role="tool",
        tool_call_id=tool_call_id,
        content=json.dumps(
            {
                "success": success,
                "output": output,
                "error": error,
                "metadata": metadata or {},
            },
            ensure_ascii=False,
        ),
    )


def test_print_tool_execution_trace_shows_success(
    capsys,
):
    messages = [
        LLMMessage(
            role="assistant",
            content="I will calculate the metric.",
            tool_calls=[
                ToolCall(
                    id="call_001",
                    name="analysis.derived_metric",
                    arguments={
                        "dataset_id": "dataset-1",
                        "left_column": "Quantity",
                        "right_column": "UnitPrice",
                    },
                )
            ],
        ),
        make_tool_message(
            "call_001",
            success=True,
            output={
                "row_count": 541909,
                "valid_row_count": 541909,
                "excluded_row_count": 0,
                "result": 9747747.934,
            },
            metadata={
                "tool_name": "analysis.derived_metric",
                "status": "success",
                "execution_time_seconds": 0.42,
                "queue_wait_seconds": 0.01,
            },
        ),
    ]

    cli.print_tool_execution_trace(messages)

    output = capsys.readouterr().out

    assert "Tool Execution Trace" in output
    assert "analysis.derived_metric" in output
    assert "call_001" in output
    assert "SUCCESS" in output

    # These values must come from the actual Tool Result.
    assert "541909" in output
    assert "valid_row_count" in output
    assert "excluded_row_count" in output
    assert "9747747.934" in output
    assert "0.42" in output


def test_print_tool_execution_trace_shows_failure(
    capsys,
):
    messages = [
        LLMMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_002",
                    name="analysis.derived_metric",
                    arguments={},
                )
            ],
        ),
        make_tool_message(
            "call_002",
            success=False,
            error="Column 'Unknown' does not exist.",
            metadata={
                "tool_name": "analysis.derived_metric",
                "status": "failed",
                "error_type": "StatisticalExecutionError",
                "execution_time_seconds": 0.12,
            },
        ),
    ]

    cli.print_tool_execution_trace(messages)

    output = capsys.readouterr().out

    assert "FAILED" in output
    assert "call_002" in output
    assert "analysis.derived_metric" in output
    assert "Column 'Unknown' does not exist." in output


def test_print_tool_execution_trace_shows_rejected_call(
    capsys,
):
    messages = [
        LLMMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_003",
                    name="analysis.derived_metric",
                    arguments={},
                )
            ],
        ),
        make_tool_message(
            "call_003",
            success=False,
            error=(
                "No tools in this batch were executed. "
                "Please replan."
            ),
            metadata={
                "reason": "max_tool_calls_per_round_exceeded",
                "executed": False,
            },
        ),
    ]

    cli.print_tool_execution_trace(messages)

    output = capsys.readouterr().out

    assert "REJECTED" in output
    assert "call_003" in output
    assert "No tools in this batch were executed." in output


def test_print_tool_execution_trace_shows_missing_result(
    capsys,
):
    messages = [
        LLMMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_004",
                    name="analysis.derived_metric",
                    arguments={},
                )
            ],
        ),
    ]

    cli.print_tool_execution_trace(messages)

    output = capsys.readouterr().out

    assert "MISSING RESULT" in output
    assert "call_004" in output

    # Missing evidence must never be presented as success.
    assert "SUCCESS" not in output


def test_print_tool_execution_trace_shows_invalid_result(
    capsys,
):
    messages = [
        LLMMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_invalid_001",
                    name="analysis.derived_metric",
                    arguments={},
                )
            ],
        ),
        LLMMessage(
            role="tool",
            tool_call_id="call_invalid_001",
            content="{invalid-json",
        ),
    ]

    cli.print_tool_execution_trace(messages)

    output = capsys.readouterr().out

    assert "INVALID RESULT" in output
    assert "call_invalid_001" in output
    assert "MISSING RESULT" not in output
    assert "SUCCESS" not in output



def test_print_tool_execution_trace_rejects_duplicate_tool_results(
    capsys,
):
    messages = [
        LLMMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_duplicate_001",
                    name="analysis.derived_metric",
                    arguments={},
                )
            ],
        ),
        make_tool_message(
            "call_duplicate_001",
            success=False,
            error="First result",
        ),
        make_tool_message(
            "call_duplicate_001",
            success=True,
            output={"result": 12345},
        ),
    ]

    cli.print_tool_execution_trace(messages)

    output = capsys.readouterr().out

    assert "INVALID RESULT" in output
    assert "call_duplicate_001" in output
    assert "SUCCESS" not in output
    assert "FAILED" not in output
    assert "First result" not in output
    assert "12345" not in output
    assert "duplicate" in output.lower()


def test_print_tool_execution_trace_ignores_messages_without_tool_calls(
    capsys,
):
    messages = [
        LLMMessage(
            role="user",
            content="Hello.",
        ),
        LLMMessage(
            role="assistant",
            content="Hello! How can I help?",
        ),
    ]

    cli.print_tool_execution_trace(messages)

    output = capsys.readouterr().out

    assert output == ""


def test_print_tool_execution_trace_preserves_multi_round_order(
    capsys,
):
    messages = [
        LLMMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_005",
                    name="analysis.profile",
                    arguments={},
                )
            ],
        ),
        make_tool_message(
            "call_005",
            success=True,
            output={"row_count": 541909},
            metadata={
                "execution_time_seconds": 0.10,
            },
        ),
        LLMMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call_006",
                    name="analysis.derived_metric",
                    arguments={},
                )
            ],
        ),
        make_tool_message(
            "call_006",
            success=True,
            output={
                "row_count": 541909,
                "valid_row_count": 541909,
                "excluded_row_count": 0,
                "result": 9747747.934,
            },
            metadata={
                "execution_time_seconds": 0.42,
            },
        ),
    ]

    cli.print_tool_execution_trace(messages)

    output = capsys.readouterr().out

    assert "call_005" in output
    assert "call_006" in output

    # Trace should follow the original execution order.
    assert output.index("call_005") < output.index("call_006")

    assert "analysis.profile" in output
    assert "analysis.derived_metric" in output
    assert "SUCCESS" in output
    assert "9747747.934" in output



def test_run_interactive_composes_and_closes_runtime(
    monkeypatch,
    tmp_path,
    capsys,
):
    captured = {}

    class FakeGateway:
        def __init__(self):
            self.closed = False
            self.reset_called = False
            self.usage_summary = SimpleNamespace(
                request_count=0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )

        async def __aenter__(self):
            return self

        async def __aexit__(
            self,
            exc_type,
            exc_value,
            traceback,
        ):
            self.closed = True

        def reset_usage(self):
            self.reset_called = True

    class FakeContext:
        def __init__(self):
            self.dataset_ids = ("dataset-1",)
            self.closed = False

        def close(self):
            self.closed = True

    class FakeAgent:
        session_id = "test-session"

        def __init__(self):
            self.inputs = []
            self.reset_called = False

        async def run(self, user_input):
            self.inputs.append(user_input)
            return make_response(user_input)

        def reset(self):
            self.reset_called = True

        def get_messages(self):
            return []

    gateway = FakeGateway()
    context = FakeContext()
    agent = FakeAgent()

    def gateway_factory():
        captured["gateway_factory_calls"] = (
            captured.get("gateway_factory_calls", 0) + 1
        )
        return gateway

    def context_factory():
        captured["context_factory_calls"] = (
            captured.get("context_factory_calls", 0) + 1
        )
        return context

    def agent_factory(
        received_gateway,
        *,
        context,
        dataset_root,
        artifact_root,
        loading_config,
    ):
        captured["gateway"] = received_gateway
        captured["context"] = context
        captured["dataset_root"] = dataset_root
        captured["artifact_root"] = artifact_root
        captured["agent_factory_calls"] = (
            captured.get("agent_factory_calls", 0) + 1
        )
        captured["loading_config"] = loading_config
        return agent

    monkeypatch.setattr(
        cli,
        "LLMGateway",
        gateway_factory,
    )
    monkeypatch.setattr(
        cli,
        "ExecutionContext",
        context_factory,
    )
    monkeypatch.setattr(
        cli,
        "create_agent",
        agent_factory,
    )
    monkeypatch.setattr(
        cli.settings,
        "DATA_DIR",
        tmp_path / "data",
    )

    inputs = iter(
        [
            "第一个问题",
            "clear",
            "第二个问题",
            "reset",
            "status",
            "exit",
        ]
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs),
    )

    asyncio.run(cli.run_interactive())

    assert captured["gateway_factory_calls"] == 1
    assert captured["context_factory_calls"] == 1
    assert captured["agent_factory_calls"] == 1

    assert captured["gateway"] is gateway
    assert captured["context"] is context

    assert captured["dataset_root"] == tmp_path / "data"
    assert captured["artifact_root"] == (
        tmp_path / "data" / "artifacts" / "visualizations"
    )

    assert agent.inputs == [
        "第一个问题",
        "第二个问题",
    ]

    assert agent.reset_called is True
    assert gateway.reset_called is True

    assert context.closed is True
    assert gateway.closed is True

    output = capsys.readouterr().out

    assert "Conversation history cleared." in output
    assert "Loaded datasets and artifacts have been preserved." in output
    assert "LLM usage statistics reset." in output
    assert "test-session" in output
    assert isinstance(
        captured["loading_config"],
        DatasetLoadingConfig,
    )
    assert captured["loading_config"].max_rows == (
        cli.settings.DATASET_MAX_ROWS
    )


def test_run_interactive_handles_eof(
    monkeypatch,
    tmp_path,
):
    class FakeGateway:
        closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(
            self,
            exc_type,
            exc_value,
            traceback,
        ):
            self.closed = True

    class FakeContext:
        closed = False
        dataset_ids = ()

        def close(self):
            self.closed = True

    class FakeAgent:
        session_id = "test-session"

        async def run(self, user_input):
            raise AssertionError(
                "Agent should not run for EOF."
            )

        def reset(self):
            pass

        def get_messages(self):
            return []

    gateway = FakeGateway()
    context = FakeContext()

    monkeypatch.setattr(
        cli,
        "LLMGateway",
        lambda: gateway,
    )
    monkeypatch.setattr(
        cli,
        "ExecutionContext",
        lambda: context,
    )
    monkeypatch.setattr(
        cli,
        "create_agent",
        lambda *args, **kwargs: FakeAgent(),
    )
    monkeypatch.setattr(
        cli.settings,
        "DATA_DIR",
        tmp_path / "data",
    )

    def raise_eof(_):
        raise EOFError

    monkeypatch.setattr(
        "builtins.input",
        raise_eof,
    )

    asyncio.run(cli.run_interactive())

    assert context.closed is True
    assert gateway.closed is True



def test_run_interactive_continues_after_agent_request_error(
    monkeypatch,
    tmp_path,
    capsys,
):
    class FakeGateway:
        def __init__(self):
            self.closed = False
            self.usage_summary = SimpleNamespace(
                request_count=0,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )

        async def __aenter__(self):
            return self

        async def __aexit__(
            self,
            exc_type,
            exc_value,
            traceback,
        ):
            self.closed = True

    class FakeContext:
        dataset_ids = ()

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeAgent:
        session_id = "test-session"

        def __init__(self):
            self.inputs = []
            self.call_count = 0

        async def run(self, user_input):
            self.inputs.append(user_input)
            self.call_count += 1

            if self.call_count == 1:
                raise cli.LLMError("simulated LLM failure")

            return make_response("第二次请求成功")

        def reset(self):
            pass

        def get_messages(self):
            return []

    gateway = FakeGateway()
    context = FakeContext()
    agent = FakeAgent()

    monkeypatch.setattr(
        cli,
        "LLMGateway",
        lambda: gateway,
    )
    monkeypatch.setattr(
        cli,
        "ExecutionContext",
        lambda: context,
    )
    monkeypatch.setattr(
        cli,
        "create_agent",
        lambda *args, **kwargs: agent,
    )
    monkeypatch.setattr(
        cli.settings,
        "DATA_DIR",
        tmp_path / "data",
    )

    inputs = iter(
        [
            "第一次请求",
            "第二次请求",
            "exit",
        ]
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs),
    )

    asyncio.run(cli.run_interactive())

    assert agent.inputs == [
        "第一次请求",
        "第二次请求",
    ]

    assert context.closed is True
    assert gateway.closed is True

    output = capsys.readouterr().out

    assert "LLM request failed: simulated LLM failure" in output
    assert "第二次请求成功" in output



def test_run_interactive_handles_keyboard_interrupt(
    monkeypatch,
    tmp_path,
    capsys,
):
    class FakeGateway:
        def __init__(self):
            self.closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(
            self,
            exc_type,
            exc_value,
            traceback,
        ):
            self.closed = True

    class FakeContext:
        dataset_ids = ()

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeAgent:
        session_id = "test-session"

        async def run(self, user_input):
            raise AssertionError(
                "Agent should not run."
            )

        def reset(self):
            pass

        def get_messages(self):
            return []

    gateway = FakeGateway()
    context = FakeContext()
    agent = FakeAgent()

    monkeypatch.setattr(
        cli,
        "LLMGateway",
        lambda: gateway,
    )
    monkeypatch.setattr(
        cli,
        "ExecutionContext",
        lambda: context,
    )
    monkeypatch.setattr(
        cli,
        "create_agent",
        lambda *args, **kwargs: agent,
    )
    monkeypatch.setattr(
        cli.settings,
        "DATA_DIR",
        tmp_path / "data",
    )

    def raise_keyboard_interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "builtins.input",
        raise_keyboard_interrupt,
    )

    asyncio.run(cli.run_interactive())

    assert context.closed is True
    assert gateway.closed is True

    output = capsys.readouterr().out

    assert "Interrupted by user." in output



def test_run_interactive_prints_only_new_tool_trace_per_request(
    monkeypatch,
    tmp_path,
    capsys,
):
    class FakeGateway:
        closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(
            self,
            exc_type,
            exc_value,
            traceback,
        ):
            self.closed = True

    class FakeContext:
        dataset_ids = ()

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeAgent:
        session_id = "trace-isolation-session"

        def __init__(self):
            self.inputs = []
            self.messages = []
            self.call_count = 0

        async def run(self, user_input):
            self.inputs.append(user_input)
            self.call_count += 1

            if self.call_count == 1:
                self.messages.extend(
                    [
                        LLMMessage(
                            role="assistant",
                            content="Calculating derived metric.",
                            tool_calls=[
                                ToolCall(
                                    id="trace_call_001",
                                    name="analysis.derived_metric",
                                    arguments={
                                        "dataset_id": "dataset-1",
                                        "left_column": "Quantity",
                                        "right_column": "UnitPrice",
                                    },
                                )
                            ],
                        ),
                        make_tool_message(
                            "trace_call_001",
                            success=True,
                            output={
                                "row_count": 541909,
                                "valid_row_count": 541909,
                                "excluded_row_count": 0,
                                "result": 9747747.934,
                            },
                            metadata={
                                "tool_name": "analysis.derived_metric",
                                "status": "success",
                                "execution_time_seconds": 0.42,
                            },
                        ),
                    ]
                )

            # The second request deliberately adds no Tool Calls.
            return make_response(user_input)

        def get_messages(self):
            return list(self.messages)

        def reset(self):
            self.messages.clear()

    gateway = FakeGateway()
    context = FakeContext()
    agent = FakeAgent()

    monkeypatch.setattr(
        cli,
        "LLMGateway",
        lambda: gateway,
    )
    monkeypatch.setattr(
        cli,
        "ExecutionContext",
        lambda: context,
    )
    monkeypatch.setattr(
        cli,
        "create_agent",
        lambda *args, **kwargs: agent,
    )
    monkeypatch.setattr(
        cli.settings,
        "DATA_DIR",
        tmp_path / "data",
    )

    inputs = iter(
        [
            "第一次请求：计算派生指标",
            "第二次请求：普通对话",
            "exit",
        ]
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs),
    )

    asyncio.run(cli.run_interactive())

    assert agent.inputs == [
        "第一次请求：计算派生指标",
        "第二次请求：普通对话",
    ]

    assert context.closed is True
    assert gateway.closed is True

    output = capsys.readouterr().out

    # The first request's Tool Execution Trace must appear exactly once.
    assert output.count("Tool Execution Trace") == 1
    assert output.count("trace_call_001") == 1

    # Actual Tool Result values must be displayed.
    assert "9747747.934" in output
    assert "541909" in output

    # Both requests should still receive their assistant responses.
    assert "第一次请求：计算派生指标" in output
    assert "第二次请求：普通对话" in output
