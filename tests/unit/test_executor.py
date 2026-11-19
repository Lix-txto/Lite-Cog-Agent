from typing import Any

import pytest

from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.execution.contracts import (
    ExecutionFailure,
    ExecutionRequest,
    ExecutionStatus,
)
from lite_cog_agent.execution.exceptions import (
    ExecutionException,
    ExecutionTimeoutError,
)
from lite_cog_agent.execution.executor import PythonExecutor
from lite_cog_agent.execution.registry import (
    BaseOperation,
    ExecutionRegistry,
)


class DummyOperation(BaseOperation):
    """A simple operation for executor testing."""

    def __init__(
        self,
        *,
        name: str = "dummy_operation",
        output: Any = "success",
        error: BaseException | None = None,
    ) -> None:
        self._name = name
        self._output = output
        self._error = error
        self.received_request = None
        self.received_context = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Dummy operation for testing."

    def execute(
        self,
        request: ExecutionRequest,
        context: ExecutionContext,
    ) -> Any:
        self.received_request = request
        self.received_context = context

        if self._error is not None:
            raise self._error

        return self._output


@pytest.fixture
def registry() -> ExecutionRegistry:
    return ExecutionRegistry()


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext()


@pytest.fixture
def executor(registry: ExecutionRegistry) -> PythonExecutor:
    return PythonExecutor(registry)


class TestPythonExecutorInitialization:
    def test_create_executor(self, registry: ExecutionRegistry) -> None:
        executor = PythonExecutor(registry)

        assert executor is not None

    def test_reject_invalid_registry(self) -> None:
        with pytest.raises(TypeError):
            PythonExecutor(None)


class TestPythonExecutorSuccess:
    def test_execute_registered_operation(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
    ) -> None:
        operation = DummyOperation(output={"value": 100})
        registry.register(operation)

        executor = PythonExecutor(registry)
        request = ExecutionRequest(operation="dummy_operation")

        result = executor.execute(request, context)

        assert result.status is ExecutionStatus.SUCCESS
        assert result.output == {"value": 100}
        assert result.error is None
        assert result.is_success is True

    def test_preserve_execution_id(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
    ) -> None:
        registry.register(DummyOperation())

        request = ExecutionRequest(
            operation="dummy_operation",
            execution_id="test-execution-001",
        )

        result = PythonExecutor(registry).execute(request, context)

        assert result.execution_id == request.execution_id

    def test_preserve_operation_name(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
    ) -> None:
        registry.register(DummyOperation())

        request = ExecutionRequest(operation="dummy_operation")

        result = PythonExecutor(registry).execute(request, context)

        assert result.operation == request.operation

    def test_pass_request_and_context_to_operation(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
    ) -> None:
        operation = DummyOperation()
        registry.register(operation)

        request = ExecutionRequest(
            operation="dummy_operation",
            parameters={"column": "revenue"},
        )

        PythonExecutor(registry).execute(request, context)

        assert operation.received_request is request
        assert operation.received_context is context

    def test_support_none_output(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
    ) -> None:
        registry.register(DummyOperation(output=None))

        request = ExecutionRequest(operation="dummy_operation")

        result = PythonExecutor(registry).execute(request, context)

        assert result.status is ExecutionStatus.SUCCESS
        assert result.output is None

    def test_record_execution_duration(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
    ) -> None:
        registry.register(DummyOperation())

        request = ExecutionRequest(operation="dummy_operation")

        result = PythonExecutor(registry).execute(request, context)

        assert result.duration_ms >= 0


    def test_dispatch_to_requested_operation(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
    ) -> None:
        first_operation = DummyOperation(
            name="first_operation",
            output={"source": "first"},
        )
        second_operation = DummyOperation(
            name="second_operation",
            output={"source": "second"},
        )

        registry.register(first_operation)
        registry.register(second_operation)

        request = ExecutionRequest(operation="second_operation")

        result = PythonExecutor(registry).execute(request, context)

        assert result.status is ExecutionStatus.SUCCESS
        assert result.output == {"source": "second"}
        assert first_operation.received_request is None
        assert second_operation.received_request is request

    def test_executor_uses_live_registry(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
    ) -> None:
        executor = PythonExecutor(registry)

        operation = DummyOperation(
            name="registered_after_executor_creation",
            output="dynamic registration",
        )
        registry.register(operation)

        request = ExecutionRequest(
            operation="registered_after_executor_creation",
        )

        result = executor.execute(request, context)

        assert result.status is ExecutionStatus.SUCCESS
        assert result.output == "dynamic registration"
        assert operation.received_request is request

    def test_preserve_empty_dict_output(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
    ) -> None:
        operation = DummyOperation(
            name="empty_dict_operation",
            output={},
        )
        registry.register(operation)

        request = ExecutionRequest(operation="empty_dict_operation")
        result = PythonExecutor(registry).execute(request, context)

        assert result.status is ExecutionStatus.SUCCESS
        assert result.output == {}
        assert result.error is None

    def test_preserve_empty_list_output(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
    ) -> None:
        operation = DummyOperation(
            name="empty_list_operation",
            output=[],
        )
        registry.register(operation)

        request = ExecutionRequest(operation="empty_list_operation")
        result = PythonExecutor(registry).execute(request, context)

        assert result.status is ExecutionStatus.SUCCESS
        assert result.output == []
        assert result.error is None

    def test_preserve_zero_output(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
    ) -> None:
        operation = DummyOperation(
            name="zero_operation",
            output=0,
        )
        registry.register(operation)

        request = ExecutionRequest(operation="zero_operation")
        result = PythonExecutor(registry).execute(request, context)

        assert result.status is ExecutionStatus.SUCCESS
        assert result.output == 0
        assert result.error is None

    def test_preserve_false_output(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
    ) -> None:
        operation = DummyOperation(
            name="false_operation",
            output=False,
        )
        registry.register(operation)

        request = ExecutionRequest(operation="false_operation")
        result = PythonExecutor(registry).execute(request, context)

        assert result.status is ExecutionStatus.SUCCESS
        assert result.output is False
        assert result.error is None

    def test_preserve_nested_output_reference(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
    ) -> None:
        output = {
            "summary": {
                "count": 3,
                "mean": 20.0,
            },
            "columns": ["age", "score"],
        }

        operation = DummyOperation(
            name="nested_output_operation",
            output=output,
        )
        registry.register(operation)

        request = ExecutionRequest(operation="nested_output_operation")
        result = PythonExecutor(registry).execute(request, context)

        assert result.status is ExecutionStatus.SUCCESS
        assert result.output == output
        assert result.output is output
        assert result.error is None

    def test_calculate_execution_duration_in_milliseconds(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        timestamps = iter([10.0, 10.125])

        monkeypatch.setattr(
            "lite_cog_agent.execution.executor.perf_counter",
            lambda: next(timestamps),
        )

        registry.register(DummyOperation())

        request = ExecutionRequest(operation="dummy_operation")

        result = PythonExecutor(registry).execute(request, context)

        assert result.status is ExecutionStatus.SUCCESS
        assert result.duration_ms == pytest.approx(125.0)

    def test_execute_without_timeout(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
    ) -> None:
        registry.register(DummyOperation())

        request = ExecutionRequest(
            operation="dummy_operation",
        )

        result = PythonExecutor(registry).execute(request, context)

        assert result.status is ExecutionStatus.SUCCESS
        assert result.error is None

    def test_execute_within_timeout(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        timestamps = iter([10.0, 12.0])

        monkeypatch.setattr(
            "lite_cog_agent.execution.executor.perf_counter",
            lambda: next(timestamps),
        )

        registry.register(DummyOperation())

        request = ExecutionRequest(
            operation="dummy_operation",
            timeout_seconds=5.0,
        )

        result = PythonExecutor(registry).execute(request, context)

        assert result.status is ExecutionStatus.SUCCESS
        assert result.duration_ms == pytest.approx(2000.0)

    def test_timeout_boundary_is_not_exceeded(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        timestamps = iter([10.0, 15.0])

        monkeypatch.setattr(
            "lite_cog_agent.execution.executor.perf_counter",
            lambda: next(timestamps),
        )

        registry.register(DummyOperation())

        request = ExecutionRequest(
            operation="dummy_operation",
            timeout_seconds=5.0,
        )

        result = PythonExecutor(registry).execute(request, context)

        assert result.status is ExecutionStatus.SUCCESS
        assert result.duration_ms == pytest.approx(5000.0)

class TestPythonExecutorFailure:
    def test_unregistered_operation_returns_failure(
        self,
        executor: PythonExecutor,
        context: ExecutionContext,
    ) -> None:
        request = ExecutionRequest(operation="unknown_operation")

        result = executor.execute(request, context)

        assert result.status is ExecutionStatus.FAILED
        assert result.error is not None
        assert result.error.code == "operation_not_found"
        assert result.is_success is False

    def test_preserve_execution_exception_details(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
    ) -> None:
        error = ExecutionException(
            "Invalid analysis parameters",
            code="invalid_parameters",
            retryable=True,
        )

        registry.register(DummyOperation(error=error))

        request = ExecutionRequest(operation="dummy_operation")

        result = PythonExecutor(registry).execute(request, context)

        assert result.status is ExecutionStatus.FAILED
        assert result.error is not None
        assert result.error.code == "invalid_parameters"
        assert result.error.message == "Invalid analysis parameters"
        assert result.error.retryable is True

    def test_convert_unexpected_exception(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
    ) -> None:
        error = ValueError("Unexpected operation failure")

        registry.register(DummyOperation(error=error))

        request = ExecutionRequest(operation="dummy_operation")

        result = PythonExecutor(registry).execute(request, context)

        assert result.status is ExecutionStatus.FAILED
        assert result.error is not None
        assert result.error.code == "operation_execution_error"
        assert result.error.message == "Unexpected operation failure"
        assert result.error.retryable is False
        assert result.error.details["exception_type"] == "ValueError"

    def test_failure_result_contains_execution_id(
        self,
        executor: PythonExecutor,
        context: ExecutionContext,
    ) -> None:
        request = ExecutionRequest(
            operation="unknown_operation",
            execution_id="failed-execution-001",
        )

        result = executor.execute(request, context)

        assert result.execution_id == request.execution_id

    def test_failure_result_records_duration(
        self,
        executor: PythonExecutor,
        context: ExecutionContext,
    ) -> None:
        request = ExecutionRequest(operation="unknown_operation")

        result = executor.execute(request, context)

        assert result.duration_ms >= 0

    def test_failure_result_contains_execution_failure(
        self,
        executor: PythonExecutor,
        context: ExecutionContext,
    ) -> None:
        request = ExecutionRequest(operation="unknown_operation")

        result = executor.execute(request, context)

        assert isinstance(result.error, ExecutionFailure)

    def test_execution_exception_with_empty_message(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
    ) -> None:
        error = ExecutionException(
            "",
            code="empty_error",
            retryable=False,
        )
        registry.register(DummyOperation(error=error))

        request = ExecutionRequest(operation="dummy_operation")

        result = PythonExecutor(registry).execute(request, context)

        assert result.status is ExecutionStatus.FAILED
        assert result.error is not None
        assert result.error.code == "empty_error"
        assert result.error.message == "ExecutionException"
        assert result.error.retryable is False

    def test_unexpected_exception_with_empty_message(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
    ) -> None:
        error = ValueError("")
        registry.register(DummyOperation(error=error))

        request = ExecutionRequest(operation="dummy_operation")

        result = PythonExecutor(registry).execute(request, context)

        assert result.status is ExecutionStatus.FAILED
        assert result.error is not None
        assert result.error.code == "operation_execution_error"
        assert result.error.message == "ValueError"
        assert result.error.retryable is False
        assert result.error.details["exception_type"] == "ValueError"

    def test_failure_result_does_not_contain_output(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
    ) -> None:
        error = ValueError("Analysis failed")

        registry.register(
            DummyOperation(
                output={"unexpected": "output"},
                error=error,
            )
        )

        request = ExecutionRequest(operation="dummy_operation")

        result = PythonExecutor(registry).execute(request, context)

        assert result.status is ExecutionStatus.FAILED
        assert result.output is None
        assert result.error is not None

    def test_keyboard_interrupt_propagates(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
    ) -> None:
        registry.register(
            DummyOperation(error=KeyboardInterrupt())
        )

        request = ExecutionRequest(operation="dummy_operation")

        with pytest.raises(KeyboardInterrupt):
            PythonExecutor(registry).execute(request, context)

    def test_system_exit_propagates(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
    ) -> None:
        registry.register(
            DummyOperation(error=SystemExit(1))
        )

        request = ExecutionRequest(operation="dummy_operation")

        with pytest.raises(SystemExit):
            PythonExecutor(registry).execute(request, context)

    def test_calculate_failure_duration_in_milliseconds(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        timestamps = iter([20.0, 20.250])

        monkeypatch.setattr(
            "lite_cog_agent.execution.executor.perf_counter",
            lambda: next(timestamps),
        )

        registry.register(
            DummyOperation(error=ValueError("Analysis failed"))
        )

        request = ExecutionRequest(operation="dummy_operation")

        result = PythonExecutor(registry).execute(request, context)

        assert result.status is ExecutionStatus.FAILED
        assert result.duration_ms == pytest.approx(250.0)

    def test_operation_exceeding_timeout_returns_failure(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        timestamps = iter([10.0, 15.1])

        monkeypatch.setattr(
            "lite_cog_agent.execution.executor.perf_counter",
            lambda: next(timestamps),
        )

        registry.register(DummyOperation())

        request = ExecutionRequest(
            operation="dummy_operation",
            timeout_seconds=5.0,
        )

        result = PythonExecutor(registry).execute(request, context)

        assert result.status is ExecutionStatus.FAILED
        assert result.error is not None
        assert result.error.code == "execution_timeout"
        assert result.error.retryable is False

    def test_timeout_failure_contains_execution_details(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        timestamps = iter([10.0, 15.1])

        monkeypatch.setattr(
            "lite_cog_agent.execution.executor.perf_counter",
            lambda: next(timestamps),
        )

        registry.register(DummyOperation())

        request = ExecutionRequest(
            operation="dummy_operation",
            timeout_seconds=5.0,
        )

        result = PythonExecutor(registry).execute(request, context)

        assert result.error is not None
        assert result.error.details["timeout_seconds"] == 5.0
        assert result.error.details["elapsed_ms"] == pytest.approx(5100.0)
        assert result.error.details["timeout_type"] == "soft"
        assert result.error.details["operation_completed"] is True

    def test_timeout_failure_does_not_contain_output(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        timestamps = iter([10.0, 15.1])

        monkeypatch.setattr(
            "lite_cog_agent.execution.executor.perf_counter",
            lambda: next(timestamps),
        )

        registry.register(DummyOperation())

        request = ExecutionRequest(
            operation="dummy_operation",
            timeout_seconds=5.0,
        )

        result = PythonExecutor(registry).execute(request, context)

        assert result.status is ExecutionStatus.FAILED
        assert result.output is None

    def test_operation_exception_takes_precedence_over_timeout(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        timestamps = iter([10.0, 16.0])

        monkeypatch.setattr(
            "lite_cog_agent.execution.executor.perf_counter",
            lambda: next(timestamps),
        )

        registry.register(
            DummyOperation(error=ValueError("Analysis failed"))
        )

        request = ExecutionRequest(
            operation="dummy_operation",
            timeout_seconds=5.0,
        )

        result = PythonExecutor(registry).execute(request, context)

        assert result.status is ExecutionStatus.FAILED
        assert result.error is not None
        assert result.error.code == "operation_execution_error"
        assert result.error.message == "Analysis failed"
        assert result.error.details["exception_type"] == "ValueError"
        assert result.error.code != "execution_timeout"
        assert result.duration_ms == pytest.approx(6000.0)

    def test_timeout_failure_preserves_execution_identity(
        self,
        registry: ExecutionRegistry,
        context: ExecutionContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        timestamps = iter([10.0, 15.1])

        monkeypatch.setattr(
            "lite_cog_agent.execution.executor.perf_counter",
            lambda: next(timestamps),
        )

        registry.register(DummyOperation())

        request = ExecutionRequest(
            operation="dummy_operation",
            timeout_seconds=5.0,
            execution_id="execution-timeout-001",
        )

        result = PythonExecutor(registry).execute(request, context)

        assert result.status is ExecutionStatus.FAILED
        assert result.execution_id == "execution-timeout-001"
        assert result.operation == "dummy_operation"
        assert result.duration_ms == pytest.approx(5100.0)

class TestPythonExecutorValidation:
    def test_reject_invalid_request(
        self,
        executor: PythonExecutor,
        context: ExecutionContext,
    ) -> None:
        with pytest.raises(TypeError):
            executor.execute(None, context)

    def test_reject_invalid_context(
        self,
        executor: PythonExecutor,
    ) -> None:
        request = ExecutionRequest(operation="dummy_operation")

        with pytest.raises(TypeError):
            executor.execute(request, None)


class TestExecutionTimeoutError:
    def test_timeout_error_is_not_retryable(self) -> None:
        error = ExecutionTimeoutError(5.0)

        assert error.code == "execution_timeout"
        assert error.retryable is False
        assert error.timeout_seconds == 5.0

    def test_timeout_error_preserves_timeout_value(self) -> None:
        error = ExecutionTimeoutError(3.5)

        assert error.timeout_seconds == 3.5
        assert "3.5" in str(error)

    def test_timeout_error_is_execution_exception(self) -> None:
        error = ExecutionTimeoutError(5.0)

        assert isinstance(error, ExecutionException)