"""Tests for execution contracts."""

from dataclasses import FrozenInstanceError

import pytest

from lite_cog_agent.execution.contracts import (
    ExecutionFailure,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
)


class TestExecutionRequest:
    def test_create_request_with_defaults(self):
        request = ExecutionRequest(operation="data.profiling")

        assert request.operation == "data.profiling"
        assert request.parameters == {}
        assert request.dataset_id is None
        assert request.timeout_seconds is None
        assert request.execution_id

    def test_create_request_with_parameters(self):
        request = ExecutionRequest(
            operation="data.cleaning",
            parameters={"strategy": "drop_nulls"},
            dataset_id="dataset-001",
            timeout_seconds=30,
        )

        assert request.parameters["strategy"] == "drop_nulls"
        assert request.dataset_id == "dataset-001"
        assert request.timeout_seconds == 30

    def test_execution_id_is_unique(self):
        first = ExecutionRequest(operation="data.profiling")
        second = ExecutionRequest(operation="data.profiling")

        assert first.execution_id != second.execution_id

    @pytest.mark.parametrize(
        "operation",
        ["", " ", None],
    )
    def test_invalid_operation(self, operation):
        with pytest.raises(ValueError):
            ExecutionRequest(operation=operation)

    def test_invalid_parameters_type(self):
        with pytest.raises(TypeError):
            ExecutionRequest(
                operation="data.profiling",
                parameters=["invalid"],
            )

    def test_parameters_are_read_only(self):
        parameters = {"strategy": "drop_nulls"}
        request = ExecutionRequest(
            operation="data.cleaning",
            parameters=parameters,
        )

        parameters["strategy"] = "fill_nulls"

        assert request.parameters["strategy"] == "drop_nulls"

        with pytest.raises(TypeError):
            request.parameters["strategy"] = "fill_nulls"

    @pytest.mark.parametrize(
        "timeout",
        [0, -1, float("inf"), float("nan"), True],
    )
    def test_invalid_timeout(self, timeout):
        with pytest.raises((ValueError, TypeError)):
            ExecutionRequest(
                operation="data.profiling",
                timeout_seconds=timeout,
            )

    def test_request_is_immutable(self):
        request = ExecutionRequest(operation="data.profiling")

        with pytest.raises(FrozenInstanceError):
            request.operation = "data.cleaning"


class TestExecutionFailure:
    def test_create_failure(self):
        failure = ExecutionFailure(
            code="operation_not_found",
            message="Operation does not exist",
            retryable=False,
            details={"operation": "data.unknown"},
        )

        assert failure.code == "operation_not_found"
        assert failure.message == "Operation does not exist"
        assert failure.retryable is False
        assert failure.details["operation"] == "data.unknown"

    def test_failure_details_are_read_only(self):
        failure = ExecutionFailure(
            code="execution_error",
            message="Something went wrong",
            details={"reason": "unknown"},
        )

        with pytest.raises(TypeError):
            failure.details["reason"] = "changed"

    def test_invalid_failure_code(self):
        with pytest.raises(ValueError):
            ExecutionFailure(
                code="",
                message="Something went wrong",
            )

    def test_invalid_failure_message(self):
        with pytest.raises(ValueError):
            ExecutionFailure(
                code="execution_error",
                message=" ",
            )


class TestExecutionResult:
    def test_create_success_result(self):
        result = ExecutionResult(
            execution_id="execution-001",
            operation="data.profiling",
            status=ExecutionStatus.SUCCESS,
            output={"rows": 100, "columns": 5},
            duration_ms=12.5,
        )

        assert result.is_success is True
        assert result.error is None
        assert result.output["rows"] == 100

    def test_create_failed_result(self):
        failure = ExecutionFailure(
            code="operation_execution_error",
            message="Analysis failed",
        )

        result = ExecutionResult(
            execution_id="execution-002",
            operation="data.profiling",
            status=ExecutionStatus.FAILED,
            error=failure,
        )

        assert result.is_success is False
        assert result.error is failure

    def test_success_result_cannot_contain_error(self):
        failure = ExecutionFailure(
            code="execution_error",
            message="Unexpected error",
        )

        with pytest.raises(ValueError):
            ExecutionResult(
                execution_id="execution-003",
                operation="data.profiling",
                status=ExecutionStatus.SUCCESS,
                error=failure,
            )

    def test_failed_result_requires_error(self):
        with pytest.raises(ValueError):
            ExecutionResult(
                execution_id="execution-004",
                operation="data.profiling",
                status=ExecutionStatus.FAILED,
            )

    def test_invalid_status(self):
        with pytest.raises(TypeError):
            ExecutionResult(
                execution_id="execution-005",
                operation="data.profiling",
                status="success",
            )

    @pytest.mark.parametrize(
        "duration",
        [-1, float("inf"), float("nan"), True],
    )
    def test_invalid_duration(self, duration):
        with pytest.raises((ValueError, TypeError)):
            ExecutionResult(
                execution_id="execution-006",
                operation="data.profiling",
                status=ExecutionStatus.SUCCESS,
                duration_ms=duration,
            )

    def test_metadata_is_read_only(self):
        result = ExecutionResult(
            execution_id="execution-007",
            operation="data.profiling",
            status=ExecutionStatus.SUCCESS,
            metadata={"source": "test"},
        )

        with pytest.raises(TypeError):
            result.metadata["source"] = "changed"
