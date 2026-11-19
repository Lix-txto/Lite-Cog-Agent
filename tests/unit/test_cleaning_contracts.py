"""Unit tests for cleaning contracts."""

from types import MappingProxyType

import pytest

from lite_cog_agent.cleaning.contracts import (
    CleaningErrorInfo,
    CleaningOperationRecord,
    CleaningRequest,
    CleaningResult,
    CleaningStatus,
    CleaningStep,
    CleaningQualityComparison,
)
from lite_cog_agent.profiling.contracts import DataQualityReport


class TestCleaningStep:
    def test_creates_step(self):
        step = CleaningStep(
            operation="fill_missing",
            parameters={"column": "age", "strategy": "mean"},
        )

        assert step.operation == "fill_missing"
        assert step.parameters["column"] == "age"

    def test_parameters_are_read_only(self):
        step = CleaningStep(
            operation="drop_missing",
            parameters={"columns": ["name"]},
        )

        assert isinstance(step.parameters, MappingProxyType)

        with pytest.raises(TypeError):
            step.parameters["columns"] = ["age"]

    def test_parameters_are_shallow_copied(self):
        parameters = {"column": "age"}
        step = CleaningStep(
            operation="fill_missing",
            parameters=parameters,
        )

        parameters["column"] = "price"

        assert step.parameters["column"] == "age"

    @pytest.mark.parametrize("operation", ["", " ", None, 123])
    def test_rejects_invalid_operation(self, operation):
        with pytest.raises((TypeError, ValueError)):
            CleaningStep(operation=operation)


class TestCleaningRequest:
    def test_creates_request(self):
        request = CleaningRequest(
            dataset_id="dataset-1",
            steps=(
                CleaningStep("drop_missing"),
                CleaningStep("drop_duplicates"),
            ),
            output_dataset_name="cleaned-data",
        )

        assert request.dataset_id == "dataset-1"
        assert len(request.steps) == 2
        assert request.output_dataset_name == "cleaned-data"

    def test_preserves_step_order(self):
        request = CleaningRequest(
            dataset_id="dataset-1",
            steps=(
                CleaningStep("drop_missing"),
                CleaningStep("convert_type"),
            ),
        )

        assert [step.operation for step in request.steps] == [
            "drop_missing",
            "convert_type",
        ]

    def test_rejects_empty_steps(self):
        with pytest.raises(ValueError, match="at least one"):
            CleaningRequest(
                dataset_id="dataset-1",
                steps=(),
            )

    def test_rejects_non_tuple_steps(self):
        with pytest.raises(TypeError, match="tuple"):
            CleaningRequest(
                dataset_id="dataset-1",
                steps=[CleaningStep("drop_missing")],
            )

    def test_rejects_invalid_step_member(self):
        with pytest.raises(TypeError):
            CleaningRequest(
                dataset_id="dataset-1",
                steps=("drop_missing",),
            )

    def test_rejects_invalid_dataset_id(self):
        with pytest.raises(ValueError):
            CleaningRequest(
                dataset_id=" ",
                steps=(CleaningStep("drop_missing"),),
            )

    def test_rejects_invalid_output_name(self):
        with pytest.raises(ValueError):
            CleaningRequest(
                dataset_id="dataset-1",
                steps=(CleaningStep("drop_missing"),),
                output_dataset_name=" ",
            )


class TestCleaningErrorInfo:
    def test_creates_error_info(self):
        error = CleaningErrorInfo(
            code="cleaning_failed",
            message="Cleaning failed",
            retryable=False,
            details={"step": "drop_missing"},
        )

        assert error.code == "cleaning_failed"
        assert error.details["step"] == "drop_missing"
        assert isinstance(error.details, MappingProxyType)

    def test_rejects_empty_code(self):
        with pytest.raises(ValueError):
            CleaningErrorInfo(code="", message="Failed")

    def test_rejects_empty_message(self):
        with pytest.raises(ValueError):
            CleaningErrorInfo(code="failed", message=" ")

    def test_rejects_invalid_retryable(self):
        with pytest.raises(TypeError):
            CleaningErrorInfo(
                code="failed",
                message="Failed",
                retryable=1,
            )


class TestCleaningOperationRecord:
    def test_creates_success_record(self):
        record = CleaningOperationRecord(
            operation="drop_missing",
            status=CleaningStatus.SUCCESS,
            rows_before=100,
            rows_after=90,
            columns_before=5,
            columns_after=5,
            duration_ms=12.5,
        )

        assert record.status is CleaningStatus.SUCCESS
        assert record.rows_before == 100
        assert record.rows_after == 90

    def test_creates_failed_record(self):
        error = CleaningErrorInfo(
            code="operation_failed",
            message="Invalid column",
        )

        record = CleaningOperationRecord(
            operation="drop_missing",
            status=CleaningStatus.FAILED,
            rows_before=100,
            columns_before=5,
            error=error,
        )

        assert record.status is CleaningStatus.FAILED
        assert record.rows_after is None
        assert record.error is error

    def test_rejects_success_record_with_error(self):
        error = CleaningErrorInfo(
            code="failed",
            message="Failed",
        )

        with pytest.raises(ValueError):
            CleaningOperationRecord(
                operation="drop_missing",
                status=CleaningStatus.SUCCESS,
                rows_before=10,
                rows_after=10,
                columns_before=2,
                columns_after=2,
                error=error,
            )

    def test_rejects_success_record_without_output_dimensions(self):
        with pytest.raises(ValueError):
            CleaningOperationRecord(
                operation="drop_missing",
                status=CleaningStatus.SUCCESS,
                rows_before=10,
                columns_before=2,
            )

    def test_rejects_failed_record_without_error(self):
        with pytest.raises(ValueError):
            CleaningOperationRecord(
                operation="drop_missing",
                status=CleaningStatus.FAILED,
                rows_before=10,
                columns_before=2,
            )

    @pytest.mark.parametrize("value", [-1, 1.5, True])
    def test_rejects_invalid_rows_before(self, value):
        with pytest.raises((TypeError, ValueError)):
            CleaningOperationRecord(
                operation="drop_missing",
                status=CleaningStatus.SUCCESS,
                rows_before=value,
                rows_after=10,
                columns_before=2,
                columns_after=2,
            )


class TestCleaningResult:
    def test_creates_success_result(self):
        record = CleaningOperationRecord(
            operation="drop_missing",
            status=CleaningStatus.SUCCESS,
            rows_before=100,
            rows_after=90,
            columns_before=5,
            columns_after=5,
        )

        result = CleaningResult(
            source_dataset_id="dataset-1",
            output_dataset_id="dataset-2",
            status=CleaningStatus.SUCCESS,
            input_row_count=100,
            input_column_count=5,
            output_row_count=90,
            output_column_count=5,
            operations=(record,),
        )

        assert result.source_dataset_id == "dataset-1"
        assert result.output_dataset_id == "dataset-2"
        assert result.output_row_count == 90

    def test_creates_failed_result(self):
        error = CleaningErrorInfo(
            code="cleaning_execution_error",
            message="Cleaning failed",
        )

        result = CleaningResult(
            source_dataset_id="dataset-1",
            status=CleaningStatus.FAILED,
            input_row_count=100,
            input_column_count=5,
            operations=(),
            error=error,
        )

        assert result.status is CleaningStatus.FAILED
        assert result.output_dataset_id is None
        assert result.output_row_count is None

    def test_rejects_success_without_output_dataset(self):
        with pytest.raises(ValueError, match="output_dataset_id"):
            CleaningResult(
                source_dataset_id="dataset-1",
                status=CleaningStatus.SUCCESS,
                input_row_count=100,
                input_column_count=5,
                output_row_count=90,
                output_column_count=5,
                operations=(),
            )

    def test_rejects_success_with_error(self):
        error = CleaningErrorInfo(
            code="failed",
            message="Failed",
        )

        with pytest.raises(ValueError):
            CleaningResult(
                source_dataset_id="dataset-1",
                output_dataset_id="dataset-2",
                status=CleaningStatus.SUCCESS,
                input_row_count=100,
                input_column_count=5,
                output_row_count=90,
                output_column_count=5,
                operations=(),
                error=error,
            )

    def test_rejects_failed_result_without_error(self):
        with pytest.raises(ValueError, match="must contain an error"):
            CleaningResult(
                source_dataset_id="dataset-1",
                status=CleaningStatus.FAILED,
                input_row_count=100,
                input_column_count=5,
                operations=(),
            )

    def test_rejects_failed_result_with_output_dataset(self):
        error = CleaningErrorInfo(
            code="failed",
            message="Failed",
        )

        with pytest.raises(ValueError, match="cannot contain"):
            CleaningResult(
                source_dataset_id="dataset-1",
                output_dataset_id="dataset-2",
                status=CleaningStatus.FAILED,
                input_row_count=100,
                input_column_count=5,
                operations=(),
                error=error,
            )

    def test_rejects_non_tuple_operations(self):
        error = CleaningErrorInfo(
            code="failed",
            message="Failed",
        )

        with pytest.raises(TypeError, match="tuple"):
            CleaningResult(
                source_dataset_id="dataset-1",
                status=CleaningStatus.FAILED,
                input_row_count=100,
                input_column_count=5,
                operations=[],
                error=error,
            )


def test_cleaning_quality_comparison_accepts_valid_reports():
    before = DataQualityReport(
        total_cells=100,
        missing_cells=20,
        completeness=80.0,
        missing_rate=20.0,
        duplicate_row_count=5,
        duplicate_rate=5.0,
        column_completeness={},
    )
    after = DataQualityReport(
        total_cells=80,
        missing_cells=0,
        completeness=100.0,
        missing_rate=0.0,
        duplicate_row_count=0,
        duplicate_rate=0.0,
        column_completeness={},
    )

    comparison = CleaningQualityComparison(
        before=before,
        after=after,
    )

    assert comparison.before is before
    assert comparison.after is after


def test_cleaning_quality_comparison_rejects_invalid_before():
    after = DataQualityReport(
        total_cells=80,
        missing_cells=0,
        completeness=100.0,
        missing_rate=0.0,
        duplicate_row_count=0,
        duplicate_rate=0.0,
        column_completeness={},
    )

    with pytest.raises(TypeError, match="before must be a DataQualityReport"):
        CleaningQualityComparison(
            before={},
            after=after,
        )


def test_cleaning_quality_comparison_rejects_invalid_after():
    before = DataQualityReport(
        total_cells=100,
        missing_cells=20,
        completeness=80.0,
        missing_rate=20.0,
        duplicate_row_count=5,
        duplicate_rate=5.0,
        column_completeness={},
    )

    with pytest.raises(TypeError, match="after must be a DataQualityReport"):
        CleaningQualityComparison(
            before=before,
            after={},
        )
