"""Application service for coordinating dataset cleaning workflows."""

from __future__ import annotations

import time

from collections.abc import Callable

import pandas as pd

from lite_cog_agent.cleaning.columns import ColumnCleaner
from lite_cog_agent.cleaning.duplicates import DuplicateCleaner
from lite_cog_agent.cleaning.missing import MissingValueCleaner
from lite_cog_agent.cleaning.type_converter import TypeConverter
from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.profiling.service import ProfilingService
from lite_cog_agent.cleaning.contracts import (
    CleaningErrorInfo,
    CleaningOperationRecord,
    CleaningQualityComparison,
    CleaningRequest,
    CleaningResult,
    CleaningStatus,
    CleaningStep,
)
from lite_cog_agent.cleaning.exceptions import (
    CleaningExecutionError,
    CleaningOperationError,
    CleaningValidationError,
)
from lite_cog_agent.execution.dataset_store import DatasetReference
from lite_cog_agent.profiling.exceptions import ProfilingException


class CleaningService:
    """Coordinate cleaning operations and dataset quality assessment."""

    def __init__(
        self,
        context: ExecutionContext,
        profiling_service: ProfilingService | None = None,
    ) -> None:
        """Initialize the cleaning service and its dependencies.

        Args:
            context: Context containing source and output datasets.
            profiling_service: Optional profiling service dependency.

        Raises:
            TypeError: If a dependency has an invalid type or interface.
        """
        if not isinstance(context, ExecutionContext):
            raise TypeError("context must be an ExecutionContext")

        if profiling_service is None:
            profiling_service = ProfilingService()

        if not callable(getattr(profiling_service, "profile", None)):
            raise TypeError(
                "profiling_service must provide a callable profile method"
            )

        self.context = context
        self.profiling_service = profiling_service

        self._missing_cleaner = MissingValueCleaner()
        self._duplicate_cleaner = DuplicateCleaner()
        self._column_cleaner = ColumnCleaner()
        self._type_converter = TypeConverter()

        self._operations: dict[
            str,
            Callable[..., pd.DataFrame],
        ] = {
            "missing.drop": self._missing_cleaner.drop,
            "missing.fill": self._missing_cleaner.fill,
            "duplicates.drop": self._duplicate_cleaner.drop_duplicates,
            "columns.select": self._column_cleaner.select,
            "columns.drop": self._column_cleaner.drop,
            "columns.rename": self._column_cleaner.rename,
            "types.convert": self._type_converter.convert,
        }

    def _build_error_info(
        self,
        exception: Exception,
        operation: str,
    ) -> CleaningErrorInfo:
        """Convert an exception into a structured cleaning error."""

        if isinstance(exception, CleaningValidationError):
            code = "cleaning_validation_error"
            retryable = False

        elif isinstance(exception, CleaningOperationError):
            code = "cleaning_operation_error"
            retryable = exception.retryable

        elif isinstance(exception, CleaningExecutionError):
            code = "cleaning_execution_error"
            retryable = exception.retryable

        else:
            code = "cleaning_execution_error"
            retryable = False

        return CleaningErrorInfo(
            code=code,
            message=str(exception) or type(exception).__name__,
            retryable=retryable,
            details={
                "operation": operation,
                "exception_type": type(exception).__name__,
            },
        )

    def _build_success_record(
        self,
        step: CleaningStep,
        dataframe_before: pd.DataFrame,
        dataframe_after: pd.DataFrame,
        duration_ms: float,
    ) -> CleaningOperationRecord:
        """Build a successful cleaning operation record."""

        return CleaningOperationRecord(
            operation=step.operation,
            status=CleaningStatus.SUCCESS,
            parameters=dict(step.parameters),
            rows_before=len(dataframe_before),
            rows_after=len(dataframe_after),
            columns_before=len(dataframe_before.columns),
            columns_after=len(dataframe_after.columns),
            duration_ms=duration_ms,
            error=None,
        )

    def _build_failure_record(
        self,
        step: CleaningStep,
        dataframe_before: pd.DataFrame,
        duration_ms: float,
        exception: Exception,
    ) -> CleaningOperationRecord:
        """Build a failed cleaning operation record."""

        return CleaningOperationRecord(
            operation=step.operation,
            status=CleaningStatus.FAILED,
            parameters=dict(step.parameters),
            rows_before=len(dataframe_before),
            rows_after=None,
            columns_before=len(dataframe_before.columns),
            columns_after=None,
            duration_ms=duration_ms,
            error=self._build_error_info(
                exception=exception,
                operation=step.operation,
            ),
        )

    def _execute_step(
        self,
        dataframe: pd.DataFrame,
        step: CleaningStep,
    ) -> tuple[pd.DataFrame | None, CleaningOperationRecord]:
        """Execute one cleaning step and build its execution record."""

        started_at = time.perf_counter()

        operation = self._operations.get(step.operation)

        if operation is None:
            exception = CleaningValidationError(
                f"Unknown cleaning operation: {step.operation}"
            )

            return None, self._build_failure_record(
                step=step,
                dataframe_before=dataframe,
                duration_ms=0.0,
                exception=exception,
            )

        try:
            result = operation(
                dataframe,
                **dict(step.parameters),
            )

            if not isinstance(result, pd.DataFrame):
                raise CleaningExecutionError(
                    "Cleaning operation must return a pandas DataFrame",
                    failed_operation=step.operation,
                )

        except (CleaningValidationError,
                CleaningOperationError,
                CleaningExecutionError,
              ) as exc:
            duration_ms = (
                time.perf_counter() - started_at
            ) * 1000

            return None, self._build_failure_record(
                step=step,
                dataframe_before=dataframe,
                duration_ms=duration_ms,
                exception=exc,
            )

        except Exception as exc:
            duration_ms = (
                time.perf_counter() - started_at
            ) * 1000

            wrapped_exception = CleaningExecutionError(
                f"Unexpected error while executing "
                f"'{step.operation}': {exc}",
                failed_operation=step.operation,
            )

            return None, self._build_failure_record(
                step=step,
                dataframe_before=dataframe,
                duration_ms=duration_ms,
                exception=wrapped_exception,
            )

        duration_ms = (
            time.perf_counter() - started_at
        ) * 1000

        record = self._build_success_record(
            step=step,
            dataframe_before=dataframe,
            dataframe_after=result,
            duration_ms=duration_ms,
        )

        return result, record

    def _get_source_dataset(
        self,
        dataset_id: str,
    ) -> tuple[pd.DataFrame, DatasetReference]:
        """Retrieve the source DataFrame and its reference."""

        try:
            dataframe = self.context.get_dataset(dataset_id)
            reference = self.context.get_dataset_reference(dataset_id)

        except KeyError as exc:
            error = CleaningValidationError(
                f"Source dataset not found: {dataset_id}"
            )
            error.code = "dataset_not_found"
            raise error from exc

        except RuntimeError as exc:
            error = CleaningExecutionError(
                f"Execution context is unavailable: {exc}"
            )
            error.code = "execution_context_error"
            raise error from exc

        if not isinstance(dataframe, pd.DataFrame):
            error = CleaningExecutionError(
                "Source dataset must be a pandas DataFrame"
            )
            error.code = "dataset_validation_error"
            raise error

        if not isinstance(reference, DatasetReference):
            error = CleaningExecutionError(
                "Source dataset reference is invalid"
            )
            error.code = "dataset_reference_error"
            raise error

        return dataframe, reference

    def _profile_dataset(
        self,
        dataframe: pd.DataFrame,
        reference: DatasetReference,
    ):
        """Generate a profiling report for a DataFrame."""

        try:
            return self.profiling_service.profile(
                dataframe,
                reference,
            )

        except ProfilingException:
            raise

        except Exception as exc:
            raise CleaningExecutionError(
                f"Unexpected profiling error: {exc}"
            ) from exc

    def _build_output_reference(
        self,
        request: CleaningRequest,
        source_reference: DatasetReference,
        dataframe: pd.DataFrame,
    ) -> DatasetReference:
        """Build a reference for the cleaned output dataset."""

        output_name = (
            request.output_dataset_name
            or f"{source_reference.name}_cleaned"
        )

        return DatasetReference(
            name=output_name,
            source="cleaning",
            file_format=None,
            row_count=len(dataframe),
            column_count=len(dataframe.columns),
            columns=tuple(dataframe.columns),
            metadata={
                "source_dataset_id": source_reference.dataset_id,
            },
        )

    def _build_workflow_error(
        self,
        exception: Exception,
        stage: str,
    ) -> CleaningErrorInfo:
        """Convert a workflow-level exception into error information."""

        if isinstance(exception, ProfilingException):
            code = "profiling_error"

        elif isinstance(exception, CleaningValidationError):
            code = "cleaning_validation_error"

        elif isinstance(exception, CleaningOperationError):
            code = "cleaning_operation_error"

        elif isinstance(exception, CleaningExecutionError):
            code = "cleaning_execution_error"

        else:
            code = "cleaning_execution_error"

        return CleaningErrorInfo(
            code=code,
            message=str(exception) or type(exception).__name__,
            retryable=False,
            details={
                "stage": stage,
                "exception_type": type(exception).__name__,
            },
        )

    def _build_failed_result(
        self,
        request: CleaningRequest,
        input_row_count: int,
        input_column_count: int,
        operations: list[CleaningOperationRecord],
        started_at: float,
        error: CleaningErrorInfo,
    ) -> CleaningResult:
        """Build a failed cleaning workflow result."""

        duration_ms = (
            time.perf_counter() - started_at
        ) * 1000

        return CleaningResult(
            source_dataset_id=request.dataset_id,
            status=CleaningStatus.FAILED,
            input_row_count=input_row_count,
            input_column_count=input_column_count,
            operations=tuple(operations),
            duration_ms=duration_ms,
            error=error,
        )

    def clean(self, request: CleaningRequest) -> CleaningResult:
        """Execute a complete dataset cleaning workflow."""

        if not isinstance(request, CleaningRequest):
            raise TypeError("request must be a CleaningRequest")

        started_at = time.perf_counter()
        operation_records: list[CleaningOperationRecord] = []

        # Step 1: Retrieve the source dataset.
        # Preflight failures are raised because input dimensions are unknown.
        source_dataframe, source_reference = self._get_source_dataset(
            request.dataset_id
        )

        input_row_count = len(source_dataframe)
        input_column_count = len(source_dataframe.columns)

        # Step 2: Profile the source dataset.
        try:
            before_report = self._profile_dataset(
                source_dataframe,
                source_reference,
            )

        except Exception as exc:
            error = self._build_workflow_error(
                exception=exc,
                stage="profiling_before",
            )

            return self._build_failed_result(
                request=request,
                input_row_count=input_row_count,
                input_column_count=input_column_count,
                operations=operation_records,
                started_at=started_at,
                error=error,
            )

        # Step 3: Execute cleaning steps sequentially.
        working_dataframe = source_dataframe

        for step in request.steps:
            cleaned_dataframe, record = self._execute_step(
                dataframe=working_dataframe,
                step=step,
            )

            operation_records.append(record)

            if record.status is CleaningStatus.FAILED:
                return self._build_failed_result(
                    request=request,
                    input_row_count=input_row_count,
                    input_column_count=input_column_count,
                    operations=operation_records,
                    started_at=started_at,
                    error=record.error,
                )

            if cleaned_dataframe is None:
                error = CleaningErrorInfo(
                    code="cleaning_execution_error",
                    message="Successful cleaning step returned no DataFrame",
                    details={
                        "stage": "cleaning_step",
                        "operation": step.operation,
                    },
                )

                return self._build_failed_result(
                    request=request,
                    input_row_count=input_row_count,
                    input_column_count=input_column_count,
                    operations=operation_records,
                    started_at=started_at,
                    error=error,
                )

            working_dataframe = cleaned_dataframe

        # Step 4: Build the output dataset reference.
        try:
            output_reference = self._build_output_reference(
                request=request,
                source_reference=source_reference,
                dataframe=working_dataframe,
            )

        except Exception as exc:
            error = self._build_workflow_error(
                exception=exc,
                stage="build_output_reference",
            )

            return self._build_failed_result(
                request=request,
                input_row_count=input_row_count,
                input_column_count=input_column_count,
                operations=operation_records,
                started_at=started_at,
                error=error,
            )

        # Step 5: Profile the cleaned dataset.
        try:
            after_report = self._profile_dataset(
                working_dataframe,
                output_reference,
            )

            quality_comparison = CleaningQualityComparison(
                before=before_report.quality_report,
                after=after_report.quality_report,
            )

        except Exception as exc:
            error = self._build_workflow_error(
                exception=exc,
                stage="profiling_after",
            )

            return self._build_failed_result(
                request=request,
                input_row_count=input_row_count,
                input_column_count=input_column_count,
                operations=operation_records,
                started_at=started_at,
                error=error,
            )

        # Step 6: Register the final output dataset.
        try:
            self.context.register_dataset(
                output_reference,
                working_dataframe,
            )

        except Exception as exc:
            error = CleaningErrorInfo(
                code="dataset_registration_error",
                message=str(exc) or type(exc).__name__,
                retryable=False,
                details={
                    "stage": "register_output_dataset",
                    "exception_type": type(exc).__name__,
                },
            )

            return self._build_failed_result(
                request=request,
                input_row_count=input_row_count,
                input_column_count=input_column_count,
                operations=operation_records,
                started_at=started_at,
                error=error,
            )

        # Step 7: Return the successful workflow result.
        duration_ms = (
            time.perf_counter() - started_at
        ) * 1000

        return CleaningResult(
            source_dataset_id=request.dataset_id,
            status=CleaningStatus.SUCCESS,
            input_row_count=input_row_count,
            input_column_count=input_column_count,
            operations=tuple(operation_records),
            duration_ms=duration_ms,
            output_dataset_id=output_reference.dataset_id,
            output_row_count=len(working_dataframe),
            output_column_count=len(working_dataframe.columns),
            error=None,
            quality_comparison=quality_comparison,
        )