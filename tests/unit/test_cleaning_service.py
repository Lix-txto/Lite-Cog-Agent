"""Unit tests for CleaningService initialization."""

from unittest.mock import Mock

import pytest

import pandas as pd

from lite_cog_agent.cleaning.service import CleaningService
from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.cleaning.contracts import (
    CleaningRequest,
    CleaningStatus,
    CleaningStep,
)
from lite_cog_agent.execution.dataset_store import DatasetReference
from lite_cog_agent.cleaning.exceptions import (
    CleaningExecutionError,
    CleaningValidationError,
)
from lite_cog_agent.profiling.exceptions import DatasetProfilingError
from lite_cog_agent.profiling.service import ProfilingService

EXPECTED_OPERATIONS = {
    "missing.drop",
    "missing.fill",
    "duplicates.drop",
    "columns.select",
    "columns.drop",
    "columns.rename",
    "types.convert",
}


def test_service_initializes_with_default_profiling_service():
    context = ExecutionContext()

    service = CleaningService(context)

    assert service.context is context
    assert service.profiling_service is not None
    assert callable(service.profiling_service.profile)


def test_service_accepts_injected_profiling_service():
    context = ExecutionContext()
    profiling_service = Mock()
    profiling_service.profile = Mock()

    service = CleaningService(
        context,
        profiling_service=profiling_service,
    )

    assert service.profiling_service is profiling_service


def test_service_registers_all_cleaning_operations():
    context = ExecutionContext()

    service = CleaningService(context)

    assert set(service._operations) == EXPECTED_OPERATIONS

    for operation in service._operations.values():
        assert callable(operation)


def test_service_rejects_invalid_context():
    with pytest.raises(
        TypeError,
        match="context must be an ExecutionContext",
    ):
        CleaningService(context={})


def test_service_rejects_invalid_profiling_service():
    context = ExecutionContext()

    with pytest.raises(
        TypeError,
        match="profiling_service must provide a callable profile method",
    ):
        CleaningService(
            context,
            profiling_service=object(),
        )

def test_execute_step_returns_success_record():
    context = ExecutionContext()
    service = CleaningService(context)

    dataframe = pd.DataFrame({
        "name": ["Alice", None],
        "age": [20, 30],
    })

    step = CleaningStep(
        operation="missing.drop",
        parameters={"how": "any"},
    )

    result, record = service._execute_step(dataframe, step)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1

    assert record.status is CleaningStatus.SUCCESS
    assert record.operation == "missing.drop"
    assert record.rows_before == 2
    assert record.rows_after == 1
    assert record.columns_before == 2
    assert record.columns_after == 2
    assert record.error is None

    # The original DataFrame must remain unchanged.
    assert len(dataframe) == 2
    assert dataframe["name"].isna().sum() == 1


def test_execute_step_returns_failure_for_unknown_operation():
    context = ExecutionContext()
    service = CleaningService(context)

    dataframe = pd.DataFrame({"value": [1, 2, 3]})

    step = CleaningStep(
        operation="unknown.operation",
    )

    result, record = service._execute_step(dataframe, step)

    assert result is None
    assert record.status is CleaningStatus.FAILED
    assert record.error is not None
    assert record.error.code == "cleaning_validation_error"
    assert record.rows_before == 3
    assert record.rows_after is None


def test_execute_step_returns_failure_for_invalid_parameters():
    context = ExecutionContext()
    service = CleaningService(context)

    dataframe = pd.DataFrame({"value": [1, None, 3]})

    step = CleaningStep(
        operation="missing.drop",
        parameters={"how": "invalid"},
    )

    result, record = service._execute_step(dataframe, step)

    assert result is None
    assert record.status is CleaningStatus.FAILED
    assert record.error is not None
    assert record.error.code == "cleaning_validation_error"

    # Failed execution must not modify the original DataFrame.
    assert len(dataframe) == 3

def create_registered_dataset(
    context: ExecutionContext,
) -> tuple[DatasetReference, pd.DataFrame]:
    dataframe = pd.DataFrame({
        "name": ["Alice", "Bob", None],
        "age": [20, 30, 40],
        "city": ["NY", "LA", "SF"],
    })

    reference = DatasetReference(
        name="users",
        source="test",
        file_format="csv",
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        columns=tuple(dataframe.columns),
    )

    context.register_dataset(reference, dataframe)

    return reference, dataframe

def test_clean_executes_complete_workflow():
    context = ExecutionContext()
    source_reference, source_dataframe = create_registered_dataset(context)

    service = CleaningService(context)

    request = CleaningRequest(
        dataset_id=source_reference.dataset_id,
        steps=(
            CleaningStep(
                operation="missing.drop",
                parameters={"how": "any"},
            ),
            CleaningStep(
                operation="columns.select",
                parameters={"columns": ["name", "age"]},
            ),
        ),
    )

    result = service.clean(request)

    assert result.status is CleaningStatus.SUCCESS
    assert result.source_dataset_id == source_reference.dataset_id

    assert len(result.operations) == 2
    assert all(
        record.status is CleaningStatus.SUCCESS
        for record in result.operations
    )

    assert result.input_row_count == 3
    assert result.input_column_count == 3

    assert result.output_row_count == 2
    assert result.output_column_count == 2

    assert result.output_dataset_id is not None
    assert result.error is None
    assert result.quality_comparison is not None

    output_dataframe = context.get_dataset(
        result.output_dataset_id
    )

    assert list(output_dataframe.columns) == ["name", "age"]
    assert len(output_dataframe) == 2

    # The original dataset must remain unchanged.
    assert len(source_dataframe) == 3
    assert source_dataframe["name"].isna().sum() == 1

def test_clean_stops_after_failed_operation():
    context = ExecutionContext()
    source_reference, source_dataframe = create_registered_dataset(context)

    # Capture the original resource state.
    original_dataframe = source_dataframe.copy(deep=True)
    original_dataset_ids = context.dataset_ids
    original_dataset_count = len(context.dataset_store)

    service = CleaningService(context)

    request = CleaningRequest(
        dataset_id=source_reference.dataset_id,
        steps=(
            CleaningStep(
                operation="missing.drop",
                parameters={"how": "any"},
            ),
            CleaningStep(
                operation="unknown.operation",
            ),
            CleaningStep(
                operation="columns.drop",
                parameters={"columns": ["city"]},
            ),
        ),
    )

    result = service.clean(request)

    # 1. Workflow should fail.
    assert result.status is CleaningStatus.FAILED

    # 2. The first operation succeeds, the second fails.
    #    The third operation must never execute.
    assert len(result.operations) == 2

    assert result.operations[0].status is CleaningStatus.SUCCESS
    assert result.operations[1].status is CleaningStatus.FAILED

    assert result.operations[0].operation == "missing.drop"
    assert result.operations[1].operation == "unknown.operation"

    assert result.error is not None
    assert result.error.code == "cleaning_validation_error"

    # 3. No output dataset should be registered.
    assert result.output_dataset_id is None
    assert result.output_row_count is None
    assert result.output_column_count is None

    assert len(context.dataset_store) == original_dataset_count
    assert context.dataset_ids == original_dataset_ids

    # 4. The source dataset must remain accessible.
    assert context.get_dataset(source_reference.dataset_id) is source_dataframe

    # 5. The source DataFrame must remain completely unchanged.
    pd.testing.assert_frame_equal(
        source_dataframe,
        original_dataframe,
    )

    # 6. The original dataset reference must remain unchanged.
    assert (
        context.get_dataset_reference(source_reference.dataset_id)
        == source_reference
    )

def test_clean_returns_failure_when_before_profiling_fails():
    context = ExecutionContext()
    source_reference, _ = create_registered_dataset(context)

    profiling_service = Mock()
    profiling_service.profile.side_effect = RuntimeError(
        "Profiling unavailable"
    )

    service = CleaningService(
        context,
        profiling_service=profiling_service,
    )

    request = CleaningRequest(
        dataset_id=source_reference.dataset_id,
        steps=(
            CleaningStep(
                operation="missing.drop",
                parameters={"how": "any"},
            ),
        ),
    )

    result = service.clean(request)

    assert result.status is CleaningStatus.FAILED
    assert result.error is not None
    assert result.error.code == "cleaning_execution_error"
    assert result.error.details["stage"] == "profiling_before"

    assert result.operations == ()
    assert result.output_dataset_id is None
    assert len(context.dataset_store) == 1

def test_clean_registers_output_dataset_reference():
    context = ExecutionContext()
    source_reference, _ = create_registered_dataset(context)

    service = CleaningService(context)

    request = CleaningRequest(
        dataset_id=source_reference.dataset_id,
        output_dataset_name="users_cleaned_v1",
        steps=(
            CleaningStep(
                operation="columns.drop",
                parameters={"columns": ["city"]},
            ),
        ),
    )

    result = service.clean(request)

    assert result.status is CleaningStatus.SUCCESS

    output_reference = context.get_dataset_reference(
        result.output_dataset_id
    )

    assert output_reference.name == "users_cleaned_v1"
    assert output_reference.source == "cleaning"
    assert output_reference.file_format is None
    assert output_reference.row_count == 3
    assert output_reference.column_count == 2
    assert output_reference.columns == ("name", "age")
    assert (
        output_reference.metadata["source_dataset_id"]
        == source_reference.dataset_id
    )

def test_clean_raises_error_when_source_dataset_does_not_exist():
    context = ExecutionContext()
    service = CleaningService(context)

    request = CleaningRequest(
        dataset_id="missing-dataset-id",
        steps=(
            CleaningStep(
                operation="missing.drop",
                parameters={"how": "any"},
            ),
        ),
    )

    with pytest.raises(
        CleaningValidationError,
        match="Source dataset not found",
    ) as exc_info:
        service.clean(request)

    assert exc_info.value.code == "dataset_not_found"
    assert len(context.dataset_store) == 0

def test_clean_raises_error_when_context_is_closed():
    context = ExecutionContext()
    source_reference, _ = create_registered_dataset(context)

    context.close()

    service = CleaningService(context)

    request = CleaningRequest(
        dataset_id=source_reference.dataset_id,
        steps=(
            CleaningStep(
                operation="missing.drop",
                parameters={"how": "any"},
            ),
        ),
    )

    with pytest.raises(
        CleaningExecutionError,
        match="Execution context is unavailable",
    ) as exc_info:
        service.clean(request)

    assert exc_info.value.code == "execution_context_error"

def test_clean_does_not_profile_when_source_dataset_is_missing():
    context = ExecutionContext()

    profiling_service = Mock()
    profiling_service.profile = Mock()

    service = CleaningService(
        context,
        profiling_service=profiling_service,
    )

    request = CleaningRequest(
        dataset_id="missing-dataset-id",
        steps=(
            CleaningStep(
                operation="missing.drop",
                parameters={"how": "any"},
            ),
        ),
    )

    with pytest.raises(CleaningValidationError):
        service.clean(request)

    profiling_service.profile.assert_not_called()

def test_clean_does_not_profile_when_context_is_closed():
    context = ExecutionContext()
    source_reference, _ = create_registered_dataset(context)

    context.close()

    profiling_service = Mock()
    profiling_service.profile = Mock()

    service = CleaningService(
        context,
        profiling_service=profiling_service,
    )

    request = CleaningRequest(
        dataset_id=source_reference.dataset_id,
        steps=(
            CleaningStep(
                operation="missing.drop",
                parameters={"how": "any"},
            ),
        ),
    )

    with pytest.raises(CleaningExecutionError):
        service.clean(request)

    profiling_service.profile.assert_not_called()

@pytest.mark.parametrize(
    (
        "profiling_exception",
        "expected_error_code",
        "expected_exception_type",
    ),
    [
        pytest.param(
            DatasetProfilingError("Dataset profiling failed"),
            "profiling_error",
            "DatasetProfilingError",
            id="profiling-exception",
        ),
        pytest.param(
            RuntimeError("Unexpected profiling failure"),
            "cleaning_execution_error",
            "CleaningExecutionError",
            id="unexpected-exception",
        ),
    ],
)
def test_clean_returns_failure_when_after_profiling_fails(
    profiling_exception,
    expected_error_code,
    expected_exception_type,
):
    context = ExecutionContext()
    source_reference, source_dataframe = create_registered_dataset(
        context
    )

    # Generate a real report for the before-profiling stage.
    real_profiling_service = ProfilingService()
    before_report = real_profiling_service.profile(
        source_dataframe,
        source_reference,
    )

    # The first call succeeds; the second call fails.
    profiling_service = Mock()
    profiling_service.profile = Mock(
        side_effect=[
            before_report,
            profiling_exception,
        ]
    )

    service = CleaningService(
        context,
        profiling_service=profiling_service,
    )

    request = CleaningRequest(
        dataset_id=source_reference.dataset_id,
        steps=(
            CleaningStep(
                operation="missing.drop",
                parameters={"how": "any"},
            ),
        ),
    )

    result = service.clean(request)

    # The workflow must fail.
    assert result.status is CleaningStatus.FAILED
    assert result.error is not None
    assert result.error.code == expected_error_code

    # The failure must be identified as an after-profiling failure.
    assert result.error.details["stage"] == "profiling_after"
    assert (
        result.error.details["exception_type"]
        == expected_exception_type
    )

    # The cleaning operation itself should have succeeded.
    assert len(result.operations) == 1
    assert result.operations[0].status is CleaningStatus.SUCCESS

    # No output dataset should be registered.
    assert result.output_dataset_id is None
    assert result.quality_comparison is None
    assert len(context.dataset_store) == 1

    # The original source dataset must remain unchanged.
    assert len(source_dataframe) == 3
    assert source_dataframe["name"].isna().sum() == 1

    # Profiling should have been called exactly twice:
    # before profiling + after profiling.
    assert profiling_service.profile.call_count == 2

def test_clean_returns_failure_when_output_registration_fails(
    monkeypatch,
):
    context = ExecutionContext()
    source_reference, source_dataframe = create_registered_dataset(
        context
    )

    original_dataframe = source_dataframe.copy(deep=True)
    original_dataset_ids = context.dataset_ids
    original_dataset_count = len(context.dataset_store)

    # Track profiling calls while keeping the real profiling behavior.
    profiling_service = Mock(wraps=ProfilingService())

    service = CleaningService(
        context,
        profiling_service=profiling_service,
    )

    # Simulate a failure at the DatasetStore registration boundary.
    registration_mock = Mock(
        side_effect=RuntimeError("Simulated registration failure")
    )
    monkeypatch.setattr(
        context.dataset_store,
        "register",
        registration_mock,
    )

    request = CleaningRequest(
        dataset_id=source_reference.dataset_id,
        steps=(
            CleaningStep(
                operation="missing.drop",
                parameters={"how": "any"},
            ),
        ),
    )

    result = service.clean(request)

    # 1. Workflow should fail with a dedicated error.
    assert result.status is CleaningStatus.FAILED
    assert result.error is not None
    assert result.error.code == "dataset_registration_error"
    assert result.error.message == "Simulated registration failure"
    assert result.error.retryable is False

    assert result.error.details["stage"] == "register_output_dataset"
    assert result.error.details["exception_type"] == "RuntimeError"

    # 2. The cleaning operation should remain recorded as successful.
    assert len(result.operations) == 1
    assert result.operations[0].status is CleaningStatus.SUCCESS

    # 3. No output dataset should be reported or registered.
    assert result.output_dataset_id is None
    assert result.output_row_count is None
    assert result.output_column_count is None
    assert result.quality_comparison is None

    assert len(context.dataset_store) == original_dataset_count
    assert context.dataset_ids == original_dataset_ids

    # 4. The source dataset must remain available and unchanged.
    assert context.get_dataset(source_reference.dataset_id) is source_dataframe

    pd.testing.assert_frame_equal(
        source_dataframe,
        original_dataframe,
    )

    # 5. Before and after profiling should both have completed.
    assert profiling_service.profile.call_count == 2

    # 6. The registration boundary should have been reached exactly once.
    registration_mock.assert_called_once()

def test_clean_does_not_register_intermediate_dataframes(monkeypatch):
    context = ExecutionContext()
    source_reference, source_dataframe = create_registered_dataset(context)

    original_dataset_count = len(context.dataset_store)
    original_dataset_ids = context.dataset_ids

    # Monitor registrations only after the source dataset is registered.
    original_register = context.dataset_store.register
    register_spy = Mock(wraps=original_register)

    monkeypatch.setattr(
        context.dataset_store,
        "register",
        register_spy,
    )

    service = CleaningService(context)

    request = CleaningRequest(
        dataset_id=source_reference.dataset_id,
        steps=(
            CleaningStep(
                operation="missing.drop",
                parameters={"how": "any"},
            ),
            CleaningStep(
                operation="unknown.operation",
            ),
            CleaningStep(
                operation="columns.drop",
                parameters={"columns": ["city"]},
            ),
        ),
    )

    result = service.clean(request)

    # The workflow should fail at the second operation.
    assert result.status is CleaningStatus.FAILED
    assert len(result.operations) == 2

    assert result.operations[0].status is CleaningStatus.SUCCESS
    assert result.operations[1].status is CleaningStatus.FAILED

    # No intermediate or final DataFrame should be registered.
    register_spy.assert_not_called()

    # Context should retain exactly its original resources.
    assert len(context.dataset_store) == original_dataset_count
    assert context.dataset_ids == original_dataset_ids

    # The source dataset must remain accessible and unchanged.
    assert context.get_dataset(source_reference.dataset_id) is source_dataframe

    assert (
        context.get_dataset_reference(source_reference.dataset_id)
        == source_reference
    )

def test_clean_failure_preserves_all_existing_datasets():
    context = ExecutionContext()

    # Register the source dataset.
    source_reference, source_dataframe = create_registered_dataset(context)

    # Register an unrelated dataset in the same context.
    unrelated_dataframe = pd.DataFrame(
        {
            "product": ["A", "B", "C"],
            "price": [10.0, 20.0, 30.0],
        }
    )

    unrelated_reference = DatasetReference(
        name="unrelated_dataset",
        source="test",
        file_format=None,
        row_count=len(unrelated_dataframe),
        column_count=len(unrelated_dataframe.columns),
        columns=tuple(unrelated_dataframe.columns),
    )

    context.register_dataset(
        unrelated_reference,
        unrelated_dataframe,
    )

    # Capture the original resource state.
    source_snapshot = source_dataframe.copy(deep=True)
    unrelated_snapshot = unrelated_dataframe.copy(deep=True)

    original_dataset_ids = set(context.dataset_ids)
    original_dataset_count = len(context.dataset_store)

    service = CleaningService(context)

    request = CleaningRequest(
        dataset_id=source_reference.dataset_id,
        steps=(
            CleaningStep(
                operation="missing.drop",
                parameters={"how": "any"},
            ),
            CleaningStep(
                operation="unknown.operation",
            ),
        ),
    )

    result = service.clean(request)

    # The cleaning workflow should fail.
    assert result.status is CleaningStatus.FAILED
    assert result.error is not None
    assert result.error.code == "cleaning_validation_error"

    # No output dataset should be registered.
    assert result.output_dataset_id is None
    assert len(context.dataset_store) == original_dataset_count
    assert set(context.dataset_ids) == original_dataset_ids

    # Both datasets must remain accessible by identity.
    assert (
        context.get_dataset(source_reference.dataset_id)
        is source_dataframe
    )
    assert (
        context.get_dataset(unrelated_reference.dataset_id)
        is unrelated_dataframe
    )

    # Both DataFrames must retain their original contents.
    pd.testing.assert_frame_equal(
        source_dataframe,
        source_snapshot,
    )
    pd.testing.assert_frame_equal(
        unrelated_dataframe,
        unrelated_snapshot,
    )

    # Both DatasetReferences must remain unchanged.
    assert (
        context.get_dataset_reference(source_reference.dataset_id)
        == source_reference
    )
    assert (
        context.get_dataset_reference(unrelated_reference.dataset_id)
        == unrelated_reference
    )

def test_context_remains_usable_after_cleaning_failure():
    context = ExecutionContext()

    source_reference, source_dataframe = create_registered_dataset(context)

    service = CleaningService(context)

    failed_request = CleaningRequest(
        dataset_id=source_reference.dataset_id,
        steps=(
            CleaningStep(
                operation="missing.drop",
                parameters={"how": "any"},
            ),
            CleaningStep(
                operation="unknown.operation",
            ),
        ),
    )

    # First workflow fails.
    failed_result = service.clean(failed_request)

    assert failed_result.status is CleaningStatus.FAILED
    assert failed_result.error is not None

    # Context must still support reading existing datasets.
    assert context.get_dataset(source_reference.dataset_id) is source_dataframe

    assert (
        context.get_dataset_reference(source_reference.dataset_id)
        == source_reference
    )

    # Context must still support registering a new dataset.
    new_dataframe = pd.DataFrame(
        {
            "name": ["Alice", "Bob"],
            "score": [90, 85],
        }
    )

    new_reference = DatasetReference(
        name="post_failure_dataset",
        source="test",
        file_format=None,
        row_count=len(new_dataframe),
        column_count=len(new_dataframe.columns),
        columns=tuple(new_dataframe.columns),
    )

    context.register_dataset(new_reference, new_dataframe)

    assert context.get_dataset(new_reference.dataset_id) is new_dataframe

    assert (
        context.get_dataset_reference(new_reference.dataset_id)
        == new_reference
    )

    # Context must also support a new successful cleaning workflow.
    recovery_request = CleaningRequest(
        dataset_id=new_reference.dataset_id,
        steps=(
            CleaningStep(
                operation="columns.select",
                parameters={"columns": ["name", "score"]},
            ),
        ),
        output_dataset_name="recovered_dataset",
    )

    recovery_result = service.clean(recovery_request)

    assert recovery_result.status is CleaningStatus.SUCCESS
    assert recovery_result.error is None
    assert recovery_result.output_dataset_id is not None

    recovered_dataframe = context.get_dataset(
        recovery_result.output_dataset_id
    )

    assert list(recovered_dataframe.columns) == ["name", "score"]

    pd.testing.assert_frame_equal(
        recovered_dataframe,
        new_dataframe,
    )

    # The original source dataset must remain unchanged.
    assert context.get_dataset(source_reference.dataset_id) is source_dataframe