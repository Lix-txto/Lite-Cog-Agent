"""End-to-end integration tests for dataset loading and cleaning."""

import pytest
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype

from lite_cog_agent.cleaning.contracts import (
    CleaningRequest,
    CleaningStatus,
    CleaningStep,
)
from lite_cog_agent.cleaning.service import CleaningService
from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.loading.contracts import DatasetLoadRequest
from lite_cog_agent.loading.service import DatasetLoadingService
from lite_cog_agent.profiling.service import ProfilingService
from lite_cog_agent.execution.dataset_store import DatasetReference
from lite_cog_agent.loading.exceptions import (
    DatasetFileNotFoundError,
    DatasetParseError,
    UnsupportedFileFormatError,
)


def test_csv_loading_cleaning_and_profiling_end_to_end(tmp_path):
    # ---------------------------------------------------------
    # Step 1: Prepare a real CSV file.
    # ---------------------------------------------------------
    csv_path = tmp_path / "sales_data.csv"

    csv_path.write_text(
        "name,age,city,score\n"
        "Alice,25,New York,90\n"
        "Bob,,Los Angeles,85\n"
        "Alice,25,New York,90\n"
        "Charlie,30,,88\n"
        "Diana,35,San Francisco,92\n",
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Step 2: Initialize the execution environment.
    # ---------------------------------------------------------
    context = ExecutionContext()

    loading_service = DatasetLoadingService()
    cleaning_service = CleaningService(context)
    profiling_service = ProfilingService()

    # ---------------------------------------------------------
    # Step 3: Load the CSV into ExecutionContext.
    # ---------------------------------------------------------
    load_request = DatasetLoadRequest(
        file_path=csv_path,
        name="sales_data",
    )

    source_reference = loading_service.load_into_context(
        load_request,
        context,
    )

    source_dataframe = context.get_dataset(
        source_reference.dataset_id
    )

    assert isinstance(source_dataframe, pd.DataFrame)

    assert source_reference.name == "sales_data"
    assert source_reference.row_count == 5
    assert source_reference.column_count == 4

    assert list(source_dataframe.columns) == [
        "name",
        "age",
        "city",
        "score",
    ]

    assert len(source_dataframe) == 5

    # Preserve the original dataset for integrity checks.
    original_dataframe = source_dataframe.copy(deep=True)

    # ---------------------------------------------------------
    # Step 4: Profile the original dataset.
    # ---------------------------------------------------------
    before_profile = profiling_service.profile(
        source_dataframe,
        source_reference,
    )

    before_quality = before_profile.quality_report

    assert before_profile.dataset_profile.row_count == 5
    assert before_profile.dataset_profile.column_count == 4

    assert before_quality.total_cells == 20
    assert before_quality.missing_cells == 2
    assert before_quality.completeness == 90.0
    assert before_quality.missing_rate == 10.0
    assert before_quality.duplicate_row_count == 1

    # ---------------------------------------------------------
    # Step 5: Execute the cleaning workflow.
    # ---------------------------------------------------------
    cleaning_request = CleaningRequest(
        dataset_id=source_reference.dataset_id,
        steps=(
            CleaningStep(
                operation="missing.drop",
                parameters={
                    "how": "any",
                },
            ),
            CleaningStep(
                operation="duplicates.drop",
                parameters={
                    "keep": "first",
                    "ignore_index": True,
                },
            ),
            CleaningStep(
                operation="types.convert",
                parameters={
                    "conversions": {
                        "age": "int",
                        "score": "float",
                    },
                },
            ),
            CleaningStep(
                operation="columns.select",
                parameters={
                    "columns": [
                        "name",
                        "age",
                        "score",
                    ],
                },
            ),
        ),
        output_dataset_name="sales_data_cleaned",
    )

    cleaning_result = cleaning_service.clean(
        cleaning_request
    )

    # ---------------------------------------------------------
    # Step 6: Validate the cleaning result.
    # ---------------------------------------------------------
    assert cleaning_result.status is CleaningStatus.SUCCESS
    assert cleaning_result.error is None

    assert len(cleaning_result.operations) == 4

    assert all(
        operation.status is CleaningStatus.SUCCESS
        for operation in cleaning_result.operations
    )

    assert cleaning_result.input_row_count == 5
    assert cleaning_result.input_column_count == 4

    assert cleaning_result.output_row_count == 2
    assert cleaning_result.output_column_count == 3

    assert cleaning_result.output_dataset_id is not None

    assert cleaning_result.quality_comparison is not None

    # ---------------------------------------------------------
    # Step 7: Retrieve the cleaned dataset from Context.
    # ---------------------------------------------------------
    cleaned_dataframe = context.get_dataset(
        cleaning_result.output_dataset_id
    )

    cleaned_reference = context.get_dataset_reference(
        cleaning_result.output_dataset_id
    )

    assert cleaned_reference.name == "sales_data_cleaned"
    assert cleaned_reference.source == "cleaning"
    assert cleaned_reference.file_format is None

    assert list(cleaned_dataframe.columns) == [
        "name",
        "age",
        "score",
    ]

    assert len(cleaned_dataframe) == 2

    # Verify actual data values.
    assert cleaned_dataframe["name"].tolist() == [
        "Alice",
        "Diana",
    ]

    assert cleaned_dataframe["age"].tolist() == [
        25,
        35,
    ]

    assert cleaned_dataframe["score"].tolist() == [
        90.0,
        92.0,
    ]

    # Verify converted data types.
    assert is_integer_dtype(cleaned_dataframe["age"])
    assert is_float_dtype(cleaned_dataframe["score"])

    # ---------------------------------------------------------
    # Step 8: Profile the cleaned dataset.
    # ---------------------------------------------------------
    after_profile = profiling_service.profile(
        cleaned_dataframe,
        cleaned_reference,
    )

    after_quality = after_profile.quality_report

    assert after_profile.dataset_profile.row_count == 2
    assert after_profile.dataset_profile.column_count == 3

    assert after_quality.total_cells == 6
    assert after_quality.missing_cells == 0
    assert after_quality.completeness == 100.0
    assert after_quality.missing_rate == 0.0
    assert after_quality.duplicate_row_count == 0

    # ---------------------------------------------------------
    # Step 9: Validate CleaningService quality comparison.
    # ---------------------------------------------------------
    comparison = cleaning_result.quality_comparison

    assert comparison.before == before_quality
    assert comparison.after == after_quality

    assert comparison.after.completeness > (
        comparison.before.completeness
    )

    assert comparison.after.missing_rate < (
        comparison.before.missing_rate
    )

    # ---------------------------------------------------------
    # Step 10: Verify source dataset integrity.
    # ---------------------------------------------------------
    assert context.get_dataset(
        source_reference.dataset_id
    ) is source_dataframe

    pd.testing.assert_frame_equal(
        source_dataframe,
        original_dataframe,
    )

    assert (
        context.get_dataset_reference(source_reference.dataset_id)
        == source_reference
    )

    # Both source and cleaned datasets should exist.
    assert context.dataset_store.contains(
        source_reference.dataset_id
    )

    assert context.dataset_store.contains(
        cleaning_result.output_dataset_id
    )

    assert len(context.dataset_store) == 2

def test_json_loading_cleaning_and_profiling_end_to_end(tmp_path):
    """Verify the complete workflow with a real JSON file."""
    json_path = tmp_path / "sales_data.json"

    json_path.write_text(
        """
        [
            {"name": "Alice", "score": 90},
            {"name": "Bob", "score": null},
            {"name": "Alice", "score": 90},
            {"name": "Diana", "score": 92}
        ]
        """,
        encoding="utf-8",
    )

    context = ExecutionContext()
    loading_service = DatasetLoadingService()
    cleaning_service = CleaningService(context)
    profiling_service = ProfilingService()

    source_reference = loading_service.load_into_context(
        DatasetLoadRequest(
            file_path=json_path,
            name="json_sales_data",
        ),
        context,
    )

    source_dataframe = context.get_dataset(
        source_reference.dataset_id
    )

    before_profile = profiling_service.profile(
        source_dataframe,
        source_reference,
    )

    assert len(source_dataframe) == 4
    assert before_profile.quality_report.missing_cells == 1

    result = cleaning_service.clean(
        CleaningRequest(
            dataset_id=source_reference.dataset_id,
            steps=(
                CleaningStep(
                    operation="missing.drop",
                    parameters={"how": "any"},
                ),
                CleaningStep(
                    operation="duplicates.drop",
                    parameters={
                        "keep": "first",
                        "ignore_index": True,
                    },
                ),
            ),
            output_dataset_name="json_sales_data_cleaned",
        )
    )

    assert result.status is CleaningStatus.SUCCESS
    assert result.output_row_count == 2
    assert result.output_column_count == 2
    assert len(result.operations) == 2

    cleaned_dataframe = context.get_dataset(
        result.output_dataset_id
    )
    cleaned_reference = context.get_dataset_reference(
        result.output_dataset_id
    )

    assert cleaned_dataframe["name"].tolist() == [
        "Alice",
        "Diana",
    ]

    after_profile = profiling_service.profile(
        cleaned_dataframe,
        cleaned_reference,
    )

    assert after_profile.quality_report.missing_cells == 0
    assert after_profile.quality_report.completeness == 100.0

    assert result.quality_comparison.before == (
        before_profile.quality_report
    )
    assert result.quality_comparison.after == (
        after_profile.quality_report
    )

    assert context.dataset_store.contains(
        source_reference.dataset_id
    )
    assert context.dataset_store.contains(
        result.output_dataset_id
    )


def test_excel_loading_cleaning_and_profiling_end_to_end(tmp_path):
    """Verify the complete workflow with a real XLSX file."""
    excel_path = tmp_path / "sales_data.xlsx"

    source_dataframe = pd.DataFrame(
        {
            "name": [
                "Alice",
                "Bob",
                "Alice",
                "Diana",
            ],
            "score": [
                90,
                None,
                90,
                92,
            ],
        }
    )

    source_dataframe.to_excel(
        excel_path,
        index=False,
        engine="openpyxl",
    )

    context = ExecutionContext()
    loading_service = DatasetLoadingService()
    cleaning_service = CleaningService(context)
    profiling_service = ProfilingService()

    source_reference = loading_service.load_into_context(
        DatasetLoadRequest(
            file_path=excel_path,
            name="excel_sales_data",
        ),
        context,
    )

    loaded_dataframe = context.get_dataset(
        source_reference.dataset_id
    )

    before_profile = profiling_service.profile(
        loaded_dataframe,
        source_reference,
    )

    assert len(loaded_dataframe) == 4
    assert before_profile.quality_report.missing_cells == 1

    result = cleaning_service.clean(
        CleaningRequest(
            dataset_id=source_reference.dataset_id,
            steps=(
                CleaningStep(
                    operation="missing.drop",
                    parameters={"how": "any"},
                ),
                CleaningStep(
                    operation="duplicates.drop",
                    parameters={
                        "keep": "first",
                        "ignore_index": True,
                    },
                ),
            ),
            output_dataset_name="excel_sales_data_cleaned",
        )
    )

    assert result.status is CleaningStatus.SUCCESS
    assert result.output_row_count == 2
    assert result.output_column_count == 2

    cleaned_dataframe = context.get_dataset(
        result.output_dataset_id
    )
    cleaned_reference = context.get_dataset_reference(
        result.output_dataset_id
    )

    assert cleaned_dataframe["name"].tolist() == [
        "Alice",
        "Diana",
    ]

    after_profile = profiling_service.profile(
        cleaned_dataframe,
        cleaned_reference,
    )

    assert after_profile.quality_report.missing_cells == 0
    assert after_profile.quality_report.completeness == 100.0

    assert result.quality_comparison.before == (
        before_profile.quality_report
    )
    assert result.quality_comparison.after == (
        after_profile.quality_report
    )

    assert context.dataset_store.contains(
        source_reference.dataset_id
    )
    assert context.dataset_store.contains(
        result.output_dataset_id
    )

def _create_context_with_existing_dataset():
    """Create a context containing one existing dataset."""
    context = ExecutionContext()

    dataframe = pd.DataFrame(
        {
            "existing_value": [10, 20],
        }
    )

    reference = DatasetReference(
        name="existing_dataset",
        source="integration_test",
        row_count=2,
        column_count=1,
        columns=("existing_value",),
    )

    context.register_dataset(reference, dataframe)

    return context, reference, dataframe

def test_loading_failure_for_missing_file_preserves_context(tmp_path):
    """A missing file must not modify the execution context."""
    context, existing_reference, existing_dataframe = (
        _create_context_with_existing_dataset()
    )

    service = DatasetLoadingService()

    missing_path = tmp_path / "missing.csv"

    initial_count = len(context.dataset_store)
    initial_references = context.dataset_store.list_references()

    with pytest.raises(DatasetFileNotFoundError):
        service.load_into_context(
            DatasetLoadRequest(file_path=missing_path),
            context,
        )

    assert len(context.dataset_store) == initial_count

    assert (
        context.dataset_store.list_references()
        == initial_references
    )

    assert context.get_dataset(
        existing_reference.dataset_id
    ) is existing_dataframe

    assert context.get_dataset_reference(
        existing_reference.dataset_id
    ) == existing_reference

def test_loading_failure_for_unsupported_format_preserves_context(
    tmp_path,
):
    """An unsupported format must not modify the execution context."""
    context, existing_reference, existing_dataframe = (
        _create_context_with_existing_dataset()
    )

    service = DatasetLoadingService()

    unsupported_path = tmp_path / "data.txt"
    unsupported_path.write_text(
        "This is not a supported dataset format.",
        encoding="utf-8",
    )

    initial_count = len(context.dataset_store)
    initial_references = context.dataset_store.list_references()

    with pytest.raises(UnsupportedFileFormatError):
        service.load_into_context(
            DatasetLoadRequest(file_path=unsupported_path),
            context,
        )

    assert len(context.dataset_store) == initial_count

    assert (
        context.dataset_store.list_references()
        == initial_references
    )

    assert context.get_dataset(
        existing_reference.dataset_id
    ) is existing_dataframe

    assert context.get_dataset_reference(
        existing_reference.dataset_id
    ) == existing_reference

def test_loading_failure_for_malformed_csv_preserves_context(
    tmp_path,
):
    """A malformed CSV must not modify the execution context."""
    context, existing_reference, existing_dataframe = (
        _create_context_with_existing_dataset()
    )

    service = DatasetLoadingService()

    malformed_path = tmp_path / "malformed.csv"

    malformed_path.write_text(
        "name,score\n"
        "Alice,90\n"
        "Bob,85,unexpected_value\n",
        encoding="utf-8",
    )

    initial_count = len(context.dataset_store)
    initial_references = context.dataset_store.list_references()

    with pytest.raises(DatasetParseError):
        service.load_into_context(
            DatasetLoadRequest(file_path=malformed_path),
            context,
        )

    assert len(context.dataset_store) == initial_count

    assert (
        context.dataset_store.list_references()
        == initial_references
    )

    assert context.get_dataset(
        existing_reference.dataset_id
    ) is existing_dataframe

    assert context.get_dataset_reference(
        existing_reference.dataset_id
    ) == existing_reference