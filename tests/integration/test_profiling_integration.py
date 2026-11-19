"""Integration tests for the dataset loading and profiling pipeline."""

import json
from pathlib import Path

import pandas as pd
import pytest

from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.loading import (
    DatasetLoadRequest,
    DatasetLoadingService,
)
from lite_cog_agent.profiling import ProfilingService


@pytest.fixture
def source_dataframe() -> pd.DataFrame:
    """Provide a dataset containing missing and duplicate values."""
    return pd.DataFrame(
        {
            "product": ["A", "B", "A", None],
            "price": [100, 200, 100, 400],
        }
    )


@pytest.fixture
def loading_service() -> DatasetLoadingService:
    return DatasetLoadingService()


@pytest.fixture
def profiling_service() -> ProfilingService:
    return ProfilingService()


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext()


def create_dataset_file(
    tmp_path: Path,
    dataframe: pd.DataFrame,
    file_format: str,
) -> Path:
    """Write a DataFrame to a supported file format."""
    file_path = tmp_path / f"sales.{file_format}"

    if file_format == "csv":
        dataframe.to_csv(file_path, index=False)
    elif file_format == "xlsx":
        dataframe.to_excel(file_path, index=False)
    elif file_format == "json":
        file_path.write_text(
            dataframe.to_json(orient="records"),
            encoding="utf-8",
        )
    else:
        raise ValueError(f"Unsupported test format: {file_format}")

    return file_path


@pytest.mark.parametrize(
    ("file_format", "expected_loader_format"),
    [
        ("csv", "csv"),
        ("xlsx", "excel"),
        ("json", "json"),
    ],
)
def test_loaded_dataset_can_be_profiled_end_to_end(
    tmp_path: Path,
    source_dataframe: pd.DataFrame,
    loading_service: DatasetLoadingService,
    profiling_service: ProfilingService,
    context: ExecutionContext,
    file_format: str,
    expected_loader_format: str,
) -> None:
    """Load a dataset and generate a complete profiling report."""
    file_path = create_dataset_file(
        tmp_path,
        source_dataframe,
        file_format,
    )

    reference = loading_service.load_into_context(
        DatasetLoadRequest(file_path=file_path),
        context,
    )

    loaded_dataframe = context.get_dataset(reference.dataset_id)

    report = profiling_service.profile(
        loaded_dataframe,
        reference,
    )

    assert report.dataset_profile.dataset_id == reference.dataset_id
    assert report.dataset_profile.dataset_name == reference.name

    assert reference.file_format == expected_loader_format
    assert report.dataset_profile.row_count == 4
    assert report.dataset_profile.column_count == 2

    assert report.dataset_profile.duplicate_row_count == 1
    assert report.dataset_profile.duplicate_percentage == 25.0

    assert report.quality_report.total_cells == 8
    assert report.quality_report.missing_cells == 1
    assert report.quality_report.completeness == 87.5
    assert report.quality_report.missing_rate == 12.5

    assert report.quality_report.duplicate_row_count == 1
    assert report.quality_report.duplicate_rate == 25.0

    assert report.quality_report.column_completeness == {
        "product": 75.0,
        "price": 100.0,
    }

    assert context.get_dataset_reference(reference.dataset_id) is reference


def test_profiling_does_not_modify_loaded_dataset(
    tmp_path: Path,
    source_dataframe: pd.DataFrame,
    loading_service: DatasetLoadingService,
    profiling_service: ProfilingService,
    context: ExecutionContext,
) -> None:
    """Profiling must not mutate the DataFrame stored in the context."""
    file_path = create_dataset_file(
        tmp_path,
        source_dataframe,
        "csv",
    )

    reference = loading_service.load_into_context(
        DatasetLoadRequest(file_path=file_path),
        context,
    )

    loaded_dataframe = context.get_dataset(reference.dataset_id)
    original_dataframe = loaded_dataframe.copy(deep=True)

    profiling_service.profile(
        loaded_dataframe,
        reference,
    )

    pd.testing.assert_frame_equal(
        loaded_dataframe,
        original_dataframe,
    )


def test_profiling_respects_context_dataset_isolation(
    tmp_path: Path,
    source_dataframe: pd.DataFrame,
    loading_service: DatasetLoadingService,
    profiling_service: ProfilingService,
) -> None:
    """A context must not access another context's dataset."""
    first_context = ExecutionContext()

    second_context = ExecutionContext(
        dataset_store=first_context.dataset_store,
    )

    try:
        file_path = create_dataset_file(
            tmp_path,
            source_dataframe,
            "csv",
        )

        reference = loading_service.load_into_context(
            DatasetLoadRequest(file_path=file_path),
            first_context,
        )

        with pytest.raises(KeyError, match="not available in context"):
            second_context.get_dataset(reference.dataset_id)

        dataframe = first_context.get_dataset(reference.dataset_id)

        report = profiling_service.profile(
            dataframe,
            reference,
        )

        assert report.dataset_profile.dataset_id == reference.dataset_id
        assert report.dataset_profile.row_count == 4

    finally:
        first_context.close()
        second_context.close()


def test_context_close_releases_dataset_after_profiling(
    tmp_path: Path,
    source_dataframe: pd.DataFrame,
    loading_service: DatasetLoadingService,
    profiling_service: ProfilingService,
) -> None:
    """Closing a context must release its registered datasets."""
    context = ExecutionContext()

    file_path = create_dataset_file(
        tmp_path,
        source_dataframe,
        "csv",
    )

    reference = loading_service.load_into_context(
        DatasetLoadRequest(file_path=file_path),
        context,
    )

    dataframe = context.get_dataset(reference.dataset_id)

    report = profiling_service.profile(dataframe, reference)

    assert report.dataset_profile.row_count == 4
    assert context.dataset_store.contains(reference.dataset_id)

    context.close()

    assert context.is_closed is True
    assert context.dataset_ids == ()
    assert context.dataset_store.contains(reference.dataset_id) is False

    with pytest.raises(RuntimeError, match="ExecutionContext is closed"):
        context.get_dataset(reference.dataset_id)
