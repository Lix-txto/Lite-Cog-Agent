"""Unit tests for DatasetProfiler."""

from unittest.mock import Mock

import pandas as pd
import pytest

from lite_cog_agent.execution.dataset_store import DatasetReference
from lite_cog_agent.profiling.dataset_profiler import DatasetProfiler
from lite_cog_agent.profiling.exceptions import (
    DatasetProfilingError,
    ProfilingValidationError,
)


@pytest.fixture
def profiler() -> DatasetProfiler:
    return DatasetProfiler()


@pytest.fixture
def reference() -> DatasetReference:
    return DatasetReference(
        dataset_id="dataset-001",
        name="sales",
        source="test",
        file_format="csv",
    )


def test_profile_returns_dataset_overview(
    profiler: DatasetProfiler,
    reference: DatasetReference,
) -> None:
    dataframe = pd.DataFrame(
        {
            "amount": [100, 200, 300],
            "category": ["A", "B", "A"],
        }
    )

    result = profiler.profile(dataframe, reference)

    assert result.dataset_id == "dataset-001"
    assert result.dataset_name == "sales"
    assert result.row_count == 3
    assert result.column_count == 2
    assert len(result.columns) == 2
    assert result.memory_usage_bytes == int(
        dataframe.memory_usage(index=True, deep=True).sum()
    )


def test_profile_handles_multiple_column_types(
    profiler: DatasetProfiler,
    reference: DatasetReference,
) -> None:
    dataframe = pd.DataFrame(
        {
            "amount": [10, 20, 30],
            "category": ["A", "B", "A"],
            "active": [True, False, True],
            "created_at": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03"]
            ),
        }
    )

    result = profiler.profile(dataframe, reference)

    assert [column.inferred_type for column in result.columns] == [
        "numeric",
        "categorical",
        "boolean",
        "datetime",
    ]


def test_profile_handles_empty_dataframe(
    profiler: DatasetProfiler,
    reference: DatasetReference,
) -> None:
    dataframe = pd.DataFrame(
        {
            "amount": pd.Series(dtype="int64"),
            "category": pd.Series(dtype="object"),
        }
    )

    result = profiler.profile(dataframe, reference)

    assert result.row_count == 0
    assert result.column_count == 2
    assert result.duplicate_row_count == 0
    assert result.duplicate_percentage == 0.0
    assert len(result.columns) == 2


def test_profile_handles_dataframe_without_columns(
    profiler: DatasetProfiler,
    reference: DatasetReference,
) -> None:
    dataframe = pd.DataFrame(index=range(3))

    result = profiler.profile(dataframe, reference)

    assert result.row_count == 3
    assert result.column_count == 0
    assert result.columns == ()
    assert result.duplicate_row_count == 0
    assert result.duplicate_percentage == 0.0


def test_profile_counts_duplicate_rows(
    profiler: DatasetProfiler,
    reference: DatasetReference,
) -> None:
    dataframe = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "name": ["A", "A", "B", "B"],
        }
    )

    result = profiler.profile(dataframe, reference)

    assert result.duplicate_row_count == 2
    assert result.duplicate_percentage == 50.0


def test_profile_counts_all_repeated_rows(
    profiler: DatasetProfiler,
    reference: DatasetReference,
) -> None:
    dataframe = pd.DataFrame(
        {
            "id": [1, 1, 1],
            "name": ["A", "A", "A"],
        }
    )

    result = profiler.profile(dataframe, reference)

    assert result.duplicate_row_count == 2
    assert result.duplicate_percentage == pytest.approx(200 / 3)


def test_profile_returns_zero_for_unique_rows(
    profiler: DatasetProfiler,
    reference: DatasetReference,
) -> None:
    dataframe = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
        }
    )

    result = profiler.profile(dataframe, reference)

    assert result.duplicate_row_count == 0
    assert result.duplicate_percentage == 0.0


def test_profile_calculates_memory_usage(
    profiler: DatasetProfiler,
    reference: DatasetReference,
) -> None:
    dataframe = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [20, 30, 40],
        }
    )

    result = profiler.profile(dataframe, reference)

    expected = int(
        dataframe.memory_usage(index=True, deep=True).sum()
    )

    assert result.memory_usage_bytes == expected


def test_profile_rejects_non_dataframe_input(
    profiler: DatasetProfiler,
    reference: DatasetReference,
) -> None:
    with pytest.raises(
        ProfilingValidationError,
        match="dataframe must be a pandas DataFrame",
    ):
        profiler.profile([1, 2, 3], reference)  # type: ignore[arg-type]


def test_profile_rejects_invalid_reference(
    profiler: DatasetProfiler,
) -> None:
    dataframe = pd.DataFrame({"value": [1, 2, 3]})

    with pytest.raises(
        ProfilingValidationError,
        match="reference must be a DatasetReference",
    ):
        profiler.profile(dataframe, None)  # type: ignore[arg-type]


def test_profile_does_not_modify_original_dataframe(
    profiler: DatasetProfiler,
    reference: DatasetReference,
) -> None:
    dataframe = pd.DataFrame(
        {
            "amount": [100, 200, 200],
            "category": ["A", "B", "B"],
        }
    )
    original = dataframe.copy(deep=True)

    profiler.profile(dataframe, reference)

    pd.testing.assert_frame_equal(dataframe, original)


def test_profile_wraps_column_profiling_errors(
    reference: DatasetReference,
) -> None:
    mock_column_profiler = Mock()
    mock_column_profiler.profile.side_effect = RuntimeError(
        "column analysis failed"
    )

    profiler = DatasetProfiler(
        column_profiler=mock_column_profiler
    )
    dataframe = pd.DataFrame({"amount": [1, 2, 3]})

    with pytest.raises(
        DatasetProfilingError,
        match="Failed to profile column",
    ):
        profiler.profile(dataframe, reference)


def test_profile_handles_duplicate_column_names(
    profiler: DatasetProfiler,
    reference: DatasetReference,
) -> None:
    dataframe = pd.DataFrame(
        [[10, 20], [30, 40]],
        columns=["value", "value"],
    )

    result = profiler.profile(dataframe, reference)

    assert result.column_count == 2
    assert len(result.columns) == 2
    assert result.columns[0].name == "value"
    assert result.columns[1].name == "value"
    assert result.columns[0].statistics["mean"] == 20.0
    assert result.columns[1].statistics["mean"] == 30.0


class TestMissingAndEmptyData:
    def test_profiles_completely_empty_dataframe(
        self,
        profiler: DatasetProfiler,
        reference: DatasetReference,
    ) -> None:
        dataframe = pd.DataFrame()

        result = profiler.profile(dataframe, reference)

        assert result.row_count == 0
        assert result.column_count == 0
        assert result.columns == ()
        assert result.duplicate_row_count == 0
        assert result.duplicate_percentage == 0.0

    def test_profiles_dataframe_with_all_missing_values(
        self,
        profiler: DatasetProfiler,
        reference: DatasetReference,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "name": [None, None],
                "age": [float("nan"), float("nan")],
            }
        )

        result = profiler.profile(dataframe, reference)

        assert result.row_count == 2
        assert result.column_count == 2

        name_profile, age_profile = result.columns

        assert name_profile.name == "name"
        assert name_profile.total_count == 2
        assert name_profile.non_null_count == 0
        assert name_profile.null_count == 2
        assert name_profile.unique_count == 0
        assert name_profile.null_percentage == 100.0
        assert name_profile.sample_values == ()

        assert age_profile.name == "age"
        assert age_profile.total_count == 2
        assert age_profile.non_null_count == 0
        assert age_profile.null_count == 2
        assert age_profile.unique_count == 0
        assert age_profile.null_percentage == 100.0
        assert age_profile.sample_values == ()

    def test_profiles_mixed_nullable_dtypes(
        self,
        profiler: DatasetProfiler,
        reference: DatasetReference,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "float_value": pd.Series(
                    [1.0, pd.NA, 3.0],
                    dtype="Float64",
                ),
                "integer_value": pd.Series(
                    [10, pd.NA, 30],
                    dtype="Int64",
                ),
                "object_value": ["A", None, "C"],
            }
        )

        result = profiler.profile(dataframe, reference)

        assert result.row_count == 3
        assert result.column_count == 3

        float_profile, integer_profile, object_profile = result.columns

        assert float_profile.inferred_type == "numeric"
        assert float_profile.total_count == 3
        assert float_profile.non_null_count == 2
        assert float_profile.null_count == 1
        assert float_profile.unique_count == 2
        assert float_profile.null_percentage == pytest.approx(100 / 3)
        assert float_profile.statistics["count"] == 2

        assert integer_profile.inferred_type == "numeric"
        assert integer_profile.total_count == 3
        assert integer_profile.non_null_count == 2
        assert integer_profile.null_count == 1
        assert integer_profile.unique_count == 2
        assert integer_profile.null_percentage == pytest.approx(100 / 3)
        assert integer_profile.statistics["count"] == 2

        assert object_profile.inferred_type == "categorical"
        assert object_profile.total_count == 3
        assert object_profile.non_null_count == 2
        assert object_profile.null_count == 1
        assert object_profile.unique_count == 2
        assert object_profile.null_percentage == pytest.approx(100 / 3)
        assert object_profile.sample_values == ("A", "C")

    def test_profiles_nullable_dataframe_without_modifying_input(
        self,
        profiler: DatasetProfiler,
        reference: DatasetReference,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "value": pd.Series(
                    [1, pd.NA, 3],
                    dtype="Int64",
                ),
                "category": ["A", None, "C"],
            }
        )
        original = dataframe.copy(deep=True)

        profiler.profile(dataframe, reference)

        pd.testing.assert_frame_equal(dataframe, original)