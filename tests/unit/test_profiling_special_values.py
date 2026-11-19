
"""Tests for mixed types and special values in profiling."""

import pandas as pd
import pytest

from lite_cog_agent.profiling.column_profiler import ColumnProfiler
from lite_cog_agent.profiling.exceptions import ColumnProfilingError


@pytest.fixture
def profiler() -> ColumnProfiler:
    return ColumnProfiler()


class TestBooleanEdgeCases:
    def test_profiles_nullable_boolean_series(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            [True, False, pd.NA, True],
            dtype="boolean",
            name="active",
        )

        result = profiler.profile(series, name="active")

        assert result.name == "active"
        assert result.inferred_type == "boolean"
        assert result.total_count == 4
        assert result.non_null_count == 3
        assert result.null_count == 1
        assert result.unique_count == 2
        assert result.null_percentage == 25.0
        assert result.sample_values == (True, False, True)

    def test_profiles_all_missing_nullable_boolean_series(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            [pd.NA, pd.NA],
            dtype="boolean",
            name="active",
        )

        result = profiler.profile(series)

        assert result.inferred_type == "boolean"
        assert result.total_count == 2
        assert result.non_null_count == 0
        assert result.null_count == 2
        assert result.unique_count == 0
        assert result.sample_values == ()

    def test_profiles_empty_nullable_boolean_series(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series([], dtype="boolean", name="active")

        result = profiler.profile(series)

        assert result.inferred_type == "boolean"
        assert result.total_count == 0
        assert result.non_null_count == 0
        assert result.null_count == 0
        assert result.unique_count == 0
        assert result.sample_values == ()

    def test_profiles_boolean_and_integer_mixed_values(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            [True, 1, False, 0],
            dtype="object",
            name="mixed_boolean_integer",
        )

        result = profiler.profile(series)

        assert result.inferred_type == "categorical"
        assert result.total_count == 4
        assert result.non_null_count == 4
        assert result.null_count == 0
        assert result.sample_values == (True, 1, False, 0)



class TestDatetimeEdgeCases:
    def test_profiles_datetime_series_with_timezone(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            pd.to_datetime(
                [
                    "2026-01-01 10:00:00+08:00",
                    "2026-01-02 10:00:00+08:00",
                    None,
                ],
                utc=True,
            )
        )

        result = profiler.profile(series, name="created_at")

        assert result.name == "created_at"
        assert result.inferred_type == "datetime"
        assert result.total_count == 3
        assert result.non_null_count == 2
        assert result.null_count == 1
        assert result.unique_count == 2
        assert result.null_percentage == pytest.approx(100 / 3)

        assert result.statistics["min"] is not None
        assert result.statistics["max"] is not None

    def test_profiles_all_missing_datetime_series(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            [pd.NaT, pd.NaT],
            dtype="datetime64[ns]",
            name="created_at",
        )

        result = profiler.profile(series)

        assert result.inferred_type == "datetime"
        assert result.total_count == 2
        assert result.non_null_count == 0
        assert result.null_count == 2
        assert result.unique_count == 0
        assert result.statistics["min"] is None
        assert result.statistics["max"] is None

    def test_profiles_empty_datetime_series_with_timezone(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            [],
            dtype="datetime64[ns, UTC]",
            name="created_at",
        )

        result = profiler.profile(series)

        assert result.inferred_type == "datetime"
        assert result.total_count == 0
        assert result.non_null_count == 0
        assert result.null_count == 0
        assert result.statistics["min"] is None
        assert result.statistics["max"] is None



class TestMixedObjectValues:
    def test_profiles_mixed_integer_and_string_values(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            [1, "two", 3, "four"],
            dtype="object",
            name="mixed_integer_string",
        )

        result = profiler.profile(series)

        assert result.inferred_type == "categorical"
        assert result.total_count == 4
        assert result.non_null_count == 4
        assert result.null_count == 0
        assert result.unique_count == 4

    def test_profiles_mixed_numeric_and_string_values(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            [1.5, "2.5", 3.5, "4.5"],
            dtype="object",
            name="mixed_numeric_string",
        )

        result = profiler.profile(series)

        assert result.inferred_type == "categorical"
        assert result.unique_count == 4

    def test_empty_string_is_not_missing(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            ["", "hello", "", None],
            dtype="object",
            name="text_value",
        )

        result = profiler.profile(series)

        assert result.total_count == 4
        assert result.non_null_count == 3
        assert result.null_count == 1
        assert result.unique_count == 2
        assert result.sample_values == ("", "hello", "")

    def test_special_strings_are_not_missing(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            ["None", "NaN", "null", "NULL", "nan"],
            dtype="object",
            name="special_strings",
        )

        result = profiler.profile(series)

        assert result.total_count == 5
        assert result.non_null_count == 5
        assert result.null_count == 0
        assert result.unique_count == 5
        assert result.sample_values == (
            "None",
            "NaN",
            "null",
            "NULL",
            "nan",
        )

    def test_profiles_object_series_with_missing_values(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            [1, None, "hello", pd.NA],
            dtype="object",
            name="mixed_object",
        )

        result = profiler.profile(series)

        assert result.inferred_type == "categorical"
        assert result.total_count == 4
        assert result.non_null_count == 2
        assert result.null_count == 2
        assert result.unique_count == 2
        assert result.sample_values == (1, "hello")



class TestComplexObjects:
    def test_rejects_list_values(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            [[1, 2], [3, 4]],
            dtype="object",
            name="list_values",
        )

        with pytest.raises(ColumnProfilingError):
            profiler.profile(series)

    def test_rejects_dict_values(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            [
                {"name": "Alice"},
                {"name": "Bob"},
            ],
            dtype="object",
            name="dict_values",
        )

        with pytest.raises(ColumnProfilingError):
            profiler.profile(series)
