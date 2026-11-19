"""Tests for ColumnProfiler."""

import pandas as pd
import pytest

from lite_cog_agent.profiling.column_profiler import ColumnProfiler
from lite_cog_agent.profiling.contracts import ColumnProfile
from lite_cog_agent.profiling.exceptions import (
    ColumnProfilingError,
    ProfilingValidationError,
)


@pytest.fixture
def profiler() -> ColumnProfiler:
    return ColumnProfiler()


class TestColumnProfilerValidation:
    def test_rejects_non_series_input(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        with pytest.raises(
            ProfilingValidationError,
            match="pandas Series",
        ):
            profiler.profile([1, 2, 3], name="sales")  # type: ignore[arg-type]

    @pytest.mark.parametrize("name", ["", " ", None])
    def test_rejects_invalid_column_name(
        self,
        profiler: ColumnProfiler,
        name: str | None,
    ) -> None:
        series = pd.Series([1, 2, 3])

        with pytest.raises(
            ProfilingValidationError,
            match="column name",
        ):
            profiler.profile(series, name=name)


class TestNumericProfiling:
    def test_profiles_numeric_series(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            [10, 20, 30, 40],
            name="sales",
        )

        result = profiler.profile(series)

        assert isinstance(result, ColumnProfile)
        assert result.name == "sales"
        assert result.dtype == "int64"
        assert result.inferred_type == "numeric"

        assert result.total_count == 4
        assert result.non_null_count == 4
        assert result.null_count == 0
        assert result.unique_count == 4

        assert result.statistics["count"] == 4
        assert result.statistics["mean"] == 25.0
        assert result.statistics["min"] == 10
        assert result.statistics["50%"] == 25.0
        assert result.statistics["max"] == 40

        assert result.sample_values == (10, 20, 30, 40)

    def test_profiles_numeric_series_with_missing_values(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            [10.0, None, 30.0, None],
            name="amount",
        )

        result = profiler.profile(series)

        assert result.total_count == 4
        assert result.non_null_count == 2
        assert result.null_count == 2
        assert result.null_percentage == 50.0
        assert result.unique_count == 2
        assert result.unique_percentage == 50.0

        assert result.statistics["count"] == 2
        assert result.statistics["mean"] == 20.0
        assert result.sample_values == (10.0, 30.0)

    def test_profiles_single_numeric_value(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        result = profiler.profile(
            pd.Series([42], name="score")
        )

        assert result.statistics["count"] == 1
        assert result.statistics["mean"] == 42.0
        assert result.statistics["std"] is None

    def test_profiles_empty_numeric_series(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        result = profiler.profile(
            pd.Series([], dtype="float64", name="amount")
        )

        assert result.total_count == 0
        assert result.non_null_count == 0
        assert result.null_count == 0
        assert result.null_percentage == 0.0
        assert result.unique_percentage == 0.0
        assert result.statistics["count"] == 0
        assert result.statistics["mean"] is None
        assert result.sample_values == ()


class TestCategoricalProfiling:
    def test_profiles_categorical_series(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            ["A", "B", "A", "C", "A"],
            name="department",
        )

        result = profiler.profile(series)

        assert result.inferred_type == "categorical"
        assert result.unique_count == 3
        assert result.statistics["unique"] == 3
        assert result.statistics["top"] == "A"
        assert result.statistics["freq"] == 3
        assert result.sample_values == ("A", "B", "A", "C", "A")

    def test_profiles_categorical_series_with_missing_values(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            ["A", None, "B", "A"],
            name="category",
        )

        result = profiler.profile(series)

        assert result.null_count == 1
        assert result.non_null_count == 3
        assert result.null_percentage == 25.0
        assert result.unique_count == 2
        assert result.statistics["top"] == "A"
        assert result.statistics["freq"] == 2
        assert result.sample_values == ("A", "B", "A")

    def test_profiles_empty_categorical_series(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        result = profiler.profile(
            pd.Series([], dtype="object", name="category")
        )

        assert result.inferred_type == "categorical"
        assert result.unique_count == 0
        assert result.statistics == {
            "unique": 0,
            "top": None,
            "freq": 0,
        }


class TestBooleanProfiling:
    def test_profiles_boolean_series(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        result = profiler.profile(
            pd.Series([True, False, True], name="active")
        )

        assert result.inferred_type == "boolean"
        assert result.unique_count == 2
        assert result.statistics["unique"] == 2
        assert result.statistics["top"] is True
        assert result.statistics["freq"] == 2


class TestDatetimeProfiling:
    def test_profiles_datetime_series(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            pd.to_datetime(
                ["2026-01-03", "2026-01-01", "2026-01-02"]
            ),
            name="created_at",
        )

        result = profiler.profile(series)

        assert result.inferred_type == "datetime"
        assert result.statistics["min"] == pd.Timestamp("2026-01-01")
        assert result.statistics["max"] == pd.Timestamp("2026-01-03")

    def test_profiles_datetime_series_with_missing_values(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            pd.to_datetime(
                ["2026-01-01", None, "2026-01-03"]
            ),
            name="created_at",
        )

        result = profiler.profile(series)

        assert result.null_count == 1
        assert result.non_null_count == 2
        assert result.statistics["min"] == pd.Timestamp("2026-01-01")
        assert result.statistics["max"] == pd.Timestamp("2026-01-03")

    def test_profiles_empty_datetime_series(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        result = profiler.profile(
            pd.Series([], dtype="datetime64[ns]", name="created_at")
        )

        assert result.inferred_type == "datetime"
        assert result.statistics == {
            "min": None,
            "max": None,
        }


class TestColumnProfilerSamples:
    def test_limits_sample_values_to_five(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        result = profiler.profile(
            pd.Series(range(10), name="value")
        )

        assert result.sample_values == (0, 1, 2, 3, 4)
        assert len(result.sample_values) == 5

    def test_samples_ignore_missing_values(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        result = profiler.profile(
            pd.Series([None, 1, None, 2, 3], name="value")
        )

        assert result.sample_values == (1, 2, 3)

    def test_does_not_modify_original_series(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            [1, None, 3],
            name="value",
        )
        original = series.copy(deep=True)

        profiler.profile(series)

        pd.testing.assert_series_equal(series, original)

    def test_supports_explicit_column_name(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series([1, 2, 3])

        result = profiler.profile(series, name="revenue")

        assert result.name == "revenue"
