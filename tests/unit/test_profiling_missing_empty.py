"""Tests for missing and empty data profiling behavior."""

import pandas as pd
import pytest

from lite_cog_agent.profiling.column_profiler import ColumnProfiler


@pytest.fixture
def profiler() -> ColumnProfiler:
    return ColumnProfiler()


class TestEmptySeries:
    def test_profiles_empty_numeric_series(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series([], dtype="float64", name="value")

        result = profiler.profile(series)

        assert result.total_count == 0
        assert result.non_null_count == 0
        assert result.null_count == 0
        assert result.unique_count == 0
        assert result.null_percentage == 0.0
        assert result.unique_percentage == 0.0

        assert result.statistics["count"] == 0
        assert result.statistics["mean"] is None
        assert result.statistics["std"] is None
        assert result.statistics["min"] is None
        assert result.statistics["max"] is None

    def test_profiles_empty_string_series(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series([], dtype="object", name="value")

        result = profiler.profile(series)

        assert result.total_count == 0
        assert result.non_null_count == 0
        assert result.null_count == 0
        assert result.unique_count == 0
        assert result.null_percentage == 0.0
        assert result.unique_percentage == 0.0


class TestAllMissingSeries:
    @pytest.mark.parametrize(
        "values",
        [
            [None, None],
            [float("nan"), float("nan")],
            [pd.NA, pd.NA],
            [pd.NaT, pd.NaT],
        ],
    )
    def test_counts_all_missing_values(
        self,
        profiler: ColumnProfiler,
        values: list,
    ) -> None:
        series = pd.Series(values, name="value")

        result = profiler.profile(series)

        assert result.total_count == 2
        assert result.non_null_count == 0
        assert result.null_count == 2
        assert result.unique_count == 0
        assert result.null_percentage == 100.0
        assert result.unique_percentage == 0.0

        assert result.sample_values == ()

    def test_all_missing_nullable_float_series(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            [pd.NA, pd.NA],
            dtype="Float64",
            name="value",
        )

        result = profiler.profile(series)

        assert result.total_count == 2
        assert result.non_null_count == 0
        assert result.null_count == 2
        assert result.unique_count == 0
        assert result.statistics["count"] == 0
        assert result.statistics["mean"] is None
        assert result.statistics["std"] is None


class TestNullableNumericSeries:
    def test_profiles_nullable_float_series(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            [1.0, pd.NA, 2.0],
            dtype="Float64",
            name="value",
        )

        result = profiler.profile(series)

        assert result.total_count == 3
        assert result.non_null_count == 2
        assert result.null_count == 1
        assert result.unique_count == 2
        assert result.null_percentage == pytest.approx(100 / 3)
        assert result.unique_percentage == pytest.approx(200 / 3)

        assert result.statistics["count"] == 2
        assert result.statistics["mean"] == pytest.approx(1.5)
        assert result.statistics["min"] == 1.0
        assert result.statistics["max"] == 2.0

    def test_profiles_nullable_integer_series(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            [10, pd.NA, 20],
            dtype="Int64",
            name="value",
        )

        result = profiler.profile(series)

        assert result.total_count == 3
        assert result.non_null_count == 2
        assert result.null_count == 1
        assert result.unique_count == 2

        assert result.statistics["count"] == 2
        assert result.statistics["mean"] == pytest.approx(15.0)


class TestMissingValueSamples:
    def test_samples_exclude_all_missing_values(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            [None, "A", pd.NA, "B", float("nan")],
            dtype="object",
            name="category",
        )

        result = profiler.profile(series)

        assert result.total_count == 5
        assert result.non_null_count == 2
        assert result.null_count == 3
        assert result.unique_count == 2
        assert result.sample_values == ("A", "B")

    def test_does_not_modify_nullable_series(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            [1.0, pd.NA, 2.0],
            dtype="Float64",
            name="value",
        )
        original = series.copy(deep=True)

        profiler.profile(series)

        pd.testing.assert_series_equal(series, original)
