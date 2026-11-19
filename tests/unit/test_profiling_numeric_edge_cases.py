"""Tests for numeric profiling edge cases."""

import math

import pandas as pd
import pytest

from lite_cog_agent.profiling.column_profiler import ColumnProfiler


@pytest.fixture
def profiler() -> ColumnProfiler:
    return ColumnProfiler()


class TestNumericNonFiniteValues:
    def test_nan_is_counted_as_missing(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        result = profiler.profile(
            pd.Series([10.0, float("nan"), 20.0], name="value")
        )

        assert result.total_count == 3
        assert result.non_null_count == 2
        assert result.null_count == 1
        assert result.unique_count == 2

    def test_positive_infinity_is_not_missing(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        result = profiler.profile(
            pd.Series([10.0, float("inf")], name="value")
        )

        assert result.total_count == 2
        assert result.non_null_count == 2
        assert result.null_count == 0
        assert result.unique_count == 2

    def test_negative_infinity_is_not_missing(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        result = profiler.profile(
            pd.Series([10.0, float("-inf")], name="value")
        )

        assert result.total_count == 2
        assert result.non_null_count == 2
        assert result.null_count == 0
        assert result.unique_count == 2

    def test_both_infinities_are_not_missing(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        result = profiler.profile(
            pd.Series(
                [
                    float("-inf"),
                    10.0,
                    float("inf"),
                    None,
                ],
                name="value",
            )
        )

        assert result.total_count == 4
        assert result.non_null_count == 3
        assert result.null_count == 1
        assert result.unique_count == 3


class TestNumericExtremeValues:
    def test_profiles_large_finite_values(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        result = profiler.profile(
            pd.Series([1e100, 2e100, 3e100], name="value")
        )

        assert result.statistics["count"] == 3
        assert math.isfinite(result.statistics["mean"])

    def test_profiles_large_positive_values(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        result = profiler.profile(
            pd.Series([1e308, 1e308], name="value")
        )

        assert result.statistics["count"] == 2

    def test_profiles_large_positive_and_negative_values(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        result = profiler.profile(
            pd.Series(
                [1e308, -1e308],
                name="value",
            )
        )

        assert result.statistics["count"] == 2

    def test_does_not_modify_original_numeric_series(
        self,
        profiler: ColumnProfiler,
    ) -> None:
        series = pd.Series(
            [1.0, float("inf"), float("-inf"), None],
            name="value",
        )
        original = series.copy(deep=True)

        profiler.profile(series)

        pd.testing.assert_series_equal(series, original)



@pytest.mark.parametrize(
    ("values", "expected_mean", "expected_min", "expected_max"),
    [
        ([10.0, 20.0, float("inf")], 15.0, 10.0, 20.0),
        ([10.0, 20.0, float("-inf")], 15.0, 10.0, 20.0),
    ],
)
def test_infinity_does_not_pollute_numeric_statistics(
    profiler: ColumnProfiler,
    values: list[float],
    expected_mean: float,
    expected_min: float,
    expected_max: float,
) -> None:
    result = profiler.profile(pd.Series(values, name="value"))

    assert result.non_null_count == 3
    assert result.unique_count == 3

    assert result.statistics["count"] == 2
    assert result.statistics["mean"] == expected_mean
    assert result.statistics["min"] == expected_min
    assert result.statistics["max"] == expected_max


def test_both_infinities_are_excluded_from_numeric_statistics(
    profiler: ColumnProfiler,
) -> None:
    result = profiler.profile(
        pd.Series(
            [float("-inf"), 10.0, float("inf"), None],
            name="value",
        )
    )

    assert result.total_count == 4
    assert result.null_count == 1
    assert result.non_null_count == 3
    assert result.unique_count == 3

    assert result.statistics["count"] == 1
    assert result.statistics["mean"] == 10.0
    assert result.statistics["std"] is None
    assert result.statistics["min"] == 10.0
    assert result.statistics["25%"] == 10.0
    assert result.statistics["50%"] == 10.0
    assert result.statistics["75%"] == 10.0
    assert result.statistics["max"] == 10.0


def test_large_positive_values_have_finite_statistics(
    profiler: ColumnProfiler,
) -> None:
    result = profiler.profile(
        pd.Series([1e308, 1e308], name="value")
    )

    assert result.statistics["count"] == 2
    assert result.statistics["mean"] == 1e308
    assert result.statistics["std"] == 0.0
    assert result.statistics["min"] == 1e308
    assert result.statistics["25%"] == 1e308
    assert result.statistics["50%"] == 1e308
    assert result.statistics["75%"] == 1e308
    assert result.statistics["max"] == 1e308


def test_large_positive_and_negative_values_are_stable(
    profiler: ColumnProfiler,
) -> None:
    result = profiler.profile(
        pd.Series([1e308, -1e308], name="value")
    )

    assert result.statistics["count"] == 2
    assert result.statistics["mean"] == 0.0

    assert result.statistics["std"] == pytest.approx(
        1.4142135623730951e308
    )

    assert result.statistics["min"] == -1e308
    assert result.statistics["25%"] == pytest.approx(-5e307)
    assert result.statistics["50%"] == 0.0
    assert result.statistics["75%"] == pytest.approx(5e307)
    assert result.statistics["max"] == 1e308


def test_numeric_statistics_contain_no_non_finite_float_values(
    profiler: ColumnProfiler,
) -> None:
    cases = [
        [10.0, 20.0, float("inf")],
        [10.0, 20.0, float("-inf")],
        [float("-inf"), 10.0, float("inf")],
        [1e308, 1e308],
        [1e308, -1e308],
    ]

    for values in cases:
        result = profiler.profile(
            pd.Series(values, name="value")
        )

        for key, value in result.statistics.items():
            if isinstance(value, float):
                assert math.isfinite(value), (
                    f"{key} is not finite: {value}"
                )