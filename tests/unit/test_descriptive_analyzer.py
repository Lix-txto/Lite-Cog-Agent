"""Unit tests for descriptive statistical analysis."""

import math

import pandas as pd
import pytest

from lite_cog_agent.statistics.descriptive import DescriptiveAnalyzer
from lite_cog_agent.statistics.exceptions import (
    StatisticalValidationError,
)


@pytest.fixture
def analyzer() -> DescriptiveAnalyzer:
    return DescriptiveAnalyzer()


class TestDescriptiveBasicStatistics:
    """Tests for ordinary descriptive statistics."""

    def test_analyzes_single_numeric_column(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"age": [20, 30, 40, 50, 60]}
        )

        results = analyzer.analyze(dataframe)

        assert len(results) == 1

        result = results[0]

        assert result.column == "age"
        assert result.count == 5
        assert result.mean == pytest.approx(40.0)
        assert result.median == pytest.approx(40.0)
        assert result.std == pytest.approx(15.8113883)
        assert result.variance == pytest.approx(250.0)
        assert result.min == 20.0
        assert result.q1 == 30.0
        assert result.q3 == 50.0
        assert result.max == 60.0
        assert result.range == 40.0

    def test_analyzes_multiple_numeric_columns(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "age": [20, 30, 40],
                "salary": [3000, 4000, 5000],
            }
        )

        results = analyzer.analyze(dataframe)

        assert tuple(item.column for item in results) == (
            "age",
            "salary",
        )

        assert results[0].mean == pytest.approx(30.0)
        assert results[1].mean == pytest.approx(4000.0)

    def test_preserves_dataframe_column_order(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "salary": [3000, 4000],
                "age": [20, 30],
                "score": [80, 90],
            }
        )

        results = analyzer.analyze(dataframe)

        assert tuple(item.column for item in results) == (
            "salary",
            "age",
            "score",
        )

    def test_analyzes_explicitly_selected_columns(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "age": [20, 30, 40],
                "salary": [3000, 4000, 5000],
                "score": [80, 90, 100],
            }
        )

        results = analyzer.analyze(
            dataframe,
            columns=["salary", "age"],
        )

        assert tuple(item.column for item in results) == (
            "salary",
            "age",
        )

    def test_returns_empty_tuple_for_dataframe_without_columns(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame()

        results = analyzer.analyze(dataframe)

        assert results == ()

    def test_returns_empty_tuple_when_no_numeric_columns_exist(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "name": ["Alice", "Bob"],
                "department": ["A", "B"],
            }
        )

        results = analyzer.analyze(dataframe)

        assert results == ()


class TestDescriptiveColumnSelection:
    """Tests for numeric column selection."""

    def test_excludes_boolean_columns_by_default(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "age": [20, 30, 40],
                "active": [True, False, True],
            }
        )

        results = analyzer.analyze(dataframe)

        assert tuple(item.column for item in results) == ("age",)

    def test_excludes_nullable_boolean_columns(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "age": [20, 30, 40],
                "active": pd.Series(
                    [True, False, pd.NA],
                    dtype="boolean",
                ),
            }
        )

        results = analyzer.analyze(dataframe)

        assert tuple(item.column for item in results) == ("age",)

    def test_rejects_explicit_boolean_column(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"active": [True, False, True]}
        )

        with pytest.raises(
            StatisticalValidationError,
            match="supported numeric dtype",
        ):
            analyzer.analyze(
                dataframe,
                columns=["active"],
            )

    def test_rejects_explicit_string_column(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"department": ["A", "B", "C"]}
        )

        with pytest.raises(
            StatisticalValidationError,
            match="supported numeric dtype",
        ):
            analyzer.analyze(
                dataframe,
                columns=["department"],
            )

    def test_excludes_complex_columns_by_default(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "age": [20, 30, 40],
                "complex_value": [1 + 2j, 3 + 4j, 5 + 6j],
            }
        )

        results = analyzer.analyze(dataframe)

        assert tuple(item.column for item in results) == ("age",)

    def test_rejects_explicit_complex_column(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"complex_value": [1 + 2j, 3 + 4j]}
        )

        with pytest.raises(
            StatisticalValidationError,
            match="supported numeric dtype",
        ):
            analyzer.analyze(
                dataframe,
                columns=["complex_value"],
            )


class TestDescriptiveMissingValues:
    """Tests for missing and non-finite values."""

    def test_ignores_nan_values(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"value": [10.0, 20.0, float("nan"), 30.0]}
        )

        result = analyzer.analyze(dataframe)[0]

        assert result.count == 3
        assert result.mean == pytest.approx(20.0)
        assert result.min == 10.0
        assert result.max == 30.0

    def test_ignores_positive_infinity(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"value": [10.0, 20.0, float("inf")]}
        )

        result = analyzer.analyze(dataframe)[0]

        assert result.count == 2
        assert result.mean == pytest.approx(15.0)
        assert result.min == 10.0
        assert result.max == 20.0

    def test_ignores_negative_infinity(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"value": [float("-inf"), 10.0, 20.0]}
        )

        result = analyzer.analyze(dataframe)[0]

        assert result.count == 2
        assert result.mean == pytest.approx(15.0)
        assert result.min == 10.0
        assert result.max == 20.0

    def test_ignores_both_infinities(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "value": [
                    float("-inf"),
                    10.0,
                    20.0,
                    float("inf"),
                    None,
                ]
            }
        )

        result = analyzer.analyze(dataframe)[0]

        assert result.count == 2
        assert result.mean == pytest.approx(15.0)
        assert result.min == 10.0
        assert result.max == 20.0

    def test_returns_none_statistics_for_all_missing_column(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "value": pd.Series(
                    [None, None, None],
                    dtype="float64",
                )
            }
        )

        result = analyzer.analyze(dataframe)[0]

        assert result.count == 0
        assert result.mean is None
        assert result.median is None
        assert result.std is None
        assert result.variance is None
        assert result.min is None
        assert result.q1 is None
        assert result.q3 is None
        assert result.max is None
        assert result.range is None

    def test_returns_none_statistics_for_empty_numeric_column(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"value": pd.Series([], dtype="float64")}
        )

        result = analyzer.analyze(dataframe)[0]

        assert result.count == 0
        assert result.mean is None
        assert result.std is None
        assert result.variance is None
        assert result.range is None


class TestDescriptiveStatisticalBoundaries:
    """Tests for sample size and numerical boundaries."""

    def test_single_value_has_no_sample_std_or_variance(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"value": [42.0]}
        )

        result = analyzer.analyze(dataframe)[0]

        assert result.count == 1
        assert result.mean == 42.0
        assert result.median == 42.0
        assert result.std is None
        assert result.variance is None
        assert result.min == 42.0
        assert result.q1 == 42.0
        assert result.q3 == 42.0
        assert result.max == 42.0
        assert result.range == 0.0

    def test_constant_column_has_zero_dispersion(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"value": [5.0, 5.0, 5.0, 5.0]}
        )

        result = analyzer.analyze(dataframe)[0]

        assert result.count == 4
        assert result.mean == 5.0
        assert result.std == 0.0
        assert result.variance == 0.0
        assert result.range == 0.0

    def test_supports_nullable_integer_dtype(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "value": pd.Series(
                    [10, 20, pd.NA, 30],
                    dtype="Int64",
                )
            }
        )

        result = analyzer.analyze(dataframe)[0]

        assert result.count == 3
        assert result.mean == pytest.approx(20.0)

    def test_handles_large_finite_values(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"value": [1e308, 1e308]}
        )

        result = analyzer.analyze(dataframe)[0]

        assert result.count == 2
        assert result.mean == 1e308
        assert result.std == 0.0
        assert result.variance == 0.0
        assert result.range == 0.0

    def test_handles_large_positive_and_negative_values(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"value": [1e308, -1e308]}
        )

        result = analyzer.analyze(dataframe)[0]

        assert result.count == 2
        assert result.mean == pytest.approx(0.0)

        assert result.std == pytest.approx(
            1.4142135623730951e308
        )

        assert result.variance is None
        assert result.min == -1e308
        assert result.median == 0.0
        assert result.max == 1e308
        assert result.range is None

    def test_no_statistic_contains_non_finite_values(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        cases = [
            [10.0, 20.0, float("inf")],
            [10.0, 20.0, float("-inf")],
            [1e308, 1e308],
            [1e308, -1e308],
        ]

        for values in cases:
            dataframe = pd.DataFrame(
                {"value": values}
            )

            result = analyzer.analyze(dataframe)[0]

            numeric_values = (
                result.mean,
                result.median,
                result.std,
                result.variance,
                result.min,
                result.q1,
                result.q3,
                result.max,
                result.range,
            )

            for value in numeric_values:
                if value is not None:
                    assert math.isfinite(value)


class TestDescriptiveInputValidation:
    """Tests for invalid input and column selection."""

    def test_rejects_non_dataframe_input(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        with pytest.raises(
            StatisticalValidationError,
            match="dataframe must be a pandas DataFrame",
        ):
            analyzer.analyze([1, 2, 3])  # type: ignore[arg-type]

    def test_rejects_non_list_columns(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"value": [1, 2, 3]}
        )

        with pytest.raises(
            StatisticalValidationError,
            match="columns must be a list",
        ):
            analyzer.analyze(
                dataframe,
                columns=("value",),  # type: ignore[arg-type]
            )

    def test_rejects_duplicate_selected_columns(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"value": [1, 2, 3]}
        )

        with pytest.raises(
            StatisticalValidationError,
            match="columns cannot contain duplicates",
        ):
            analyzer.analyze(
                dataframe,
                columns=["value", "value"],
            )

    def test_rejects_missing_selected_column(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"value": [1, 2, 3]}
        )

        with pytest.raises(
            StatisticalValidationError,
            match="columns not found",
        ):
            analyzer.analyze(
                dataframe,
                columns=["missing"],
            )

    def test_rejects_empty_column_name(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"value": [1, 2, 3]}
        )

        with pytest.raises(
            StatisticalValidationError,
            match="non-empty strings",
        ):
            analyzer.analyze(
                dataframe,
                columns=[""],
            )

    def test_rejects_duplicate_dataframe_column_names(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            [[1, 2, 3]],
            columns=["value", "value", "other"],
        )

        with pytest.raises(
            StatisticalValidationError,
            match="duplicate column names",
        ):
            analyzer.analyze(dataframe)


class TestDescriptiveInputImmutability:
    """Tests that analysis does not mutate the source DataFrame."""

    def test_does_not_modify_original_dataframe(
        self,
        analyzer: DescriptiveAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "value": [
                    10.0,
                    20.0,
                    float("inf"),
                    float("-inf"),
                    None,
                ],
                "other": [1, 2, 3, 4, 5],
            }
        )

        original = dataframe.copy(deep=True)

        analyzer.analyze(dataframe)

        pd.testing.assert_frame_equal(
            dataframe,
            original,
        )
