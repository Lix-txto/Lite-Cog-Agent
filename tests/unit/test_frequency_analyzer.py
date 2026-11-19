"""Unit tests for frequency statistical analysis."""

import pandas as pd
import pytest

from lite_cog_agent.statistics.contracts import FrequencyItem
from lite_cog_agent.statistics.exceptions import (
    FrequencyAnalysisError,
    StatisticalValidationError,
)
from lite_cog_agent.statistics.frequency import FrequencyAnalyzer


@pytest.fixture
def analyzer() -> FrequencyAnalyzer:
    return FrequencyAnalyzer()


class TestFrequencyBasicAnalysis:
    """Tests for ordinary frequency analysis."""

    def test_analyzes_single_categorical_column(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"department": ["A", "B", "A", "C", "B", "A"]}
        )

        results = analyzer.analyze(
            dataframe,
            column="department",
        )

        assert len(results) == 3
        assert results[0].value == "A"
        assert results[0].count == 3
        assert results[0].percentage == pytest.approx(50.0)
        assert results[0].cumulative_percentage == pytest.approx(50.0)

        assert results[1].value == "B"
        assert results[1].count == 2
        assert results[1].percentage == pytest.approx(100 / 3)
        assert results[1].cumulative_percentage == pytest.approx(
            100 / 3 + 50
        )

        assert results[2].value == "C"
        assert results[2].count == 1
        assert results[2].percentage == pytest.approx(100 / 6)
        assert results[2].cumulative_percentage == pytest.approx(100.0)

    def test_returns_frequency_items_in_descending_order(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"category": ["A", "B", "C", "A", "B", "A"]}
        )

        results = analyzer.analyze(
            dataframe,
            column="category",
        )

        counts = [item.count for item in results]

        assert counts == sorted(counts, reverse=True)

    def test_returns_frequency_item_instances(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"category": ["A", "B", "A"]}
        )

        results = analyzer.analyze(
            dataframe,
            column="category",
        )

        assert all(
            isinstance(item, FrequencyItem)
            for item in results
        )

    def test_analyzes_numeric_categories(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"score": [80, 90, 80, 100, 90, 80]}
        )

        results = analyzer.analyze(
            dataframe,
            column="score",
        )

        assert results[0].value == 80
        assert results[0].count == 3

        assert results[1].value == 90
        assert results[1].count == 2

    def test_analyzes_boolean_categories(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"active": [True, False, True, True, False]}
        )

        results = analyzer.analyze(
            dataframe,
            column="active",
        )

        assert results[0].value is True
        assert results[0].count == 3

        assert results[1].value is False
        assert results[1].count == 2

    def test_analyzes_datetime_categories(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dates = pd.to_datetime(
            [
                "2026-01-01",
                "2026-01-02",
                "2026-01-01",
            ]
        )

        dataframe = pd.DataFrame({"date": dates})

        results = analyzer.analyze(
            dataframe,
            column="date",
        )

        assert len(results) == 2
        assert results[0].value == pd.Timestamp("2026-01-01")
        assert results[0].count == 2


class TestFrequencyTopN:
    """Tests for top-N frequency selection."""

    def test_returns_top_n_categories(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "category": [
                    "A", "B", "A", "C", "B",
                    "A", "D", "A", "B", "A",
                ]
            }
        )

        results = analyzer.analyze(
            dataframe,
            column="category",
            top_n=2,
        )

        assert len(results) == 2
        assert results[0].value == "A"
        assert results[0].count == 5
        assert results[1].value == "B"
        assert results[1].count == 3

    def test_uses_full_population_as_percentage_denominator(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "category": [
                    "A", "A", "A", "B", "B",
                    "C", "C", "D", "E", "F",
                ]
            }
        )

        results = analyzer.analyze(
            dataframe,
            column="category",
            top_n=2,
        )

        assert results[0].percentage == pytest.approx(30.0)
        assert results[1].percentage == pytest.approx(20.0)

        assert results[0].cumulative_percentage == pytest.approx(30.0)
        assert results[1].cumulative_percentage == pytest.approx(50.0)

    def test_returns_all_categories_when_top_n_is_none(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"category": ["A", "B", "C", "D"]}
        )

        results = analyzer.analyze(
            dataframe,
            column="category",
            top_n=None,
        )

        assert len(results) == 4

    def test_returns_all_categories_when_top_n_exceeds_category_count(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"category": ["A", "B", "A"]}
        )

        results = analyzer.analyze(
            dataframe,
            column="category",
            top_n=100,
        )

        assert len(results) == 2

    def test_top_n_one_returns_only_most_frequent_category(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"category": ["A", "B", "A", "C", "A"]}
        )

        results = analyzer.analyze(
            dataframe,
            column="category",
            top_n=1,
        )

        assert len(results) == 1
        assert results[0].value == "A"
        assert results[0].count == 3
        assert results[0].percentage == pytest.approx(60.0)
        assert results[0].cumulative_percentage == pytest.approx(60.0)


class TestFrequencyMissingValues:
    """Tests for missing-value handling."""

    def test_excludes_missing_values_by_default(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"category": ["A", "B", None, "A", None]}
        )

        results = analyzer.analyze(
            dataframe,
            column="category",
        )

        assert len(results) == 2
        assert sum(item.count for item in results) == 3

        assert results[0].value == "A"
        assert results[0].count == 2
        assert results[0].percentage == pytest.approx(200 / 3)


    def test_includes_missing_values_when_dropna_is_false(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"category": ["A", "B", None, "A", None]}
        )

        results = analyzer.analyze(
            dataframe,
            column="category",
            dropna=False,
        )

        assert len(results) == 3

        missing_item = next(
            item for item in results
            if item.value is None
        )

        assert missing_item.count == 2
        assert missing_item.percentage == pytest.approx(40.0)


    def test_counts_nullable_integer_missing_values(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "value": pd.Series(
                    [1, 2, pd.NA, 1],
                    dtype="Int64",
                )
            }
        )

        results = analyzer.analyze(
            dataframe,
            column="value",
            dropna=False,
        )

        assert len(results) == 3
        assert results[-1].value is None
        assert results[-1].count == 1

    def test_returns_empty_tuple_for_empty_dataframe(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"category": pd.Series(dtype="object")}
        )

        results = analyzer.analyze(
            dataframe,
            column="category",
        )

        assert results == ()

    def test_returns_empty_tuple_for_all_missing_values_with_dropna(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"category": [None, None, None]}
        )

        results = analyzer.analyze(
            dataframe,
            column="category",
            dropna=True,
        )

        assert results == ()

    def test_returns_one_missing_category_when_all_values_are_missing(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"category": [None, None, None]}
        )

        results = analyzer.analyze(
            dataframe,
            column="category",
            dropna=False,
        )

        assert len(results) == 1
        assert results[0].value is None
        assert results[0].count == 3
        assert results[0].percentage == pytest.approx(100.0)
        assert results[0].cumulative_percentage == pytest.approx(100.0)


class TestFrequencySpecialValues:
    """Tests for special values and data types."""

    def test_counts_positive_infinity_as_valid_category(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"value": [10.0, float("inf"), 10.0, float("inf")]}
        )

        results = analyzer.analyze(
            dataframe,
            column="value",
        )

        assert len(results) == 2
        assert results[0].value == 10.0
        assert results[0].count == 2

        assert results[1].value == float("inf")
        assert results[1].count == 2

    def test_counts_negative_infinity_as_valid_category(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"value": [float("-inf"), 10.0, float("-inf")]}
        )

        results = analyzer.analyze(
            dataframe,
            column="value",
        )

        assert results[0].value == float("-inf")
        assert results[0].count == 2

    def test_counts_both_infinities_as_separate_categories(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "value": [
                    float("inf"),
                    float("-inf"),
                    float("inf"),
                    float("-inf"),
                ]
            }
        )

        results = analyzer.analyze(
            dataframe,
            column="value",
        )

        assert len(results) == 2
        assert results[0].count == 2
        assert results[1].count == 2
        assert results[0].value != results[1].value

    def test_supports_nullable_boolean_dtype(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "active": pd.Series(
                    [True, False, pd.NA, True],
                    dtype="boolean",
                )
            }
        )

        results = analyzer.analyze(
            dataframe,
            column="active",
            dropna=False,
        )

        assert len(results) == 3
        assert results[-1].value is None
        assert results[-1].count == 1

    def test_normalizes_numpy_scalar_values(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"value": pd.Series([1, 2, 1], dtype="int64")}
        )

        results = analyzer.analyze(
            dataframe,
            column="value",
        )

        assert all(
            type(item.value) is int
            for item in results
        )


class TestFrequencyInputValidation:
    """Tests for invalid input and parameters."""

    def test_rejects_non_dataframe_input(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        with pytest.raises(
            StatisticalValidationError,
            match="dataframe must be a pandas DataFrame",
        ):
            analyzer.analyze(
                [1, 2, 3],  # type: ignore[arg-type]
                column="value",
            )

    def test_rejects_duplicate_dataframe_columns(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            [[1, 2], [3, 4]],
            columns=["value", "value"],
        )

        with pytest.raises(
            StatisticalValidationError,
            match="duplicate columns",
        ):
            analyzer.analyze(
                dataframe,
                column="value",
            )

    @pytest.mark.parametrize(
        "column",
        ["", " ", None, 123],
    )
    def test_rejects_invalid_column_name(
        self,
        analyzer: FrequencyAnalyzer,
        column: object,
    ) -> None:
        dataframe = pd.DataFrame(
            {"value": [1, 2, 3]}
        )

        with pytest.raises(
            StatisticalValidationError,
            match="column must be a non-empty string",
        ):
            analyzer.analyze(
                dataframe,
                column=column,  # type: ignore[arg-type]
            )

    def test_rejects_missing_column(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"value": [1, 2, 3]}
        )

        with pytest.raises(
            StatisticalValidationError,
            match="does not exist",
        ):
            analyzer.analyze(
                dataframe,
                column="missing",
            )

    @pytest.mark.parametrize(
        "top_n",
        [0, -1, True, False, 1.5, "2"],
    )
    def test_rejects_invalid_top_n(
        self,
        analyzer: FrequencyAnalyzer,
        top_n: object,
    ) -> None:
        dataframe = pd.DataFrame(
            {"value": [1, 2, 3]}
        )

        with pytest.raises(
            StatisticalValidationError,
            match="top_n must be a positive integer",
        ):
            analyzer.analyze(
                dataframe,
                column="value",
                top_n=top_n,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "dropna",
        [None, 0, 1, "true"],
    )
    def test_rejects_non_boolean_dropna(
        self,
        analyzer: FrequencyAnalyzer,
        dropna: object,
    ) -> None:
        dataframe = pd.DataFrame(
            {"value": [1, 2, 3]}
        )

        with pytest.raises(
            StatisticalValidationError,
            match="dropna must be a boolean",
        ):
            analyzer.analyze(
                dataframe,
                column="value",
                dropna=dropna,  # type: ignore[arg-type]
            )


class TestFrequencyErrorHandling:
    """Tests for frequency calculation errors."""

    def test_raises_frequency_analysis_error_for_unhashable_values(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "value": pd.Series(
                    [["A"], ["B"], ["A"]],
                    dtype="object",
                )
            }
        )

        with pytest.raises(FrequencyAnalysisError):
            analyzer.analyze(
                dataframe,
                column="value",
            )

    def test_does_not_wrap_validation_errors_as_analysis_errors(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"value": [1, 2, 3]}
        )

        with pytest.raises(StatisticalValidationError):
            analyzer.analyze(
                dataframe,
                column="missing",
            )


class TestFrequencyDataIntegrity:
    """Tests for source data integrity."""

    def test_does_not_modify_original_dataframe(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "category": ["A", "B", None, "A"],
                "value": [10, 20, 30, 40],
            }
        )

        original = dataframe.copy(deep=True)

        analyzer.analyze(
            dataframe,
            column="category",
            dropna=False,
        )

        pd.testing.assert_frame_equal(
            dataframe,
            original,
        )

    def test_does_not_modify_dataframe_when_top_n_is_used(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"category": ["A", "B", "A", "C", "B"]}
        )

        original = dataframe.copy(deep=True)

        analyzer.analyze(
            dataframe,
            column="category",
            top_n=2,
        )

        pd.testing.assert_frame_equal(
            dataframe,
            original,
        )

    def test_returns_immutable_tuple(
        self,
        analyzer: FrequencyAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {"category": ["A", "B", "A"]}
        )

        results = analyzer.analyze(
            dataframe,
            column="category",
        )

        assert isinstance(results, tuple)

        with pytest.raises(AttributeError):
            results.append(results[0])  # type: ignore[attr-defined]
