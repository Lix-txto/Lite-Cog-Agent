from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lite_cog_agent.statistics.correlation import CorrelationAnalyzer
from lite_cog_agent.statistics.exceptions import (
    CorrelationAnalysisError,
    StatisticalValidationError,
)


@pytest.fixture
def analyzer() -> CorrelationAnalyzer:
    return CorrelationAnalyzer()


class TestCorrelationBasicAnalysis:
    """Test basic Pearson and Spearman correlation analysis."""

    def test_pearson_correlation(self, analyzer: CorrelationAnalyzer) -> None:
        dataframe = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5],
                "y": [2, 4, 6, 8, 10],
                "z": [10, 8, 6, 4, 2],
            }
        )

        matrix, pair_counts = analyzer.analyze(dataframe)

        assert matrix[0][0] == pytest.approx(1.0)
        assert matrix[1][1] == pytest.approx(1.0)
        assert matrix[2][2] == pytest.approx(1.0)

        assert matrix[0][1] == pytest.approx(1.0)
        assert matrix[0][2] == pytest.approx(-1.0)
        assert matrix[1][2] == pytest.approx(-1.0)

        assert pair_counts == (
            (5, 5, 5),
            (5, 5, 5),
            (5, 5, 5),
        )

    def test_spearman_correlation(self, analyzer: CorrelationAnalyzer) -> None:
        dataframe = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5],
                "positive": [10, 20, 30, 40, 50],
                "negative": [50, 40, 30, 20, 10],
            }
        )

        matrix, _ = analyzer.analyze(
            dataframe,
            method="spearman",
        )

        assert matrix[0][1] == pytest.approx(1.0)
        assert matrix[0][2] == pytest.approx(-1.0)
        assert matrix[1][2] == pytest.approx(-1.0)

    def test_default_method_is_pearson(
        self,
        analyzer: CorrelationAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "y": [2, 4, 6, 8],
            }
        )

        default_result = analyzer.analyze(dataframe)
        explicit_result = analyzer.analyze(
            dataframe,
            method="pearson",
        )

        assert default_result == explicit_result

    def test_single_numeric_column(
        self,
        analyzer: CorrelationAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "value": [1, 2, 3, 4],
            }
        )

        matrix, pair_counts = analyzer.analyze(dataframe)

        assert matrix == ((1.0,),)
        assert pair_counts == ((4,),)


class TestCorrelationColumnSelection:
    """Test numeric column selection."""

    def test_default_selects_numeric_columns(
        self,
        analyzer: CorrelationAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "age": [20, 25, 30, 35],
                "salary": [3000, 4000, 5000, 6000],
                "department": ["A", "B", "A", "B"],
                "active": [True, False, True, False],
            }
        )

        matrix, pair_counts = analyzer.analyze(dataframe)

        assert len(matrix) == 2
        assert all(len(row) == 2 for row in matrix)

        assert pair_counts == (
            (4, 4),
            (4, 4),
        )

    def test_explicit_columns(
        self,
        analyzer: CorrelationAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "age": [20, 25, 30, 35],
                "salary": [3000, 4000, 5000, 6000],
                "score": [60, 70, 80, 90],
            }
        )

        matrix, pair_counts = analyzer.analyze(
            dataframe,
            columns=["age", "salary"],
        )

        assert len(matrix) == 2
        assert matrix[0][1] == pytest.approx(1.0)
        assert pair_counts == (
            (4, 4),
            (4, 4),
        )

    def test_explicit_non_numeric_column_raises(
        self,
        analyzer: CorrelationAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "age": [20, 25, 30],
                "department": ["A", "B", "A"],
            }
        )

        with pytest.raises(StatisticalValidationError):
            analyzer.analyze(
                dataframe,
                columns=["age", "department"],
            )

    def test_missing_column_raises(
        self,
        analyzer: CorrelationAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "age": [20, 25, 30],
            }
        )

        with pytest.raises(StatisticalValidationError):
            analyzer.analyze(
                dataframe,
                columns=["salary"],
            )

    def test_duplicate_requested_columns_raise(
        self,
        analyzer: CorrelationAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "age": [20, 25, 30],
                "salary": [3000, 4000, 5000],
            }
        )

        with pytest.raises(StatisticalValidationError):
            analyzer.analyze(
                dataframe,
                columns=["age", "age"],
            )

    def test_duplicate_dataframe_columns_raise(
        self,
        analyzer: CorrelationAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            [[1, 2], [3, 4]],
            columns=["value", "value"],
        )

        with pytest.raises(StatisticalValidationError):
            analyzer.analyze(dataframe)


class TestCorrelationMissingValues:
    """Test missing and non-finite value handling."""

    def test_missing_values_use_pairwise_observations(
        self,
        analyzer: CorrelationAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, np.nan],
                "b": [2.0, 4.0, np.nan, 8.0],
            }
        )

        matrix, pair_counts = analyzer.analyze(dataframe)

        assert pair_counts == (
            (3, 2),
            (2, 3),
        )

        assert matrix[0][1] == pytest.approx(1.0)

    def test_infinity_is_excluded(
        self,
        analyzer: CorrelationAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "a": [1.0, 2.0, np.inf, 4.0],
                "b": [2.0, 4.0, 6.0, -np.inf],
            }
        )

        matrix, pair_counts = analyzer.analyze(dataframe)

        assert pair_counts == (
            (3, 2),
            (2, 3),
        )

        assert matrix[0][1] == pytest.approx(1.0)

    def test_constant_column_returns_none(
        self,
        analyzer: CorrelationAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "constant": [1.0, 1.0, 1.0, 1.0],
                "variable": [1.0, 2.0, 3.0, 4.0],
            }
        )

        matrix, pair_counts = analyzer.analyze(dataframe)

        assert matrix == (
            (None, None),
            (None, 1.0),
        )

        assert pair_counts == (
            (4, 4),
            (4, 4),
        )

    def test_all_missing_column(
        self,
        analyzer: CorrelationAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "empty": [np.nan, np.nan, np.nan],
                "value": [1.0, 2.0, 3.0],
            }
        )

        matrix, pair_counts = analyzer.analyze(dataframe)

        assert matrix == (
            (None, None),
            (None, 1.0),
        )

        assert pair_counts == (
            (0, 0),
            (0, 3),
        )


class TestCorrelationMinPeriods:
    """Test minimum valid observation requirements."""

    def test_min_periods_rejects_insufficient_pairs(
        self,
        analyzer: CorrelationAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "a": [1.0, 2.0, np.nan, np.nan],
                "b": [2.0, 4.0, 6.0, 8.0],
            }
        )

        matrix, pair_counts = analyzer.analyze(
            dataframe,
            min_periods=3,
        )

        assert pair_counts == (
            (2, 2),
            (2, 4),
        )

        assert matrix[0][1] is None

    def test_min_periods_boundary(
        self,
        analyzer: CorrelationAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "a": [1.0, 2.0, np.nan],
                "b": [2.0, 4.0, 6.0],
            }
        )

        matrix, _ = analyzer.analyze(
            dataframe,
            min_periods=2,
        )

        assert matrix[0][1] == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "min_periods",
        [
            0,
            -1,
            1.5,
            True,
            "2",
            None,
        ],
    )
    def test_invalid_min_periods_raises(
        self,
        analyzer: CorrelationAnalyzer,
        min_periods: object,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "b": [2.0, 4.0, 6.0],
            }
        )

        with pytest.raises(StatisticalValidationError):
            analyzer.analyze(
                dataframe,
                min_periods=min_periods,
            )


class TestCorrelationNullableNumeric:
    """Test Pandas nullable numeric dtypes."""

    def test_nullable_integer(
        self,
        analyzer: CorrelationAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "a": pd.Series(
                    [1, 2, 3, None],
                    dtype="Int64",
                ),
                "b": pd.Series(
                    [2, 4, 6, 8],
                    dtype="Int64",
                ),
            }
        )

        matrix, pair_counts = analyzer.analyze(dataframe)

        assert matrix[0][1] == pytest.approx(1.0)
        assert pair_counts == (
            (3, 3),
            (3, 4),
        )

    def test_nullable_float(
        self,
        analyzer: CorrelationAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "a": pd.Series(
                    [1.0, 2.0, 3.0],
                    dtype="Float64",
                ),
                "b": pd.Series(
                    [2.0, 4.0, 6.0],
                    dtype="Float64",
                ),
            }
        )

        matrix, pair_counts = analyzer.analyze(dataframe)

        assert matrix[0][1] == pytest.approx(1.0)
        assert pair_counts == (
            (3, 3),
            (3, 3),
        )


class TestCorrelationEmptyData:
    """Test empty datasets and datasets without numeric columns."""

    def test_empty_dataframe(
        self,
        analyzer: CorrelationAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame()

        matrix, pair_counts = analyzer.analyze(dataframe)

        assert matrix == ()
        assert pair_counts == ()

    def test_dataframe_without_numeric_columns(
        self,
        analyzer: CorrelationAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "name": ["Alice", "Bob"],
                "department": ["A", "B"],
                "active": [True, False],
            }
        )

        matrix, pair_counts = analyzer.analyze(dataframe)

        assert matrix == ()
        assert pair_counts == ()

    def test_empty_numeric_dataframe(
        self,
        analyzer: CorrelationAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "a": pd.Series(dtype="float64"),
                "b": pd.Series(dtype="float64"),
            }
        )

        matrix, pair_counts = analyzer.analyze(dataframe)

        assert matrix == (
            (None, None),
            (None, None),
        )

        assert pair_counts == (
            (0, 0),
            (0, 0),
        )


class TestCorrelationInputValidation:
    """Test invalid input parameters."""

    def test_non_dataframe_input_raises(
        self,
        analyzer: CorrelationAnalyzer,
    ) -> None:
        with pytest.raises(StatisticalValidationError):
            analyzer.analyze([1, 2, 3])  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "method",
        [
            "kendall",
            "Kendall",
            "invalid",
            "",
            None,
        ],
    )
    def test_unsupported_method_raises(
        self,
        analyzer: CorrelationAnalyzer,
        method: object,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [2, 4, 6],
            }
        )

        with pytest.raises(StatisticalValidationError):
            analyzer.analyze(
                dataframe,
                method=method,
            )


class TestCorrelationMatrixIntegrity:
    """Test matrix symmetry, immutability, and source integrity."""

    def test_correlation_matrix_is_symmetric(
        self,
        analyzer: CorrelationAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, np.nan],
                "b": [2.0, 4.0, np.nan, 8.0],
                "c": [3.0, 6.0, 9.0, 12.0],
            }
        )

        matrix, pair_counts = analyzer.analyze(dataframe)

        for row_index in range(len(matrix)):
            for column_index in range(len(matrix)):
                assert matrix[row_index][column_index] == (
                    matrix[column_index][row_index]
                )

                assert pair_counts[row_index][column_index] == (
                    pair_counts[column_index][row_index]
                )

    def test_result_matrices_are_immutable(
        self,
        analyzer: CorrelationAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "b": [2.0, 4.0, 6.0],
            }
        )

        matrix, pair_counts = analyzer.analyze(dataframe)

        assert isinstance(matrix, tuple)
        assert isinstance(pair_counts, tuple)

        assert isinstance(matrix[0], tuple)
        assert isinstance(pair_counts[0], tuple)

        with pytest.raises(TypeError):
            matrix[0][0] = 0.0  # type: ignore[index]

        with pytest.raises(TypeError):
            pair_counts[0][0] = 0  # type: ignore[index]

    def test_original_dataframe_is_not_modified(
        self,
        analyzer: CorrelationAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "a": [1.0, 2.0, np.inf, 4.0],
                "b": [2.0, 4.0, 6.0, np.nan],
            }
        )

        original = dataframe.copy(deep=True)

        analyzer.analyze(dataframe)

        pd.testing.assert_frame_equal(dataframe, original)

    def test_duplicate_row_index(
        self,
        analyzer: CorrelationAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0],
                "b": [2.0, 4.0, 6.0, 8.0],
            },
            index=[0, 0, 1, 1],
        )

        matrix, pair_counts = analyzer.analyze(dataframe)

        assert matrix == (
            (1.0, 1.0),
            (1.0, 1.0),
        )

        assert pair_counts == (
            (4, 4),
            (4, 4),
        )


class TestCorrelationErrorHandling:
    """Test unexpected internal errors."""

    def test_internal_pandas_error_is_wrapped(
        self,
        analyzer: CorrelationAnalyzer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "b": [2.0, 4.0, 6.0],
            }
        )

        def raise_runtime_error(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated pandas failure")

        monkeypatch.setattr(
            pd.DataFrame,
            "corr",
            raise_runtime_error,
        )

        with pytest.raises(CorrelationAnalysisError):
            analyzer.analyze(dataframe)
