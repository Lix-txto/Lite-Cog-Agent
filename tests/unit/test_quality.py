"""Unit tests for DataQualityAnalyzer."""

import pandas as pd
import pytest

from lite_cog_agent.profiling.exceptions import (
    DataQualityAnalysisError,
    ProfilingValidationError,
)
from lite_cog_agent.profiling.quality import DataQualityAnalyzer


@pytest.fixture
def analyzer() -> DataQualityAnalyzer:
    return DataQualityAnalyzer()


def test_analyze_calculates_dataset_quality(
    analyzer: DataQualityAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "name": ["Alice", "Bob", None, "David"],
            "age": [20, None, 22, 25],
            "score": [90, 80, 80, 70],
        }
    )

    result = analyzer.analyze(dataframe)

    assert result.total_cells == 12
    assert result.missing_cells == 2
    assert result.completeness == pytest.approx(10 / 12 * 100)
    assert result.missing_rate == pytest.approx(2 / 12 * 100)
    assert result.duplicate_row_count == 0
    assert result.duplicate_rate == 0.0


def test_analyze_calculates_column_completeness(
    analyzer: DataQualityAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "name": ["Alice", "Bob", None, "David"],
            "age": [20, None, 22, 25],
            "score": [90, 80, 80, 70],
        }
    )

    result = analyzer.analyze(dataframe)

    assert dict(result.column_completeness) == {
        "name": 75.0,
        "age": 75.0,
        "score": 100.0,
    }


def test_analyze_counts_duplicate_rows(
    analyzer: DataQualityAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "name": ["A", "A", "B", "B"],
        }
    )

    result = analyzer.analyze(dataframe)

    assert result.duplicate_row_count == 2
    assert result.duplicate_rate == 50.0


def test_analyze_counts_all_repeated_rows(
    analyzer: DataQualityAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "id": [1, 1, 1],
            "name": ["A", "A", "A"],
        }
    )

    result = analyzer.analyze(dataframe)

    assert result.duplicate_row_count == 2
    assert result.duplicate_rate == pytest.approx(200 / 3)


def test_analyze_returns_zero_duplicate_rate_without_duplicates(
    analyzer: DataQualityAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
        }
    )

    result = analyzer.analyze(dataframe)

    assert result.duplicate_row_count == 0
    assert result.duplicate_rate == 0.0


def test_analyze_handles_empty_dataframe_with_columns(
    analyzer: DataQualityAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "name": pd.Series(dtype="object"),
            "age": pd.Series(dtype="int64"),
        }
    )

    result = analyzer.analyze(dataframe)

    assert result.total_cells == 0
    assert result.missing_cells == 0
    assert result.completeness == 0.0
    assert result.missing_rate == 0.0
    assert result.duplicate_row_count == 0
    assert result.duplicate_rate == 0.0
    assert dict(result.column_completeness) == {
        "name": 0.0,
        "age": 0.0,
    }


def test_analyze_handles_dataframe_without_columns(
    analyzer: DataQualityAnalyzer,
) -> None:
    dataframe = pd.DataFrame(index=range(3))

    result = analyzer.analyze(dataframe)

    assert result.total_cells == 0
    assert result.missing_cells == 0
    assert result.completeness == 0.0
    assert result.missing_rate == 0.0
    assert result.duplicate_row_count == 0
    assert result.duplicate_rate == 0.0
    assert dict(result.column_completeness) == {}


def test_analyze_handles_dataframe_without_rows(
    analyzer: DataQualityAnalyzer,
) -> None:
    dataframe = pd.DataFrame(columns=["name", "age"])

    result = analyzer.analyze(dataframe)

    assert result.total_cells == 0
    assert result.missing_cells == 0
    assert result.column_completeness == {
        "name": 0.0,
        "age": 0.0,
    }


def test_analyze_handles_all_missing_values(
    analyzer: DataQualityAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "name": [None, None],
            "age": [None, None],
        }
    )

    result = analyzer.analyze(dataframe)

    assert result.total_cells == 4
    assert result.missing_cells == 4
    assert result.completeness == 0.0
    assert result.missing_rate == 100.0
    assert dict(result.column_completeness) == {
        "name": 0.0,
        "age": 0.0,
    }


def test_analyze_handles_no_missing_values(
    analyzer: DataQualityAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "name": ["Alice", "Bob"],
            "age": [20, 30],
        }
    )

    result = analyzer.analyze(dataframe)

    assert result.missing_cells == 0
    assert result.completeness == 100.0
    assert result.missing_rate == 0.0
    assert dict(result.column_completeness) == {
        "name": 100.0,
        "age": 100.0,
    }


def test_analyze_rejects_non_dataframe_input(
    analyzer: DataQualityAnalyzer,
) -> None:
    with pytest.raises(
        ProfilingValidationError,
        match="dataframe must be a pandas DataFrame",
    ):
        analyzer.analyze([1, 2, 3])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "columns",
    [
        ["name", ""],
        ["name", " "],
        ["name", None],
        ["name", 123],
    ],
)
def test_analyze_rejects_invalid_column_names(
    analyzer: DataQualityAnalyzer,
    columns: list,
) -> None:
    dataframe = pd.DataFrame(
        [[1, 2]],
        columns=columns,
    )

    with pytest.raises(
        ProfilingValidationError,
        match="column names must be non-empty strings",
    ):
        analyzer.analyze(dataframe)


def test_analyze_rejects_duplicate_column_names(
    analyzer: DataQualityAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        [[1, 2]],
        columns=["value", "value"],
    )

    with pytest.raises(
        ProfilingValidationError,
        match="duplicate column name",
    ):
        analyzer.analyze(dataframe)


def test_analyze_does_not_modify_original_dataframe(
    analyzer: DataQualityAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "name": ["Alice", None, "Bob"],
            "age": [20, 30, None],
        }
    )
    original = dataframe.copy(deep=True)

    analyzer.analyze(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)


def test_analyze_wraps_unexpected_errors(
    analyzer: DataQualityAnalyzer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataframe = pd.DataFrame({"value": [1, 2, 3]})

    def raise_error(_: pd.DataFrame) -> int:
        raise RuntimeError("unexpected analysis failure")

    monkeypatch.setattr(
        analyzer,
        "_calculate_missing_cells",
        raise_error,
    )

    with pytest.raises(
        DataQualityAnalysisError,
        match="Failed to analyze data quality",
    ) as exc_info:
        analyzer.analyze(dataframe)

    assert isinstance(exc_info.value.__cause__, RuntimeError)


class TestMissingAndEmptyData:
    def test_analyzes_empty_dataframe_with_columns(
        self,
        analyzer: DataQualityAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(columns=["name", "age"])

        result = analyzer.analyze(dataframe)

        assert result.total_cells == 0
        assert result.missing_cells == 0
        assert result.completeness == 0.0
        assert result.missing_rate == 0.0
        assert result.duplicate_row_count == 0
        assert result.duplicate_rate == 0.0

        assert result.column_completeness == {
            "name": 0.0,
            "age": 0.0,
        }

    def test_analyzes_dataframe_without_columns(
        self,
        analyzer: DataQualityAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(index=[0, 1, 2])

        result = analyzer.analyze(dataframe)

        assert result.total_cells == 0
        assert result.missing_cells == 0
        assert result.completeness == 0.0
        assert result.missing_rate == 0.0
        assert result.duplicate_row_count == 0
        assert result.duplicate_rate == 0.0
        assert dict(result.column_completeness) == {}

    def test_analyzes_dataframe_with_all_missing_values(
        self,
        analyzer: DataQualityAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "name": [None, None],
                "age": [float("nan"), float("nan")],
            }
        )

        result = analyzer.analyze(dataframe)

        assert result.total_cells == 4
        assert result.missing_cells == 4
        assert result.completeness == 0.0
        assert result.missing_rate == 100.0
        assert result.column_completeness == {
            "name": 0.0,
            "age": 0.0,
        }

    def test_analyzes_mixed_missing_value_types(
        self,
        analyzer: DataQualityAnalyzer,
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
                "object_value": [
                    "A",
                    None,
                    "C",
                ],
            }
        )

        result = analyzer.analyze(dataframe)

        assert result.total_cells == 9
        assert result.missing_cells == 3
        assert result.completeness == pytest.approx(
            200 / 3
        )
        assert result.missing_rate == pytest.approx(
            100 / 3
        )

        assert result.column_completeness == {
            "float_value": pytest.approx(200 / 3),
            "integer_value": pytest.approx(200 / 3),
            "object_value": pytest.approx(200 / 3),
        }

    def test_empty_dataframe_analysis_does_not_modify_input(
        self,
        analyzer: DataQualityAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "value": pd.Series([], dtype="Float64"),
            }
        )
        original = dataframe.copy(deep=True)

        analyzer.analyze(dataframe)

        pd.testing.assert_frame_equal(dataframe, original)

    def test_all_missing_dataframe_analysis_does_not_modify_input(
        self,
        analyzer: DataQualityAnalyzer,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "value": pd.Series(
                    [pd.NA, pd.NA],
                    dtype="Float64",
                )
            }
        )
        original = dataframe.copy(deep=True)

        analyzer.analyze(dataframe)

        pd.testing.assert_frame_equal(dataframe, original)