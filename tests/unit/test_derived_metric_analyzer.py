"""Unit tests for controlled derived metric analysis."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from lite_cog_agent.statistics.derived_metric import (
    DerivedMetricAnalyzer,
)
from lite_cog_agent.statistics.exceptions import (
    DerivedMetricAnalysisError,
    StatisticalValidationError,
)


@pytest.fixture
def analyzer() -> DerivedMetricAnalyzer:
    return DerivedMetricAnalyzer()


@pytest.fixture
def sales_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "country": ["France", "France", "Germany", "Germany"],
            "category": ["A", "B", "A", "B"],
            "quantity": [2, 3, 4, 5],
            "unit_price": [10.0, 20.0, 30.0, 40.0],
        }
    )


def run_analysis(
    analyzer: DerivedMetricAnalyzer,
    dataframe: pd.DataFrame,
    **kwargs: object,
) -> tuple[Mapping[str, object], ...]:
    return analyzer.analyze(
        dataframe,
        left_column="quantity",
        right_column="unit_price",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Basic calculation
# ---------------------------------------------------------------------------


def test_calculates_product_sum_without_grouping(
    analyzer: DerivedMetricAnalyzer,
    sales_dataframe: pd.DataFrame,
) -> None:
    records = run_analysis(analyzer, sales_dataframe)

    assert len(records) == 1
    assert records[0]["valid_row_count"] == 4
    assert records[0]["result"] == pytest.approx(400.0)


def test_preserves_negative_values(
    analyzer: DerivedMetricAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "quantity": [2, -3, 4],
            "unit_price": [10.0, 20.0, -5.0],
        }
    )

    records = run_analysis(analyzer, dataframe)

    assert records[0]["valid_row_count"] == 3
    assert records[0]["result"] == pytest.approx(-60.0)


def test_uses_stable_floating_point_sum(
    analyzer: DerivedMetricAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "quantity": [1.0, 1.0, 1.0],
            "unit_price": [1e16, 1.0, -1e16],
        }
    )

    records = run_analysis(analyzer, dataframe)

    assert records[0]["result"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. Missing and non-finite values
# ---------------------------------------------------------------------------


def test_excludes_rows_with_missing_operands(
    analyzer: DerivedMetricAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "quantity": [2.0, np.nan, 4.0],
            "unit_price": [10.0, 20.0, np.nan],
        }
    )

    records = run_analysis(analyzer, dataframe)

    assert records[0]["valid_row_count"] == 1
    assert records[0]["result"] == pytest.approx(20.0)


def test_excludes_non_finite_operands(
    analyzer: DerivedMetricAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "quantity": [2.0, np.inf, 4.0, 5.0],
            "unit_price": [10.0, 20.0, -np.inf, 3.0],
        }
    )

    records = run_analysis(analyzer, dataframe)

    assert records[0]["valid_row_count"] == 2
    assert records[0]["result"] == pytest.approx(35.0)


def test_excludes_rows_when_multiplication_overflows(
    analyzer: DerivedMetricAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "quantity": [np.finfo(np.float64).max, 2.0],
            "unit_price": [2.0, 3.0],
        }
    )

    records = run_analysis(analyzer, dataframe)

    assert records[0]["valid_row_count"] == 1
    assert records[0]["result"] == pytest.approx(6.0)


def test_returns_none_when_all_ungrouped_rows_are_invalid(
    analyzer: DerivedMetricAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "quantity": [np.nan, np.inf],
            "unit_price": [10.0, 20.0],
        }
    )

    records = run_analysis(analyzer, dataframe)

    assert len(records) == 1
    assert records[0]["valid_row_count"] == 0
    assert records[0]["result"] is None


# ---------------------------------------------------------------------------
# 3. Grouped calculation
# ---------------------------------------------------------------------------


def test_groups_by_one_column(
    analyzer: DerivedMetricAnalyzer,
    sales_dataframe: pd.DataFrame,
) -> None:
    records = run_analysis(
        analyzer,
        sales_dataframe,
        group_by=["country"],
    )

    assert len(records) == 2
    assert records[0]["country"] == "France"
    assert records[0]["valid_row_count"] == 2
    assert records[0]["result"] == pytest.approx(80.0)

    assert records[1]["country"] == "Germany"
    assert records[1]["valid_row_count"] == 2
    assert records[1]["result"] == pytest.approx(320.0)


def test_groups_by_multiple_columns(
    analyzer: DerivedMetricAnalyzer,
    sales_dataframe: pd.DataFrame,
) -> None:
    records = run_analysis(
        analyzer,
        sales_dataframe,
        group_by=["country", "category"],
    )

    assert len(records) == 4
    assert [
        (record["country"], record["category"])
        for record in records
    ] == [
        ("France", "A"),
        ("France", "B"),
        ("Germany", "A"),
        ("Germany", "B"),
    ]

    assert [record["result"] for record in records] == [
        20.0,
        60.0,
        120.0,
        200.0,
    ]


def test_keeps_groups_with_no_valid_metric_rows(
    analyzer: DerivedMetricAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "country": ["France", "Germany", "Germany"],
            "quantity": [2.0, np.nan, np.nan],
            "unit_price": [10.0, 20.0, 30.0],
        }
    )

    records = run_analysis(
        analyzer,
        dataframe,
        group_by=["country"],
    )

    assert len(records) == 2

    assert records[0]["country"] == "France"
    assert records[0]["valid_row_count"] == 1
    assert records[0]["result"] == pytest.approx(20.0)

    assert records[1]["country"] == "Germany"
    assert records[1]["valid_row_count"] == 0
    assert records[1]["result"] is None


def test_dropna_true_excludes_missing_group_keys(
    analyzer: DerivedMetricAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "country": ["France", None, "Germany"],
            "quantity": [2.0, 3.0, 4.0],
            "unit_price": [10.0, 20.0, 30.0],
        }
    )

    records = run_analysis(
        analyzer,
        dataframe,
        group_by=["country"],
        dropna=True,
    )

    assert len(records) == 2
    assert [record["country"] for record in records] == [
        "France",
        "Germany",
    ]


def test_dropna_false_keeps_missing_group_key(
    analyzer: DerivedMetricAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "country": ["France", None, "Germany"],
            "quantity": [2.0, 3.0, 4.0],
            "unit_price": [10.0, 20.0, 30.0],
        }
    )

    records = run_analysis(
        analyzer,
        dataframe,
        group_by=["country"],
        dropna=False,
    )

    assert len(records) == 3
    missing_group = next(
        record for record in records
        if record["country"] is None
    )
    assert missing_group["valid_row_count"] == 1
    assert missing_group["result"] == pytest.approx(60.0)


def test_group_with_only_invalid_rows_is_kept_when_dropna_false(
    analyzer: DerivedMetricAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "country": [None, None, "France"],
            "quantity": [np.nan, np.nan, 2.0],
            "unit_price": [10.0, 20.0, 10.0],
        }
    )

    records = run_analysis(
        analyzer,
        dataframe,
        group_by=["country"],
        dropna=False,
    )

    assert len(records) == 2

    missing_group = next(
        record for record in records
        if record["country"] is None
    )
    assert missing_group["valid_row_count"] == 0
    assert missing_group["result"] is None


# ---------------------------------------------------------------------------
# 4. Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "operator",
    [
        "divide",
        "",
        None,
        [],
        {},
    ],
)
def test_rejects_unsupported_or_invalid_operators(
    analyzer: DerivedMetricAnalyzer,
    sales_dataframe: pd.DataFrame,
    operator: object,
) -> None:
    with pytest.raises(StatisticalValidationError):
        run_analysis(
            analyzer,
            sales_dataframe,
            operator=operator,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "aggregation",
    [
        "mean",
        "",
        None,
        [],
        {},
    ],
)
def test_rejects_unsupported_or_invalid_aggregations(
    analyzer: DerivedMetricAnalyzer,
    sales_dataframe: pd.DataFrame,
    aggregation: object,
) -> None:
    with pytest.raises(StatisticalValidationError):
        run_analysis(
            analyzer,
            sales_dataframe,
            aggregation=aggregation,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "group_by",
    [
        "country",
        ["country", "country"],
        ["country", ""],
        ["country", None],
        ["result"],
        ["valid_row_count"],
    ],
)
def test_rejects_invalid_group_by(
    analyzer: DerivedMetricAnalyzer,
    sales_dataframe: pd.DataFrame,
    group_by: object,
) -> None:
    with pytest.raises(StatisticalValidationError):
        run_analysis(
            analyzer,
            sales_dataframe,
            group_by=group_by,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "dropna",
    [
        1,
        0,
        "true",
        None,
    ],
)
def test_rejects_non_boolean_dropna(
    analyzer: DerivedMetricAnalyzer,
    sales_dataframe: pd.DataFrame,
    dropna: object,
) -> None:
    with pytest.raises(StatisticalValidationError):
        run_analysis(
            analyzer,
            sales_dataframe,
            dropna=dropna,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "column",
    [
        "missing_column",
        "",
        None,
    ],
)
def test_rejects_invalid_left_column(
    analyzer: DerivedMetricAnalyzer,
    sales_dataframe: pd.DataFrame,
    column: object,
) -> None:
    with pytest.raises(StatisticalValidationError):
        analyzer.analyze(
            sales_dataframe,
            left_column=column,  # type: ignore[arg-type]
            right_column="unit_price",
        )


@pytest.mark.parametrize(
    "dataframe",
    [
        pd.DataFrame(
            {
                "quantity": [True, False],
                "unit_price": [10.0, 20.0],
            }
        ),
        pd.DataFrame(
            {
                "quantity": [1 + 2j, 3 + 4j],
                "unit_price": [10.0, 20.0],
            }
        ),
        pd.DataFrame(
            {
                "quantity": ["2", "3"],
                "unit_price": [10.0, 20.0],
            }
        ),
    ],
)
def test_rejects_non_real_numeric_operand_columns(
    analyzer: DerivedMetricAnalyzer,
    dataframe: pd.DataFrame,
) -> None:
    with pytest.raises(StatisticalValidationError):
        run_analysis(analyzer, dataframe)


def test_rejects_duplicate_dataframe_columns(
    analyzer: DerivedMetricAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        [[1, 2]],
        columns=["quantity", "quantity"],
    )

    with pytest.raises(StatisticalValidationError):
        run_analysis(analyzer, dataframe)


def test_rejects_non_dataframe_input(
    analyzer: DerivedMetricAnalyzer,
) -> None:
    with pytest.raises(StatisticalValidationError):
        analyzer.analyze(
            [],  # type: ignore[arg-type]
            left_column="quantity",
            right_column="unit_price",
        )


# ---------------------------------------------------------------------------
# 5. Empty data, immutability, and input protection
# ---------------------------------------------------------------------------


def test_empty_dataframe_without_grouping_returns_empty_result(
    analyzer: DerivedMetricAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "quantity": pd.Series(dtype="float64"),
            "unit_price": pd.Series(dtype="float64"),
        }
    )

    records = run_analysis(analyzer, dataframe)

    assert len(records) == 1
    assert records[0]["valid_row_count"] == 0
    assert records[0]["result"] is None


def test_empty_dataframe_with_grouping_returns_no_records(
    analyzer: DerivedMetricAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "country": pd.Series(dtype="object"),
            "quantity": pd.Series(dtype="float64"),
            "unit_price": pd.Series(dtype="float64"),
        }
    )

    records = run_analysis(
        analyzer,
        dataframe,
        group_by=["country"],
    )

    assert records == ()


def test_result_records_are_immutable(
    analyzer: DerivedMetricAnalyzer,
    sales_dataframe: pd.DataFrame,
) -> None:
    records = run_analysis(
        analyzer,
        sales_dataframe,
        group_by=["country"],
    )

    assert isinstance(records, tuple)
    assert isinstance(records[0], MappingProxyType)

    with pytest.raises(TypeError):
        records[0]["result"] = 999  # type: ignore[index]


def test_input_dataframe_is_not_modified(
    analyzer: DerivedMetricAnalyzer,
    sales_dataframe: pd.DataFrame,
) -> None:
    original = sales_dataframe.copy(deep=True)

    run_analysis(
        analyzer,
        sales_dataframe,
        group_by=["country"],
    )

    pd.testing.assert_frame_equal(sales_dataframe, original)


def test_grouped_result_is_json_compatible(
    analyzer: DerivedMetricAnalyzer,
    sales_dataframe: pd.DataFrame,
) -> None:
    import json

    records = run_analysis(
        analyzer,
        sales_dataframe,
        group_by=["country"],
    )

    serialized = json.dumps(
        [dict(record) for record in records],
        ensure_ascii=False,
        allow_nan=False,
    )

    assert isinstance(serialized, str)


def test_grouping_failure_is_wrapped_in_domain_exception(
    analyzer: DerivedMetricAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": [["A"], ["B"]],
            "quantity": [2.0, 3.0],
            "unit_price": [10.0, 20.0],
        }
    )

    with pytest.raises(DerivedMetricAnalysisError):
        run_analysis(
            analyzer,
            dataframe,
            group_by=["group"],
        )
