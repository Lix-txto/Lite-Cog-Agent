from __future__ import annotations

import pandas as pd
import pytest

from lite_cog_agent.statistics.exceptions import (
    GroupedAnalysisError,
    StatisticalValidationError,
)
from lite_cog_agent.statistics.grouped import GroupedAnalyzer


@pytest.fixture
def analyzer() -> GroupedAnalyzer:
    return GroupedAnalyzer()


@pytest.fixture
def employee_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "department": [
                "Engineering",
                "Engineering",
                "Marketing",
                "Marketing",
                "Finance",
            ],
            "region": [
                "East",
                "West",
                "East",
                "West",
                "East",
            ],
            "salary": [100.0, 120.0, 80.0, 90.0, 110.0],
            "age": [25, 30, 28, 32, 35],
        }
    )


# ---------------------------------------------------------------------------
# 1. Basic grouping
# ---------------------------------------------------------------------------


def test_single_column_grouping(
    analyzer: GroupedAnalyzer,
    employee_dataframe: pd.DataFrame,
) -> None:
    records = analyzer.analyze(
        employee_dataframe,
        group_by=["department"],
        aggregations={"salary": ["mean"]},
    )

    assert len(records) == 3

    engineering = next(
        record
        for record in records
        if record["department"] == "Engineering"
    )

    assert engineering["salary_mean"] == 110.0


def test_multiple_column_grouping(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "department": ["A", "A", "A", "B"],
            "region": ["East", "East", "West", "East"],
            "sales": [100, 200, 300, 400],
        }
    )

    records = analyzer.analyze(
        dataframe,
        group_by=["department", "region"],
        aggregations={"sales": ["sum", "count"]},
    )

    assert len(records) == 3

    east_record = next(
        record
        for record in records
        if record["department"] == "A"
        and record["region"] == "East"
    )

    assert east_record["sales_sum"] == 300.0
    assert east_record["sales_count"] == 2


def test_grouping_respects_sort_option(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": ["C", "A", "B"],
            "value": [30, 10, 20],
        }
    )

    sorted_records = analyzer.analyze(
        dataframe,
        group_by=["group"],
        aggregations={"value": ["sum"]},
        sort=True,
    )

    unsorted_records = analyzer.analyze(
        dataframe,
        group_by=["group"],
        aggregations={"value": ["sum"]},
        sort=False,
    )

    assert [r["group"] for r in sorted_records] == ["A", "B", "C"]
    assert [r["group"] for r in unsorted_records] == ["C", "A", "B"]


# ---------------------------------------------------------------------------
# 2. Aggregation operations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("count", 3),
        ("sum", 60.0),
        ("mean", 20.0),
        ("median", 20.0),
        ("min", 10.0),
        ("max", 30.0),
        ("std", 10.0),
    ],
)
def test_supported_aggregation_operations(
    analyzer: GroupedAnalyzer,
    operation: str,
    expected: float,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": ["A", "A", "A"],
            "value": [10.0, 20.0, 30.0],
        }
    )

    records = analyzer.analyze(
        dataframe,
        group_by=["group"],
        aggregations={"value": [operation]},
    )

    assert records[0][f"value_{operation}"] == pytest.approx(expected)


def test_multiple_aggregations_for_same_column(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": ["A", "A", "B"],
            "value": [10, 20, 30],
        }
    )

    records = analyzer.analyze(
        dataframe,
        group_by=["group"],
        aggregations={
            "value": ["count", "sum", "mean", "median", "min", "max", "std"]
        },
    )

    record_a = next(
        record for record in records if record["group"] == "A"
    )

    assert record_a["value_count"] == 2
    assert record_a["value_sum"] == 30.0
    assert record_a["value_mean"] == 15.0
    assert record_a["value_median"] == 15.0
    assert record_a["value_min"] == 10.0
    assert record_a["value_max"] == 20.0
    assert record_a["value_std"] == pytest.approx(7.0710678118654755)


# ---------------------------------------------------------------------------
# 3. Missing values and empty data
# ---------------------------------------------------------------------------


def test_dropna_true_excludes_missing_group_keys(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": ["A", None, "B"],
            "value": [10, 20, 30],
        }
    )

    records = analyzer.analyze(
        dataframe,
        group_by=["group"],
        aggregations={"value": ["sum"]},
        dropna=True,
    )

    assert len(records) == 2
    assert all(record["group"] is not None for record in records)


def test_dropna_false_keeps_missing_group_keys(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": ["A", None, "B"],
            "value": [10, 20, 30],
        }
    )

    records = analyzer.analyze(
        dataframe,
        group_by=["group"],
        aggregations={"value": ["sum"]},
        dropna=False,
    )

    assert len(records) == 3
    assert any(record["group"] is None for record in records)


def test_aggregation_ignores_missing_values(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": ["A", "A", "A"],
            "value": [10.0, None, 30.0],
        }
    )

    records = analyzer.analyze(
        dataframe,
        group_by=["group"],
        aggregations={"value": ["count", "sum", "mean"]},
    )

    assert records[0]["value_count"] == 2
    assert records[0]["value_sum"] == 40.0
    assert records[0]["value_mean"] == 20.0


def test_empty_dataframe_returns_empty_tuple(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": pd.Series(dtype="object"),
            "value": pd.Series(dtype="float64"),
        }
    )

    records = analyzer.analyze(
        dataframe,
        group_by=["group"],
        aggregations={"value": ["mean"]},
    )

    assert records == ()
    assert isinstance(records, tuple)


def test_empty_dataframe_still_validates_arguments(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": pd.Series(dtype="object"),
            "value": pd.Series(dtype="float64"),
        }
    )

    with pytest.raises(StatisticalValidationError):
        analyzer.analyze(
            dataframe,
            group_by=["missing_column"],
            aggregations={"value": ["mean"]},
        )


def test_all_missing_aggregation_values_return_none(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": ["A", "A"],
            "value": pd.Series([None, None],dtype="Float64"),
        }
    )

    records = analyzer.analyze(
        dataframe,
        group_by=["group"],
        aggregations={"value": ["sum"]},
    )

    assert records[0]["value_sum"] is None


# ---------------------------------------------------------------------------
# 4. Numeric boundaries
# ---------------------------------------------------------------------------


def test_infinity_is_counted_but_excluded_from_numeric_aggregations(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": ["A", "A", "A"],
            "value": [1.0, float("inf"), float("-inf")],
        }
    )

    records = analyzer.analyze(
        dataframe,
        group_by=["group"],
        aggregations={"value": ["count", "sum", "mean", "min", "max"]},
    )

    record = records[0]

    assert record["value_count"] == 3
    assert record["value_sum"] == 1.0
    assert record["value_mean"] == 1.0
    assert record["value_min"] == 1.0
    assert record["value_max"] == 1.0


def test_single_observation_std_returns_none(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": ["A"],
            "value": [10.0],
        }
    )

    records = analyzer.analyze(
        dataframe,
        group_by=["group"],
        aggregations={"value": ["std"]},
    )

    assert records[0]["value_std"] is None


def test_constant_group_std_is_zero(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": ["A", "A", "A"],
            "value": [5.0, 5.0, 5.0],
        }
    )

    records = analyzer.analyze(
        dataframe,
        group_by=["group"],
        aggregations={"value": ["std"]},
    )

    assert records[0]["value_std"] == 0.0


def test_nullable_integer_column(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": ["A", "A", "B"],
            "value": pd.Series([10, None, 30], dtype="Int64"),
        }
    )

    records = analyzer.analyze(
        dataframe,
        group_by=["group"],
        aggregations={"value": ["count", "sum", "mean"]},
    )

    record_a = next(
        record for record in records if record["group"] == "A"
    )

    assert record_a["value_count"] == 1
    assert record_a["value_sum"] == 10.0
    assert record_a["value_mean"] == 10.0


def test_nullable_float_column(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": ["A", "A", "B"],
            "value": pd.Series([1.5, None, 3.5], dtype="Float64"),
        }
    )

    records = analyzer.analyze(
        dataframe,
        group_by=["group"],
        aggregations={"value": ["mean"]},
    )

    record_a = next(
        record for record in records if record["group"] == "A"
    )

    assert record_a["value_mean"] == 1.5


# ---------------------------------------------------------------------------
# 5. Data type handling
# ---------------------------------------------------------------------------


def test_string_min_and_max(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": ["A", "A", "A"],
            "name": ["Charlie", "Alice", "Bob"],
        }
    )

    records = analyzer.analyze(
        dataframe,
        group_by=["group"],
        aggregations={"name": ["min", "max"]},
    )

    assert records[0]["name_min"] == "Alice"
    assert records[0]["name_max"] == "Charlie"


def test_datetime_min_and_max_are_iso_strings(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": ["A", "A"],
            "created_at": pd.to_datetime(
                ["2026-01-02", "2026-01-01"]
            ),
        }
    )

    records = analyzer.analyze(
        dataframe,
        group_by=["group"],
        aggregations={"created_at": ["min", "max"]},
    )

    assert records[0]["created_at_min"] == "2026-01-01T00:00:00"
    assert records[0]["created_at_max"] == "2026-01-02T00:00:00"


def test_boolean_column_only_supports_count(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": ["A", "A"],
            "active": [True, False],
        }
    )

    records = analyzer.analyze(
        dataframe,
        group_by=["group"],
        aggregations={"active": ["count"]},
    )

    assert records[0]["active_count"] == 2

    with pytest.raises(StatisticalValidationError):
        analyzer.analyze(
            dataframe,
            group_by=["group"],
            aggregations={"active": ["sum"]},
        )


def test_complex_column_is_rejected_for_numeric_aggregation(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": ["A", "A"],
            "value": [1 + 2j, 3 + 4j],
        }
    )

    with pytest.raises(StatisticalValidationError):
        analyzer.analyze(
            dataframe,
            group_by=["group"],
            aggregations={"value": ["mean"]},
        )


# ---------------------------------------------------------------------------
# 6. Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "group_by",
    [
        [],
        (),
        "group",
        [""],
        ["   "],
        [1],
        ["group", "group"],
        ["missing"],
    ],
)
def test_invalid_group_by_is_rejected(
    analyzer: GroupedAnalyzer,
    employee_dataframe: pd.DataFrame,
    group_by: object,
) -> None:
    with pytest.raises(StatisticalValidationError):
        analyzer.analyze(
            employee_dataframe,
            group_by=group_by,  # type: ignore[arg-type]
            aggregations={"salary": ["mean"]},
        )


@pytest.mark.parametrize(
    "aggregations",
    [
        {},
        [],
        None,
        {"missing": ["mean"]},
        {"salary": []},
        {"salary": "mean"},
        {"salary": ["unknown"]},
        {"salary": ["mean", "mean"]},
        {"": ["mean"]},
    ],
)
def test_invalid_aggregations_are_rejected(
    analyzer: GroupedAnalyzer,
    employee_dataframe: pd.DataFrame,
    aggregations: object,
) -> None:
    with pytest.raises(StatisticalValidationError):
        analyzer.analyze(
            employee_dataframe,
            group_by=["department"],
            aggregations=aggregations,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("dropna", "sort"),
    [
        (1, True),
        (True, 1),
        ("true", False),
        (False, None),
    ],
)
def test_invalid_options_are_rejected(
    analyzer: GroupedAnalyzer,
    employee_dataframe: pd.DataFrame,
    dropna: object,
    sort: object,
) -> None:
    with pytest.raises(StatisticalValidationError):
        analyzer.analyze(
            employee_dataframe,
            group_by=["department"],
            aggregations={"salary": ["mean"]},
            dropna=dropna,  # type: ignore[arg-type]
            sort=sort,  # type: ignore[arg-type]
        )


def test_duplicate_dataframe_columns_are_rejected(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        [[1, 2]],
        columns=["group", "group"],
    )

    with pytest.raises(StatisticalValidationError):
        analyzer.analyze(
            dataframe,
            group_by=["group"],
            aggregations={"group": ["count"]},
        )


def test_grouping_and_aggregation_output_collision_is_rejected(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": ["A", "A"],
            "value": [10, 20],
            "value_mean": [1, 2],
        }
    )

    with pytest.raises(StatisticalValidationError):
        analyzer.analyze(
            dataframe,
            group_by=["group", "value_mean"],
            aggregations={"value": ["mean"]},
        )


# ---------------------------------------------------------------------------
# 7. Result immutability and input protection
# ---------------------------------------------------------------------------


def test_result_records_are_immutable(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": ["A", "A"],
            "value": [10, 20],
        }
    )

    records = analyzer.analyze(
        dataframe,
        group_by=["group"],
        aggregations={"value": ["sum"]},
    )

    with pytest.raises(TypeError):
        records[0]["value_sum"] = 999


def test_result_is_a_tuple(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": ["A", "B"],
            "value": [10, 20],
        }
    )

    records = analyzer.analyze(
        dataframe,
        group_by=["group"],
        aggregations={"value": ["sum"]},
    )

    assert isinstance(records, tuple)


def test_original_dataframe_is_not_modified(
    analyzer: GroupedAnalyzer,
    employee_dataframe: pd.DataFrame,
) -> None:
    original = employee_dataframe.copy(deep=True)

    analyzer.analyze(
        employee_dataframe,
        group_by=["department"],
        aggregations={
            "salary": ["mean", "std"],
            "age": ["min", "max"],
        },
    )

    pd.testing.assert_frame_equal(
        employee_dataframe,
        original,
    )


# ---------------------------------------------------------------------------
# 8. Categorical grouping
# ---------------------------------------------------------------------------


def test_categorical_grouping_ignores_unused_categories(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": pd.Categorical(
                ["A", "B", "A"],
                categories=["A", "B", "C"],
            ),
            "value": [10, 20, 30],
        }
    )

    records = analyzer.analyze(
        dataframe,
        group_by=["group"],
        aggregations={"value": ["sum"]},
    )

    assert len(records) == 2
    assert {record["group"] for record in records} == {"A", "B"}

def test_groupby_execution_error_is_wrapped(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": [["A"], ["A"]],
            "value": [10, 20],
        }
    )

    with pytest.raises(GroupedAnalysisError):
        analyzer.analyze(
            dataframe,
            group_by=["group"],
            aggregations={"value": ["sum"]},
        )


def test_tuple_valued_single_column_grouping(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": pd.Series(
                [("A", 1), ("A", 1), ("B", 2)],
                dtype="object",
            ),
            "value": [10, 20, 30],
        }
    )

    records = analyzer.analyze(
        dataframe,
        group_by=["group"],
        aggregations={"value": ["sum"]},
    )

    assert len(records) == 2

    record_a = next(
        record
        for record in records
        if record["group"] == ("A", 1)
    )

    assert record_a["value_sum"] == 30.0


def test_large_numeric_values_do_not_produce_non_finite_results(
    analyzer: GroupedAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": ["A", "A", "A"],
            "value": [1e150, 2e150, 3e150],
        }
    )

    records = analyzer.analyze(
        dataframe,
        group_by=["group"],
        aggregations={
            "value": ["sum", "mean", "std"],
        },
    )

    record = records[0]

    assert record["value_sum"] == pytest.approx(6e150)
    assert record["value_mean"] == pytest.approx(2e150)
    assert pd.notna(record["value_std"])

    for key in ("value_sum", "value_mean", "value_std"):
        value = record[key]
        if value is not None:
            assert value != float("inf")
            assert value != float("-inf")


@pytest.mark.parametrize(
    "group_dtype",
    ["Int64", "string"],
)
def test_nullable_grouping_key_with_missing_values(
    analyzer: GroupedAnalyzer,
    group_dtype: str,
) -> None:
    dataframe = pd.DataFrame(
        {
            "group": pd.Series(
                ["1", None, "2"] if group_dtype == "string"
                else [1, None, 2],
                dtype=group_dtype,
            ),
            "value": [10, 20, 30],
        }
    )

    records = analyzer.analyze(
        dataframe,
        group_by=["group"],
        aggregations={"value": ["sum"]},
        dropna=False,
    )

    assert len(records) == 3
    assert any(record["group"] is None for record in records)

def test_accepts_tuple_aggregation_operations():
    dataframe = pd.DataFrame(
        {
            "department": ["A", "A", "B"],
            "salary": [5000, 6000, 7000],
        }
    )

    result = GroupedAnalyzer().analyze(
        dataframe,
        group_by=["department"],
        aggregations={"salary": ("count", "mean")},
    )

    assert len(result) == 2
    assert result[0]["salary_count"] == 2
    assert result[0]["salary_mean"] == 5500