"""Tests for derived metric analysis contracts."""

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from lite_cog_agent.statistics.contracts import (
    DerivedMetricAnalysisResult,
)
from lite_cog_agent.statistics.exceptions import (
    StatisticalValidationError,
)


def make_result(**overrides):
    values = {
        "dataset_id": "dataset-1",
        "dataset_name": "online-retail",
        "row_count": 5,
        "valid_row_count": 4,
        "excluded_row_count": 1,
        "left_column": "Quantity",
        "right_column": "UnitPrice",
        "operator": "multiply",
        "aggregation": "sum",
        "group_by": (),
        "records": (
            {
                "valid_row_count": 4,
                "result": 125.0,
            },
        ),
    }
    values.update(overrides)
    return DerivedMetricAnalysisResult(**values)


def test_derived_metric_result_creation():
    result = make_result()

    assert result.row_count == 5
    assert result.valid_row_count == 4
    assert result.excluded_row_count == 1
    assert result.operator == "multiply"
    assert result.aggregation == "sum"
    assert result.records[0]["result"] == 125.0


def test_derived_metric_result_supports_grouped_records():
    result = make_result(
        row_count=5,
        valid_row_count=4,
        excluded_row_count=1,
        group_by=("Country",),
        records=(
            {"Country": "France", "valid_row_count": 2, "result": 50.0},
            {"Country": "Germany", "valid_row_count": 2, "result": 75.0},
        ),
    )

    assert len(result.records) == 2
    assert sum(
        record["valid_row_count"] for record in result.records
    ) == result.valid_row_count


def test_derived_metric_result_allows_no_valid_rows():
    result = make_result(
        row_count=5,
        valid_row_count=0,
        excluded_row_count=5,
        records=(
            {"valid_row_count": 0, "result": None},
        ),
    )

    assert result.valid_row_count == 0
    assert result.records[0]["result"] is None


def test_derived_metric_result_allows_empty_grouped_records():
    result = make_result(
        row_count=0,
        valid_row_count=0,
        excluded_row_count=0,
        group_by=("Country",),
        records=(),
    )

    assert result.records == ()


def test_derived_metric_records_are_read_only():
    result = make_result()

    assert isinstance(result.records[0], MappingProxyType)

    with pytest.raises(TypeError):
        result.records[0]["result"] = 999.0


def test_derived_metric_result_is_frozen():
    result = make_result()

    with pytest.raises(FrozenInstanceError):
        result.row_count = 10


@pytest.mark.parametrize(
    "overrides",
    [
        {"row_count": -1},
        {"row_count": True},
        {"valid_row_count": -1},
        {"valid_row_count": True},
        {"excluded_row_count": -1},
        {"excluded_row_count": True},
        {"row_count": 5, "valid_row_count": 4, "excluded_row_count": 0},
        {"row_count": 5, "valid_row_count": 6, "excluded_row_count": 0},
    ],
)
def test_derived_metric_result_rejects_invalid_counts(overrides):
    with pytest.raises(StatisticalValidationError):
        make_result(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"operator": "divide"},
        {"operator": ""},
        {"aggregation": "mean"},
        {"aggregation": ""},
    ],
)
def test_derived_metric_result_rejects_unsupported_operations(overrides):
    with pytest.raises(StatisticalValidationError):
        make_result(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"group_by": ["Country"]},
        {"group_by": ("Country", "Country")},
        {"group_by": ("",)},
        {"group_by": (None,)},
    ],
)
def test_derived_metric_result_rejects_invalid_group_by(overrides):
    with pytest.raises(StatisticalValidationError):
        make_result(**overrides)


@pytest.mark.parametrize(
    "records",
    [
        ({"result": 10.0},),
        ({"valid_row_count": 1},),
        ({"valid_row_count": -1, "result": 10.0},),
        ({"valid_row_count": True, "result": 10.0},),
        ({"valid_row_count": 1, "result": float("nan")},),
        ({"valid_row_count": 1, "result": float("inf")},),
    ],
)
def test_derived_metric_result_rejects_invalid_records(records):
    with pytest.raises(StatisticalValidationError):
        make_result(
            row_count=1,
            valid_row_count=1,
            excluded_row_count=0,
            records=records,
        )


def test_derived_metric_result_rejects_record_count_mismatch():
    with pytest.raises(StatisticalValidationError):
        make_result(
            valid_row_count=4,
            records=(
                {"valid_row_count": 2, "result": 50.0},
            ),
        )


def test_ungrouped_result_requires_exactly_one_record():
    with pytest.raises(StatisticalValidationError):
        make_result(
            records=(
                {"valid_row_count": 2, "result": 50.0},
                {"valid_row_count": 2, "result": 75.0},
            ),
        )


@pytest.mark.parametrize(
    ("record_valid_count", "record_result"),
    [
        (0, 10.0),
        (1, None),
    ],
)
def test_derived_metric_result_rejects_inconsistent_record_result(
    record_valid_count,
    record_result,
):
    with pytest.raises(StatisticalValidationError):
        make_result(
            row_count=1,
            valid_row_count=record_valid_count,
            excluded_row_count=1 - record_valid_count,
            records=(
                {
                    "valid_row_count": record_valid_count,
                    "result": record_result,
                },
            ),
        )


def test_derived_metric_result_allows_zero_sum_with_valid_rows():
    result = make_result(
        row_count=1,
        valid_row_count=1,
        excluded_row_count=0,
        records=(
            {
                "valid_row_count": 1,
                "result": 0.0,
            },
        ),
    )

    assert result.records[0]["valid_row_count"] == 1
    assert result.records[0]["result"] == 0.0
