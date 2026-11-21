"""Data contracts for statistical analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isclose, isfinite
from types import MappingProxyType
from typing import Any

from lite_cog_agent.statistics.exceptions import (
    StatisticalValidationError,
)


def _validate_non_empty_string(value: Any, field_name: str) -> None:
    """Validate that a value is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise StatisticalValidationError(
            f"{field_name} must be a non-empty string"
        )


def _validate_non_negative_integer(
    value: Any,
    field_name: str,
) -> None:
    """Validate that a value is a non-negative integer."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise StatisticalValidationError(
            f"{field_name} must be a non-negative integer"
        )


def _validate_positive_integer(
    value: Any,
    field_name: str,
) -> None:
    """Validate that a value is a positive integer."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise StatisticalValidationError(
            f"{field_name} must be a positive integer"
        )


def _validate_finite_number(
    value: Any,
    field_name: str,
    *,
    allow_none: bool = False,
) -> None:
    """Validate that a value is a finite numeric value."""
    if value is None and allow_none:
        return

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
    ):
        raise StatisticalValidationError(
            f"{field_name} must be a finite number"
            + (" or None" if allow_none else "")
        )


def _validate_percentage(
    value: Any,
    field_name: str,
) -> None:
    """Validate a percentage between 0 and 100."""
    _validate_finite_number(value, field_name)

    if not 0 <= value <= 100:
        raise StatisticalValidationError(
            f"{field_name} must be between 0 and 100"
        )


def _validate_string_tuple(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool = True,
) -> None:
    """Validate a tuple containing unique non-empty strings."""
    if not isinstance(value, tuple):
        raise StatisticalValidationError(
            f"{field_name} must be a tuple"
        )

    if not allow_empty and not value:
        raise StatisticalValidationError(
            f"{field_name} cannot be empty"
        )

    if any(
        not isinstance(item, str) or not item.strip()
        for item in value
    ):
        raise StatisticalValidationError(
            f"all {field_name} items must be non-empty strings"
        )

    if len(value) != len(set(value)):
        raise StatisticalValidationError(
            f"{field_name} cannot contain duplicates"
        )


def _immutable_mapping(
    value: Any,
    field_name: str,
) -> Mapping[str, Any]:
    """Validate and create a shallow read-only mapping copy."""
    if not isinstance(value, Mapping):
        raise StatisticalValidationError(
            f"{field_name} must be a mapping"
        )

    copied = dict(value)

    if any(not isinstance(key, str) for key in copied):
        raise StatisticalValidationError(
            f"{field_name} keys must be strings"
        )

    return MappingProxyType(copied)


def _validate_optional_statistics(
    values: tuple[Any, ...],
    field_names: tuple[str, ...],
) -> None:
    """Validate a sequence of optional finite statistical values."""
    for value, field_name in zip(values, field_names):
        _validate_finite_number(
            value,
            field_name,
            allow_none=True,
        )


@dataclass(frozen=True, slots=True)
class DescriptiveStatistics:
    """Descriptive statistics for a single numeric column."""

    column: str
    count: int
    mean: float | None
    median: float | None
    std: float | None
    variance: float | None
    min: float | None
    q1: float | None
    q3: float | None
    max: float | None
    range: float | None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.column, "column")
        _validate_non_negative_integer(self.count, "count")

        values = (
            self.mean,
            self.median,
            self.std,
            self.variance,
            self.min,
            self.q1,
            self.q3,
            self.max,
            self.range,
        )
        names = (
            "mean",
            "median",
            "std",
            "variance",
            "min",
            "q1",
            "q3",
            "max",
            "range",
        )

        _validate_optional_statistics(values, names)

        if self.count == 0 and any(
            value is not None for value in values
        ):
            raise StatisticalValidationError(
                "statistics must be None when count is zero"
            )

        if self.count == 1 and (
            self.std is not None or self.variance is not None
        ):
            raise StatisticalValidationError(
                "std and variance must be None when count is one"
            )

        if self.std is not None and self.std < 0:
            raise StatisticalValidationError(
                "std cannot be negative"
            )

        if self.variance is not None and self.variance < 0:
            raise StatisticalValidationError(
                "variance cannot be negative"
            )

        if self.range is not None and self.range < 0:
            raise StatisticalValidationError(
                "range cannot be negative"
            )

        if (
            self.min is not None
            and self.max is not None
            and self.min > self.max
        ):
            raise StatisticalValidationError(
                "min cannot exceed max"
            )

        if (
            self.q1 is not None
            and self.q3 is not None
            and self.q1 > self.q3
        ):
            raise StatisticalValidationError(
                "q1 cannot exceed q3"
            )

        if (
            self.min is not None
            and self.max is not None
            and self.range is not None
            and not isclose(
                self.range,
                self.max - self.min,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise StatisticalValidationError(
                "range must equal max - min"
            )


@dataclass(frozen=True, slots=True)
class DescriptiveStatisticsResult:
    """Descriptive statistics for a dataset."""

    dataset_id: str
    dataset_name: str
    row_count: int
    statistics: tuple[DescriptiveStatistics, ...]

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.dataset_id, "dataset_id")
        _validate_non_empty_string(self.dataset_name, "dataset_name")
        _validate_non_negative_integer(self.row_count, "row_count")

        if not isinstance(self.statistics, tuple):
            raise StatisticalValidationError(
                "statistics must be a tuple"
            )

        if any(
            not isinstance(item, DescriptiveStatistics)
            for item in self.statistics
        ):
            raise StatisticalValidationError(
                "all statistics items must be DescriptiveStatistics"
            )

        columns = tuple(item.column for item in self.statistics)

        if len(columns) != len(set(columns)):
            raise StatisticalValidationError(
                "statistics cannot contain duplicate columns"
            )

        if any(item.count > self.row_count for item in self.statistics):
            raise StatisticalValidationError(
                "column count cannot exceed dataset row_count"
            )


@dataclass(frozen=True, slots=True)
class FrequencyItem:
    """Frequency statistics for a single category value."""

    value: Any
    count: int
    percentage: float
    cumulative_percentage: float

    def __post_init__(self) -> None:
        _validate_non_negative_integer(self.count, "count")
        _validate_percentage(self.percentage, "percentage")
        _validate_percentage(
            self.cumulative_percentage,
            "cumulative_percentage",
        )

        if self.cumulative_percentage < self.percentage:
            raise StatisticalValidationError(
                "cumulative_percentage cannot be less than percentage"
            )


@dataclass(frozen=True, slots=True)
class FrequencyAnalysisResult:
    """Frequency analysis result for a dataset column."""

    dataset_id: str
    dataset_name: str
    column: str
    total_count: int
    items: tuple[FrequencyItem, ...]

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.dataset_id, "dataset_id")
        _validate_non_empty_string(self.dataset_name, "dataset_name")
        _validate_non_empty_string(self.column, "column")
        _validate_non_negative_integer(
            self.total_count,
            "total_count",
        )

        if not isinstance(self.items, tuple):
            raise StatisticalValidationError(
                "items must be a tuple"
            )

        if any(
            not isinstance(item, FrequencyItem)
            for item in self.items
        ):
            raise StatisticalValidationError(
                "all items must be FrequencyItem instances"
            )

        if sum(item.count for item in self.items) > self.total_count:
            raise StatisticalValidationError(
                "frequency item counts cannot exceed total_count"
            )

        if self.total_count == 0 and self.items:
            raise StatisticalValidationError(
                "items must be empty when total_count is zero"
            )

        for item in self.items:
            expected_percentage = (
                item.count / self.total_count * 100
                if self.total_count > 0
                else 0.0
            )

            if not isclose(
                item.percentage,
                expected_percentage,
                rel_tol=1e-7,
                abs_tol=1e-7,
            ):
                raise StatisticalValidationError(
                    "percentage does not match count / total_count"
                )

        cumulative = 0.0

        for item in self.items:
            cumulative += item.percentage

            if item.cumulative_percentage > 100:
                raise StatisticalValidationError(
                    "cumulative_percentage cannot exceed 100"
                )

            if not isclose(
                item.cumulative_percentage,
                cumulative,
                rel_tol=1e-7,
                abs_tol=1e-7,
            ):
                raise StatisticalValidationError(
                    "cumulative_percentage is inconsistent "
                    "with item percentages"
                )


@dataclass(frozen=True, slots=True)
class CorrelationAnalysisResult:
    """Correlation matrix and pairwise observation counts."""

    dataset_id: str
    dataset_name: str
    method: str
    columns: tuple[str, ...]
    matrix: tuple[tuple[float | None, ...], ...]
    pair_counts: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.dataset_id, "dataset_id")
        _validate_non_empty_string(self.dataset_name, "dataset_name")

        if self.method not in ("pearson", "spearman"):
            raise StatisticalValidationError(
                "method must be 'pearson' or 'spearman'"
            )

        _validate_string_tuple(self.columns, "columns")

        size = len(self.columns)

        if not isinstance(self.matrix, tuple):
            raise StatisticalValidationError(
                "matrix must be a tuple"
            )

        if not isinstance(self.pair_counts, tuple):
            raise StatisticalValidationError(
                "pair_counts must be a tuple"
            )

        if len(self.matrix) != size or len(self.pair_counts) != size:
            raise StatisticalValidationError(
                "matrix dimensions must match columns"
            )

        for row in self.matrix:
            if not isinstance(row, tuple) or len(row) != size:
                raise StatisticalValidationError(
                    "matrix must be square"
                )

            for value in row:
                _validate_finite_number(
                    value,
                    "correlation coefficient",
                    allow_none=True,
                )

                if value is not None and not -1 <= value <= 1:
                    raise StatisticalValidationError(
                        "correlation coefficients must be between -1 and 1"
                    )

        for row in self.pair_counts:
            if not isinstance(row, tuple) or len(row) != size:
                raise StatisticalValidationError(
                    "pair_counts must be square"
                )

            for value in row:
                _validate_non_negative_integer(
                    value,
                    "pair_count",
                )

        for row_index in range(size):
            for column_index in range(size):
                left = self.matrix[row_index][column_index]
                right = self.matrix[column_index][row_index]

                if left is None or right is None:
                    if left is not right:
                        raise StatisticalValidationError(
                            "correlation matrix must be symmetric"
                        )
                elif not isclose(
                    left,
                    right,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise StatisticalValidationError(
                        "correlation matrix must be symmetric"
                    )

                if (
                    self.pair_counts[row_index][column_index]
                    != self.pair_counts[column_index][row_index]
                ):
                    raise StatisticalValidationError(
                        "pair_counts must be symmetric"
                    )


@dataclass(frozen=True, slots=True)
class GroupedAnalysisResult:
    """Grouped aggregation results."""

    dataset_id: str
    dataset_name: str
    group_by: tuple[str, ...]
    aggregations: Mapping[str, tuple[str, ...]]
    group_count: int
    records: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.dataset_id, "dataset_id")
        _validate_non_empty_string(self.dataset_name, "dataset_name")
        _validate_string_tuple(
            self.group_by,
            "group_by",
            allow_empty=False,
        )
        _validate_non_negative_integer(
            self.group_count,
            "group_count",
        )

        raw_aggregations = _immutable_mapping(
            self.aggregations,
            "aggregations",
        )

        normalized_aggregations: dict[str, tuple[str, ...]] = {}

        for column, operations in raw_aggregations.items():
            _validate_non_empty_string(column, "aggregation column")

            if not isinstance(operations, tuple):
                raise StatisticalValidationError(
                    "aggregation operations must be tuples"
                )

            if not operations:
                raise StatisticalValidationError(
                    "aggregation operations cannot be empty"
                )

            _validate_string_tuple(
                operations,
                f"aggregations[{column}]",
                allow_empty=False,
            )

            normalized_aggregations[column] = operations

        object.__setattr__(
            self,
            "aggregations",
            MappingProxyType(normalized_aggregations),
        )

        if not isinstance(self.records, tuple):
            raise StatisticalValidationError(
                "records must be a tuple"
            )

        if len(self.records) != self.group_count:
            raise StatisticalValidationError(
                "group_count must equal len(records)"
            )

        normalized_records = tuple(
            _immutable_mapping(record, "record")
            for record in self.records
        )

        object.__setattr__(self, "records", normalized_records)


@dataclass(frozen=True, slots=True)
class HistogramBin:
    """A single histogram interval."""

    lower_bound: float
    upper_bound: float
    count: int
    percentage: float

    def __post_init__(self) -> None:
        _validate_finite_number(self.lower_bound, "lower_bound")
        _validate_finite_number(self.upper_bound, "upper_bound")
        _validate_non_negative_integer(self.count, "count")
        _validate_percentage(self.percentage, "percentage")

        if self.lower_bound > self.upper_bound:
            raise StatisticalValidationError(
                "lower_bound cannot exceed upper_bound"
            )


@dataclass(frozen=True, slots=True)
class DistributionAnalysisResult:
    """Distribution analysis result for one numeric column."""

    dataset_id: str
    dataset_name: str
    column: str
    count: int
    mean: float | None
    median: float | None
    std: float | None
    skewness: float | None
    kurtosis: float | None
    quantiles: Mapping[str, float | None]
    histogram: tuple[HistogramBin, ...]

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.dataset_id, "dataset_id")
        _validate_non_empty_string(self.dataset_name, "dataset_name")
        _validate_non_empty_string(self.column, "column")
        _validate_non_negative_integer(self.count, "count")

        _validate_optional_statistics(
            (
                self.mean,
                self.median,
                self.std,
                self.skewness,
                self.kurtosis,
            ),
            (
                "mean",
                "median",
                "std",
                "skewness",
                "kurtosis",
            ),
        )

        if self.std is not None and self.std < 0:
            raise StatisticalValidationError(
                "std cannot be negative"
            )

        quantiles = _immutable_mapping(
            self.quantiles,
            "quantiles",
        )

        for name, value in quantiles.items():
            _validate_non_empty_string(name, "quantile name")
            _validate_finite_number(
                value,
                f"quantiles[{name}]",
                allow_none=True,
            )

        object.__setattr__(self, "quantiles", quantiles)

        if not isinstance(self.histogram, tuple):
            raise StatisticalValidationError(
                "histogram must be a tuple"
            )

        if any(
            not isinstance(item, HistogramBin)
            for item in self.histogram
        ):
            raise StatisticalValidationError(
                "all histogram items must be HistogramBin instances"
            )

        if sum(item.count for item in self.histogram) > self.count:
            raise StatisticalValidationError(
                "histogram counts cannot exceed count"
            )


@dataclass(frozen=True, slots=True)
class DerivedMetricAnalysisResult:
    """Result of a controlled derived metric analysis."""

    dataset_id: str
    dataset_name: str
    row_count: int
    valid_row_count: int
    excluded_row_count: int
    left_column: str
    right_column: str
    operator: str
    aggregation: str
    group_by: tuple[str, ...]
    records: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.dataset_id, "dataset_id")
        _validate_non_empty_string(self.dataset_name, "dataset_name")
        _validate_non_empty_string(self.left_column, "left_column")
        _validate_non_empty_string(self.right_column, "right_column")

        for field_name in (
            "row_count",
            "valid_row_count",
            "excluded_row_count",
        ):
            _validate_non_negative_integer(
                getattr(self, field_name),
                field_name,
            )

        if self.valid_row_count > self.row_count:
            raise StatisticalValidationError(
                "valid_row_count cannot exceed row_count"
            )

        if (
            self.valid_row_count + self.excluded_row_count
            != self.row_count
        ):
            raise StatisticalValidationError(
                "valid_row_count + excluded_row_count "
                "must equal row_count"
            )

        if self.operator != "multiply":
            raise StatisticalValidationError(
                "operator must be 'multiply'"
            )

        if self.aggregation != "sum":
            raise StatisticalValidationError(
                "aggregation must be 'sum'"
            )

        _validate_string_tuple(
            self.group_by,
            "group_by",
        )

        if not isinstance(self.records, tuple):
            raise StatisticalValidationError(
                "records must be a tuple"
            )

        if not self.group_by and len(self.records) != 1:
            raise StatisticalValidationError(
                "ungrouped analysis must contain exactly one record"
            )

        normalized_records = tuple(
            _immutable_mapping(record, "record")
            for record in self.records
        )

        total_record_valid_rows = 0

        for index, record in enumerate(normalized_records):
            if "valid_row_count" not in record:
                raise StatisticalValidationError(
                    f"records[{index}] must contain valid_row_count"
                )

            if "result" not in record:
                raise StatisticalValidationError(
                    f"records[{index}] must contain result"
                )

            record_valid_count = record["valid_row_count"]
            _validate_non_negative_integer(
                record_valid_count,
                f"records[{index}].valid_row_count",
            )

            record_result = record["result"]

            _validate_finite_number(
                record_result,
                f"records[{index}].result",
                allow_none=True,
            )

            if record_valid_count == 0 and record_result is not None:
                raise StatisticalValidationError(
                    f"records[{index}].result must be None "
                    "when valid_row_count is zero"
                )

            if record_valid_count > 0 and record_result is None:
                raise StatisticalValidationError(
                    f"records[{index}].result cannot be None "
                    "when valid_row_count is positive"
                )

            total_record_valid_rows += record_valid_count

        if total_record_valid_rows != self.valid_row_count:
            raise StatisticalValidationError(
                "sum of record valid_row_count values must equal "
                "valid_row_count"
            )

        object.__setattr__(self, "records", normalized_records)
