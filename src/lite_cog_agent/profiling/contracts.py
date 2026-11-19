"""Data contracts for dataset profiling."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from types import MappingProxyType
from typing import Any, Mapping

from lite_cog_agent.profiling.exceptions import ProfilingValidationError


def _validate_non_empty_string(value: Any, field_name: str) -> None:
    """Validate that a value is a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ProfilingValidationError(
            f"{field_name} must be a non-empty string"
        )


def _validate_non_negative_integer(
    value: Any,
    field_name: str,
) -> None:
    """Validate that a value is a non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProfilingValidationError(
            f"{field_name} must be a non-negative integer"
        )


def _validate_percentage(value: Any, field_name: str) -> None:
    """Validate that a value is a finite percentage between 0 and 100."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0 <= value <= 100
    ):
        raise ProfilingValidationError(
            f"{field_name} must be a finite number between 0 and 100"
        )


def _validate_string_mapping(
    value: Any,
    field_name: str,
) -> Mapping[str, Any]:
    """Validate and copy a mapping with string keys."""
    if not isinstance(value, Mapping):
        raise ProfilingValidationError(
            f"{field_name} must be a mapping"
        )

    copied = dict(value)

    for key in copied:
        if not isinstance(key, str):
            raise ProfilingValidationError(
                f"{field_name} keys must be strings"
            )

    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """Profiling result for a single dataset column."""

    name: str
    dtype: str
    inferred_type: str
    total_count: int
    non_null_count: int
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    statistics: Mapping[str, Any]
    sample_values: tuple[Any, ...]

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.name, "name")
        _validate_non_empty_string(self.dtype, "dtype")
        _validate_non_empty_string(self.inferred_type, "inferred_type")

        _validate_non_negative_integer(self.total_count, "total_count")
        _validate_non_negative_integer(
            self.non_null_count, "non_null_count"
        )
        _validate_non_negative_integer(self.null_count, "null_count")
        _validate_non_negative_integer(self.unique_count, "unique_count")

        _validate_percentage(
            self.null_percentage, "null_percentage"
        )
        _validate_percentage(
            self.unique_percentage, "unique_percentage"
        )

        if self.total_count != self.non_null_count + self.null_count:
            raise ProfilingValidationError(
                "total_count must equal non_null_count + null_count"
            )

        if self.unique_count > self.non_null_count:
            raise ProfilingValidationError(
                "unique_count cannot exceed non_null_count"
            )

        expected_null_percentage = (
            self.null_count / self.total_count * 100
            if self.total_count > 0
            else 0.0
        )
        expected_unique_percentage = (
            self.unique_count / self.total_count * 100
            if self.total_count > 0
            else 0.0
        )

        if not isclose(
            self.null_percentage,
            expected_null_percentage,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ProfilingValidationError(
                "null_percentage does not match null_count / total_count"
            )

        if not isclose(
            self.unique_percentage,
            expected_unique_percentage,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ProfilingValidationError(
                "unique_percentage does not match unique_count / total_count"
            )

        statistics = _validate_string_mapping(
            self.statistics,
            "statistics",
        )
        object.__setattr__(self, "statistics", statistics)

        if not isinstance(self.sample_values, tuple):
            raise ProfilingValidationError(
                "sample_values must be a tuple"
            )


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """Profiling result for an entire dataset."""

    dataset_id: str
    dataset_name: str
    row_count: int
    column_count: int
    memory_usage_bytes: int
    columns: tuple[ColumnProfile, ...]
    duplicate_row_count: int
    duplicate_percentage: float

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.dataset_id, "dataset_id")
        _validate_non_empty_string(self.dataset_name, "dataset_name")

        _validate_non_negative_integer(self.row_count, "row_count")
        _validate_non_negative_integer(self.column_count, "column_count")
        _validate_non_negative_integer(
            self.memory_usage_bytes,
            "memory_usage_bytes",
        )
        _validate_non_negative_integer(
            self.duplicate_row_count,
            "duplicate_row_count",
        )
        _validate_percentage(
            self.duplicate_percentage,
            "duplicate_percentage",
        )

        if not isinstance(self.columns, tuple):
            raise ProfilingValidationError(
                "columns must be a tuple"
            )

        if any(
            not isinstance(column, ColumnProfile)
            for column in self.columns
        ):
            raise ProfilingValidationError(
                "all columns must be ColumnProfile instances"
            )

        if self.column_count != len(self.columns):
            raise ProfilingValidationError(
                "column_count must equal len(columns)"
            )

        if self.duplicate_row_count > self.row_count:
            raise ProfilingValidationError(
                "duplicate_row_count cannot exceed row_count"
            )

        expected_duplicate_percentage = (
            self.duplicate_row_count / self.row_count * 100
            if self.row_count > 0
            else 0.0
        )

        if not isclose(
            self.duplicate_percentage,
            expected_duplicate_percentage,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ProfilingValidationError(
                "duplicate_percentage does not match duplicate_row_count"
            )


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    """Data quality metrics for a dataset."""

    total_cells: int
    missing_cells: int
    completeness: float
    missing_rate: float
    duplicate_row_count: int
    duplicate_rate: float
    column_completeness: Mapping[str, float]

    def __post_init__(self) -> None:
        _validate_non_negative_integer(self.total_cells, "total_cells")
        _validate_non_negative_integer(
            self.missing_cells, "missing_cells"
        )
        _validate_non_negative_integer(
            self.duplicate_row_count,
            "duplicate_row_count",
        )

        _validate_percentage(self.completeness, "completeness")
        _validate_percentage(self.missing_rate, "missing_rate")
        _validate_percentage(self.duplicate_rate, "duplicate_rate")

        if self.missing_cells > self.total_cells:
            raise ProfilingValidationError(
                "missing_cells cannot exceed total_cells"
            )

        expected_completeness = (
            (self.total_cells - self.missing_cells)
            / self.total_cells
            * 100
            if self.total_cells > 0
            else 0.0
        )
        expected_missing_rate = (
            self.missing_cells / self.total_cells * 100
            if self.total_cells > 0
            else 0.0
        )

        if not isclose(
            self.completeness,
            expected_completeness,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ProfilingValidationError(
                "completeness does not match the cell counts"
            )

        if not isclose(
            self.missing_rate,
            expected_missing_rate,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ProfilingValidationError(
                "missing_rate does not match the cell counts"
            )

        if not isclose(
            self.completeness + self.missing_rate,
            100.0 if self.total_cells > 0 else 0.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ProfilingValidationError(
                "completeness and missing_rate are inconsistent"
            )

        column_completeness = _validate_string_mapping(
            self.column_completeness,
            "column_completeness",
        )

        for name, percentage in column_completeness.items():
            _validate_non_empty_string(
                name,
                "column_completeness key",
            )
            _validate_percentage(
                percentage,
                f"column_completeness[{name!r}]",
            )

        object.__setattr__(
            self,
            "column_completeness",
            column_completeness,
        )


@dataclass(frozen=True, slots=True)
class ProfilingReport:
    """Combined dataset profiling and quality report."""

    dataset_profile: DatasetProfile
    quality_report: DataQualityReport

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_profile, DatasetProfile):
            raise ProfilingValidationError(
                "dataset_profile must be a DatasetProfile"
            )

        if not isinstance(self.quality_report, DataQualityReport):
            raise ProfilingValidationError(
                "quality_report must be a DataQualityReport"
            )
