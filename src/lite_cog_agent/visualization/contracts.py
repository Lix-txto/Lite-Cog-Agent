"""Data contracts for visualization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import fsum, isclose, isfinite
from types import MappingProxyType
from typing import Any, Mapping

from lite_cog_agent.statistics.contracts import HistogramBin
from lite_cog_agent.visualization.exceptions import (
    VisualizationValidationError,
)


class ChartType(str, Enum):
    """Supported chart types."""

    BAR = "bar"
    LINE = "line"
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    PIE = "pie"
    BOX = "box"


_BOX_STATISTIC_KEYS = frozenset(
    {"min", "q1", "median", "q3", "max"}
)


def _validate_non_empty_string(
    value: Any,
    field_name: str,
) -> None:
    """Validate a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise VisualizationValidationError(
            f"{field_name} must be a non-empty string"
        )


def _validate_finite_number(
    value: Any,
    field_name: str,
    *,
    allow_none: bool = False,
) -> None:
    """Validate a finite numeric value."""
    if value is None and allow_none:
        return

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise VisualizationValidationError(
            f"{field_name} must be a finite number"
            + (" or None" if allow_none else "")
        )

    try:
        finite = isfinite(value)
    except (OverflowError, TypeError):
        finite = False

    if not finite:
        raise VisualizationValidationError(
            f"{field_name} must be a finite number"
            + (" or None" if allow_none else "")
        )


def _immutable_mapping(
    value: Any,
    field_name: str,
) -> Mapping[str, Any]:
    """Validate and create a shallow read-only mapping copy."""
    if not isinstance(value, Mapping):
        raise VisualizationValidationError(
            f"{field_name} must be a mapping"
        )

    copied = dict(value)

    if any(not isinstance(key, str) for key in copied):
        raise VisualizationValidationError(
            f"{field_name} keys must be strings"
        )

    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class ChartSeries:
    """A single data series used by a chart."""

    name: str
    x_values: tuple[str | int | float | None, ...] = ()
    y_values: tuple[float | None, ...] = ()
    statistics: Mapping[str, float | None] | None = None
    outliers: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.name, "name")

        if not isinstance(self.x_values, tuple):
            raise VisualizationValidationError(
                "x_values must be a tuple"
            )

        if not isinstance(self.y_values, tuple):
            raise VisualizationValidationError(
                "y_values must be a tuple"
            )

        if len(self.x_values) != len(self.y_values):
            raise VisualizationValidationError(
                "x_values and y_values must have the same length"
            )

        for value in self.x_values:
            if value is None or isinstance(value, str):
                continue

            _validate_finite_number(value, "x_values item")

        for value in self.y_values:
            _validate_finite_number(
                value,
                "y_values item",
                allow_none=True,
            )

        statistics = (
            {}
            if self.statistics is None
            else self.statistics
        )
        immutable_statistics = _immutable_mapping(
            statistics,
            "statistics",
        )

        for name, value in immutable_statistics.items():
            _validate_non_empty_string(
                name,
                "statistics key",
            )
            _validate_finite_number(
                value,
                f"statistics[{name}]",
                allow_none=True,
            )

        object.__setattr__(
            self,
            "statistics",
            immutable_statistics,
        )

        if not isinstance(self.outliers, tuple):
            raise VisualizationValidationError(
                "outliers must be a tuple"
            )

        for value in self.outliers:
            _validate_finite_number(value, "outliers item")


@dataclass(frozen=True, slots=True)
class ChartData:
    """Chart data and its source dataset metadata."""

    dataset_id: str
    dataset_name: str
    series: tuple[ChartSeries, ...] = ()
    histogram_bins: tuple[HistogramBin, ...] = ()
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(
            self.dataset_id,
            "dataset_id",
        )
        _validate_non_empty_string(
            self.dataset_name,
            "dataset_name",
        )

        if not isinstance(self.series, tuple):
            raise VisualizationValidationError(
                "series must be a tuple"
            )

        if any(
            not isinstance(item, ChartSeries)
            for item in self.series
        ):
            raise VisualizationValidationError(
                "all series items must be ChartSeries instances"
            )

        series_names = tuple(item.name for item in self.series)

        if len(series_names) != len(set(series_names)):
            raise VisualizationValidationError(
                "series names must be unique"
            )

        if not isinstance(self.histogram_bins, tuple):
            raise VisualizationValidationError(
                "histogram_bins must be a tuple"
            )

        if any(
            not isinstance(item, HistogramBin)
            for item in self.histogram_bins
        ):
            raise VisualizationValidationError(
                "all histogram_bins items must be HistogramBin instances"
            )

        if self.series and self.histogram_bins:
            raise VisualizationValidationError(
                "series and histogram_bins cannot both contain data"
            )

        if not self.series and not self.histogram_bins:
            raise VisualizationValidationError(
                "chart data must contain at least one series or histogram bin"
            )

        if self.histogram_bins:
            total_count = sum(
                item.count for item in self.histogram_bins
            )
            total_percentage = fsum(
                item.percentage for item in self.histogram_bins
            )

            if total_count > 0 and not isclose(
                total_percentage,
                100.0,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise VisualizationValidationError(
                    "histogram percentages must sum to 100"
                )

            if total_count == 0 and not isclose(
                total_percentage,
                0.0,
                abs_tol=1e-9,
            ):
                raise VisualizationValidationError(
                    "empty histogram percentages must sum to zero"
                )

        metadata = (
            {}
            if self.metadata is None
            else self.metadata
        )
        object.__setattr__(
            self,
            "metadata",
            _immutable_mapping(metadata, "metadata"),
        )


@dataclass(frozen=True, slots=True)
class ChartSpecification:
    """Complete chart specification without rendering details."""

    chart_type: ChartType
    title: str
    data: ChartData
    x_label: str | None = None
    y_label: str | None = None
    options: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.chart_type, ChartType):
            raise VisualizationValidationError(
                "chart_type must be a ChartType"
            )

        _validate_non_empty_string(self.title, "title")

        if not isinstance(self.data, ChartData):
            raise VisualizationValidationError(
                "data must be a ChartData instance"
            )

        if self.x_label is not None:
            _validate_non_empty_string(self.x_label, "x_label")

        if self.y_label is not None:
            _validate_non_empty_string(self.y_label, "y_label")

        options = (
            {}
            if self.options is None
            else self.options
        )
        object.__setattr__(
            self,
            "options",
            _immutable_mapping(options, "options"),
        )

        self._validate_chart_data()

    def _validate_chart_data(self) -> None:
        """Validate data layout against the selected chart type."""
        data = self.data

        if self.chart_type is ChartType.HISTOGRAM:
            if data.series or not data.histogram_bins:
                raise VisualizationValidationError(
                    "histogram charts require histogram_bins "
                    "and cannot contain series"
                )
            return

        if data.histogram_bins:
            raise VisualizationValidationError(
                "only histogram charts can use histogram_bins"
            )

        if not data.series:
            raise VisualizationValidationError(
                f"{self.chart_type.value} charts require at least one series"
            )

        if self.chart_type is ChartType.BOX:
            for series in data.series:
                if series.x_values or series.y_values:
                    raise VisualizationValidationError(
                        "box chart series cannot contain x_values or y_values"
                    )

                if (
                    series.statistics is None
                    or set(series.statistics) != _BOX_STATISTIC_KEYS
                ):
                    raise VisualizationValidationError(
                        "box chart series must contain exactly "
                        "min, q1, median, q3, and max statistics"
                    )

                values = [
                    series.statistics[key]
                    for key in (
                        "min",
                        "q1",
                        "median",
                        "q3",
                        "max",
                    )
                ]

                if any(value is None for value in values):
                    raise VisualizationValidationError(
                        "box chart statistics cannot contain None"
                    )

                numeric_values = [
                    float(value) for value in values
                ]

                if numeric_values != sorted(numeric_values):
                    raise VisualizationValidationError(
                        "box chart statistics must be ordered "
                        "min <= q1 <= median <= q3 <= max"
                    )
            return

        if self.chart_type is ChartType.PIE: 
            if len(data.series) != 1:
                raise VisualizationValidationError(
                    "pie charts require exactly one series"
                )

            pie_series = data.series[0]

            if any(
                value is None or value < 0
                for value in pie_series.y_values
            ):
                raise VisualizationValidationError(
                    "pie chart values must be non-negative numbers"
                )
            
        for series in data.series:
            if series.statistics or series.outliers:
                raise VisualizationValidationError(
                    "only box charts can use statistics or outliers"
                )

            if not series.x_values or not series.y_values:
                raise VisualizationValidationError(
                    "this chart type requires non-empty x_values and y_values"
                )

            if self.chart_type is ChartType.SCATTER:
                for value in series.x_values:
                    if value is None:
                        continue

                    if isinstance(value, str):
                        raise VisualizationValidationError(
                            "scatter chart x_values must be numeric or None"
                        )

                    _validate_finite_number(
                        value,
                        "scatter x_values item",
                    )


@dataclass(frozen=True, slots=True)
class VisualizationResult:
    """Structured visualization result."""

    visualization_id: str
    dataset_id: str
    dataset_name: str
    specification: ChartSpecification
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_string(
            self.visualization_id,
            "visualization_id",
        )
        _validate_non_empty_string(
            self.dataset_id,
            "dataset_id",
        )
        _validate_non_empty_string(
            self.dataset_name,
            "dataset_name",
        )

        if not isinstance(
            self.specification,
            ChartSpecification,
        ):
            raise VisualizationValidationError(
                "specification must be a ChartSpecification"
            )

        if self.specification.data.dataset_id != self.dataset_id:
            raise VisualizationValidationError(
                "result dataset_id must match specification dataset_id"
            )

        if self.specification.data.dataset_name != self.dataset_name:
            raise VisualizationValidationError(
                "result dataset_name must match specification dataset_name"
            )

        metadata = (
            {}
            if self.metadata is None
            else self.metadata
        )
        object.__setattr__(
            self,
            "metadata",
            _immutable_mapping(metadata, "metadata"),
        )
