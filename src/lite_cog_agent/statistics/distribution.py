"""Distribution analysis for numeric dataset columns."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_numeric_dtype,
)

from lite_cog_agent.statistics.contracts import HistogramBin
from lite_cog_agent.statistics.exceptions import (
    DistributionAnalysisError,
    StatisticalValidationError,
)


@dataclass(frozen=True, slots=True)
class DistributionAnalysisData:
    """Immutable distribution analysis data without dataset metadata."""

    count: int
    mean: float | None
    median: float | None
    std: float | None
    skewness: float | None
    kurtosis: float | None
    quantiles: Mapping[str, float | None]
    histogram: tuple[HistogramBin, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count < 0
        ):
            raise StatisticalValidationError(
                "count must be a non-negative integer"
            )

        numeric_fields = (
            self.mean,
            self.median,
            self.std,
            self.skewness,
            self.kurtosis,
        )

        for value in numeric_fields:
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise StatisticalValidationError(
                    "distribution statistics must be finite numbers or None"
                )

        if self.std is not None and self.std < 0:
            raise StatisticalValidationError(
                "std cannot be negative"
            )

        if not isinstance(self.quantiles, Mapping):
            raise StatisticalValidationError(
                "quantiles must be a mapping"
            )

        copied_quantiles = dict(self.quantiles)

        for name, value in copied_quantiles.items():
            if not isinstance(name, str) or not name.strip():
                raise StatisticalValidationError(
                    "quantile names must be non-empty strings"
                )

            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise StatisticalValidationError(
                    f"quantile '{name}' must be finite or None"
                )

        object.__setattr__(
            self,
            "quantiles",
            MappingProxyType(copied_quantiles),
        )

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

        if sum(item.count for item in self.histogram) != self.count:
            raise StatisticalValidationError(
                "histogram counts must equal count"
            )

        if self.histogram and not math.isclose(
            math.fsum(item.percentage for item in self.histogram),
            100.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise StatisticalValidationError(
                "histogram percentages must sum to 100"
            )

        for item in self.histogram:
            expected_percentage = item.count / self.count * 100

            if not math.isclose(
                item.percentage,
                expected_percentage,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise StatisticalValidationError(
                    "histogram percentage must match bin count"
                )

        if self.count == 0:
            if any(value is not None for value in numeric_fields):
                raise StatisticalValidationError(
                    "statistics must be None when count is zero"
                )

            if any(value is not None for value in copied_quantiles.values()):
                raise StatisticalValidationError(
                    "quantiles must be None when count is zero"
                )

            if self.histogram:
                raise StatisticalValidationError(
                    "histogram must be empty when count is zero"
                )



class DistributionAnalyzer:
    """Analyze the distribution of one numeric DataFrame column."""

    def analyze(
        self,
        dataframe: pd.DataFrame,
        *,
        column: str,
        bins: int = 10,
        quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
    ) -> DistributionAnalysisData:
        """Calculate descriptive distribution statistics and a histogram.

        Args:
            dataframe: Source DataFrame. It is never modified.
            column: Numeric column to analyze.
            bins: Number of equal-width histogram bins.
            quantiles: Unique quantile probabilities in the [0, 1] range.

        Returns:
            Immutable distribution analysis data.

        Raises:
            StatisticalValidationError: Invalid input parameters.
            DistributionAnalysisError: Distribution calculation fails.
        """
        self._validate_inputs(
            dataframe,
            column=column,
            bins=bins,
            quantiles=quantiles,
        )

        try:
            values = self._get_finite_values(dataframe[column])

            if values.size == 0:
                return self._build_empty_result(quantiles)

            scale = float(np.max(np.abs(values)))

            # Scaling keeps intermediate calculations within a
            # predictable numeric range and reduces overflow risk.
            if scale == 0.0:
                normalized = np.zeros(values.shape, dtype=np.float64)
            else:
                normalized = values / scale

            normalized_series = pd.Series(
                normalized,
                dtype="float64",
            )

            mean = self._scale_result(
                math.fsum(float(value) for value in normalized) / len(normalized),
                scale,
            )

            quantile_values = np.quantile(
                normalized,
                quantiles,
                method="linear",
            )

            normalized_std = (
                float(np.std(normalized, ddof=1))
                if len(normalized) >= 2
                else None
            )

            std = (
                self._scale_result(normalized_std, scale)
                if normalized_std is not None
                else None
            )

            skewness = self._finite_or_none(
                float(normalized_series.skew())
            )

            kurtosis = self._finite_or_none(
                float(normalized_series.kurt())
            )

            normalized_histogram, normalized_edges = np.histogram(
                normalized,
                bins=bins,
            )

            histogram = self._build_histogram(
                counts=normalized_histogram,
                edges=normalized_edges,
                scale=scale,
                total_count=len(normalized),
            )

            normalized_quantiles = {
                self._quantile_name(probability): self._scale_result(
                    float(value),
                    scale,
                )
                for probability, value in zip(
                    quantiles,
                    quantile_values,
                )
            }

            median = self._scale_result(
            float(np.quantile(normalized, 0.5, method="linear")),
            scale,
            )

            return DistributionAnalysisData(
                count=int(len(normalized)),
                mean=mean,
                median=median,
                std=std,
                skewness=skewness,
                kurtosis=kurtosis,
                quantiles=normalized_quantiles,
                histogram=histogram,
            )

        except StatisticalValidationError:
            raise
        except DistributionAnalysisError:
            raise
        except Exception as exc:
            raise DistributionAnalysisError(
                f"Failed to analyze distribution for column '{column}': "
                f"{exc}"
            ) from exc

    @staticmethod
    def _validate_inputs(
        dataframe: pd.DataFrame,
        *,
        column: str,
        bins: int,
        quantiles: tuple[float, ...],
    ) -> None:
        """Validate DataFrame, selected column, and analysis options."""
        if not isinstance(dataframe, pd.DataFrame):
            raise StatisticalValidationError(
                "dataframe must be a pandas DataFrame"
            )

        if not dataframe.columns.is_unique:
            raise StatisticalValidationError(
                "dataframe cannot contain duplicate columns"
            )

        if not isinstance(column, str) or not column.strip():
            raise StatisticalValidationError(
                "column must be a non-empty string"
            )

        if column not in dataframe.columns:
            raise StatisticalValidationError(
                f"column '{column}' does not exist in dataframe"
            )

        if (
            isinstance(bins, bool)
            or not isinstance(bins, int)
            or bins <= 0
        ):
            raise StatisticalValidationError(
                "bins must be a positive integer"
            )

        if not isinstance(quantiles, tuple):
            raise StatisticalValidationError(
                "quantiles must be a tuple of probabilities"
            )

        for probability in quantiles:
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise StatisticalValidationError(
                    "quantiles must contain finite probabilities between 0 and 1"
                )

        if len(quantiles) != len(set(quantiles)):
            raise StatisticalValidationError(
                "quantiles cannot contain duplicates"
            )

        series = dataframe[column]

        if (
            is_bool_dtype(series.dtype)
            or is_complex_dtype(series.dtype)
            or not is_numeric_dtype(series.dtype)
        ):
            raise StatisticalValidationError(
                f"column '{column}' must have a supported numeric dtype"
            )

    @staticmethod
    def _get_finite_values(series: pd.Series) -> np.ndarray:
        """Convert a numeric Series into finite float64 values."""
        try:
            values = series.to_numpy(
                dtype="float64",
                na_value=np.nan,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise StatisticalValidationError(
                "column values cannot be represented as float64"
            ) from exc

        return values[np.isfinite(values)]

    @staticmethod
    def _finite_or_none(value: float) -> float | None:
        """Return a finite float or None for non-finite values."""
        if math.isfinite(value):
            return float(value)
        return None

    @classmethod
    def _scale_result(
        cls,
        normalized_value: float,
        scale: float,
    ) -> float | None:
        """Restore the original scale without returning infinity."""
        return cls._finite_or_none(normalized_value * scale)

    @staticmethod
    def _quantile_name(probability: float) -> str:
        """Create a stable string key for a quantile probability."""
        return repr(float(probability))

    @classmethod
    def _build_histogram(
        cls,
        *,
        counts: np.ndarray,
        edges: np.ndarray,
        scale: float,
        total_count: int,
    ) -> tuple[HistogramBin, ...]:
        """Build immutable histogram bins on the original data scale."""
        result: list[HistogramBin] = []

        for index, count in enumerate(counts):
            lower_bound = cls._scale_result(
                float(edges[index]),
                scale,
            )
            upper_bound = cls._scale_result(
                float(edges[index + 1]),
                scale,
            )

            # Histogram edges are derived from finite normalized values.
            # A failed conversion indicates an unrepresentable boundary.
            if lower_bound is None or upper_bound is None:
                raise DistributionAnalysisError(
                    "histogram boundary exceeds the finite float64 range"
                )

            normalized_count = int(count)

            result.append(
                HistogramBin(
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    count=normalized_count,
                    percentage=normalized_count / total_count * 100,
                )
            )

        return tuple(result)

    @staticmethod
    def _build_empty_result(
        quantiles: tuple[float, ...],
    ) -> DistributionAnalysisData:
        """Build a consistent result for a column without finite values."""
        return DistributionAnalysisData(
            count=0,
            mean=None,
            median=None,
            std=None,
            skewness=None,
            kurtosis=None,
            quantiles={
                DistributionAnalyzer._quantile_name(probability): None
                for probability in quantiles
            },
            histogram=(),
        )
