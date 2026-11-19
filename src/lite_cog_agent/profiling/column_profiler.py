"""Column-level data profiling."""
from __future__ import annotations

import math

from typing import Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

from lite_cog_agent.profiling.contracts import ColumnProfile
from lite_cog_agent.profiling.exceptions import (
    ColumnProfilingError,
    ProfilingValidationError,
)


class ColumnProfiler:
    """Profile a single pandas Series."""

    SAMPLE_LIMIT = 5

    def profile(
        self,
        series: pd.Series,
        name: str | None = None,
    ) -> ColumnProfile:
        """Generate a profile for a pandas Series.

        Args:
            series: Column data to analyze.
            name: Optional column name. Defaults to series.name.

        Returns:
            A validated ColumnProfile.

        Raises:
            ProfilingValidationError: If the input is invalid.
            ColumnProfilingError: If profiling fails.
        """
        if not isinstance(series, pd.Series):
            raise ProfilingValidationError(
                "series must be a pandas Series"
            )

        column_name = name if name is not None else series.name

        if not isinstance(column_name, str) or not column_name.strip():
            raise ProfilingValidationError(
                "column name must be a non-empty string"
            )

        try:
            return self._build_profile(series, column_name)
        except (ProfilingValidationError, ColumnProfilingError):
            raise
        except Exception as exc:
            raise ColumnProfilingError(
                f"Failed to profile column '{column_name}': {exc}"
            ) from exc

    def _build_profile(
        self,
        series: pd.Series,
        name: str,
    ) -> ColumnProfile:
        """Build a ColumnProfile from a Series."""
        total_count = len(series)
        null_count = int(series.isna().sum())
        non_null_count = total_count - null_count

        unique_count = int(series.nunique(dropna=True))

        null_percentage = self._percentage(
            null_count,
            total_count,
        )
        unique_percentage = self._percentage(
            unique_count,
            total_count,
        )

        inferred_type = self._infer_type(series)

        statistics = self._calculate_statistics(
            series,
            inferred_type,
        )

        sample_values = tuple(
            self._to_python_value(value)
            for value in series.dropna().head(self.SAMPLE_LIMIT).tolist()
        )

        return ColumnProfile(
            name=name,
            dtype=str(series.dtype),
            inferred_type=inferred_type,
            total_count=total_count,
            non_null_count=non_null_count,
            null_count=null_count,
            null_percentage=null_percentage,
            unique_count=unique_count,
            unique_percentage=unique_percentage,
            statistics=statistics,
            sample_values=sample_values,
        )

    @staticmethod
    def _percentage(
        numerator: int,
        denominator: int,
    ) -> float:
        """Calculate a percentage, returning zero for an empty denominator."""
        if denominator == 0:
            return 0.0

        return numerator / denominator * 100.0

    @staticmethod
    def _infer_type(series: pd.Series) -> str:
        """Infer a conservative logical type from the pandas dtype."""
        dtype = series.dtype

        # Boolean must be checked before numeric.
        if is_bool_dtype(dtype):
            return "boolean"

        if is_datetime64_any_dtype(dtype):
            return "datetime"

        # Complex numbers are not included in the initial numeric statistics.
        if is_complex_dtype(dtype):
            return "categorical"

        if is_numeric_dtype(dtype):
            return "numeric"

        return "categorical"

    def _calculate_statistics(
        self,
        series: pd.Series,
        inferred_type: str,
    ) -> dict[str, Any]:
        """Calculate statistics according to the inferred type."""
        if inferred_type == "numeric":
            return self._numeric_statistics(series)

        if inferred_type == "datetime":
            return self._datetime_statistics(series)

        return self._categorical_statistics(series)

    @staticmethod
    def _finite_or_none(value: float) -> float | None:
        """Return a finite float, or None when it cannot be represented."""
        if math.isfinite(value):
            return value
        return None

    @staticmethod
    def _interpolate_percentile(
        sorted_values: list[float],
        percentile: float,
    ) -> float:
        """Calculate a linearly interpolated percentile on scaled values."""
        if len(sorted_values) == 1:
            return sorted_values[0]

        position = (len(sorted_values) - 1) * percentile
        lower_index = math.floor(position)
        upper_index = math.ceil(position)

        lower_value = sorted_values[lower_index]
        upper_value = sorted_values[upper_index]

        if lower_index == upper_index:
            return lower_value

        fraction = position - lower_index

        return lower_value + (
            upper_value - lower_value
        ) * fraction

    def _numeric_statistics(
        self,
        series: pd.Series,
    ) -> dict[str, Any]:
        """Calculate numeric statistics using finite values only."""
        finite_values: list[float] = []

        for value in series.tolist():
            if pd.isna(value):
                continue

            numeric_value = float(value)

            if math.isfinite(numeric_value):
                finite_values.append(numeric_value)

        count = len(finite_values)

        if count == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "25%": None,
                "50%": None,
                "75%": None,
                "max": None,
            }

        # Scale values into [-1, 1] before calculations.
        # This avoids overflow in sums, differences and squared deviations.
        scale = max(abs(value) for value in finite_values)

        if scale == 0.0:
            scaled_values = [0.0] * count
        else:
            scaled_values = [
                value / scale
                for value in finite_values
            ]

        mean_scaled = math.fsum(scaled_values) / count

        mean = self._finite_or_none(mean_scaled * scale)

        if count > 1:
            squared_deviations = [
                (value - mean_scaled) ** 2
                for value in scaled_values
            ]

            variance_scaled = (
                math.fsum(squared_deviations) / (count - 1)
            )

            std_scaled = math.sqrt(variance_scaled)
            std = self._finite_or_none(std_scaled * scale)
        else:
            std = None

        sorted_scaled_values = sorted(scaled_values)

        def restore_scale(value: float) -> float | None:
            if scale == 0.0:
                return 0.0

            return self._finite_or_none(value * scale)

        percentile_25 = restore_scale(
            self._interpolate_percentile(
                sorted_scaled_values,
                0.25,
            )
        )

        percentile_50 = restore_scale(
            self._interpolate_percentile(
                sorted_scaled_values,
                0.50,
            )
        )

        percentile_75 = restore_scale(
            self._interpolate_percentile(
                sorted_scaled_values,
                0.75,
            )
        )

        return {
            "count": count,
            "mean": mean,
            "std": std,
            "min": min(finite_values),
            "25%": percentile_25,
            "50%": percentile_50,
            "75%": percentile_75,
            "max": max(finite_values),
        }

    def _categorical_statistics(
        self,
        series: pd.Series,
    ) -> dict[str, Any]:
        """Calculate frequency statistics for categorical columns."""
        values = series.dropna()

        if values.empty:
            return {
                "unique": 0,
                "top": None,
                "freq": 0,
            }

        value_counts = values.value_counts(dropna=True)

        return {
            "unique": int(value_counts.size),
            "top": self._to_python_value(value_counts.index[0]),
            "freq": int(value_counts.iloc[0]),
        }

    def _datetime_statistics(
        self,
        series: pd.Series,
    ) -> dict[str, Any]:
        """Calculate date range statistics."""
        values = series.dropna()

        if values.empty:
            return {
                "min": None,
                "max": None,
            }

        return {
            "min": self._to_python_value(values.min()),
            "max": self._to_python_value(values.max()),
        }

    @staticmethod
    def _to_python_value(value: Any) -> Any:
        """Convert NumPy scalar values into native Python values."""
        if hasattr(value, "item") and not isinstance(
            value,
            (pd.Timestamp, pd.Timedelta),
        ):
            try:
                return value.item()
            except (ValueError, TypeError):
                pass

        return value
