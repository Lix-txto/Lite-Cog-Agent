"""Descriptive statistical analysis for numeric dataset columns."""

from __future__ import annotations

import math

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_numeric_dtype,
)

from lite_cog_agent.profiling.column_profiler import ColumnProfiler
from lite_cog_agent.profiling.exceptions import ColumnProfilingError
from lite_cog_agent.statistics.contracts import DescriptiveStatistics
from lite_cog_agent.statistics.exceptions import (
    DescriptiveAnalysisError,
    StatisticalValidationError,
)


class DescriptiveAnalyzer:
    """Calculate descriptive statistics for numeric DataFrame columns.

    The analyzer is intentionally independent of ExecutionContext.
    Dataset metadata and result assembly belong to the service layer.
    """

    def __init__(
        self,
        column_profiler: ColumnProfiler | None = None,
    ) -> None:
        self._column_profiler = (
            column_profiler
            if column_profiler is not None
            else ColumnProfiler()
        )

    def analyze(
        self,
        dataframe: pd.DataFrame,
        *,
        columns: list[str] | None = None,
    ) -> tuple[DescriptiveStatistics, ...]:
        """Analyze selected numeric columns.

        Args:
            dataframe: Source DataFrame. It is never modified.
            columns: Optional list of columns to analyze. When omitted,
                all eligible numeric columns are selected.

        Returns:
            An immutable tuple of descriptive statistics.

        Raises:
            StatisticalValidationError: If the input or selected columns
                are invalid.
            DescriptiveAnalysisError: If statistical analysis fails.
        """
        self._validate_dataframe(dataframe)
        self._validate_unique_dataframe_columns(dataframe)

        selected_columns = self._select_columns(
            dataframe,
            columns,
        )

        results: list[DescriptiveStatistics] = []

        for column in selected_columns:
            series = dataframe[column]

            try:
                profile = self._column_profiler.profile(
                    series,
                    name=column,
                )
                results.append(
                    self._build_statistics(
                        column=column,
                        values=profile.statistics,
                    )
                )
            except StatisticalValidationError:
                raise
            except ColumnProfilingError as exc:
                raise DescriptiveAnalysisError(
                    f"Failed to analyze column '{column}': {exc}"
                ) from exc
            except Exception as exc:
                raise DescriptiveAnalysisError(
                    f"Unexpected error analyzing column '{column}': {exc}"
                ) from exc

        return tuple(results)

    @staticmethod
    def _validate_dataframe(dataframe: pd.DataFrame) -> None:
        """Validate the DataFrame input."""
        if not isinstance(dataframe, pd.DataFrame):
            raise StatisticalValidationError(
                "dataframe must be a pandas DataFrame"
            )

    @staticmethod
    def _validate_unique_dataframe_columns(
        dataframe: pd.DataFrame,
    ) -> None:
        """Reject duplicate column names."""
        if not dataframe.columns.is_unique:
            raise StatisticalValidationError(
                "dataframe cannot contain duplicate column names"
            )

    def _select_columns(
        self,
        dataframe: pd.DataFrame,
        columns: list[str] | None,
    ) -> tuple[str, ...]:
        """Select eligible numeric columns."""
        if columns is not None:
            if not isinstance(columns, list):
                raise StatisticalValidationError(
                    "columns must be a list of strings or None"
                )

            if any(
                not isinstance(column, str) or not column.strip()
                for column in columns
            ):
                raise StatisticalValidationError(
                    "all selected column names must be non-empty strings"
                )

            if len(columns) != len(set(columns)):
                raise StatisticalValidationError(
                    "columns cannot contain duplicates"
                )

            missing_columns = [
                column
                for column in columns
                if column not in dataframe.columns
            ]

            if missing_columns:
                raise StatisticalValidationError(
                    f"columns not found: {missing_columns}"
                )

            selected = columns
        else:
            selected = [
                column
                for column in dataframe.columns
                if self._is_supported_numeric_column(
                    dataframe[column]
                )
            ]

        for column in selected:
            if not self._is_supported_numeric_column(
                dataframe[column]
            ):
                raise StatisticalValidationError(
                    f"column '{column}' must have a supported numeric dtype"
                )

        return tuple(selected)

    @staticmethod
    def _is_supported_numeric_column(
        series: pd.Series,
    ) -> bool:
        """Return whether a Series has a supported numeric dtype."""
        dtype = series.dtype

        # Boolean is technically numeric in some libraries,
        # but it is not treated as a quantitative variable here.
        if is_bool_dtype(dtype):
            return False

        # Complex values do not have the ordinary ordering semantics
        # required by min, max, quantiles and range.
        if is_complex_dtype(dtype):
            return False

        return is_numeric_dtype(dtype)

    @staticmethod
    def _finite_or_none(value: float) -> float | None:
        """Return a finite float or None when the result overflows."""
        if math.isfinite(value):
            return value
        return None

    @classmethod
    def _calculate_variance(
        cls,
        standard_deviation: float | None,
    ) -> float | None:
        """Calculate sample variance while preventing infinity."""
        if standard_deviation is None:
            return None

        return cls._finite_or_none(
            standard_deviation * standard_deviation
        )

    @classmethod
    def _calculate_range(
        cls,
        minimum: float | None,
        maximum: float | None,
    ) -> float | None:
        """Calculate range while preventing subtraction overflow."""
        if minimum is None or maximum is None:
            return None

        return cls._finite_or_none(maximum - minimum)

    @classmethod
    def _build_statistics(
        cls,
        *,
        column: str,
        values: dict[str, object],
    ) -> DescriptiveStatistics:
        """Convert ColumnProfiler output into a statistics contract."""
        standard_deviation = values["std"]
        minimum = values["min"]
        maximum = values["max"]

        if standard_deviation is not None:
            standard_deviation = float(standard_deviation)

        if minimum is not None:
            minimum = float(minimum)

        if maximum is not None:
            maximum = float(maximum)

        return DescriptiveStatistics(
            column=column,
            count=int(values["count"]),
            mean=values["mean"],
            median=values["50%"],
            std=standard_deviation,
            variance=cls._calculate_variance(
                standard_deviation,
            ),
            min=minimum,
            q1=values["25%"],
            q3=values["75%"],
            max=maximum,
            range=cls._calculate_range(
                minimum,
                maximum,
            ),
        )
