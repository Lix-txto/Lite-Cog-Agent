"""Correlation analysis for numeric dataset columns."""

from __future__ import annotations

import math

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_numeric_dtype,
)

from lite_cog_agent.statistics.exceptions import (
    CorrelationAnalysisError,
    StatisticalValidationError,
)


class CorrelationAnalyzer:
    """Calculate pairwise correlations for numeric DataFrame columns.

    The analyzer is intentionally independent of ExecutionContext.
    Dataset metadata and result assembly belong to the service layer.

    Missing and non-finite values are excluded pairwise. Each matrix
    position therefore uses its own set of valid observations.
    """

    SUPPORTED_METHODS = ("pearson", "spearman")

    def analyze(
        self,
        dataframe: pd.DataFrame,
        *,
        columns: list[str] | None = None,
        method: str = "pearson",
        min_periods: int = 2,
    ) -> tuple[
        tuple[tuple[float | None, ...], ...],
        tuple[tuple[int, ...], ...],
    ]:
        """Calculate a correlation matrix and pairwise observation counts.

        Args:
            dataframe: Source DataFrame. It is never modified.
            columns: Optional list of numeric columns to analyze.
                When omitted, all supported numeric columns are selected.
            method: Correlation method: "pearson" or "spearman".
            min_periods: Minimum number of valid observations required
                to calculate a correlation coefficient.

        Returns:
            A pair containing:
                - An immutable correlation coefficient matrix.
                - An immutable pairwise observation count matrix.

            Uncomputable correlation coefficients are represented by None.

        Raises:
            StatisticalValidationError: If the input or selected columns
                are invalid.
            CorrelationAnalysisError: If correlation analysis fails.
        """
        self._validate_dataframe(dataframe)
        self._validate_unique_dataframe_columns(dataframe)
        self._validate_method(method)
        self._validate_min_periods(min_periods)

        selected_columns = self._select_columns(
            dataframe,
            columns,
        )

        if not selected_columns:
            return (), ()

        try:
            numeric_dataframe = self._build_numeric_dataframe(
                dataframe,
                selected_columns,
            )

            finite_mask = numeric_dataframe.notna()

            pair_counts_dataframe = (
                finite_mask.astype("int64").T.dot(
                    finite_mask.astype("int64")
                )
            )

            correlation_dataframe = numeric_dataframe.corr(
                method=method,
                min_periods=min_periods,
            )

            correlation_matrix = self._build_correlation_matrix(
                correlation_dataframe,
            )

            pair_counts = self._build_pair_counts(
                pair_counts_dataframe,
            )

            return correlation_matrix, pair_counts

        except StatisticalValidationError:
            raise
        except CorrelationAnalysisError:
            raise
        except Exception as exc:
            raise CorrelationAnalysisError(
                f"Failed to calculate correlation matrix: {exc}"
            ) from exc

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
        """Reject duplicate DataFrame column names."""
        if not dataframe.columns.is_unique:
            raise StatisticalValidationError(
                "dataframe cannot contain duplicate column names"
            )

    @classmethod
    def _validate_method(cls, method: str) -> None:
        """Validate the correlation method."""
        if (
            not isinstance(method, str)
            or method not in cls.SUPPORTED_METHODS
        ):
            raise StatisticalValidationError(
                "method must be 'pearson' or 'spearman'"
            )

    @staticmethod
    def _validate_min_periods(min_periods: int) -> None:
        """Validate the minimum number of observations."""
        if (
            isinstance(min_periods, bool)
            or not isinstance(min_periods, int)
            or min_periods < 1
        ):
            raise StatisticalValidationError(
                "min_periods must be a positive integer"
            )

    def _select_columns(
        self,
        dataframe: pd.DataFrame,
        columns: list[str] | None,
    ) -> tuple[str, ...]:
        """Select supported numeric columns."""
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

            selected_columns = columns

        else:
            selected_columns = [
                column
                for column in dataframe.columns
                if self._is_supported_numeric_column(
                    dataframe[column]
                )
            ]

        for column in selected_columns:
            if not self._is_supported_numeric_column(
                dataframe[column]
            ):
                raise StatisticalValidationError(
                    f"column '{column}' must have a supported numeric dtype"
                )

        return tuple(selected_columns)

    @staticmethod
    def _is_supported_numeric_column(
        series: pd.Series,
    ) -> bool:
        """Return whether a Series has a supported numeric dtype."""
        dtype = series.dtype

        # Boolean values are not treated as quantitative variables.
        if is_bool_dtype(dtype):
            return False

        # Complex values are not supported by ordinary correlation analysis.
        if is_complex_dtype(dtype):
            return False

        return is_numeric_dtype(dtype)

    @staticmethod
    def _build_numeric_dataframe(
        dataframe: pd.DataFrame,
        columns: tuple[str, ...],
    ) -> pd.DataFrame:
        """Build a float DataFrame and normalize non-finite values.

        Using to_numpy with na_value also supports Pandas nullable
        numeric dtypes such as Int64 and Float64.
        """
        values = dataframe.loc[:, list(columns)].to_numpy(
            dtype="float64",
            na_value=float("nan"),
        )

        numeric_dataframe = pd.DataFrame(
            values,
            index=dataframe.index,
            columns=list(columns),
        )

        return numeric_dataframe.replace(
            [float("inf"), float("-inf")],
            float("nan"),
        )

    @staticmethod
    def _build_correlation_matrix(
        correlation_dataframe: pd.DataFrame,
    ) -> tuple[tuple[float | None, ...], ...]:
        """Convert Pandas correlation output to an immutable matrix."""
        matrix: list[tuple[float | None, ...]] = []

        for row_index in range(len(correlation_dataframe.columns)):
            row: list[float | None] = []

            for column_index in range(len(correlation_dataframe.columns)):
                value = correlation_dataframe.iat[
                    row_index,
                    column_index,
                ]

                if pd.isna(value):
                    row.append(None)
                    continue

                numeric_value = float(value)

                if not math.isfinite(numeric_value):
                    row.append(None)
                else:
                    row.append(numeric_value)

            matrix.append(tuple(row))

        return tuple(matrix)

    @staticmethod
    def _build_pair_counts(
        pair_counts_dataframe: pd.DataFrame,
    ) -> tuple[tuple[int, ...], ...]:
        """Convert pairwise observation counts to an immutable matrix."""
        return tuple(
            tuple(
                int(pair_counts_dataframe.iat[row_index, column_index])
                for column_index in range(len(pair_counts_dataframe.columns))
            )
            for row_index in range(len(pair_counts_dataframe.index))
        )
