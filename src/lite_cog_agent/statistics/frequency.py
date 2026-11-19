"""Frequency analysis for dataset columns."""

from __future__ import annotations

from typing import Any

import pandas as pd

from lite_cog_agent.statistics.contracts import FrequencyItem
from lite_cog_agent.statistics.exceptions import (
    FrequencyAnalysisError,
    StatisticalValidationError,
)


class FrequencyAnalyzer:
    """Analyze category frequencies in a DataFrame column."""

    def analyze(
        self,
        dataframe: pd.DataFrame,
        *,
        column: str,
        top_n: int | None = None,
        dropna: bool = True,
    ) -> tuple[FrequencyItem, ...]:
        """Calculate frequency statistics for one column.

        Args:
            dataframe: Source DataFrame. It is never modified.
            column: Column to analyze.
            top_n: Maximum number of categories to return.
            dropna: Whether to exclude missing values.

        Returns:
            Frequency items ordered by descending frequency.

        Raises:
            StatisticalValidationError: Invalid input parameters.
            FrequencyAnalysisError: Frequency calculation fails.
        """
        self._validate_inputs(
            dataframe,
            column=column,
            top_n=top_n,
            dropna=dropna,
        )

        series = dataframe[column]

        total_count = (
            int(series.notna().sum())
            if dropna
            else len(series)
        )

        if total_count == 0:
            return ()

        try:
            self._validate_hashable_values(series)

            counts = series.value_counts(
                dropna=dropna,
                sort=True,
            )

            if top_n is not None:
                counts = counts.head(top_n)

            items: list[FrequencyItem] = []
            cumulative_percentage = 0.0

            for value, count in counts.items():
                normalized_value = self._normalize_value(value)

                percentage = int(count) / total_count * 100
                cumulative_percentage += percentage

                # Avoid floating-point accumulation producing a value
                # slightly greater than 100.
                cumulative_percentage = min(
                    cumulative_percentage,
                    100.0,
                )

                items.append(
                    FrequencyItem(
                        value=normalized_value,
                        count=int(count),
                        percentage=percentage,
                        cumulative_percentage=cumulative_percentage,
                    )
                )

            return tuple(items)

        except (StatisticalValidationError,
                FrequencyAnalysisError,
        ):
            raise
        except Exception as exc:
            raise FrequencyAnalysisError(
                f"Failed to analyze frequencies for column '{column}': "
                f"{exc}"
            ) from exc

    @staticmethod
    def _validate_inputs(
        dataframe: pd.DataFrame,
        *,
        column: str,
        top_n: int | None,
        dropna: bool,
    ) -> None:
        """Validate DataFrame and analysis parameters."""
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

        if top_n is not None:
            if (
                isinstance(top_n, bool)
                or not isinstance(top_n, int)
                or top_n <= 0
            ):
                raise StatisticalValidationError(
                    "top_n must be a positive integer or None"
                )

        if not isinstance(dropna, bool):
            raise StatisticalValidationError(
                "dropna must be a boolean"
            )


    @staticmethod
    def _validate_hashable_values(series: pd.Series) -> None:
        """Reject non-missing category values that cannot be hashed."""
        for value in series.array:
            if (
                pd.api.types.is_scalar(value)
                and pd.isna(value)
            ):
                continue

            try:
                hash(value)
            except TypeError as exc:
                raise FrequencyAnalysisError(
                    "frequency categories must contain hashable values"
                ) from exc


    @staticmethod
    def _normalize_value(value: Any) -> Any:
        """Convert missing values and NumPy scalars to Python values."""
        if pd.api.types.is_scalar(value) and pd.isna(value):
            return None

        # Convert NumPy scalar values into native Python values.
        if hasattr(value, "item") and callable(value.item):
            try:
                return value.item()
            except (ValueError, TypeError):
                pass

        return value
