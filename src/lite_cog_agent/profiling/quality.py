"""Dataset-level data quality analysis."""

from __future__ import annotations

import pandas as pd

from lite_cog_agent.profiling.contracts import DataQualityReport
from lite_cog_agent.profiling.exceptions import (
    DataQualityAnalysisError,
    ProfilingValidationError,
)


class DataQualityAnalyzer:
    """Analyze missing values, completeness, and duplicate rows."""

    def analyze(
        self,
        dataframe: pd.DataFrame,
    ) -> DataQualityReport:
        """Generate a data quality report for a DataFrame.

        Args:
            dataframe: Dataset to analyze.

        Returns:
            A validated DataQualityReport.

        Raises:
            ProfilingValidationError: If the input is invalid.
            DataQualityAnalysisError: If quality analysis fails.
        """
        self._validate_input(dataframe)

        try:
            row_count, column_count = dataframe.shape
            total_cells = row_count * column_count
            missing_cells = self._calculate_missing_cells(dataframe)
            duplicate_row_count = self._calculate_duplicate_rows(dataframe)

            return DataQualityReport(
                total_cells=total_cells,
                missing_cells=missing_cells,
                completeness=self._percentage(
                    total_cells - missing_cells,
                    total_cells,
                ),
                missing_rate=self._percentage(
                    missing_cells,
                    total_cells,
                ),
                duplicate_row_count=duplicate_row_count,
                duplicate_rate=self._percentage(
                    duplicate_row_count,
                    row_count,
                ),
                column_completeness=self._calculate_column_completeness(
                    dataframe
                ),
            )
        except ProfilingValidationError:
            raise
        except Exception as exc:
            raise DataQualityAnalysisError(
                f"Failed to analyze data quality: {exc}"
            ) from exc

    @staticmethod
    def _validate_input(dataframe: pd.DataFrame) -> None:
        """Validate the DataFrame and its column names."""
        if not isinstance(dataframe, pd.DataFrame):
            raise ProfilingValidationError(
                "dataframe must be a pandas DataFrame"
            )

        seen_names: set[str] = set()

        for column_name in dataframe.columns:
            if (
                not isinstance(column_name, str)
                or not column_name.strip()
            ):
                raise ProfilingValidationError(
                    "column names must be non-empty strings"
                )

            if column_name in seen_names:
                raise ProfilingValidationError(
                    f"duplicate column name: {column_name!r}"
                )

            seen_names.add(column_name)

    @staticmethod
    def _calculate_missing_cells(
        dataframe: pd.DataFrame,
    ) -> int:
        """Count all missing cells in the DataFrame."""
        return int(dataframe.isna().sum().sum())

    @staticmethod
    def _calculate_duplicate_rows(
        dataframe: pd.DataFrame,
    ) -> int:
        """Count duplicate rows, excluding the first occurrence.

        A DataFrame without rows or columns has no comparable
        row values and is treated as having no duplicate rows.
        """
        if dataframe.empty or dataframe.shape[1] == 0:
            return 0

        return int(dataframe.duplicated().sum())

    @classmethod
    def _calculate_column_completeness(
        cls,
        dataframe: pd.DataFrame,
    ) -> dict[str, float]:
        """Calculate the non-missing percentage for each column."""
        row_count = len(dataframe)

        return {
            column_name: cls._percentage(
                int(dataframe[column_name].notna().sum()),
                row_count,
            )
            for column_name in dataframe.columns
        }

    @staticmethod
    def _percentage(
        numerator: int,
        denominator: int,
    ) -> float:
        """Calculate a percentage, returning zero for an empty denominator."""
        if denominator == 0:
            return 0.0

        return numerator / denominator * 100.0
