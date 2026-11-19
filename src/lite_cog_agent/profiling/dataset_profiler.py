"""Dataset-level data profiling."""

from __future__ import annotations

import pandas as pd

from lite_cog_agent.execution.dataset_store import DatasetReference
from lite_cog_agent.profiling.column_profiler import ColumnProfiler
from lite_cog_agent.profiling.contracts import DatasetProfile
from lite_cog_agent.profiling.exceptions import (
    DatasetProfilingError,
    ProfilingValidationError,
)


class DatasetProfiler:
    """Generate a profiling summary for a pandas DataFrame."""

    def __init__(
        self,
        column_profiler: ColumnProfiler | None = None,
    ) -> None:
        """Initialize the profiler with an optional column profiler."""
        self.column_profiler = (
            column_profiler
            if column_profiler is not None
            else ColumnProfiler()
        )

        if not callable(getattr(self.column_profiler, "profile", None)):
            raise ProfilingValidationError(
                "column_profiler must provide a callable profile method"
            )

    def profile(
        self,
        dataframe: pd.DataFrame,
        reference: DatasetReference,
    ) -> DatasetProfile:
        """Generate a dataset-level profiling result.

        Args:
            dataframe: Dataset to analyze.
            reference: Metadata reference identifying the dataset.

        Returns:
            A validated DatasetProfile.

        Raises:
            ProfilingValidationError: If the input is invalid.
            DatasetProfilingError: If dataset profiling fails.
        """
        self._validate_input(dataframe, reference)

        try:
            columns = self._profile_columns(dataframe)
            row_count, column_count = dataframe.shape
            duplicate_row_count = self._calculate_duplicates(dataframe)

            return DatasetProfile(
                dataset_id=reference.dataset_id,
                dataset_name=reference.name,
                row_count=row_count,
                column_count=column_count,
                memory_usage_bytes=self._calculate_memory_usage(dataframe),
                columns=columns,
                duplicate_row_count=duplicate_row_count,
                duplicate_percentage=self._percentage(
                    duplicate_row_count,
                    row_count,
                ),
            )
        except (ProfilingValidationError, DatasetProfilingError):
            raise
        except Exception as exc:
            raise DatasetProfilingError(
                f"Failed to profile dataset '{reference.name}': {exc}"
            ) from exc

    @staticmethod
    def _validate_input(
        dataframe: pd.DataFrame,
        reference: DatasetReference,
    ) -> None:
        """Validate the input objects."""
        if not isinstance(dataframe, pd.DataFrame):
            raise ProfilingValidationError(
                "dataframe must be a pandas DataFrame"
            )

        if not isinstance(reference, DatasetReference):
            raise ProfilingValidationError(
                "reference must be a DatasetReference"
            )

    def _profile_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple:
        """Profile columns in their original positional order."""
        profiles = []

        for index in range(dataframe.shape[1]):
            series = dataframe.iloc[:, index]
            raw_name = dataframe.columns[index]
            column_name = str(raw_name).strip()

            if not column_name:
                column_name = f"column_{index}"

            try:
                column_profile = self.column_profiler.profile(
                    series,
                    name=column_name,
                )
            except Exception as exc:
                raise DatasetProfilingError(
                    f"Failed to profile column at position {index} "
                    f"({column_name!r}): {exc}"
                ) from exc

            profiles.append(column_profile)

        return tuple(profiles)

    @staticmethod
    def _calculate_memory_usage(
        dataframe: pd.DataFrame,
    ) -> int:
        """Calculate the DataFrame memory usage in bytes."""
        return int(
            dataframe.memory_usage(
                index=True,
                deep=True,
            ).sum()
        )

    @staticmethod
    def _calculate_duplicates(
        dataframe: pd.DataFrame,
    ) -> int:
        """Count duplicate rows, excluding the first occurrence.

        A DataFrame without columns has no comparable row values,
        so it is treated as having no duplicate rows.
        """
        if dataframe.empty or dataframe.shape[1] == 0:
            return 0

        return int(dataframe.duplicated().sum())

    @staticmethod
    def _percentage(
        numerator: int,
        denominator: int,
    ) -> float:
        """Calculate a percentage, returning zero for an empty denominator."""
        if denominator == 0:
            return 0.0

        return numerator / denominator * 100.0
