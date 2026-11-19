"""Missing value cleaning operations."""

from __future__ import annotations

import warnings

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from lite_cog_agent.cleaning.exceptions import (
    CleaningOperationError,
    CleaningValidationError,
)


class MissingValueCleaner:
    """Clean missing values in pandas DataFrames.

    All operations return a new DataFrame and leave the input unchanged.
    """

    def drop(
        self,
        dataframe: pd.DataFrame,
        *,
        how: str | None = None,
        subset: Sequence[str] | None = None,
        thresh: int | None = None,
    ) -> pd.DataFrame:
        """Drop rows containing missing values.

        Args:
            dataframe: Source DataFrame.
            how: "any" or "all". Defaults to "any".
                Must be None when thresh is supplied.
            subset: Optional column names used to identify missing values.
            thresh: Minimum number of non-missing values required to keep
                a row. Must be used instead of how.

        Returns:
            A cleaned copy of the DataFrame.

        Raises:
            CleaningValidationError: If arguments are invalid.
            CleaningOperationError: If pandas fails to perform the operation.
        """
        self._validate_dataframe(dataframe)

        if how is not None and how not in ("any", "all"):
            raise CleaningValidationError(
                "how must be either 'any', 'all', or None"
            )

        if thresh is not None:
            if isinstance(thresh, bool) or not isinstance(thresh, int):
                raise CleaningValidationError(
                    "thresh must be a non-negative integer"
                )

            if thresh < 0:
                raise CleaningValidationError(
                    "thresh must be non-negative"
                )

            if how is not None:
                raise CleaningValidationError(
                    "how and thresh cannot be specified together"
                )

        selected_columns = self._validate_columns(
            dataframe,
            subset,
            field_name="subset",
        )

        result = dataframe.copy(deep=True)

        try:
            if thresh is not None:
                return result.dropna(
                    axis=0,
                    thresh=thresh,
                    subset=selected_columns,
                )

            return result.dropna(
                axis=0,
                how=how or "any",
                subset=selected_columns,
            )
        except Exception as exc:
            raise CleaningOperationError(
                f"Failed to drop missing values: {exc}",
                operation="missing.drop",
            ) from exc

    def fill(
        self,
        dataframe: pd.DataFrame,
        *,
        value: Any,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Fill missing values with a scalar or per-column mapping.

        Args:
            dataframe: Source DataFrame.
            value: Scalar fill value or mapping of column names to values.
            columns: Optional columns to fill. When omitted, all applicable
                columns are considered.

        Returns:
            A cleaned copy of the DataFrame.

        Raises:
            CleaningValidationError: If arguments are invalid.
            CleaningOperationError: If pandas fails to perform the operation.
        """
        self._validate_dataframe(dataframe)

        selected_columns = self._validate_columns(
            dataframe,
            columns,
            field_name="columns",
        )

        fill_values = self._validate_fill_value(value)

        if isinstance(fill_values, Mapping):
            unknown_columns = set(fill_values) - set(dataframe.columns)

            if unknown_columns:
                raise CleaningValidationError(
                    "fill value contains unknown columns: "
                    f"{sorted(unknown_columns)}"
                )

            if selected_columns is not None:
                unselected_columns = (
                    set(fill_values) - set(selected_columns)
                )

                if unselected_columns:
                    raise CleaningValidationError(
                        "fill value contains columns outside the selected "
                        f"columns: {sorted(unselected_columns)}"
                    )

            target_columns = (
                selected_columns
                if selected_columns is not None
                else tuple(fill_values.keys())
            )

            effective_values = {
                column: fill_values[column]
                for column in target_columns
                if column in fill_values
            }
        else:
            target_columns = (
                selected_columns
                if selected_columns is not None
                else tuple(dataframe.columns)
            )
            effective_values = fill_values

        result = dataframe.copy(deep=True)

        try:
            if isinstance(effective_values, Mapping):
                for column, fill_value in effective_values.items():
                    result[column] = self._fill_series(
                        result[column],
                        fill_value,
                    )

                return result

            for column in target_columns:
                result[column] = self._fill_series(
                    result[column],
                    effective_values,
                )

            return result

        except Exception as exc:
            raise CleaningOperationError(
                f"Failed to fill missing values: {exc}",
                operation="missing.fill",
            ) from exc

    @staticmethod
    def _validate_dataframe(dataframe: pd.DataFrame) -> None:
        """Validate DataFrame type and column names."""
        if not isinstance(dataframe, pd.DataFrame):
            raise CleaningValidationError(
                "dataframe must be a pandas DataFrame"
            )

        seen_columns: set[str] = set()

        for column in dataframe.columns:
            if not isinstance(column, str) or not column.strip():
                raise CleaningValidationError(
                    "column names must be non-empty strings"
                )

            if column in seen_columns:
                raise CleaningValidationError(
                    f"duplicate column name: {column!r}"
                )

            seen_columns.add(column)

    @classmethod
    def _validate_columns(
        cls,
        dataframe: pd.DataFrame,
        columns: Sequence[str] | None,
        *,
        field_name: str,
    ) -> tuple[str, ...] | None:
        """Validate an optional sequence of column names."""
        if columns is None:
            return None

        if isinstance(columns, (str, bytes)) or not isinstance(
            columns, Sequence
        ):
            raise CleaningValidationError(
                f"{field_name} must be a sequence of column names"
            )

        if not columns:
            raise CleaningValidationError(
                f"{field_name} must not be empty"
            )

        if not all(
            isinstance(column, str) and column.strip()
            for column in columns
        ):
            raise CleaningValidationError(
                f"{field_name} must contain non-empty string names"
            )

        if len(set(columns)) != len(columns):
            raise CleaningValidationError(
                f"{field_name} must not contain duplicate columns"
            )

        unknown_columns = set(columns) - set(dataframe.columns)

        if unknown_columns:
            raise CleaningValidationError(
                f"{field_name} contains unknown columns: "
                f"{sorted(unknown_columns)}"
            )

        return tuple(columns)

    @staticmethod
    def _fill_series(
        series: pd.Series,
        value: Any,
    ) -> pd.Series:
        """Fill a Series while handling incompatible pandas dtypes."""
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                result = series.fillna(value=value)

            incompatible_fill = any(
                "not supported as a fill value" in str(warning.message)
                for warning in caught_warnings
            )

            if incompatible_fill:
                return series.astype(object).fillna(value=value)

            return result

        except (TypeError, ValueError):
            return series.astype(object).fillna(value=value)
    @staticmethod
    def _validate_fill_value(value: Any) -> Any:
        """Validate a scalar or per-column fill mapping."""
        if isinstance(value, Mapping):
            if not value:
                raise CleaningValidationError(
                    "fill value mapping must not be empty"
                )

            for column, fill_value in value.items():
                if not isinstance(column, str) or not column.strip():
                    raise CleaningValidationError(
                        "fill value mapping keys must be non-empty strings"
                    )

                if fill_value is None:
                    raise CleaningValidationError(
                        "fill values must not be None"
                    )

                if not pd.api.types.is_scalar(fill_value):
                    raise CleaningValidationError(
                        "each mapped fill value must be a scalar"
                    )

                if bool(pd.isna(fill_value)):
                    raise CleaningValidationError(
                        "fill values must not themselves be missing"
                    )

            return dict(value)

        if value is None or not pd.api.types.is_scalar(value):
            raise CleaningValidationError(
                "value must be a non-missing scalar or a column mapping"
            )

        if bool(pd.isna(value)):
            raise CleaningValidationError(
                "fill value must not itself be missing"
            )

        return value
