from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date, datetime
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

from lite_cog_agent.statistics.exceptions import (
    GroupedAnalysisError,
    StatisticalValidationError,
)


class GroupedAnalyzer:
    """Perform controlled grouped statistical analysis."""

    SUPPORTED_AGGREGATIONS = (
        "count",
        "sum",
        "mean",
        "median",
        "min",
        "max",
        "std",
    )

    _NUMERIC_AGGREGATIONS = {
        "sum",
        "mean",
        "median",
        "std",
    }

    _ORDERED_AGGREGATIONS = {
        "min",
        "max",
    }

    def analyze(
        self,
        dataframe: pd.DataFrame,
        *,
        group_by: list[str],
        aggregations: Mapping[str, list[str] | tuple[str, ...]],
        dropna: bool = True,
        sort: bool = True,
    ) -> tuple[Mapping[str, Any], ...]:
        """Group a DataFrame and calculate configured aggregations."""
        self._validate_dataframe(dataframe)
        self._validate_options(dropna=dropna, sort=sort)

        normalized_group_by = self._validate_group_by(
            dataframe,
            group_by,
        )
        normalized_aggregations = self._validate_aggregations(
            dataframe,
            aggregations,
        )

        self._validate_output_columns(
            normalized_group_by,
            normalized_aggregations,
        )

        if dataframe.empty:
            return ()

        try:
            grouped = dataframe.groupby(
                list(normalized_group_by),
                dropna=dropna,
                sort=sort,
                observed=True,
            )

            records: list[Mapping[str, Any]] = []

            for group_key, group_dataframe in grouped:
                record = self._build_record(
                    group_key,
                    group_dataframe,
                    normalized_group_by,
                    normalized_aggregations,
                )
                records.append(MappingProxyType(record))

            return tuple(records)

        except StatisticalValidationError:
            raise
        except Exception as exc:
            raise GroupedAnalysisError(
                f"grouped analysis failed: {exc}"
            ) from exc

    def _validate_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise StatisticalValidationError(
                "dataframe must be a pandas DataFrame"
            )

        if not dataframe.columns.is_unique:
            raise StatisticalValidationError(
                "dataframe must not contain duplicate column names"
            )

    def _validate_options(
        self,
        *,
        dropna: bool,
        sort: bool,
    ) -> None:
        if not isinstance(dropna, bool):
            raise StatisticalValidationError(
                "dropna must be a boolean"
            )

        if not isinstance(sort, bool):
            raise StatisticalValidationError(
                "sort must be a boolean"
            )

    def _validate_group_by(
        self,
        dataframe: pd.DataFrame,
        group_by: list[str],
    ) -> tuple[str, ...]:
        if not isinstance(group_by, list) or not group_by:
            raise StatisticalValidationError(
                "group_by must be a non-empty list"
            )

        if any(
            not isinstance(column, str) or not column.strip()
            for column in group_by
        ):
            raise StatisticalValidationError(
                "group_by must contain non-empty strings"
            )

        if len(group_by) != len(set(group_by)):
            raise StatisticalValidationError(
                "group_by must not contain duplicate columns"
            )

        missing_columns = [
            column
            for column in group_by
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise StatisticalValidationError(
                f"grouping columns do not exist: {missing_columns}"
            )

        return tuple(group_by)

    def _validate_aggregations(
        self,
        dataframe: pd.DataFrame,
        aggregations: Mapping[str, list[str] | tuple[str, ...]],
    ) -> dict[str, tuple[str, ...]]:
        if not isinstance(aggregations, Mapping) or not aggregations:
            raise StatisticalValidationError(
                "aggregations must be a non-empty mapping"
            )

        normalized: dict[str, tuple[str, ...]] = {}

        for column, operations in aggregations.items():
            if not isinstance(column, str) or not column.strip():
                raise StatisticalValidationError(
                    "aggregation column names must be non-empty strings"
                )

            if column not in dataframe.columns:
                raise StatisticalValidationError(
                    f"aggregation column does not exist: {column}"
                )

            if not isinstance(operations, (list, tuple)) or not operations:
                raise StatisticalValidationError(
                    f"aggregations for '{column}' must be a non-empty list"
                )

            if any(
                not isinstance(operation, str)
                or operation not in self.SUPPORTED_AGGREGATIONS
                for operation in operations
            ):
                raise StatisticalValidationError(
                    f"unsupported aggregation for '{column}': "
                    f"{operations}"
                )

            if len(operations) != len(set(operations)):
                raise StatisticalValidationError(
                    f"duplicate aggregation operations for '{column}'"
                )

            for operation in operations:
                self._validate_aggregation_dtype(
                    dataframe[column],
                    column=column,
                    operation=operation,
                )

            normalized[column] = tuple(operations)

        return normalized

    def _validate_aggregation_dtype(
        self,
        series: pd.Series,
        *,
        column: str,
        operation: str,
    ) -> None:
        if operation == "count":
            return

        dtype = series.dtype

        is_boolean = is_bool_dtype(dtype)
        is_numeric = (
            is_numeric_dtype(dtype)
            and not is_boolean
            and not is_complex_dtype(dtype)
        )
        is_datetime = is_datetime64_any_dtype(dtype)

        if operation in self._NUMERIC_AGGREGATIONS:
            if not is_numeric:
                raise StatisticalValidationError(
                    f"aggregation '{operation}' requires a numeric "
                    f"column: '{column}'"
                )
            return

        if operation in self._ORDERED_AGGREGATIONS:
            if is_numeric or is_datetime:
                return

            if self._is_string_series(series):
                return

            raise StatisticalValidationError(
                f"aggregation '{operation}' requires a numeric, "
                f"string, or datetime column: '{column}'"
            )

    @staticmethod
    def _is_string_series(series: pd.Series) -> bool:
        non_missing = series.dropna()

        return all(
            isinstance(value, str)
            for value in non_missing.tolist()
        )

    def _validate_output_columns(
        self,
        group_by: tuple[str, ...],
        aggregations: dict[str, tuple[str, ...]],
    ) -> None:
        output_columns = list(group_by)

        for column, operations in aggregations.items():
            output_columns.extend(
                f"{column}_{operation}"
                for operation in operations
            )

        if len(output_columns) != len(set(output_columns)):
            raise StatisticalValidationError(
                "grouping and aggregation output columns must be unique"
            )

    def _build_record(
        self,
        group_key: Any,
        group_dataframe: pd.DataFrame,
        group_by: tuple[str, ...],
        aggregations: dict[str, tuple[str, ...]],
    ) -> dict[str, Any]:
        if len(group_by) == 1:
            if isinstance(group_key, tuple):
                group_values = group_key
            else:
                group_values = (group_key,)
        else:
            group_values = tuple(group_key)

        record: dict[str, Any] = {
            column: self._normalize_value(value)
            for column, value in zip(
                group_by,
                group_values,
                strict=True,
            )
        }

        for column, operations in aggregations.items():
            series = group_dataframe[column]

            for operation in operations:
                value = self._aggregate(
                    series,
                    operation,
                )
                record[f"{column}_{operation}"] = (
                    self._normalize_value(value)
                )

        return record

    def _aggregate(
        self,
        series: pd.Series,
        operation: str,
    ) -> Any:
        if operation == "count":
            return int(series.count())

        if operation in self._ORDERED_AGGREGATIONS:
            values = series.dropna()

            if values.empty:
                return None

            if is_numeric_dtype(series.dtype):
                values = self._finite_numeric_values(series)

                if values.empty:
                    return None

            if self._is_string_series(series):
                string_values = values.tolist()

                if operation == "min":
                    return min(string_values)

                return max(string_values)

            if operation == "min":
                return values.min()

            return values.max()

        values = self._finite_numeric_values(series)

        if values.empty:
            return None

        numbers = values.tolist()

        try:
            if operation == "sum":
                return math.fsum(numbers)

            if operation == "mean":
                return math.fsum(numbers) / len(numbers)

            if operation == "median":
                return float(np.median(numbers))

            if operation == "std":
                if len(numbers) < 2:
                    return None

                return float(np.std(numbers, ddof=1))

        except (ArithmeticError, OverflowError, ValueError):
            return None

        raise StatisticalValidationError(
            f"unsupported aggregation: {operation}"
        )

    @staticmethod
    def _finite_numeric_values(
        series: pd.Series,
    ) -> pd.Series:
        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        numeric = numeric.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        return numeric.dropna()

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if value is None or value is pd.NA or value is pd.NaT:
            return None

        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, float):
            if not math.isfinite(value):
                return None

        if isinstance(value, (pd.Timestamp, datetime, date)):
            return value.isoformat()

        if isinstance(value, (pd.Timedelta,)):
            return value.isoformat()

        return value
