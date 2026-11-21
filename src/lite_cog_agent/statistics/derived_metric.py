"""Controlled derived metric analysis."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_numeric_dtype,
)

from lite_cog_agent.statistics.exceptions import (
    DerivedMetricAnalysisError,
    StatisticalValidationError,
)


class DerivedMetricAnalyzer:
    """Calculate controlled derived metrics from DataFrame columns."""

    SUPPORTED_OPERATORS = ("multiply",)
    SUPPORTED_AGGREGATIONS = ("sum",)

    _INTERNAL_COLUMN = "__lite_cog_derived_metric_value__"

    def analyze(
        self,
        dataframe: pd.DataFrame,
        *,
        left_column: str,
        right_column: str,
        operator: str = "multiply",
        aggregation: str = "sum",
        group_by: list[str] | tuple[str, ...] = (),
        dropna: bool = True,
    ) -> tuple[Mapping[str, Any], ...]:
        """Calculate a derived metric and optionally aggregate by groups.

        Rows with missing, non-numeric, or non-finite operands are excluded.
        Rows whose multiplication result is non-finite are also excluded.

        When grouping is enabled, rows with missing group keys are excluded
        if ``dropna`` is True.
        """
        self._validate_dataframe(dataframe)
        self._validate_options(
            left_column=left_column,
            right_column=right_column,
            operator=operator,
            aggregation=aggregation,
            group_by=group_by,
            dropna=dropna,
        )

        normalized_group_by = tuple(group_by)

        self._validate_columns(
            dataframe,
            left_column=left_column,
            right_column=right_column,
            group_by=normalized_group_by,
        )

        left_values = self._to_finite_numeric_array(
            dataframe[left_column],
            column=left_column,
        )
        right_values = self._to_finite_numeric_array(
            dataframe[right_column],
            column=right_column,
        )

        with np.errstate(over="ignore", invalid="ignore"):
            derived_values = np.multiply(left_values, right_values)

        valid_mask = (
            np.isfinite(left_values)
            & np.isfinite(right_values)
            & np.isfinite(derived_values)
        )

        if not normalized_group_by:
            values = derived_values[valid_mask]

            return (
                MappingProxyType(
                    {
                        "valid_row_count": len(values),
                        "result": self._sum(values),
                    }
                ),
            )

        return self._build_grouped_records(
            dataframe,
            derived_values=derived_values,
            valid_mask=valid_mask,
            group_by=normalized_group_by,
            dropna=dropna,
        )

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
        left_column: str,
        right_column: str,
        operator: str,
        aggregation: str,
        group_by: list[str] | tuple[str, ...],
        dropna: bool,
    ) -> None:
        for name, value in (
            ("left_column", left_column),
            ("right_column", right_column),
        ):
            if not isinstance(value, str) or not value.strip():
                raise StatisticalValidationError(
                    f"{name} must be a non-empty string"
                )

        if (
            not isinstance(operator, str)
            or operator not in self.SUPPORTED_OPERATORS
        ):
            raise StatisticalValidationError(
                f"unsupported operator: {operator}"
            )

        if (
            not isinstance(aggregation, str)
            or aggregation not in self.SUPPORTED_AGGREGATIONS
        ):
            raise StatisticalValidationError(
                f"unsupported aggregation: {aggregation}"
            )

        if not isinstance(group_by, (list, tuple)):
            raise StatisticalValidationError(
                "group_by must be a list or tuple"
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

        reserved_output_columns = {"valid_row_count", "result"}
        conflicting_columns = (
            reserved_output_columns.intersection(group_by)
        )
        if conflicting_columns:
            raise StatisticalValidationError(
                "group_by contains reserved output columns: "
                f"{sorted(conflicting_columns)}"
            )

        if not isinstance(dropna, bool):
            raise StatisticalValidationError(
                "dropna must be a boolean"
            )

    def _validate_columns(
        self,
        dataframe: pd.DataFrame,
        *,
        left_column: str,
        right_column: str,
        group_by: tuple[str, ...],
    ) -> None:
        required_columns = (
            left_column,
            right_column,
            *group_by,
        )

        missing_columns = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise StatisticalValidationError(
                f"columns do not exist: {missing_columns}"
            )

        for column in (left_column, right_column):
            dtype = dataframe[column].dtype

            if (
                not is_numeric_dtype(dtype)
                or is_bool_dtype(dtype)
                or is_complex_dtype(dtype)
            ):
                raise StatisticalValidationError(
                    f"column must have a real numeric dtype: '{column}'"
                )

    def _to_finite_numeric_array(
        self,
        series: pd.Series,
        *,
        column: str,
    ) -> np.ndarray:
        try:
            numeric = pd.to_numeric(
                series,
                errors="coerce",
            )

            values = numeric.to_numpy(
                dtype=np.float64,
                na_value=np.nan,
            )

            return values

        except (TypeError, ValueError, OverflowError) as exc:
            raise DerivedMetricAnalysisError(
                f"failed to convert numeric column '{column}'"
            ) from exc

    def _build_grouped_records(
        self,
        dataframe: pd.DataFrame,
        *,
        derived_values: np.ndarray,
        valid_mask: np.ndarray,
        group_by: tuple[str, ...],
        dropna: bool,
    ) -> tuple[Mapping[str, Any], ...]:
        metric_column = self._INTERNAL_COLUMN

        while metric_column in dataframe.columns:
            metric_column = f"_{metric_column}"

        work = dataframe.loc[:, list(group_by)].copy()
        work[metric_column] = np.where(
            valid_mask,
            derived_values,
            np.nan,
        )

        if work.empty:
            return ()

        try:
            grouped = work.groupby(
                list(group_by),
                dropna=dropna,
                sort=True,
                observed=True,
            )

            records: list[Mapping[str, Any]] = []

            for group_key, group_dataframe in grouped:
                if len(group_by) == 1:
                    group_values = (
                        group_key
                        if isinstance(group_key, tuple)
                        else (group_key,)
                    )
                else:
                    group_values = tuple(group_key)

                values = group_dataframe[metric_column].to_numpy(
                    dtype=np.float64,
                )
                valid_values = values[np.isfinite(values)]

                record: dict[str, Any] = {
                    column: self._normalize_value(value)
                    for column, value in zip(
                        group_by,
                        group_values,
                        strict=True,
                    )
                }
                record["valid_row_count"] = len(valid_values)
                record["result"] = self._sum(valid_values)

                records.append(MappingProxyType(record))

            return tuple(records)

        except StatisticalValidationError:
            raise
        except Exception as exc:
            raise DerivedMetricAnalysisError(
                f"derived metric grouping failed: {exc}"
            ) from exc

    @staticmethod
    def _sum(values: np.ndarray) -> float | None:
        if len(values) == 0:
            return None

        try:
            result = math.fsum(float(value) for value in values)
        except (ArithmeticError, OverflowError, ValueError) as exc:
            raise DerivedMetricAnalysisError(
                "derived metric aggregation overflowed"
            ) from exc

        if not math.isfinite(result):
            raise DerivedMetricAnalysisError(
                "derived metric aggregation produced a non-finite result"
            )

        return result

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if value is None or value is pd.NA or value is pd.NaT:
            return None

        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, float) and not math.isfinite(value):
            return None

        if isinstance(value, (pd.Timestamp,)):
            return value.isoformat()

        return value
