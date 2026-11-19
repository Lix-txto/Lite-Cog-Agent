"""Data type conversion utilities."""

from collections.abc import Mapping
from typing import Any

import pandas as pd

from lite_cog_agent.cleaning.exceptions import (
    CleaningOperationError,
    CleaningValidationError,
)


class TypeConverter:
    """Convert DataFrame columns to explicitly supported data types."""

    _SUPPORTED_TYPES = {
        "int",
        "float",
        "string",
        "boolean",
        "datetime",
    }

    _BOOLEAN_VALUES = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
    }

    def convert(
        self,
        dataframe: pd.DataFrame,
        *,
        conversions: Mapping[str, str],
        errors: str = "raise",
        datetime_format: str | None = None,
    ) -> pd.DataFrame:
        """Convert selected columns and return an independent DataFrame.

        Args:
            dataframe: Source DataFrame.
            conversions: Mapping of column names to target types.
            errors: Either "raise" or "coerce".
            datetime_format: Optional format string for datetime parsing.

        Returns:
            A converted copy of the source DataFrame.

        Raises:
            CleaningValidationError: If arguments are invalid.
            CleaningOperationError: If a conversion fails.
        """
        self._validate_dataframe(dataframe)
        self._validate_errors(errors)
        self._validate_datetime_format(datetime_format)

        validated_conversions = self._validate_conversions(
            dataframe,
            conversions,
        )

        result = dataframe.copy(deep=True)

        try:
            for column, target_type in validated_conversions.items():
                result[column] = self._convert_series(
                    result[column],
                    target_type,
                    errors=errors,
                    datetime_format=datetime_format,
                )

            return result

        except Exception as exc:
            raise CleaningOperationError(
                f"Failed to convert column data types: {exc}",
                operation="types.convert",
            ) from exc

    @staticmethod
    def _validate_dataframe(dataframe: Any) -> None:
        """Validate the input DataFrame and its column names."""
        if not isinstance(dataframe, pd.DataFrame):
            raise CleaningValidationError(
                "dataframe must be a pandas DataFrame"
            )

        if not dataframe.columns.is_unique:
            raise CleaningValidationError(
                "dataframe columns must be unique"
            )

        if any(
            not isinstance(column, str) or not column.strip()
            for column in dataframe.columns
        ):
            raise CleaningValidationError(
                "dataframe columns must be non-empty strings"
            )

    @staticmethod
    def _validate_errors(errors: Any) -> None:
        """Validate the conversion error policy."""
        if errors not in ("raise", "coerce"):
            raise CleaningValidationError(
                "errors must be either 'raise' or 'coerce'"
            )

    @staticmethod
    def _validate_datetime_format(
        datetime_format: Any,
    ) -> None:
        """Validate an optional datetime format string."""
        if datetime_format is not None and (
            not isinstance(datetime_format, str)
            or not datetime_format.strip()
        ):
            raise CleaningValidationError(
                "datetime_format must be a non-empty string or None"
            )

    @classmethod
    def _validate_conversions(
        cls,
        dataframe: pd.DataFrame,
        conversions: Mapping[str, str],
    ) -> dict[str, str]:
        """Validate requested columns and target data types."""
        if not isinstance(conversions, Mapping):
            raise CleaningValidationError(
                "conversions must be a mapping"
            )

        if not conversions:
            raise CleaningValidationError(
                "conversions must not be empty"
            )

        validated: dict[str, str] = {}

        for column, target_type in conversions.items():
            if not isinstance(column, str) or not column.strip():
                raise CleaningValidationError(
                    "conversion column names must be non-empty strings"
                )

            if column not in dataframe.columns:
                raise CleaningValidationError(
                    f"unknown column: {column!r}"
                )

            if (
                not isinstance(target_type, str)
                or target_type not in cls._SUPPORTED_TYPES
            ):
                raise CleaningValidationError(
                    f"unsupported target type for {column!r}: "
                    f"{target_type!r}"
                )

            validated[column] = target_type

        return validated

    @classmethod
    def _convert_series(
        cls,
        series: pd.Series,
        target_type: str,
        *,
        errors: str,
        datetime_format: str | None,
    ) -> pd.Series:
        """Convert one Series to the requested target type."""
        if target_type == "int":
            return cls._to_integer(series, errors=errors)

        if target_type == "float":
            return cls._to_float(series, errors=errors)

        if target_type == "string":
            return series.astype("string")

        if target_type == "boolean":
            return cls._to_boolean(series, errors=errors)

        if target_type == "datetime":
            return pd.to_datetime(
                series,
                errors=errors,
                format=datetime_format,
            )

        # This branch should be unreachable after argument validation.
        raise ValueError(f"unsupported target type: {target_type!r}")

    @staticmethod
    def _to_integer(
        series: pd.Series,
        *,
        errors: str,
    ) -> pd.Series:
        """Convert values to nullable 64-bit integers."""
        numeric = pd.to_numeric(series, errors=errors)

        fractional = numeric.notna() & (numeric % 1 != 0)

        if fractional.any():
            if errors == "raise":
                raise ValueError(
                    "integer conversion encountered fractional values"
                )

            numeric = numeric.mask(fractional)

        return numeric.astype("Int64")

    @staticmethod
    def _to_float(
        series: pd.Series,
        *,
        errors: str,
    ) -> pd.Series:
        """Convert values to 64-bit floating-point numbers."""
        numeric = pd.to_numeric(series, errors=errors)
        return numeric.astype("float64")

    @classmethod
    def _to_boolean(
        cls,
        series: pd.Series,
        *,
        errors: str,
    ) -> pd.Series:
        """Convert explicitly recognized values to nullable booleans."""
        def convert_value(value: Any) -> Any:
            if value is None or value is pd.NA or value is pd.NaT:
                return pd.NA

            missing = pd.isna(value)

            if isinstance(missing, bool) and missing:
                return pd.NA

            if isinstance(value, bool):
                return value

            if isinstance(value, (int, float)) and value in (0, 1):
                return bool(value)

            if isinstance(value, str):
                normalized = value.strip().lower()

                if normalized in cls._BOOLEAN_VALUES:
                    return cls._BOOLEAN_VALUES[normalized]

            if errors == "coerce":
                return pd.NA

            raise ValueError(
                f"cannot convert value {value!r} to boolean"
            )

        converted = series.map(convert_value)
        return converted.astype("boolean")
