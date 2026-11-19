"""Column selection, deletion, and renaming utilities."""

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from lite_cog_agent.cleaning.exceptions import (
    CleaningOperationError,
    CleaningValidationError,
)


class ColumnCleaner:
    """Clean DataFrame columns without modifying the source DataFrame."""

    def select(
        self,
        dataframe: pd.DataFrame,
        *,
        columns: Sequence[str],
    ) -> pd.DataFrame:
        """Return a DataFrame containing only the selected columns."""
        self._validate_dataframe(dataframe)

        selected_columns = self._validate_columns(
            dataframe,
            columns,
            field_name="columns",
        )

        try:
            return dataframe.loc[:, list(selected_columns)].copy(
                deep=True
            )
        except Exception as exc:
            raise CleaningOperationError(
                f"Failed to select columns: {exc}",
                operation="columns.select",
            ) from exc

    def drop(
        self,
        dataframe: pd.DataFrame,
        *,
        columns: Sequence[str],
    ) -> pd.DataFrame:
        """Return a DataFrame without the specified columns."""
        self._validate_dataframe(dataframe)

        dropped_columns = self._validate_columns(
            dataframe,
            columns,
            field_name="columns",
        )

        try:
            return dataframe.drop(
                columns=list(dropped_columns),
            ).copy(deep=True)
        except Exception as exc:
            raise CleaningOperationError(
                f"Failed to drop columns: {exc}",
                operation="columns.drop",
            ) from exc

    def rename(
        self,
        dataframe: pd.DataFrame,
        *,
        mapping: Mapping[str, str],
    ) -> pd.DataFrame:
        """Return a DataFrame with columns renamed according to mapping."""
        self._validate_dataframe(dataframe)

        validated_mapping = self._validate_mapping(
            dataframe,
            mapping,
        )

        try:
            return dataframe.rename(
                columns=dict(validated_mapping),
            ).copy(deep=True)
        except Exception as exc:
            raise CleaningOperationError(
                f"Failed to rename columns: {exc}",
                operation="columns.rename",
            ) from exc

    @staticmethod
    def _validate_dataframe(dataframe: Any) -> None:
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
        columns: Sequence[str],
        *,
        field_name: str,
    ) -> tuple[str, ...]:
        """Validate a non-empty sequence of existing column names."""
        if isinstance(columns, (str, bytes)) or not isinstance(
            columns,
            Sequence,
        ):
            raise CleaningValidationError(
                f"{field_name} must be a sequence of column names"
            )

        if not columns:
            raise CleaningValidationError(
                f"{field_name} must not be empty"
            )

        if any(
            not isinstance(column, str) or not column.strip()
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
    def _validate_mapping(
        dataframe: pd.DataFrame,
        mapping: Mapping[str, str],
    ) -> dict[str, str]:
        """Validate source names and final renamed column uniqueness."""
        if not isinstance(mapping, Mapping):
            raise CleaningValidationError(
                "mapping must be a mapping of old names to new names"
            )

        if not mapping:
            raise CleaningValidationError(
                "mapping must not be empty"
            )

        validated_mapping: dict[str, str] = {}

        for old_name, new_name in mapping.items():
            if not isinstance(old_name, str) or not old_name.strip():
                raise CleaningValidationError(
                    "mapping keys must be non-empty strings"
                )

            if not isinstance(new_name, str) or not new_name.strip():
                raise CleaningValidationError(
                    "mapping values must be non-empty strings"
                )

            validated_mapping[old_name] = new_name

        unknown_columns = (
            set(validated_mapping) - set(dataframe.columns)
        )

        if unknown_columns:
            raise CleaningValidationError(
                "mapping contains unknown columns: "
                f"{sorted(unknown_columns)}"
            )

        final_columns = [
            validated_mapping.get(column, column)
            for column in dataframe.columns
        ]

        if len(final_columns) != len(set(final_columns)):
            raise CleaningValidationError(
                "renaming would produce duplicate column names"
            )

        return validated_mapping
