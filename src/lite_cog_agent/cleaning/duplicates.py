"""Duplicate row cleaning utilities."""

from collections.abc import Sequence
from typing import Any

import pandas as pd

from lite_cog_agent.cleaning.exceptions import (
    CleaningOperationError,
    CleaningValidationError,
)


class DuplicateCleaner:
    """Clean duplicate rows from pandas DataFrames."""

    _VALID_KEEP_VALUES = ("first", "last", False)

    def drop_duplicates(
        self,
        dataframe: pd.DataFrame,
        *,
        subset: Sequence[str] | None = None,
        keep: str | bool = "first",
        ignore_index: bool = False,
    ) -> pd.DataFrame:
        """Return a copy of the DataFrame without duplicate rows."""
        self._validate_dataframe(dataframe)
        self._validate_keep(keep)
        self._validate_ignore_index(ignore_index)

        validated_subset = self._validate_subset(
            dataframe,
            subset,
        )

        try:
            result = dataframe.drop_duplicates(
                subset=(
                    list(validated_subset)
                    if validated_subset is not None
                    else None
                ),
                keep=keep,
                ignore_index=ignore_index,
            )

            return result.copy(deep=True)

        except Exception as exc:
            raise CleaningOperationError(
                f"Failed to drop duplicate rows: {exc}",
                operation="duplicates.drop",
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

    @classmethod
    def _validate_keep(cls, keep: Any) -> None:
        """Validate the duplicate retention strategy."""
        if not any(
            type(keep) is type(candidate) and keep == candidate
            for candidate in cls._VALID_KEEP_VALUES
        ):
            raise CleaningValidationError(
                "keep must be 'first', 'last', or False"
            )

    @staticmethod
    def _validate_ignore_index(ignore_index: Any) -> None:
        """Validate the index reset option."""
        if not isinstance(ignore_index, bool):
            raise CleaningValidationError(
                "ignore_index must be a boolean"
            )

    @classmethod
    def _validate_subset(
        cls,
        dataframe: pd.DataFrame,
        subset: Sequence[str] | None,
    ) -> tuple[str, ...] | None:
        """Validate selected columns used for duplicate detection."""
        if subset is None:
            return None

        if isinstance(subset, (str, bytes)) or not isinstance(
            subset,
            Sequence,
        ):
            raise CleaningValidationError(
                "subset must be a sequence of column names"
            )

        if not subset:
            raise CleaningValidationError(
                "subset must not be empty"
            )

        if any(
            not isinstance(column, str) or not column.strip()
            for column in subset
        ):
            raise CleaningValidationError(
                "subset must contain non-empty string column names"
            )

        if len(set(subset)) != len(subset):
            raise CleaningValidationError(
                "subset must not contain duplicate columns"
            )

        unknown_columns = set(subset) - set(dataframe.columns)

        if unknown_columns:
            raise CleaningValidationError(
                f"subset contains unknown columns: "
                f"{sorted(unknown_columns)}"
            )

        return tuple(subset)
