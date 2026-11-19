"""Execution operations for data cleaning."""

from __future__ import annotations

from abc import ABC
from typing import Any

from lite_cog_agent.cleaning.columns import ColumnCleaner
from lite_cog_agent.cleaning.duplicates import DuplicateCleaner
from lite_cog_agent.cleaning.exceptions import CleaningValidationError
from lite_cog_agent.cleaning.missing import MissingValueCleaner
from lite_cog_agent.cleaning.type_converter import TypeConverter
from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.execution.contracts import ExecutionRequest
from lite_cog_agent.execution.registry import BaseOperation


class BaseCleaningOperation(BaseOperation, ABC):
    """Base class for operations that transform a dataset."""

    def _get_dataframe(
        self,
        request: ExecutionRequest,
        context: ExecutionContext,
    ) -> Any:
        """Retrieve the source DataFrame from the execution context."""
        if request.dataset_id is None:
            raise CleaningValidationError(
                "dataset_id is required for cleaning operations"
            )

        try:
            return context.get_dataset(request.dataset_id)
        except KeyError as exc:
            raise CleaningValidationError(
                f"dataset is not available in context: "
                f"{request.dataset_id}"
            ) from exc

    @staticmethod
    def _validate_parameters(
        request: ExecutionRequest,
        *,
        allowed: set[str],
        required: set[str] | None = None,
    ) -> None:
        """Reject unknown parameters and check required parameters."""
        parameters = request.parameters

        unknown = set(parameters) - allowed

        if unknown:
            raise CleaningValidationError(
                f"unexpected parameters: {sorted(unknown)}"
            )

        missing = (required or set()) - set(parameters)

        if missing:
            raise CleaningValidationError(
                f"missing required parameters: {sorted(missing)}"
            )


class MissingDropOperation(BaseCleaningOperation):
    """Drop rows containing missing values."""

    _ALLOWED_PARAMETERS = {"how", "subset", "thresh"}

    def __init__(
        self,
        cleaner: MissingValueCleaner | None = None,
    ) -> None:
        self._cleaner = cleaner or MissingValueCleaner()

    @property
    def name(self) -> str:
        return "missing.drop"

    @property
    def description(self) -> str:
        return "Drop rows containing missing values from a dataset."

    def execute(
        self,
        request: ExecutionRequest,
        context: ExecutionContext,
    ) -> Any:
        self._validate_parameters(
            request,
            allowed=self._ALLOWED_PARAMETERS,
        )

        dataframe = self._get_dataframe(request, context)

        return self._cleaner.drop(
            dataframe,
            **dict(request.parameters),
        )


class MissingFillOperation(BaseCleaningOperation):
    """Fill missing values in selected columns."""

    _ALLOWED_PARAMETERS = {"value", "columns"}

    def __init__(
        self,
        cleaner: MissingValueCleaner | None = None,
    ) -> None:
        self._cleaner = cleaner or MissingValueCleaner()

    @property
    def name(self) -> str:
        return "missing.fill"

    @property
    def description(self) -> str:
        return "Fill missing values in a dataset."

    def execute(
        self,
        request: ExecutionRequest,
        context: ExecutionContext,
    ) -> Any:
        self._validate_parameters(
            request,
            allowed=self._ALLOWED_PARAMETERS,
            required={"value"},
        )

        dataframe = self._get_dataframe(request, context)

        return self._cleaner.fill(
            dataframe,
            **dict(request.parameters),
        )


class DuplicateDropOperation(BaseCleaningOperation):
    """Remove duplicate rows from a dataset."""

    _ALLOWED_PARAMETERS = {
        "subset",
        "keep",
        "ignore_index",
    }

    def __init__(
        self,
        cleaner: DuplicateCleaner | None = None,
    ) -> None:
        self._cleaner = cleaner or DuplicateCleaner()

    @property
    def name(self) -> str:
        return "duplicates.drop"

    @property
    def description(self) -> str:
        return "Remove duplicate rows from a dataset."

    def execute(
        self,
        request: ExecutionRequest,
        context: ExecutionContext,
    ) -> Any:
        self._validate_parameters(
            request,
            allowed=self._ALLOWED_PARAMETERS,
        )

        dataframe = self._get_dataframe(request, context)

        return self._cleaner.drop_duplicates(
            dataframe,
            **dict(request.parameters),
        )


class ColumnSelectOperation(BaseCleaningOperation):
    """Select a subset of columns."""

    _ALLOWED_PARAMETERS = {"columns"}

    def __init__(
        self,
        cleaner: ColumnCleaner | None = None,
    ) -> None:
        self._cleaner = cleaner or ColumnCleaner()

    @property
    def name(self) -> str:
        return "columns.select"

    @property
    def description(self) -> str:
        return "Select specified columns from a dataset."

    def execute(
        self,
        request: ExecutionRequest,
        context: ExecutionContext,
    ) -> Any:
        self._validate_parameters(
            request,
            allowed=self._ALLOWED_PARAMETERS,
            required={"columns"},
        )

        dataframe = self._get_dataframe(request, context)

        return self._cleaner.select(
            dataframe,
            **dict(request.parameters),
        )


class ColumnDropOperation(BaseCleaningOperation):
    """Remove selected columns."""

    _ALLOWED_PARAMETERS = {"columns"}

    def __init__(
        self,
        cleaner: ColumnCleaner | None = None,
    ) -> None:
        self._cleaner = cleaner or ColumnCleaner()

    @property
    def name(self) -> str:
        return "columns.drop"

    @property
    def description(self) -> str:
        return "Remove specified columns from a dataset."

    def execute(
        self,
        request: ExecutionRequest,
        context: ExecutionContext,
    ) -> Any:
        self._validate_parameters(
            request,
            allowed=self._ALLOWED_PARAMETERS,
            required={"columns"},
        )

        dataframe = self._get_dataframe(request, context)

        return self._cleaner.drop(
            dataframe,
            **dict(request.parameters),
        )


class ColumnRenameOperation(BaseCleaningOperation):
    """Rename selected columns."""

    _ALLOWED_PARAMETERS = {"mapping"}

    def __init__(
        self,
        cleaner: ColumnCleaner | None = None,
    ) -> None:
        self._cleaner = cleaner or ColumnCleaner()

    @property
    def name(self) -> str:
        return "columns.rename"

    @property
    def description(self) -> str:
        return "Rename dataset columns using a name mapping."

    def execute(
        self,
        request: ExecutionRequest,
        context: ExecutionContext,
    ) -> Any:
        self._validate_parameters(
            request,
            allowed=self._ALLOWED_PARAMETERS,
            required={"mapping"},
        )

        dataframe = self._get_dataframe(request, context)

        return self._cleaner.rename(
            dataframe,
            **dict(request.parameters),
        )


class TypeConvertOperation(BaseCleaningOperation):
    """Convert selected columns to specified data types."""

    _ALLOWED_PARAMETERS = {
        "conversions",
        "errors",
        "datetime_format",
    }

    def __init__(
        self,
        converter: TypeConverter | None = None,
    ) -> None:
        self._converter = converter or TypeConverter()

    @property
    def name(self) -> str:
        return "types.convert"

    @property
    def description(self) -> str:
        return "Convert selected columns to specified data types."

    def execute(
        self,
        request: ExecutionRequest,
        context: ExecutionContext,
    ) -> Any:
        self._validate_parameters(
            request,
            allowed=self._ALLOWED_PARAMETERS,
            required={"conversions"},
        )

        dataframe = self._get_dataframe(request, context)

        return self._converter.convert(
            dataframe,
            **dict(request.parameters),
        )


__all__ = [
    "BaseCleaningOperation",
    "MissingDropOperation",
    "MissingFillOperation",
    "DuplicateDropOperation",
    "ColumnSelectOperation",
    "ColumnDropOperation",
    "ColumnRenameOperation",
    "TypeConvertOperation",
]
