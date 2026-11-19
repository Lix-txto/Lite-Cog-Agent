"""Contracts for the data cleaning module."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping
from lite_cog_agent.profiling.contracts import DataQualityReport


def _validate_identifier(value: str, field_name: str) -> None:
    """Validate a non-empty string identifier."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_non_negative_int(value: int, field_name: str) -> None:
    """Validate a non-negative integer, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")

    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _validate_duration(value: float, field_name: str) -> None:
    """Validate a finite, non-negative duration."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")

    if not isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")


def _immutable_mapping(
    value: Mapping[str, Any],
    field_name: str,
) -> Mapping[str, Any]:
    """Return a read-only shallow copy of a mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")

    copied = dict(value)

    if not all(isinstance(key, str) for key in copied):
        raise TypeError(f"{field_name} keys must be strings")

    return MappingProxyType(copied)


class CleaningStatus(str, Enum):
    """Possible states of a cleaning operation or workflow."""

    SUCCESS = "success"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CleaningQualityComparison:
    """Data quality comparison before and after cleaning."""

    before: DataQualityReport
    after: DataQualityReport

    def __post_init__(self) -> None:
        if not isinstance(self.before, DataQualityReport):
            raise TypeError("before must be a DataQualityReport")

        if not isinstance(self.after, DataQualityReport):
            raise TypeError("after must be a DataQualityReport")


@dataclass(frozen=True, slots=True)
class CleaningStep:
    """A single cleaning operation and its parameters."""

    operation: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.operation, "operation")

        object.__setattr__(
            self,
            "parameters",
            _immutable_mapping(self.parameters, "parameters"),
        )


@dataclass(frozen=True, slots=True)
class CleaningRequest:
    """A request to clean a dataset using an ordered sequence of steps."""

    dataset_id: str
    steps: tuple[CleaningStep, ...]
    output_dataset_name: str | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.dataset_id, "dataset_id")

        if not isinstance(self.steps, tuple):
            raise TypeError("steps must be a tuple")

        if not self.steps:
            raise ValueError("steps must contain at least one cleaning step")

        if not all(isinstance(step, CleaningStep) for step in self.steps):
            raise TypeError("all steps must be CleaningStep instances")

        if self.output_dataset_name is not None:
            if (
                not isinstance(self.output_dataset_name, str)
                or not self.output_dataset_name.strip()
            ):
                raise ValueError(
                    "output_dataset_name must be a non-empty string "
                    "or None"
                )


@dataclass(frozen=True, slots=True)
class CleaningErrorInfo:
    """Structured error information for a cleaning workflow or step."""

    code: str
    message: str
    retryable: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.code, "code")

        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message must be a non-empty string")

        if not isinstance(self.retryable, bool):
            raise TypeError("retryable must be a boolean")

        object.__setattr__(
            self,
            "details",
            _immutable_mapping(self.details, "details"),
        )


@dataclass(frozen=True, slots=True)
class CleaningOperationRecord:
    """Execution record for one cleaning step."""

    operation: str
    status: CleaningStatus
    parameters: Mapping[str, Any] = field(default_factory=dict)
    rows_before: int = 0
    rows_after: int | None = None
    columns_before: int = 0
    columns_after: int | None = None
    duration_ms: float = 0.0
    error: CleaningErrorInfo | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.operation, "operation")

        if not isinstance(self.status, CleaningStatus):
            raise TypeError("status must be a CleaningStatus")

        _validate_non_negative_int(self.rows_before, "rows_before")
        _validate_non_negative_int(self.columns_before, "columns_before")

        if self.rows_after is not None:
            _validate_non_negative_int(self.rows_after, "rows_after")

        if self.columns_after is not None:
            _validate_non_negative_int(self.columns_after, "columns_after")

        _validate_duration(self.duration_ms, "duration_ms")

        if self.error is not None and not isinstance(
            self.error, CleaningErrorInfo
        ):
            raise TypeError("error must be a CleaningErrorInfo or None")

        if self.status is CleaningStatus.SUCCESS:
            if self.error is not None:
                raise ValueError(
                    "successful cleaning operation cannot contain an error"
                )

            if self.rows_after is None or self.columns_after is None:
                raise ValueError(
                    "successful cleaning operation must contain "
                    "rows_after and columns_after"
                )

        if self.status is CleaningStatus.FAILED and self.error is None:
            raise ValueError(
                "failed cleaning operation must contain an error"
            )

        object.__setattr__(
            self,
            "parameters",
            _immutable_mapping(self.parameters, "parameters"),
        )


@dataclass(frozen=True, slots=True)
class CleaningResult:
    """Final result of a dataset cleaning workflow."""

    source_dataset_id: str
    status: CleaningStatus
    input_row_count: int
    input_column_count: int
    operations: tuple[CleaningOperationRecord, ...]
    duration_ms: float = 0.0
    output_dataset_id: str | None = None
    output_row_count: int | None = None
    output_column_count: int | None = None
    error: CleaningErrorInfo | None = None
    quality_comparison: CleaningQualityComparison | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.source_dataset_id, "source_dataset_id")

        if not isinstance(self.status, CleaningStatus):
            raise TypeError("status must be a CleaningStatus")

        _validate_non_negative_int(
            self.input_row_count,
            "input_row_count",
        )
        _validate_non_negative_int(
            self.input_column_count,
            "input_column_count",
        )
        _validate_duration(self.duration_ms, "duration_ms")

        if not isinstance(self.operations, tuple):
            raise TypeError("operations must be a tuple")

        if not all(
            isinstance(record, CleaningOperationRecord)
            for record in self.operations
        ):
            raise TypeError(
                "all operations must be CleaningOperationRecord instances"
            )

        if self.output_dataset_id is not None:
            _validate_identifier(
                self.output_dataset_id,
                "output_dataset_id",
            )

        if self.output_row_count is not None:
            _validate_non_negative_int(
                self.output_row_count,
                "output_row_count",
            )

        if self.output_column_count is not None:
            _validate_non_negative_int(
                self.output_column_count,
                "output_column_count",
            )

        if self.error is not None and not isinstance(
            self.error, CleaningErrorInfo
        ):
            raise TypeError("error must be a CleaningErrorInfo or None")

        if self.quality_comparison is not None and not isinstance(
            self.quality_comparison,
            CleaningQualityComparison,
        ):
            raise TypeError(
                "quality_comparison must be a CleaningQualityComparison or None"
            )

        if self.status is CleaningStatus.SUCCESS:
            if self.output_dataset_id is None:
                raise ValueError(
                    "successful cleaning result must contain "
                    "output_dataset_id"
                )

            if (
                self.output_row_count is None
                or self.output_column_count is None
            ):
                raise ValueError(
                    "successful cleaning result must contain "
                    "output row and column counts"
                )

            if self.error is not None:
                raise ValueError(
                    "successful cleaning result cannot contain an error"
                )

        if self.status is CleaningStatus.FAILED:
            if self.error is None:
                raise ValueError(
                    "failed cleaning result must contain an error"
                )

            if self.output_dataset_id is not None:
                raise ValueError(
                    "failed cleaning result cannot contain "
                    "output_dataset_id"
                )

            if (
                self.output_row_count is not None
                or self.output_column_count is not None
            ):
                raise ValueError(
                    "failed cleaning result cannot contain "
                    "published output dimensions"
                )


