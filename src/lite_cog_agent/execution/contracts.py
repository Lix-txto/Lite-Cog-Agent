"""Execution contracts for the Python execution infrastructure."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4


def _validate_identifier(value: str, field_name: str) -> None:
    """Validate a non-empty string identifier."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


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


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    """A request to execute a registered analysis operation."""

    operation: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    dataset_id: str | None = None
    timeout_seconds: float | None = None
    execution_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        _validate_identifier(self.operation, "operation")
        _validate_identifier(self.execution_id, "execution_id")

        if self.dataset_id is not None:
            _validate_identifier(self.dataset_id, "dataset_id")

        if self.timeout_seconds is not None:
            if isinstance(self.timeout_seconds, bool) or not isinstance(
                self.timeout_seconds, (int, float)
            ):
                raise TypeError("timeout_seconds must be a number")

            if (
                not isfinite(self.timeout_seconds)
                or self.timeout_seconds <= 0
            ):
                raise ValueError("timeout_seconds must be greater than 0")

        object.__setattr__(
            self,
            "parameters",
            _immutable_mapping(self.parameters, "parameters"),
        )


class ExecutionStatus(str, Enum):
    """Possible execution states."""

    SUCCESS = "success"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ExecutionFailure:
    """Structured failure information returned by an execution."""

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
class ExecutionResult:
    """The result of a Python execution operation."""

    execution_id: str
    operation: str
    status: ExecutionStatus
    output: Any = None
    error: ExecutionFailure | None = None
    duration_ms: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.execution_id, "execution_id")
        _validate_identifier(self.operation, "operation")

        if not isinstance(self.status, ExecutionStatus):
            raise TypeError("status must be an ExecutionStatus")

        if self.error is not None and not isinstance(
            self.error, ExecutionFailure
        ):
            raise TypeError("error must be an ExecutionFailure or None")

        if isinstance(self.duration_ms, bool) or not isinstance(
            self.duration_ms, (int, float)
        ):
            raise TypeError("duration_ms must be a number")

        if not isfinite(self.duration_ms) or self.duration_ms < 0:
            raise ValueError("duration_ms must be non-negative")

        if self.status is ExecutionStatus.SUCCESS and self.error is not None:
            raise ValueError("successful execution cannot contain an error")

        if self.status is ExecutionStatus.FAILED and self.error is None:
            raise ValueError("failed execution must contain an error")

        object.__setattr__(
            self,
            "metadata",
            _immutable_mapping(self.metadata, "metadata"),
        )

    @property
    def is_success(self) -> bool:
        """Whether the execution completed successfully."""
        return self.status is ExecutionStatus.SUCCESS
