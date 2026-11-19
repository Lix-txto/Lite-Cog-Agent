"""Exceptions for the data cleaning module."""

from __future__ import annotations

from lite_cog_agent.execution.exceptions import ExecutionException


class CleaningException(ExecutionException):
    """Base exception for data cleaning operations."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "cleaning_error",
        retryable: bool = False,
    ) -> None:
        super().__init__(
            message,
            code=code,
            retryable=retryable,
        )


class CleaningValidationError(CleaningException):
    """Raised when a cleaning request or its parameters are invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            code="cleaning_validation_error",
            retryable=False,
        )


class CleaningOperationError(CleaningException):
    """Raised when an individual cleaning operation fails."""

    def __init__(
        self,
        message: str,
        *,
        operation: str,
        retryable: bool = False,
    ) -> None:
        super().__init__(
            message,
            code="cleaning_operation_error",
            retryable=retryable,
        )
        self.operation = operation


class CleaningExecutionError(CleaningException):
    """Raised when the overall cleaning workflow fails."""

    def __init__(
        self,
        message: str,
        *,
        failed_operation: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(
            message,
            code="cleaning_execution_error",
            retryable=retryable,
        )
        self.failed_operation = failed_operation
