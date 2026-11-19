"""Unit tests for cleaning exceptions."""

import pytest

from lite_cog_agent.cleaning.exceptions import (
    CleaningException,
    CleaningExecutionError,
    CleaningOperationError,
    CleaningValidationError,
)
from lite_cog_agent.execution.exceptions import ExecutionException


def test_cleaning_exception_inherits_execution_exception():
    error = CleaningException("Cleaning failed")

    assert isinstance(error, ExecutionException)
    assert error.code == "cleaning_error"
    assert error.retryable is False


def test_cleaning_exception_supports_custom_code_and_retryable():
    error = CleaningException(
        "Temporary failure",
        code="temporary_failure",
        retryable=True,
    )

    assert error.code == "temporary_failure"
    assert error.retryable is True


def test_cleaning_validation_error():
    error = CleaningValidationError("Invalid cleaning parameters")

    assert isinstance(error, CleaningException)
    assert error.code == "cleaning_validation_error"
    assert error.retryable is False


def test_cleaning_operation_error():
    error = CleaningOperationError(
        "Column does not exist",
        operation="drop_missing",
    )

    assert error.code == "cleaning_operation_error"
    assert error.operation == "drop_missing"
    assert error.retryable is False


def test_cleaning_operation_error_can_be_retryable():
    error = CleaningOperationError(
        "Temporary operation failure",
        operation="convert_type",
        retryable=True,
    )

    assert error.retryable is True


def test_cleaning_execution_error():
    error = CleaningExecutionError(
        "Workflow failed",
        failed_operation="drop_duplicates",
    )

    assert error.code == "cleaning_execution_error"
    assert error.failed_operation == "drop_duplicates"
    assert error.retryable is False


def test_cleaning_execution_error_without_failed_operation():
    error = CleaningExecutionError("Workflow failed")

    assert error.failed_operation is None


def test_cleaning_execution_error_can_be_retryable():
    error = CleaningExecutionError(
        "Temporary workflow failure",
        retryable=True,
    )

    assert error.retryable is True


def test_cleaning_exceptions_are_execution_exceptions():
    errors = [
        CleaningException("base"),
        CleaningValidationError("validation"),
        CleaningOperationError(
            "operation",
            operation="drop_missing",
        ),
        CleaningExecutionError("execution"),
    ]

    assert all(isinstance(error, ExecutionException) for error in errors)
