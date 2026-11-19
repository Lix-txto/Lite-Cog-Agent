"""Exceptions raised by the execution infrastructure."""


class ExecutionException(Exception):
    """Base exception for execution infrastructure errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "execution_error",
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class ExecutionValidationError(ExecutionException):
    """Raised when an execution request is invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            code="execution_validation_error",
            retryable=False,
        )


class OperationNotFoundError(ExecutionException):
    """Raised when the requested operation is not registered."""

    def __init__(self, operation: str) -> None:
        super().__init__(
            f"Operation not found: {operation}",
            code="operation_not_found",
            retryable=False,
        )
        self.operation = operation


class ExecutionTimeoutError(ExecutionException):
    """Raised when an execution exceeds its time limit."""

    def __init__(self, timeout_seconds: float) -> None:
        super().__init__(
            f"Execution timed out after {timeout_seconds} seconds",
            code="execution_timeout",
            retryable=False,
        )
        self.timeout_seconds = timeout_seconds


class OperationExecutionError(ExecutionException):
    """Raised when an operation fails during execution."""

    def __init__(self, message: str) -> None:
        super().__init__(
            message,
            code="operation_execution_error",
            retryable=False,
        )
