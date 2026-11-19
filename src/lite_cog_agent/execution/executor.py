"""Python executor for registered data analysis operations."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.execution.contracts import (
    ExecutionFailure,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
)
from lite_cog_agent.execution.exceptions import (
    ExecutionException,
    ExecutionTimeoutError,
    )
from lite_cog_agent.execution.registry import ExecutionRegistry


class PythonExecutor:
    """Execute registered data analysis operations."""

    def __init__(self, registry: ExecutionRegistry) -> None:
        if not isinstance(registry, ExecutionRegistry):
            raise TypeError("registry must be an ExecutionRegistry")

        self._registry = registry

    def execute(
        self,
        request: ExecutionRequest,
        context: ExecutionContext,
    ) -> ExecutionResult:
        """Execute an operation and return a structured result.

        Execution infrastructure exceptions are converted into failed
        ExecutionResult objects. Invalid method arguments are rejected
        before execution and raise TypeError.
        """

        if not isinstance(request, ExecutionRequest):
            raise TypeError("request must be an ExecutionRequest")

        if not isinstance(context, ExecutionContext):
            raise TypeError("context must be an ExecutionContext")

        start_time = perf_counter()

        try:
            operation = self._registry.get(request.operation)

            output = operation.execute(request, context)

            duration_ms = (perf_counter() - start_time) * 1000

            if (
                request.timeout_seconds is not None
                and duration_ms / 1000 > request.timeout_seconds
            ):
                timeout_error = ExecutionTimeoutError(
                    request.timeout_seconds
                )

                return self._build_failure_result(
                    request=request,
                    code=timeout_error.code,
                    message=str(timeout_error),
                    retryable=timeout_error.retryable,
                    duration_ms=duration_ms,
                    details={
                        "timeout_seconds": request.timeout_seconds,
                        "elapsed_ms": duration_ms,
                        "timeout_type": "soft",
                        "operation_completed": True,
                    },
                )

            return ExecutionResult(
                execution_id=request.execution_id,
                operation=request.operation,
                status=ExecutionStatus.SUCCESS,
                output=output,
                duration_ms=duration_ms,
            )

        except ExecutionException as exc:
            duration_ms = (perf_counter() - start_time) * 1000

            return self._build_failure_result(
                request=request,
                code=exc.code,
                message=str(exc) or type(exc).__name__,
                retryable=exc.retryable,
                duration_ms=duration_ms,
            )

        except Exception as exc:
            duration_ms = (perf_counter() - start_time) * 1000

            return self._build_failure_result(
                request=request,
                code="operation_execution_error",
                message=str(exc) or type(exc).__name__,
                retryable=False,
                duration_ms=duration_ms,
                details={
                    "exception_type": type(exc).__name__,
                },
            )

    @staticmethod
    def _build_failure_result(
        *,
        request: ExecutionRequest,
        code: str,
        message: str,
        retryable: bool,
        duration_ms: float,
        details: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """Build a standardized failed execution result."""

        failure = ExecutionFailure(
            code=code,
            message=message,
            retryable=retryable,
            details=details or {},
        )

        return ExecutionResult(
            execution_id=request.execution_id,
            operation=request.operation,
            status=ExecutionStatus.FAILED,
            error=failure,
            duration_ms=duration_ms,
        )
