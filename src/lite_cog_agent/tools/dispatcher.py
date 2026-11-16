from __future__ import annotations

import asyncio
import time
from typing import Any

from lite_cog_agent.llm.schemas import ToolCall
from lite_cog_agent.tools.exceptions import (
    ToolExecutionError,
    ToolExecutionTimeoutError,
    ToolParameterValidationError,
)
from lite_cog_agent.tools.execution_config import ToolExecutionConfig
from lite_cog_agent.tools.parameter_validator import (
    ToolParameterValidator,
)
from lite_cog_agent.tools.registry import (
    ToolNotFoundError,
    ToolRegistry,
)
from lite_cog_agent.tools.schemas import ToolResult


class ToolDispatcher:
    """Dispatch LLM tool calls with execution control."""

    def __init__(
        self,
        registry: ToolRegistry,
        validator: ToolParameterValidator | None = None,
        config: ToolExecutionConfig | None = None,
    ) -> None:
        self._registry = registry
        self._validator = validator or ToolParameterValidator()
        self._config = config or ToolExecutionConfig()

        self._semaphore = asyncio.Semaphore(
            self._config.max_concurrency
        )

    @staticmethod
    def _build_failure_result(
        error: str,
        *,
        tool_name: str | None = None,
        error_type: str | None = None,
        execution_time_seconds: float = 0.0,
        queue_wait_seconds: float = 0.0,
    ) -> ToolResult:
        """Build a standardized failed tool result."""

        return ToolResult(
            success=False,
            error=error,
            metadata={
                "tool_name": tool_name,
                "status": "failed",
                "execution_time_seconds": execution_time_seconds,
                "queue_wait_seconds": queue_wait_seconds,
                "error_type": error_type,
            },
        )

    async def dispatch(
        self,
        tool_call: ToolCall,
    ) -> ToolResult:
        """Find, validate, and execute a tool call."""

        if not isinstance(tool_call, ToolCall):
            return self._build_failure_result(
                "Invalid tool call: expected a ToolCall instance.",
                error_type="InvalidToolCallError",
            )

        try:
            tool = self._registry.get(tool_call.name)
        except ToolNotFoundError:
            return self._build_failure_result(
                f"Tool not found: {tool_call.name}",
                tool_name=tool_call.name,
                error_type="ToolNotFoundError",
            )

        try:
            self._validator.validate_arguments(
                tool.parameters,
                tool_call.arguments,
            )
        except ToolParameterValidationError as exc:
            return self._build_failure_result(
                str(exc),
                tool_name=tool.name,
                error_type=type(exc).__name__,
            )

        queued_at = time.perf_counter()

        async with self._semaphore:
            started_at = time.perf_counter()

            queue_wait_seconds = started_at - queued_at

            try:
                result = await asyncio.wait_for(
                    tool.execute(tool_call.arguments),
                    timeout=self._config.timeout_seconds,
                )

            except TimeoutError as exc:
                timeout_error = ToolExecutionTimeoutError(
                    (
                        f"Tool '{tool.name}' execution timed out "
                        f"after {self._config.timeout_seconds} seconds."
                    ),
                    tool_name=tool.name,
                    timeout_seconds=self._config.timeout_seconds,
                )

                execution_time_seconds = (
                    time.perf_counter() - started_at
                )

                return self._build_failure_result(
                    str(timeout_error),
                    tool_name=tool.name,
                    error_type=type(timeout_error).__name__,
                    execution_time_seconds=execution_time_seconds,
                    queue_wait_seconds=queue_wait_seconds,
                )

            except ToolExecutionError as exc:
                execution_time_seconds = (
                    time.perf_counter() - started_at
                )

                return self._build_failure_result(
                    str(exc),
                    tool_name=tool.name,
                    error_type=type(exc).__name__,
                    execution_time_seconds=execution_time_seconds,
                    queue_wait_seconds=queue_wait_seconds,
                )

            except Exception as exc:
                execution_time_seconds = (
                    time.perf_counter() - started_at
                )

                return self._build_failure_result(
                    str(exc),
                    tool_name=tool.name,
                    error_type=type(exc).__name__,
                    execution_time_seconds=execution_time_seconds,
                    queue_wait_seconds=queue_wait_seconds,
                )

            execution_time_seconds = (
                time.perf_counter() - started_at
            )

        if not isinstance(result, ToolResult):
            return self._build_failure_result(
                f"Tool '{tool.name}' returned an invalid result.",
                tool_name=tool.name,
                error_type="InvalidToolResultError",
                execution_time_seconds=execution_time_seconds,
                queue_wait_seconds=queue_wait_seconds,
            )

        execution_metadata: dict[str, Any] = {
            "tool_name": tool.name,
            "status": "success" if result.success else "failed",
            "execution_time_seconds": execution_time_seconds,
            "queue_wait_seconds": queue_wait_seconds,
            "error_type": (
                None if result.success else "ToolReportedFailure"
            ),
        }

        metadata = {
            **result.metadata,
            **execution_metadata,
        }

        return ToolResult(
            success=result.success,
            output=result.output,
            error=result.error,
            metadata=metadata,
        )