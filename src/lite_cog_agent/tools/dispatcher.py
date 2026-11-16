from __future__ import annotations

import asyncio

from lite_cog_agent.llm.schemas import ToolCall
from lite_cog_agent.tools.exceptions import (
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
    """Dispatch LLM tool calls to registered tools."""

    def __init__(
        self,
        registry: ToolRegistry,
        validator: ToolParameterValidator | None = None,
        config: ToolExecutionConfig | None = None,
    ) -> None:
        self._registry = registry
        self._validator = validator or ToolParameterValidator()
        self._config = config or ToolExecutionConfig()

        # Limit the number of concurrently executing tools.
        self._semaphore = asyncio.Semaphore(
            self._config.max_concurrency
        )

    async def dispatch(
        self,
        tool_call: ToolCall,
    ) -> ToolResult:
        """Find, validate, and execute a tool call."""

        if not isinstance(tool_call, ToolCall):
            return ToolResult.fail(
                "Invalid tool call: expected a ToolCall instance."
            )

        try:
            tool = self._registry.get(tool_call.name)
        except ToolNotFoundError:
            return ToolResult.fail(
                f"Tool not found: {tool_call.name}"
            )

        try:
            self._validator.validate_arguments(
                tool.parameters,
                tool_call.arguments,
            )
        except ToolParameterValidationError as exc:
            return ToolResult.fail(str(exc))

        # Wait for an available execution slot.
        async with self._semaphore:
            try:
                result = await asyncio.wait_for(
                    tool.execute(tool_call.arguments),
                    timeout=self._config.timeout_seconds,
                )
            except TimeoutError as exc:
                raise ToolExecutionTimeoutError(
                    (
                        f"Tool '{tool.name}' execution timed out "
                        f"after {self._config.timeout_seconds} seconds."
                    ),
                    tool_name=tool.name,
                    timeout_seconds=self._config.timeout_seconds,
                ) from exc

        if not isinstance(result, ToolResult):
            return ToolResult.fail(
                f"Tool '{tool.name}' returned an invalid result."
            )

        return result