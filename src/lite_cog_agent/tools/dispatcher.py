from __future__ import annotations

from lite_cog_agent.llm.schemas import ToolCall
from lite_cog_agent.tools.base import BaseTool
from lite_cog_agent.tools.exceptions import (
    ToolParameterValidationError,
)
from lite_cog_agent.tools.parameter_validator import (
    ToolParameterValidator,
)
from lite_cog_agent.tools.registry import ToolRegistry, ToolNotFoundError
from lite_cog_agent.tools.schemas import ToolResult


class ToolDispatcher:
    """Dispatch LLM tool calls to registered tools."""

    def __init__(
        self,
        registry: ToolRegistry,
        validator: ToolParameterValidator | None = None,
    ) -> None:
        self._registry = registry
        self._validator = validator or ToolParameterValidator()


    async def dispatch(
        self,
        tool_call: ToolCall,
    ) -> ToolResult:
        """Find, validate, and execute a tool call."""

        # 1. Validate the ToolCall object.
        if not isinstance(tool_call, ToolCall):
            return ToolResult.fail(
                "Invalid tool call: expected a ToolCall instance."
            )

        # 2. Find the registered tool.
        try:
            tool = self._registry.get(tool_call.name)
        except ToolNotFoundError:
            return ToolResult.fail(
                f"Tool not found: {tool_call.name}"
            )

        # 3. Validate tool arguments.
        try:
            self._validator.validate_arguments(
                tool.parameters,
                tool_call.arguments,
            )
        except ToolParameterValidationError as exc:
            return ToolResult.fail(str(exc))

        # 4. Execute the tool.
        result = await tool.execute(tool_call.arguments)

        # 5. Verify the execution result.
        if not isinstance(result, ToolResult):
            return ToolResult.fail(
                f"Tool '{tool.name}' returned an invalid result."
            )

        return result

