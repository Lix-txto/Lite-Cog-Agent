from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from lite_cog_agent.core.tool_calling import ToolCallingService
from lite_cog_agent.core.tool_calling_config import ToolCallingConfig
from lite_cog_agent.llm.gateway import LLMGateway
from lite_cog_agent.tools.base import BaseTool
from lite_cog_agent.tools.dispatcher import ToolDispatcher
from lite_cog_agent.tools.execution_config import ToolExecutionConfig
from lite_cog_agent.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ToolRuntime:
    """Composed tool execution components."""

    tools: tuple[BaseTool, ...]
    registry: ToolRegistry
    dispatcher: ToolDispatcher
    tool_calling_service: ToolCallingService


def create_tool_runtime(
    gateway: LLMGateway,
    tools: Sequence[BaseTool],
    *,
    execution_config: ToolExecutionConfig | None = None,
    calling_config: ToolCallingConfig | None = None,
) -> ToolRuntime:
    """Compose tools, registry, dispatcher and tool-calling service.

    The caller owns the lifecycle of the gateway and all external resources.

    By default, each tool-calling round is limited to one tool call.
    This keeps dependent data-analysis operations sequential.
    """
    if not isinstance(gateway, LLMGateway):
        raise TypeError("gateway must be an LLMGateway instance.")

    if isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence):
        raise TypeError("tools must be a sequence of BaseTool instances.")

    tool_snapshot = tuple(tools)

    if not tool_snapshot:
        raise ValueError("tools must not be empty.")

    if any(not isinstance(tool, BaseTool) for tool in tool_snapshot):
        raise TypeError("every item in tools must be a BaseTool instance.")

    if execution_config is not None and not isinstance(
        execution_config, ToolExecutionConfig
    ):
        raise TypeError(
            "execution_config must be a ToolExecutionConfig instance."
        )

    if calling_config is not None and not isinstance(
        calling_config, ToolCallingConfig
    ):
        raise TypeError(
            "calling_config must be a ToolCallingConfig instance."
        )

    registry = ToolRegistry()

    for tool in tool_snapshot:
        registry.register(tool)

    dispatcher = ToolDispatcher(
        registry,
        config=execution_config,
    )

    effective_calling_config = calling_config or ToolCallingConfig(
        max_tool_calls_per_round=1,
    )

    tool_calling_service = ToolCallingService(
        gateway=gateway,
        registry=registry,
        dispatcher=dispatcher,
        config=effective_calling_config,
    )

    return ToolRuntime(
        tools=tool_snapshot,
        registry=registry,
        dispatcher=dispatcher,
        tool_calling_service=tool_calling_service,
    )
