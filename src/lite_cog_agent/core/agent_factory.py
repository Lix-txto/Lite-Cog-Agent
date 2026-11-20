"""Factory for composing a complete Lite-Cog-Agent instance."""

from __future__ import annotations

from pathlib import Path

from lite_cog_agent.config.settings import (
    AGENT_MAX_ITERATIONS,
    AGENT_MAX_TOOL_CALLS,
)
from lite_cog_agent.core.agent import Agent
from lite_cog_agent.core.agent_budget import AgentBudget
from lite_cog_agent.core.agent_loop import AgentLoop
from lite_cog_agent.core.agent_state import AgentState
from lite_cog_agent.core.prompts import AGENT_SYSTEM_PROMPT
from lite_cog_agent.core.tool_calling_config import ToolCallingConfig
from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.llm.gateway import LLMGateway
from lite_cog_agent.loading.config import DatasetLoadingConfig
from lite_cog_agent.tools.builtin.factory import (
    create_builtin_tool_services,
    create_builtin_tools,
)
from lite_cog_agent.tools.execution_config import ToolExecutionConfig
from lite_cog_agent.tools.factory import create_tool_runtime


def create_agent(
    gateway: LLMGateway,
    *,
    context: ExecutionContext,
    dataset_root: Path | str,
    session_id: str | None = None,
    budget: AgentBudget | None = None,
    execution_config: ToolExecutionConfig | None = None,
    calling_config: ToolCallingConfig | None = None,
    system_prompt: str = AGENT_SYSTEM_PROMPT,
    artifact_root: Path | str | None = None,
    loading_config: DatasetLoadingConfig | None = None,
) -> Agent:
    """Create a fully composed Agent instance.

    This factory composes the builtin analysis services, tools,
    tool runtime, AgentState, AgentLoop, and public Agent facade.

    Resource ownership remains with the caller:
    - The caller owns the LLMGateway lifecycle.
    - The caller owns the ExecutionContext lifecycle.
    - The caller owns the configured dataset and artifact directories.

    The factory does not close or otherwise take ownership of
    externally supplied resources.

    Args:
        gateway: Configured LLM Gateway.
        context: Shared execution context for one Agent session.
        dataset_root: Allowed root directory for dataset loading.
        session_id: Optional session identifier. Generated when omitted.
        budget: Optional Agent execution budget. Defaults to application
            settings.
        execution_config: Optional tool execution configuration.
        calling_config: Optional Tool Calling configuration.
        system_prompt: System prompt used by AgentLoop.
        artifact_root: Optional visualization artifact directory.
        loading_config: Optional dataset loading configuration.

    Returns:
        A ready-to-use Agent instance.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If an argument has an invalid value.
        RuntimeError: If the supplied ExecutionContext is closed.
    """

    # Validate external dependencies before constructing services.
    if not isinstance(gateway, LLMGateway):
        raise TypeError("gateway must be an LLMGateway instance.")

    if not isinstance(context, ExecutionContext):
        raise TypeError("context must be an ExecutionContext instance.")

    if context.is_closed:
        raise RuntimeError(
            "Cannot create an Agent with a closed ExecutionContext."
        )

    if not isinstance(dataset_root, (Path, str)):
        raise TypeError("dataset_root must be a Path or string.")

    if not str(dataset_root).strip():
        raise ValueError("dataset_root must not be empty.")

    if session_id is not None:
        if not isinstance(session_id, str):
            raise TypeError("session_id must be a string or None.")

        if not session_id.strip():
            raise ValueError("session_id must not be empty.")

    if budget is not None and not isinstance(budget, AgentBudget):
        raise TypeError("budget must be an AgentBudget instance or None.")

    if (
        execution_config is not None
        and not isinstance(execution_config, ToolExecutionConfig)
    ):
        raise TypeError(
            "execution_config must be a ToolExecutionConfig instance "
            "or None."
        )

    if (
        calling_config is not None
        and not isinstance(calling_config, ToolCallingConfig)
    ):
        raise TypeError(
            "calling_config must be a ToolCallingConfig instance or None."
        )

    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt must be a non-empty string.")

    if artifact_root is not None:
        if not isinstance(artifact_root, (Path, str)):
            raise TypeError("artifact_root must be a Path, string, or None.")

        if not str(artifact_root).strip():
            raise ValueError("artifact_root must not be empty.")

    if (
        loading_config is not None
        and not isinstance(loading_config, DatasetLoadingConfig)
    ):
        raise TypeError(
            "loading_config must be a DatasetLoadingConfig instance or None."
        )

    # Resolve the Agent budget from explicit input or application settings.
    effective_budget = (
        budget
        if budget is not None
        else AgentBudget(
            max_iterations=AGENT_MAX_ITERATIONS,
            max_tool_calls=AGENT_MAX_TOOL_CALLS,
        )
    )

    # 1. Construct shared builtin service dependencies.
    builtin_services = create_builtin_tool_services(
        context,
        loading_config=loading_config,
        artifact_root=artifact_root,
    )

    # 2. Construct builtin tools using the shared service bundle.
    builtin_tools = create_builtin_tools(
        builtin_services,
        dataset_root=dataset_root,
    )

    # 3. Compose registry, dispatcher, and ToolCallingService.
    tool_runtime = create_tool_runtime(
        gateway,
        builtin_tools,
        execution_config=execution_config,
        calling_config=calling_config,
    )

    # 4. Create the state for this Agent session.
    state = AgentState(
        session_id=session_id,
    )

    # 5. Compose the Agent request lifecycle.
    agent_loop = AgentLoop(
        tool_calling_service=tool_runtime.tool_calling_service,
        state=state,
        system_prompt=system_prompt,
        budget=effective_budget,
    )

    # 6. Return the public Agent facade.
    return Agent(agent_loop)
