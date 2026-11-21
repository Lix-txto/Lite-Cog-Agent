"""Unit tests for the complete Agent composition factory."""

from __future__ import annotations

from pathlib import Path

import pytest

from lite_cog_agent.config.settings import (
    AGENT_MAX_ITERATIONS,
    AGENT_MAX_TOOL_CALLS,
)
from lite_cog_agent.core.agent import Agent
from lite_cog_agent.core.agent_budget import AgentBudget
from lite_cog_agent.core.agent_factory import create_agent
from lite_cog_agent.core.agent_loop import AgentLoop
from lite_cog_agent.core.agent_state import AgentState
from lite_cog_agent.core.tool_calling_config import ToolCallingConfig
from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.llm.gateway import LLMGateway
from lite_cog_agent.llm.schemas import LLMRequest, LLMResponse
from lite_cog_agent.loading.config import DatasetLoadingConfig
from lite_cog_agent.tools.dispatcher import ToolDispatcher
from lite_cog_agent.tools.execution_config import ToolExecutionConfig
from lite_cog_agent.tools.registry import ToolRegistry

EXPECTED_TOOL_NAMES = (
    "dataset.load",
    "dataset.clean",
    "analysis.profile",
    "analysis.descriptive",
    "analysis.frequency",
    "analysis.correlation",
    "analysis.grouped",
    "analysis.derived_metric",
    "analysis.distribution",
    "visualization.bar",
    "visualization.grouped_bar",
    "visualization.line",
    "visualization.histogram",
    "visualization.scatter",
    "visualization.pie",
    "visualization.box",
)


class StubProvider:
    """Provider stub that must never be called by factory tests."""

    async def chat_completion(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        raise AssertionError(
            "Agent Factory tests must not invoke the LLM provider."
        )

    async def close(self) -> None:
        return None


@pytest.fixture
def gateway() -> LLMGateway:
    return LLMGateway(provider=StubProvider())


@pytest.fixture
def context():
    with ExecutionContext() as execution_context:
        yield execution_context


def build_agent(
    gateway: LLMGateway,
    context: ExecutionContext,
    tmp_path: Path,
    **overrides,
) -> Agent:
    """Create an Agent with isolated test storage directories."""

    arguments = {
        "context": context,
        "dataset_root": tmp_path / "datasets",
        "artifact_root": tmp_path / "artifacts",
    }
    arguments.update(overrides)

    return create_agent(
        gateway,
        **arguments,
    )


# ---------------------------------------------------------------------------
# Normal composition
# ---------------------------------------------------------------------------


def test_factory_creates_agent(
    gateway: LLMGateway,
    context: ExecutionContext,
    tmp_path: Path,
) -> None:
    agent = build_agent(gateway, context, tmp_path)

    assert isinstance(agent, Agent)
    assert isinstance(agent._loop, AgentLoop)
    assert isinstance(agent._loop._state, AgentState)
    assert isinstance(agent.session_id, str)
    assert agent.session_id


def test_factory_creates_all_builtin_tools(
    gateway: LLMGateway,
    context: ExecutionContext,
    tmp_path: Path,
) -> None:
    agent = build_agent(gateway, context, tmp_path)

    tool_calling_service = agent._loop._tool_calling_service
    registry = tool_calling_service._registry

    assert isinstance(registry, ToolRegistry)
    assert len(registry) == 16

    assert tuple(
        tool.name for tool in registry.list_tools()
    ) == EXPECTED_TOOL_NAMES


def test_factory_connects_tool_runtime_dependencies(
    gateway: LLMGateway,
    context: ExecutionContext,
    tmp_path: Path,
) -> None:
    agent = build_agent(gateway, context, tmp_path)

    tool_calling_service = agent._loop._tool_calling_service
    registry = tool_calling_service._registry
    dispatcher = tool_calling_service._dispatcher

    assert isinstance(dispatcher, ToolDispatcher)

    assert dispatcher._registry is registry
    assert tool_calling_service._gateway is gateway


def test_factory_uses_single_tool_call_per_round_by_default(
    gateway: LLMGateway,
    context: ExecutionContext,
    tmp_path: Path,
) -> None:
    agent = build_agent(gateway, context, tmp_path)

    config = agent._loop._tool_calling_service._config

    assert config.max_tool_calls_per_round == 1


def test_factory_uses_application_default_budget(
    gateway: LLMGateway,
    context: ExecutionContext,
    tmp_path: Path,
) -> None:
    agent = build_agent(gateway, context, tmp_path)

    budget = agent._loop._budget

    assert budget == AgentBudget(
        max_iterations=AGENT_MAX_ITERATIONS,
        max_tool_calls=AGENT_MAX_TOOL_CALLS,
    )


def test_factory_generates_session_id_when_omitted(
    gateway: LLMGateway,
    context: ExecutionContext,
    tmp_path: Path,
) -> None:
    agent = build_agent(gateway, context, tmp_path)

    assert agent.session_id
    assert isinstance(agent.session_id, str)


# ---------------------------------------------------------------------------
# Custom configuration
# ---------------------------------------------------------------------------


def test_factory_accepts_custom_configuration(
    gateway: LLMGateway,
    context: ExecutionContext,
    tmp_path: Path,
) -> None:
    budget = AgentBudget(
        max_iterations=4,
        max_tool_calls=7,
    )

    execution_config = ToolExecutionConfig(
        timeout_seconds=12.0,
        max_concurrency=2,
    )

    calling_config = ToolCallingConfig(
        max_rounds=6,
        max_tool_calls_per_round=2,
    )

    loading_config = DatasetLoadingConfig(
        max_file_size_bytes=10 * 1024 * 1024,
        max_rows=5000,
        max_columns=100,
    )

    agent = build_agent(
        gateway,
        context,
        tmp_path,
        session_id="custom-session",
        budget=budget,
        execution_config=execution_config,
        calling_config=calling_config,
        system_prompt="You are a data analysis agent.",
        loading_config=loading_config,
    )

    loop = agent._loop
    tool_calling_service = loop._tool_calling_service

    assert agent.session_id == "custom-session"
    assert loop._budget is budget
    assert loop._system_prompt == "You are a data analysis agent."

    assert tool_calling_service._config is calling_config
    assert tool_calling_service._dispatcher._config is execution_config


def test_factory_accepts_custom_artifact_root(
    gateway: LLMGateway,
    context: ExecutionContext,
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "custom-artifacts"

    build_agent(
        gateway,
        context,
        tmp_path,
        artifact_root=artifact_root,
    )

    assert artifact_root.exists()
    assert artifact_root.is_dir()


# ---------------------------------------------------------------------------
# Dependency and resource ownership
# ---------------------------------------------------------------------------


def test_factory_reuses_caller_execution_context(
    gateway: LLMGateway,
    context: ExecutionContext,
    tmp_path: Path,
) -> None:
    agent = build_agent(gateway, context, tmp_path)

    tools = agent._loop._tool_calling_service._registry.list_tools()

    assert all(
        getattr(tool, "_context", None) is context
        for tool in tools
    )


def test_factory_does_not_close_execution_context(
    gateway: LLMGateway,
    context: ExecutionContext,
    tmp_path: Path,
) -> None:
    build_agent(gateway, context, tmp_path)

    assert context.is_closed is False


def test_factory_does_not_close_gateway(
    gateway: LLMGateway,
    context: ExecutionContext,
    tmp_path: Path,
) -> None:
    build_agent(gateway, context, tmp_path)

    assert gateway._closed is False


def test_factory_rejects_closed_execution_context(
    gateway: LLMGateway,
    context: ExecutionContext,
    tmp_path: Path,
) -> None:
    context.close()

    with pytest.raises(
        RuntimeError,
        match="closed ExecutionContext",
    ):
        build_agent(gateway, context, tmp_path)


# ---------------------------------------------------------------------------
# Invalid dependency types
# ---------------------------------------------------------------------------


def test_factory_rejects_invalid_gateway(
    context: ExecutionContext,
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="gateway"):
        build_agent(
            object(),
            context,
            tmp_path,
        )


def test_factory_rejects_invalid_context(
    gateway: LLMGateway,
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="context"):
        create_agent(
            gateway,
            context=object(),
            dataset_root=tmp_path / "datasets",
        )


@pytest.mark.parametrize(
    "dataset_root",
    [
        None,
        123,
        object(),
    ],
)
def test_factory_rejects_invalid_dataset_root_type(
    gateway: LLMGateway,
    context: ExecutionContext,
    dataset_root: object,
) -> None:
    with pytest.raises(TypeError, match="dataset_root"):
        create_agent(
            gateway,
            context=context,
            dataset_root=dataset_root,
        )


@pytest.mark.parametrize(
    "dataset_root",
    [
        "",
        "   ",
    ],
)
def test_factory_rejects_empty_dataset_root(
    gateway: LLMGateway,
    context: ExecutionContext,
    dataset_root: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="dataset_root must not be empty",
    ):
        create_agent(
            gateway,
            context=context,
            dataset_root=dataset_root,
        )


@pytest.mark.parametrize(
    "session_id",
    [
        None,
        "",
        "   ",
    ],
)
def test_factory_handles_optional_or_empty_session_id(
    gateway: LLMGateway,
    context: ExecutionContext,
    tmp_path: Path,
    session_id: str | None,
) -> None:
    if session_id is None:
        agent = build_agent(
            gateway,
            context,
            tmp_path,
            session_id=None,
        )
        assert agent.session_id
        return

    with pytest.raises(ValueError, match="session_id"):
        build_agent(
            gateway,
            context,
            tmp_path,
            session_id=session_id,
        )


def test_factory_rejects_invalid_session_id_type(
    gateway: LLMGateway,
    context: ExecutionContext,
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="session_id"):
        build_agent(
            gateway,
            context,
            tmp_path,
            session_id=123,
        )


def test_factory_rejects_invalid_budget(
    gateway: LLMGateway,
    context: ExecutionContext,
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="budget"):
        build_agent(
            gateway,
            context,
            tmp_path,
            budget=object(),
        )


def test_factory_rejects_invalid_execution_config(
    gateway: LLMGateway,
    context: ExecutionContext,
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="execution_config"):
        build_agent(
            gateway,
            context,
            tmp_path,
            execution_config=object(),
        )


def test_factory_rejects_invalid_calling_config(
    gateway: LLMGateway,
    context: ExecutionContext,
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="calling_config"):
        build_agent(
            gateway,
            context,
            tmp_path,
            calling_config=object(),
        )


@pytest.mark.parametrize(
    "system_prompt",
    [
        "",
        "   ",
        None,
        123,
    ],
)
def test_factory_rejects_invalid_system_prompt(
    gateway: LLMGateway,
    context: ExecutionContext,
    tmp_path: Path,
    system_prompt: object,
) -> None:
    with pytest.raises(ValueError, match="system_prompt"):
        build_agent(
            gateway,
            context,
            tmp_path,
            system_prompt=system_prompt,
        )


@pytest.mark.parametrize(
    "artifact_root",
    [
        None,
        "",
        "   ",
    ],
)
def test_factory_handles_optional_or_empty_artifact_root(
    gateway: LLMGateway,
    context: ExecutionContext,
    tmp_path: Path,
    artifact_root: str | None,
) -> None:
    if artifact_root is None:
        build_agent(
            gateway,
            context,
            tmp_path,
            artifact_root=None,
        )
        return

    with pytest.raises(ValueError, match="artifact_root"):
        build_agent(
            gateway,
            context,
            tmp_path,
            artifact_root=artifact_root,
        )


def test_factory_rejects_invalid_artifact_root_type(
    gateway: LLMGateway,
    context: ExecutionContext,
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="artifact_root"):
        build_agent(
            gateway,
            context,
            tmp_path,
            artifact_root=123,
        )


def test_factory_rejects_invalid_loading_config(
    gateway: LLMGateway,
    context: ExecutionContext,
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="loading_config"):
        build_agent(
            gateway,
            context,
            tmp_path,
            loading_config=object(),
        )
