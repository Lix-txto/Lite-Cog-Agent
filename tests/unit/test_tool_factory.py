from __future__ import annotations

import pytest

from lite_cog_agent.core.tool_calling import ToolCallingService
from lite_cog_agent.core.tool_calling_config import ToolCallingConfig
from lite_cog_agent.llm.gateway import LLMGateway
from lite_cog_agent.llm.schemas import LLMRequest, LLMResponse
from lite_cog_agent.tools.base import BaseTool
from lite_cog_agent.tools.dispatcher import ToolDispatcher
from lite_cog_agent.tools.execution_config import ToolExecutionConfig
from lite_cog_agent.tools.factory import (
    ToolRuntime,
    create_tool_runtime,
)
from lite_cog_agent.tools.registry import (
    DuplicateToolError,
    ToolRegistry,
)
from lite_cog_agent.tools.schemas import ToolResult


class StubProvider:
    """Provider stub for composition tests."""

    async def chat_completion(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        raise NotImplementedError("This provider must not be called in these tests.")

    async def close(self) -> None:
        return None


class ExampleTool(BaseTool):
    """Minimal tool implementation for factory tests."""

    def __init__(self, tool_name: str) -> None:
        self._name = tool_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Test tool: {self._name}"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult.ok(output={})


@pytest.fixture
def gateway() -> LLMGateway:
    """Create a gateway without initializing a real provider."""
    return LLMGateway(provider=StubProvider())


@pytest.fixture
def tools() -> tuple[ExampleTool, ...]:
    return (
        ExampleTool("first_tool"),
        ExampleTool("second_tool"),
        ExampleTool("third_tool"),
    )


def test_factory_creates_tool_runtime(gateway, tools):
    runtime = create_tool_runtime(gateway, tools)

    assert isinstance(runtime, ToolRuntime)
    assert isinstance(runtime.registry, ToolRegistry)
    assert isinstance(runtime.dispatcher, ToolDispatcher)
    assert isinstance(runtime.tool_calling_service, ToolCallingService)


def test_factory_preserves_tool_instances(gateway, tools):
    runtime = create_tool_runtime(gateway, tools)

    assert runtime.tools == tools

    for actual, expected in zip(runtime.tools, tools, strict=True):
        assert actual is expected


def test_factory_registers_all_tools(gateway, tools):
    runtime = create_tool_runtime(gateway, tools)

    assert len(runtime.registry) == len(tools)
    assert len(runtime.registry.list_tools()) == len(tools)


def test_factory_preserves_registration_order(gateway, tools):
    runtime = create_tool_runtime(gateway, tools)

    registered_tools = runtime.registry.list_tools()

    assert registered_tools == list(tools)
    assert [tool.name for tool in registered_tools] == [
        "first_tool",
        "second_tool",
        "third_tool",
    ]


def test_factory_dispatcher_uses_same_registry(gateway, tools):
    runtime = create_tool_runtime(gateway, tools)

    assert runtime.dispatcher._registry is runtime.registry


def test_factory_tool_calling_service_uses_same_registry(gateway, tools):
    runtime = create_tool_runtime(gateway, tools)

    assert runtime.tool_calling_service._registry is runtime.registry


def test_factory_tool_calling_service_uses_same_dispatcher(
    gateway,
    tools,
):
    runtime = create_tool_runtime(gateway, tools)

    assert runtime.tool_calling_service._dispatcher is runtime.dispatcher


def test_factory_tool_calling_service_uses_same_gateway(gateway, tools):
    runtime = create_tool_runtime(gateway, tools)

    assert runtime.tool_calling_service._gateway is gateway


def test_factory_uses_single_tool_call_per_round_by_default(
    gateway,
    tools,
):
    runtime = create_tool_runtime(gateway, tools)

    config = runtime.tool_calling_service._config

    assert config.max_tool_calls_per_round == 1


def test_factory_uses_default_max_rounds(gateway, tools):
    runtime = create_tool_runtime(gateway, tools)

    config = runtime.tool_calling_service._config

    assert config.max_rounds == ToolCallingConfig().max_rounds


def test_factory_accepts_custom_calling_config(gateway, tools):
    config = ToolCallingConfig(
        max_rounds=5,
        max_tool_calls_per_round=3,
    )

    runtime = create_tool_runtime(
        gateway,
        tools,
        calling_config=config,
    )

    assert runtime.tool_calling_service._config is config


def test_factory_accepts_custom_execution_config(gateway, tools):
    config = ToolExecutionConfig(
        timeout_seconds=12.0,
        max_concurrency=2,
    )

    runtime = create_tool_runtime(
        gateway,
        tools,
        execution_config=config,
    )

    assert runtime.dispatcher._config is config


def test_factory_rejects_invalid_gateway(tools):
    with pytest.raises(TypeError, match="gateway"):
        create_tool_runtime(object(), tools)


def test_factory_rejects_none_gateway(tools):
    with pytest.raises(TypeError, match="gateway"):
        create_tool_runtime(None, tools)


@pytest.mark.parametrize(
    "invalid_tools",
    [
        None,
        "not-a-tool-list",
        b"not-a-tool-list",
        123,
        object(),
    ],
)
def test_factory_rejects_invalid_tools(gateway, invalid_tools):
    with pytest.raises(TypeError, match="tools"):
        create_tool_runtime(gateway, invalid_tools)


def test_factory_rejects_empty_tools(gateway):
    with pytest.raises(ValueError, match="must not be empty"):
        create_tool_runtime(gateway, [])


def test_factory_rejects_non_base_tool_items(gateway):
    with pytest.raises(TypeError, match="BaseTool"):
        create_tool_runtime(gateway, [object()])


def test_factory_rejects_duplicate_tool_names(gateway):
    tools = [
        ExampleTool("duplicate"),
        ExampleTool("duplicate"),
    ]

    with pytest.raises(DuplicateToolError):
        create_tool_runtime(gateway, tools)


@pytest.mark.parametrize(
    "config",
    [
        object(),
        None,
    ],
)
def test_factory_rejects_invalid_execution_config_type(
    gateway,
    tools,
    config,
):
    if config is None:
        return

    with pytest.raises(TypeError, match="execution_config"):
        create_tool_runtime(
            gateway,
            tools,
            execution_config=config,
        )


def test_factory_rejects_invalid_calling_config_type(gateway, tools):
    with pytest.raises(TypeError, match="calling_config"):
        create_tool_runtime(
            gateway,
            tools,
            calling_config=object(),
        )


def test_runtime_is_frozen(gateway, tools):
    runtime = create_tool_runtime(gateway, tools)

    with pytest.raises(AttributeError):
        runtime.tools = ()


def test_runtime_tools_are_immutable_tuple(gateway, tools):
    runtime = create_tool_runtime(gateway, tools)

    assert isinstance(runtime.tools, tuple)

    with pytest.raises(AttributeError):
        runtime.tools.append(ExampleTool("fourth_tool"))


def test_factory_does_not_close_gateway(gateway, tools):
    runtime = create_tool_runtime(gateway, tools)

    assert runtime.tool_calling_service._gateway is gateway
    assert gateway._closed is False
