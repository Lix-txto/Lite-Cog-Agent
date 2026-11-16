import pytest

from lite_cog_agent.llm.schemas import ToolDefinition
from lite_cog_agent.tools.base import BaseTool
from lite_cog_agent.tools.registry import (
    DuplicateToolError,
    ToolNotFoundError,
    ToolRegistry,
)
from lite_cog_agent.tools.schemas import ToolResult


class ExampleTool(BaseTool):
    """A simple tool for registry testing."""

    def __init__(
        self,
        tool_name: str = "example_tool",
    ) -> None:
        self._name = tool_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "An example tool for testing."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
            },
            "required": ["value"],
            "additionalProperties": False,
        }

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult.ok(
            output={"received": arguments["value"]}
        )


class InvalidNameTool(ExampleTool):
    """A tool with an invalid name."""

    @property
    def name(self) -> str:
        return "   "


def test_register_tool():
    registry = ToolRegistry()
    tool = ExampleTool()

    registry.register(tool)

    assert len(registry) == 1
    assert "example_tool" in registry
    assert registry.get("example_tool") is tool


def test_register_multiple_tools():
    registry = ToolRegistry()

    first = ExampleTool("first_tool")
    second = ExampleTool("second_tool")

    registry.register(first)
    registry.register(second)

    assert len(registry) == 2
    assert registry.get("first_tool") is first
    assert registry.get("second_tool") is second


def test_register_rejects_non_base_tool():
    registry = ToolRegistry()

    with pytest.raises(TypeError):
        registry.register(object())


def test_register_rejects_empty_tool_name():
    registry = ToolRegistry()

    with pytest.raises(ValueError):
        registry.register(InvalidNameTool())


def test_register_rejects_duplicate_name():
    registry = ToolRegistry()

    registry.register(ExampleTool())

    with pytest.raises(DuplicateToolError):
        registry.register(ExampleTool())


def test_get_unknown_tool():
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError):
        registry.get("unknown_tool")


def test_get_rejects_non_string_name():
    registry = ToolRegistry()

    with pytest.raises(TypeError):
        registry.get(123)


def test_unregister_existing_tool():
    registry = ToolRegistry()
    tool = ExampleTool()

    registry.register(tool)

    removed = registry.unregister("example_tool")

    assert removed is True
    assert len(registry) == 0
    assert "example_tool" not in registry


def test_unregister_unknown_tool():
    registry = ToolRegistry()

    removed = registry.unregister("unknown_tool")

    assert removed is False
    assert len(registry) == 0


def test_unregister_rejects_non_string_name():
    registry = ToolRegistry()

    with pytest.raises(TypeError):
        registry.unregister(123)


def test_list_tools_preserves_registration_order():
    registry = ToolRegistry()

    first = ExampleTool("first_tool")
    second = ExampleTool("second_tool")
    third = ExampleTool("third_tool")

    registry.register(first)
    registry.register(second)
    registry.register(third)

    tools = registry.list_tools()

    assert tools == [first, second, third]


def test_list_tools_returns_new_list():
    registry = ToolRegistry()
    tool = ExampleTool()

    registry.register(tool)

    tools = registry.list_tools()
    tools.clear()

    assert len(registry) == 1
    assert registry.get("example_tool") is tool


def test_get_definitions():
    registry = ToolRegistry()
    registry.register(ExampleTool())

    definitions = registry.get_definitions()

    assert len(definitions) == 1
    assert isinstance(definitions[0], ToolDefinition)
    assert definitions[0].name == "example_tool"
    assert definitions[0].description == (
        "An example tool for testing."
    )
    assert definitions[0].parameters["type"] == "object"


def test_get_definitions_preserves_order():
    registry = ToolRegistry()

    registry.register(ExampleTool("first_tool"))
    registry.register(ExampleTool("second_tool"))

    definitions = registry.get_definitions()

    assert [item.name for item in definitions] == [
        "first_tool",
        "second_tool",
    ]


def test_contains_unknown_tool():
    registry = ToolRegistry()

    assert "unknown_tool" not in registry


def test_contains_non_string_value():
    registry = ToolRegistry()

    assert 123 not in registry


def test_empty_registry():
    registry = ToolRegistry()

    assert len(registry) == 0
    assert registry.list_tools() == []
    assert registry.get_definitions() == []
