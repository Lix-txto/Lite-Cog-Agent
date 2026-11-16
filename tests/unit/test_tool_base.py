import pytest

from lite_cog_agent.llm.schemas import ToolDefinition
from lite_cog_agent.tools.base import BaseTool
from lite_cog_agent.tools.schemas import ToolResult


class ExampleTool(BaseTool):
    """A simple tool used for testing the BaseTool contract."""

    @property
    def name(self) -> str:
        return "example_tool"

    @property
    def description(self) -> str:
        return "An example tool for testing."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                }
            },
            "required": ["value"],
            "additionalProperties": False,
        }

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult.ok(output={"received": arguments["value"]})


def test_base_tool_metadata():
    tool = ExampleTool()

    assert tool.name == "example_tool"
    assert tool.description == "An example tool for testing."
    assert tool.parameters["type"] == "object"


def test_base_tool_to_definition():
    tool = ExampleTool()

    definition = tool.to_definition()

    assert isinstance(definition, ToolDefinition)
    assert definition.name == "example_tool"
    assert definition.description == "An example tool for testing."
    assert definition.parameters["type"] == "object"


@pytest.mark.asyncio
async def test_base_tool_execute():
    tool = ExampleTool()

    result = await tool.execute({"value": "hello"})

    assert result.success is True
    assert result.output == {"received": "hello"}


def test_base_tool_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseTool()


def test_incomplete_tool_cannot_be_instantiated():
    class IncompleteTool(BaseTool):
        pass

    with pytest.raises(TypeError):
        IncompleteTool()
