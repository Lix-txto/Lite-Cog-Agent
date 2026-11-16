import pytest

from lite_cog_agent.tools.base import BaseTool
from lite_cog_agent.tools.exceptions import (
    ToolParameterValidationError,
    ToolSchemaValidationError,
)
from lite_cog_agent.tools.parameter_validator import (
    ToolParameterValidator,
)
from lite_cog_agent.tools.schemas import ToolResult


VALID_SCHEMA = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
        },
        "include_summary": {
            "type": "boolean",
        },
    },
    "required": ["file_path"],
    "additionalProperties": False,
}


def test_validate_valid_schema():
    ToolParameterValidator.validate_schema(VALID_SCHEMA)


def test_validate_rejects_non_dictionary_schema():
    with pytest.raises(ToolSchemaValidationError):
        ToolParameterValidator.validate_schema("invalid")


def test_validate_rejects_non_object_root_schema():
    with pytest.raises(ToolSchemaValidationError):
        ToolParameterValidator.validate_schema(
            {"type": "string"}
        )


def test_validate_rejects_invalid_json_schema():
    invalid_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "invalid_type",
            }
        },
    }

    with pytest.raises(ToolSchemaValidationError):
        ToolParameterValidator.validate_schema(invalid_schema)


def test_validate_valid_arguments():
    arguments = {
        "file_path": "data/sales.csv",
        "limit": 100,
        "include_summary": True,
    }

    ToolParameterValidator.validate_arguments(
        VALID_SCHEMA,
        arguments,
    )


def test_validate_arguments_with_only_required_field():
    arguments = {
        "file_path": "data/sales.csv",
    }

    ToolParameterValidator.validate_arguments(
        VALID_SCHEMA,
        arguments,
    )


def test_validate_rejects_non_dictionary_arguments():
    with pytest.raises(ToolParameterValidationError):
        ToolParameterValidator.validate_arguments(
            VALID_SCHEMA,
            "invalid",
        )


def test_validate_rejects_missing_required_field():
    with pytest.raises(ToolParameterValidationError):
        ToolParameterValidator.validate_arguments(
            VALID_SCHEMA,
            {"limit": 100},
        )


def test_validate_rejects_wrong_string_type():
    with pytest.raises(ToolParameterValidationError):
        ToolParameterValidator.validate_arguments(
            VALID_SCHEMA,
            {"file_path": 123},
        )


def test_validate_rejects_wrong_integer_type():
    with pytest.raises(ToolParameterValidationError):
        ToolParameterValidator.validate_arguments(
            VALID_SCHEMA,
            {
                "file_path": "data/sales.csv",
                "limit": "100",
            },
        )


def test_validate_rejects_boolean_as_integer():
    with pytest.raises(ToolParameterValidationError):
        ToolParameterValidator.validate_arguments(
            VALID_SCHEMA,
            {
                "file_path": "data/sales.csv",
                "limit": True,
            },
        )


def test_validate_rejects_additional_properties():
    with pytest.raises(ToolParameterValidationError):
        ToolParameterValidator.validate_arguments(
            VALID_SCHEMA,
            {
                "file_path": "data/sales.csv",
                "unknown_field": "unexpected",
            },
        )


def test_validate_rejects_value_below_minimum():
    with pytest.raises(ToolParameterValidationError):
        ToolParameterValidator.validate_arguments(
            VALID_SCHEMA,
            {
                "file_path": "data/sales.csv",
                "limit": 0,
            },
        )


def test_validate_tool_schema():
    class ExampleTool(BaseTool):
        @property
        def name(self):
            return "example_tool"

        @property
        def description(self):
            return "An example tool."

        @property
        def parameters(self):
            return VALID_SCHEMA

        async def execute(self, arguments):
            return ToolResult.ok(output=arguments)

    ToolParameterValidator.validate_tool(ExampleTool())


def test_validate_tool_rejects_invalid_schema():
    class InvalidTool(BaseTool):
        @property
        def name(self):
            return "invalid_tool"

        @property
        def description(self):
            return "A tool with an invalid schema."

        @property
        def parameters(self):
            return {"type": "string"}

        async def execute(self, arguments):
            return ToolResult.ok(output=arguments)

    with pytest.raises(ToolSchemaValidationError):
        ToolParameterValidator.validate_tool(InvalidTool())
