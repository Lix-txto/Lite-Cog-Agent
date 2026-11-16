"""
JSON Schema-based validation for tool definitions and arguments.

This module validates tool parameter schemas and model-generated
arguments before the tool execution stage.
"""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from lite_cog_agent.tools.exceptions import (
    ToolParameterValidationError,
    ToolSchemaValidationError,
)


class ToolParameterValidator:
    """Validate tool schemas and tool-call arguments."""

    @staticmethod
    def validate_schema(schema: dict[str, Any]) -> None:
        """
        Validate the structure of a tool parameter JSON Schema.

        Tool parameter schemas must be JSON objects at the root.
        """

        if not isinstance(schema, dict):
            raise ToolSchemaValidationError(
                "Tool parameter schema must be a dictionary."
            )

        if schema.get("type") != "object":
            raise ToolSchemaValidationError(
                "Tool parameter schema root type must be 'object'."
            )

        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            raise ToolSchemaValidationError(
                f"Invalid tool parameter schema: {exc.message}"
            ) from exc

    @classmethod
    def validate_arguments(
        cls,
        schema: dict[str, Any],
        arguments: dict[str, Any],
    ) -> None:
        """
        Validate model-generated arguments against a tool schema.

        Raises:
            ToolSchemaValidationError:
                If the tool schema itself is invalid.
            ToolParameterValidationError:
                If the supplied arguments are invalid.
        """

        cls.validate_schema(schema)

        if not isinstance(arguments, dict):
            raise ToolParameterValidationError(
                "Tool arguments must be a dictionary."
            )

        validator = Draft202012Validator(schema)

        errors = sorted(
            validator.iter_errors(arguments),
            key=lambda error: (
                list(str(part) for part in error.absolute_path),
                error.message,
            ),
        )

        if errors:
            details = []

            for error in errors:
                path = ".".join(
                    str(part) for part in error.absolute_path
                )

                location = path or "<root>"
                details.append(f"{location}: {error.message}")

            raise ToolParameterValidationError(
                "Invalid tool arguments: " + "; ".join(details)
            )

    @classmethod
    def validate_tool(cls, tool: Any) -> None:
        """
        Validate the parameter schema exposed by a BaseTool-like object.

        This method intentionally validates tool metadata without
        executing the tool.
        """

        parameters = getattr(tool, "parameters", None)
        cls.validate_schema(parameters)
