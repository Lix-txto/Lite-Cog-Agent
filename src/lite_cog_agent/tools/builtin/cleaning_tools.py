"""Builtin tools for dataset cleaning workflows."""

from __future__ import annotations

import asyncio
from typing import Any

from lite_cog_agent.cleaning.contracts import (
    CleaningRequest,
    CleaningStatus,
    CleaningStep,
)
from lite_cog_agent.cleaning.service import CleaningService
from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.tools.base import BaseTool
from lite_cog_agent.tools.builtin.common import to_json_compatible
from lite_cog_agent.tools.schemas import ToolResult


class DatasetCleaningTool(BaseTool):
    """Clean a dataset through the existing CleaningService."""

    _ALLOWED_OPERATIONS = (
        "missing.drop",
        "missing.fill",
        "duplicates.drop",
        "columns.select",
        "columns.drop",
        "columns.rename",
        "types.convert",
    )

    _ALLOWED_ARGUMENTS = {
        "dataset_id",
        "steps",
        "output_dataset_name",
    }

    def __init__(
        self,
        context: ExecutionContext,
        cleaning_service: CleaningService,
    ) -> None:
        if not isinstance(context, ExecutionContext):
            raise TypeError("context must be an ExecutionContext")

        if context.is_closed:
            raise RuntimeError("ExecutionContext is closed")

        if not isinstance(cleaning_service, CleaningService):
            raise TypeError(
                "cleaning_service must be a CleaningService"
            )

        if cleaning_service.context is not context:
            raise ValueError(
                "cleaning_service must use the same ExecutionContext"
            )

        self._context = context
        self._cleaning_service = cleaning_service

    @property
    def name(self) -> str:
        return "dataset.clean"

    @property
    def description(self) -> str:
        return (
            "Clean a dataset by executing an ordered sequence of "
            "registered cleaning operations. Cleaning creates a new "
            "dataset. Use the returned output_dataset_id for subsequent "
            "analysis instead of the original dataset_id."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "ID of a dataset already loaded in the current "
                        "execution context."
                    ),
                },
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "description": (
                        "Ordered cleaning operations. Each operation "
                        "runs against the result of the previous step."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": list(self._ALLOWED_OPERATIONS),
                                "description": (
                                    "Registered cleaning operation name."
                                ),
                            },
                            "parameters": {
                                "type": "object",
                                "description": (
                                    "Parameters for this cleaning operation. "
                                    "Operation-specific validation is handled "
                                    "by CleaningService."
                                ),
                                "additionalProperties": True,
                            },
                        },
                        "required": ["operation", "parameters"],
                        "additionalProperties": False,
                    },
                },
                "output_dataset_name": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Optional name for the cleaned output dataset."
                    ),
                },
            },
            "required": ["dataset_id", "steps"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Execute a cleaning workflow asynchronously."""

        if not isinstance(arguments, dict):
            return ToolResult.fail(
                "Tool arguments must be an object.",
                metadata={"tool": self.name},
            )

        unknown = set(arguments) - self._ALLOWED_ARGUMENTS
        if unknown:
            return ToolResult.fail(
                f"Unexpected arguments: {sorted(unknown)}.",
                metadata={"tool": self.name},
            )

        if self._context.is_closed:
            return ToolResult.fail(
                "ExecutionContext is closed.",
                metadata={"tool": self.name},
            )

        dataset_id: str | None = None

        try:
            request = self._build_request(arguments)
            dataset_id = request.dataset_id

            result = await asyncio.to_thread(
                self._cleaning_service.clean,
                request,
            )

            if result.status is CleaningStatus.FAILED:
                serialized_result = to_json_compatible(result)
                error_message = (
                    result.error.message
                    if result.error is not None
                    else "Dataset cleaning failed."
                )

                return ToolResult.fail(
                    error=error_message,
                    metadata={
                        "tool": self.name,
                        "dataset_id": dataset_id,
                        "cleaning_result": serialized_result,
                    },
                )

            if result.status is not CleaningStatus.SUCCESS:
                return ToolResult.fail(
                    "Cleaning service returned an unsupported result status.",
                    metadata={
                        "tool": self.name,
                        "dataset_id": dataset_id,
                    },
                )

            return ToolResult.ok(
                output=to_json_compatible(result),
                metadata={
                    "tool": self.name,
                    "dataset_id": dataset_id,
                    "output_dataset_id": result.output_dataset_id,
                    "status": result.status.value,
                },
            )

        except (TypeError, ValueError) as exc:
            return ToolResult.fail(
                str(exc),
                metadata={
                    "tool": self.name,
                    "dataset_id": dataset_id,
                    "error_type": type(exc).__name__,
                },
            )

        except KeyError:
            return ToolResult.fail(
                "Dataset is not available in the current execution context.",
                metadata={
                    "tool": self.name,
                    "dataset_id": dataset_id,
                    "error_type": "KeyError",
                },
            )

        except RuntimeError as exc:
            return ToolResult.fail(
                str(exc),
                metadata={
                    "tool": self.name,
                    "dataset_id": dataset_id,
                    "error_type": type(exc).__name__,
                },
            )

        except Exception as exc:
            return ToolResult.fail(
                f"Dataset cleaning failed: {type(exc).__name__}.",
                metadata={
                    "tool": self.name,
                    "dataset_id": dataset_id,
                    "error_type": type(exc).__name__,
                },
            )

    def _build_request(
        self,
        arguments: dict[str, Any],
    ) -> CleaningRequest:
        """Validate tool arguments and build a CleaningRequest."""

        dataset_id = arguments.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise ValueError(
                "dataset_id must be a non-empty string."
            )

        raw_steps = arguments.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError(
                "steps must be a non-empty list."
            )

        steps: list[CleaningStep] = []

        for index, raw_step in enumerate(raw_steps):
            if not isinstance(raw_step, dict):
                raise ValueError(
                    f"steps[{index}] must be an object."
                )

            unknown_step_fields = (
                set(raw_step) - {"operation", "parameters"}
            )
            if unknown_step_fields:
                raise ValueError(
                    f"steps[{index}] contains unexpected fields: "
                    f"{sorted(unknown_step_fields)}."
                )

            operation = raw_step.get("operation")
            if (
                not isinstance(operation, str)
                or operation not in self._ALLOWED_OPERATIONS
            ):
                raise ValueError(
                    f"steps[{index}].operation must be one of: "
                    f"{', '.join(self._ALLOWED_OPERATIONS)}."
                )

            parameters = raw_step.get("parameters")
            if not isinstance(parameters, dict):
                raise ValueError(
                    f"steps[{index}].parameters must be an object."
                )

            steps.append(
                CleaningStep(
                    operation=operation,
                    parameters=parameters,
                )
            )

        output_dataset_name = arguments.get("output_dataset_name")
        if output_dataset_name is not None:
            if (
                not isinstance(output_dataset_name, str)
                or not output_dataset_name.strip()
            ):
                raise ValueError(
                    "output_dataset_name must be a non-empty string."
                )

        return CleaningRequest(
            dataset_id=dataset_id,
            steps=tuple(steps),
            output_dataset_name=output_dataset_name,
        )
