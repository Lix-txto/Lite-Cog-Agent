"""Builtin tools for creating and storing dataset visualizations."""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import Any

from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.storage.file_manager import (
    FileManager,
    FileManagerError,
)
from lite_cog_agent.tools.base import BaseTool, ToolResult
from lite_cog_agent.tools.builtin.common import to_json_compatible
from lite_cog_agent.visualization.exceptions import VisualizationException
from lite_cog_agent.visualization.outputs import VisualizationOutput
from lite_cog_agent.visualization.renderer import OutputFormat
from lite_cog_agent.visualization.service import VisualizationService


def _validate_arguments(
    arguments: dict[str, Any],
) -> ToolResult | None:
    if not isinstance(arguments, dict):
        return ToolResult.fail("Tool arguments must be an object.")

    return None


def _required_string(
    arguments: dict[str, Any],
    name: str,
) -> str:
    value = arguments.get(name)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")

    return value


def _optional_string(
    arguments: dict[str, Any],
    name: str,
) -> str | None:
    value = arguments.get(name)

    if value is not None and (
        not isinstance(value, str) or not value.strip()
    ):
        raise ValueError(f"{name} must be a non-empty string or null.")

    return value


def _optional_positive_integer(
    arguments: dict[str, Any],
    name: str,
    default: int | None = None,
) -> int | None:
    value = arguments.get(name, default)

    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer or null.")

    return value


def _get_output_format(
    arguments: dict[str, Any],
) -> OutputFormat:
    value = arguments.get("output_format", "png")

    if not isinstance(value, str):
        raise ValueError("output_format must be 'png' or 'svg'.")

    try:
        return OutputFormat(value.lower())
    except ValueError as exc:
        raise ValueError(
            "output_format must be 'png' or 'svg'."
        ) from exc


def _base_properties() -> dict[str, Any]:
    return {
        "dataset_id": {
            "type": "string",
            "minLength": 1,
            "description": "ID of the dataset to visualize.",
        },
        "output_format": {
            "type": "string",
            "enum": ["png", "svg"],
            "default": "png",
        },
        "title": {
            "type": ["string", "null"],
            "description": "Optional chart title.",
        },
    }


def _build_schema(
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


class _BaseVisualizationTool(BaseTool):
    """Shared execution and artifact-persistence behavior."""

    def __init__(
        self,
        context: ExecutionContext,
        visualization_service: VisualizationService,
        file_manager: FileManager,
    ) -> None:
        if not isinstance(context, ExecutionContext):
            raise TypeError("context must be an ExecutionContext")

        if context.is_closed:
            raise ValueError("context is already closed")

        if not isinstance(
            visualization_service,
            VisualizationService,
        ):
            raise TypeError(
                "visualization_service must be a VisualizationService"
            )

        if not isinstance(file_manager, FileManager):
            raise TypeError("file_manager must be a FileManager")

        self._context = context
        self._visualization_service = visualization_service
        self._file_manager = file_manager

    @property
    def description(self) -> str:
        return "Create a chart from a registered dataset and store it as an artifact."

    @property
    def parameters(self) -> dict[str, Any]:
        properties = _base_properties()
        properties.update(self._specific_properties())

        return _build_schema(
            properties,
            ["dataset_id", *self._specific_required()],
        )

    @abstractmethod
    def _specific_properties(self) -> dict[str, Any]:
        """Return chart-specific JSON Schema properties."""

    @abstractmethod
    def _specific_required(self) -> list[str]:
        """Return chart-specific required parameters."""

    @abstractmethod
    def _create_chart(
        self,
        dataset_id: str,
        arguments: dict[str, Any],
        output_format: OutputFormat,
        title: str | None,
    ) -> VisualizationOutput:
        """Create a chart using the visualization service."""

    def _create_and_store(
        self,
        dataset_id: str,
        arguments: dict[str, Any],
        output_format: OutputFormat,
        title: str | None,
    ) -> tuple[VisualizationOutput, Any]:
        """Create a chart and persist its rendered bytes."""

        output = self._create_chart(
            dataset_id,
            arguments,
            output_format,
            title,
        )

        if not isinstance(output, VisualizationOutput):
            raise TypeError(
                "VisualizationService returned an invalid output."
            )

        rendered = output.rendered_chart

        if rendered.output_format is not output_format:
            raise ValueError(
                "Rendered chart format does not match requested format."
            )

        extension = f".{output_format.value}"

        stored_file = self._file_manager.save_bytes(
            rendered.image_data,
            extension=extension,
            media_type=rendered.mime_type,
        )

        return output, stored_file

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolResult:
        invalid = _validate_arguments(arguments)

        if invalid is not None:
            return invalid

        dataset_id: str | None = None

        try:
            if self._context.is_closed:
                raise RuntimeError("Execution context is closed.")

            dataset_id = _required_string(
                arguments,
                "dataset_id",
            )

            output_format = _get_output_format(arguments)
            title = _optional_string(arguments, "title")

            output, stored_file = await asyncio.to_thread(
                self._create_and_store,
                dataset_id,
                arguments,
                output_format,
                title,
            )

            visualization_result = to_json_compatible(
                output.result
            )

            artifact = {
                "artifact_id": stored_file.artifact_id,
                "filename": stored_file.filename,
                "media_type": stored_file.media_type,
                "size_bytes": stored_file.size_bytes,
                "format": output_format.value,
            }

            return ToolResult.ok(
                output={
                    "visualization": visualization_result,
                    "artifact": artifact,
                },
                metadata={
                    "tool": self.name,
                    "dataset_id": dataset_id,
                    "visualization_id": output.result.visualization_id,
                    "artifact_id": stored_file.artifact_id,
                },
            )

        except (
            VisualizationException,
            FileManagerError,
            ValueError,
            KeyError,
            RuntimeError,
        ) as exc:
            return ToolResult.fail(
                str(exc),
                metadata={
                    "tool": self.name,
                    "dataset_id": dataset_id,
                },
            )

        except Exception as exc:
            return ToolResult.fail(
                "Visualization failed: "
                f"{type(exc).__name__}.",
                metadata={
                    "tool": self.name,
                    "dataset_id": dataset_id,
                },
            )


class BarChartTool(_BaseVisualizationTool):
    """Create a category frequency bar chart."""

    @property
    def name(self) -> str:
        return "visualization.bar"

    @property
    def description(self) -> str:
        return (
            "Create a bar chart showing category frequencies "
            "for a dataset column."
        )

    def _specific_properties(self) -> dict[str, Any]:
        return {
            "column": {
                "type": "string",
                "minLength": 1,
            },
            "top_n": {
                "type": ["integer", "null"],
                "minimum": 1,
                "description": "Optional maximum number of categories.",
            },
        }

    def _specific_required(self) -> list[str]:
        return ["column"]

    def _create_chart(
        self,
        dataset_id: str,
        arguments: dict[str, Any],
        output_format: OutputFormat,
        title: str | None,
    ) -> VisualizationOutput:
        column = _required_string(arguments, "column")
        top_n = _optional_positive_integer(arguments, "top_n")

        return self._visualization_service.create_bar_chart(
            self._context,
            dataset_id,
            column=column,
            top_n=top_n,
            output_format=output_format,
            title=title,
        )


class GroupedBarChartTool(_BaseVisualizationTool):
    """Create a grouped aggregation bar chart."""

    @property
    def name(self) -> str:
        return "visualization.grouped_bar"

    @property
    def description(self) -> str:
        return (
            "Create a grouped bar chart from one or more grouping "
            "columns and configured numeric aggregations."
        )

    def _specific_properties(self) -> dict[str, Any]:
        return {
            "group_by": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "string",
                    "minLength": 1,
                },
                "uniqueItems": True,
            },
            "aggregations": {
                "type": "object",
                "minProperties": 1,
                "description": (
                    "Mapping of dataset columns to aggregation operations."
                ),
                "additionalProperties": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "string",
                        "enum": [
                            "count",
                            "sum",
                            "mean",
                            "median",
                            "min",
                            "max",
                            "std",
                        ],
                    },
                    "uniqueItems": True,
                },
            },
        }

    def _specific_required(self) -> list[str]:
        return ["group_by", "aggregations"]

    def _create_chart(
        self,
        dataset_id: str,
        arguments: dict[str, Any],
        output_format: OutputFormat,
        title: str | None,
    ) -> VisualizationOutput:
        group_by = arguments.get("group_by")
        aggregations = arguments.get("aggregations")

        if (
            not isinstance(group_by, list)
            or not group_by
            or any(
                not isinstance(item, str) or not item.strip()
                for item in group_by
            )
        ):
            raise ValueError(
                "group_by must be a non-empty list of column names."
            )

        if len(group_by) != len(set(group_by)):
            raise ValueError("group_by cannot contain duplicates.")

        if not isinstance(aggregations, dict) or not aggregations:
            raise ValueError("aggregations must be a non-empty object.")

        supported = {
            "count",
            "sum",
            "mean",
            "median",
            "min",
            "max",
            "std",
        }

        for column, operations in aggregations.items():
            if not isinstance(column, str) or not column.strip():
                raise ValueError(
                    "Aggregation column names must be non-empty strings."
                )

            if (
                not isinstance(operations, list)
                or not operations
                or any(
                    operation not in supported
                    for operation in operations
                )
            ):
                raise ValueError(
                    f"Invalid aggregation operations for '{column}'."
                )

            if len(operations) != len(set(operations)):
                raise ValueError(
                    f"Duplicate aggregation operations for '{column}'."
                )

        return self._visualization_service.create_grouped_bar_chart(
            self._context,
            dataset_id,
            group_by=group_by,
            aggregations=aggregations,
            output_format=output_format,
            title=title,
        )


class LineChartTool(_BaseVisualizationTool):
    """Create a line chart from two dataset columns."""

    @property
    def name(self) -> str:
        return "visualization.line"

    @property
    def description(self) -> str:
        return (
            "Create a line chart using one dataset column as X "
            "and another column as Y."
        )

    def _specific_properties(self) -> dict[str, Any]:
        return {
            "x_column": {"type": "string", "minLength": 1},
            "y_column": {"type": "string", "minLength": 1},
        }

    def _specific_required(self) -> list[str]:
        return ["x_column", "y_column"]

    def _create_chart(
        self,
        dataset_id: str,
        arguments: dict[str, Any],
        output_format: OutputFormat,
        title: str | None,
    ) -> VisualizationOutput:
        return self._visualization_service.create_line_chart(
            self._context,
            dataset_id,
            x_column=_required_string(arguments, "x_column"),
            y_column=_required_string(arguments, "y_column"),
            output_format=output_format,
            title=title,
        )


class HistogramTool(_BaseVisualizationTool):
    """Create a numeric histogram."""

    @property
    def name(self) -> str:
        return "visualization.histogram"

    @property
    def description(self) -> str:
        return (
            "Create a histogram for a numeric dataset column "
            "using configurable bin count."
        )

    def _specific_properties(self) -> dict[str, Any]:
        return {
            "column": {"type": "string", "minLength": 1},
            "bins": {
                "type": "integer",
                "minimum": 1,
                "default": 10,
            },
        }

    def _specific_required(self) -> list[str]:
        return ["column"]

    def _create_chart(
        self,
        dataset_id: str,
        arguments: dict[str, Any],
        output_format: OutputFormat,
        title: str | None,
    ) -> VisualizationOutput:
        bins = arguments.get("bins", 10)

        if (
            isinstance(bins, bool)
            or not isinstance(bins, int)
            or bins <= 0
        ):
            raise ValueError("bins must be a positive integer.")

        return self._visualization_service.create_histogram(
            self._context,
            dataset_id,
            column=_required_string(arguments, "column"),
            bins=bins,
            output_format=output_format,
            title=title,
        )


class ScatterChartTool(_BaseVisualizationTool):
    """Create a scatter chart from two numeric columns."""

    @property
    def name(self) -> str:
        return "visualization.scatter"

    @property
    def description(self) -> str:
        return (
            "Create a scatter chart showing the relationship "
            "between two numeric dataset columns."
        )

    def _specific_properties(self) -> dict[str, Any]:
        return {
            "x_column": {"type": "string", "minLength": 1},
            "y_column": {"type": "string", "minLength": 1},
        }

    def _specific_required(self) -> list[str]:
        return ["x_column", "y_column"]

    def _create_chart(
        self,
        dataset_id: str,
        arguments: dict[str, Any],
        output_format: OutputFormat,
        title: str | None,
    ) -> VisualizationOutput:
        return self._visualization_service.create_scatter_chart(
            self._context,
            dataset_id,
            x_column=_required_string(arguments, "x_column"),
            y_column=_required_string(arguments, "y_column"),
            output_format=output_format,
            title=title,
        )


class PieChartTool(_BaseVisualizationTool):
    """Create a category proportion pie chart."""

    @property
    def name(self) -> str:
        return "visualization.pie"

    @property
    def description(self) -> str:
        return (
            "Create a pie chart showing category proportions "
            "for a dataset column."
        )

    def _specific_properties(self) -> dict[str, Any]:
        return {
            "column": {"type": "string", "minLength": 1},
            "top_n": {
                "type": ["integer", "null"],
                "minimum": 1,
                "description": "Optional maximum number of categories.",
            },
        }

    def _specific_required(self) -> list[str]:
        return ["column"]

    def _create_chart(
        self,
        dataset_id: str,
        arguments: dict[str, Any],
        output_format: OutputFormat,
        title: str | None,
    ) -> VisualizationOutput:
        return self._visualization_service.create_pie_chart(
            self._context,
            dataset_id,
            column=_required_string(arguments, "column"),
            top_n=_optional_positive_integer(arguments, "top_n"),
            output_format=output_format,
            title=title,
        )


class BoxChartTool(_BaseVisualizationTool):
    """Create a box plot from descriptive statistics."""

    @property
    def name(self) -> str:
        return "visualization.box"

    @property
    def description(self) -> str:
        return (
            "Create a box plot for a numeric dataset column "
            "using its descriptive statistics."
        )

    def _specific_properties(self) -> dict[str, Any]:
        return {
            "column": {"type": "string", "minLength": 1},
        }

    def _specific_required(self) -> list[str]:
        return ["column"]

    def _create_chart(
        self,
        dataset_id: str,
        arguments: dict[str, Any],
        output_format: OutputFormat,
        title: str | None,
    ) -> VisualizationOutput:
        return self._visualization_service.create_box_chart(
            self._context,
            dataset_id,
            column=_required_string(arguments, "column"),
            output_format=output_format,
            title=title,
        )


def create_visualization_tools(
    context: ExecutionContext,
    visualization_service: VisualizationService,
    file_manager: FileManager,
) -> tuple[_BaseVisualizationTool, ...]:
    """Create all builtin visualization tools sharing the same services."""

    return (
        BarChartTool(context, visualization_service, file_manager),
        GroupedBarChartTool(context, visualization_service, file_manager),
        LineChartTool(context, visualization_service, file_manager),
        HistogramTool(context, visualization_service, file_manager),
        ScatterChartTool(context, visualization_service, file_manager),
        PieChartTool(context, visualization_service, file_manager),
        BoxChartTool(context, visualization_service, file_manager),
    )