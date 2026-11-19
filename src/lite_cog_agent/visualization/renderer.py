"""Visualization renderer contracts and interfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol

from lite_cog_agent.visualization.contracts import ChartSpecification
from lite_cog_agent.visualization.exceptions import (
    VisualizationValidationError,
)


class OutputFormat(str, Enum):
    """Supported chart output formats."""

    PNG = "png"
    SVG = "svg"


def _immutable_mapping(
    value: Mapping[str, Any] | None,
    field_name: str,
) -> Mapping[str, Any]:
    """Validate and create a shallow read-only mapping."""

    if value is None:
        return MappingProxyType({})

    if not isinstance(value, Mapping):
        raise VisualizationValidationError(
            f"{field_name} must be a mapping"
        )

    if any(not isinstance(key, str) for key in value):
        raise VisualizationValidationError(
            f"{field_name} keys must be strings"
        )

    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class RenderedChart:
    """Immutable result produced by a chart renderer."""

    image_data: bytes
    output_format: OutputFormat
    width: int
    height: int
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.image_data, bytes):
            raise VisualizationValidationError(
                "image_data must be bytes"
            )

        if not self.image_data:
            raise VisualizationValidationError(
                "image_data must not be empty"
            )

        if not isinstance(self.output_format, OutputFormat):
            raise VisualizationValidationError(
                "output_format must be an OutputFormat"
            )

        if (
            not isinstance(self.width, int)
            or isinstance(self.width, bool)
            or self.width <= 0
        ):
            raise VisualizationValidationError(
                "width must be a positive integer"
            )

        if (
            not isinstance(self.height, int)
            or isinstance(self.height, bool)
            or self.height <= 0
        ):
            raise VisualizationValidationError(
                "height must be a positive integer"
            )

        object.__setattr__(
            self,
            "metadata",
            _immutable_mapping(self.metadata, "metadata"),
        )

    @property
    def mime_type(self) -> str:
        """Return the MIME type corresponding to the output format."""

        if self.output_format is OutputFormat.PNG:
            return "image/png"

        if self.output_format is OutputFormat.SVG:
            return "image/svg+xml"

        # Defensive fallback for future enum extensions.
        raise VisualizationValidationError(
            f"unsupported output format: {self.output_format}"
        )


class ChartRenderer(Protocol):
    """Structural interface for chart rendering implementations."""

    def render(
        self,
        specification: ChartSpecification,
        *,
        output_format: OutputFormat = OutputFormat.PNG,
    ) -> RenderedChart:
        """Render a chart specification into image bytes.

        Args:
            specification: Validated chart specification.
            output_format: Requested image format.

        Returns:
            Rendered chart data and metadata.

        Raises:
            VisualizationValidationError:
                If the specification or rendering options are invalid.
            VisualizationExecutionError:
                If rendering fails during execution.
        """
        ...
