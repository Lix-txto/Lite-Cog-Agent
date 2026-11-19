"""Output contracts for visualization services."""

from dataclasses import dataclass

from lite_cog_agent.visualization.contracts import VisualizationResult
from lite_cog_agent.visualization.exceptions import (
    VisualizationValidationError,
)
from lite_cog_agent.visualization.renderer import RenderedChart


@dataclass(frozen=True, slots=True)
class VisualizationOutput:
    """Combined logical and rendered visualization output."""

    result: VisualizationResult
    rendered_chart: RenderedChart

    def __post_init__(self) -> None:
        """Validate the components of the visualization output."""
        if not isinstance(self.result, VisualizationResult):
            raise VisualizationValidationError(
                "result must be a VisualizationResult instance"
            )

        if not isinstance(self.rendered_chart, RenderedChart):
            raise VisualizationValidationError(
                "rendered_chart must be a RenderedChart instance"
            )
