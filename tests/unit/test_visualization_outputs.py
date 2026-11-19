"""Tests for visualization output contracts."""

from dataclasses import FrozenInstanceError

import pytest

from lite_cog_agent.visualization.contracts import (
    ChartData,
    ChartSeries,
    ChartSpecification,
    ChartType,
    VisualizationResult,
)
from lite_cog_agent.visualization.exceptions import (
    VisualizationValidationError,
)
from lite_cog_agent.visualization.outputs import VisualizationOutput
from lite_cog_agent.visualization.renderer import (
    OutputFormat,
    RenderedChart,
)


@pytest.fixture
def visualization_result() -> VisualizationResult:
    """Create a valid visualization result."""
    series = ChartSeries(
        name="sales",
        x_values=("A", "B", "C"),
        y_values=(100.0, 200.0, 150.0),
    )

    data = ChartData(
        dataset_id="dataset-001",
        dataset_name="sales.csv",
        series=(series,),
    )

    specification = ChartSpecification(
        chart_type=ChartType.BAR,
        title="Sales",
        data=data,
    )

    return VisualizationResult(
        visualization_id="visualization-001",
        dataset_id="dataset-001",
        dataset_name="sales.csv",
        specification=specification,
    )


@pytest.fixture
def rendered_chart() -> RenderedChart:
    """Create a valid rendered chart."""
    return RenderedChart(
        image_data=b"fake-image-data",
        output_format=OutputFormat.PNG,
        width=1200,
        height=800,
    )


def test_creates_visualization_output(
    visualization_result: VisualizationResult,
    rendered_chart: RenderedChart,
) -> None:
    output = VisualizationOutput(
        result=visualization_result,
        rendered_chart=rendered_chart,
    )

    assert output.result is visualization_result
    assert output.rendered_chart is rendered_chart


def test_preserves_result_reference(
    visualization_result: VisualizationResult,
    rendered_chart: RenderedChart,
) -> None:
    output = VisualizationOutput(
        result=visualization_result,
        rendered_chart=rendered_chart,
    )

    assert output.result.visualization_id == "visualization-001"
    assert output.result.dataset_id == "dataset-001"


def test_preserves_rendered_chart_reference(
    visualization_result: VisualizationResult,
    rendered_chart: RenderedChart,
) -> None:
    output = VisualizationOutput(
        result=visualization_result,
        rendered_chart=rendered_chart,
    )

    assert output.rendered_chart.image_data == b"fake-image-data"
    assert output.rendered_chart.output_format is OutputFormat.PNG
    assert output.rendered_chart.width == 1200
    assert output.rendered_chart.height == 800


def test_rejects_invalid_result_type(
    rendered_chart: RenderedChart,
) -> None:
    with pytest.raises(
        VisualizationValidationError,
        match="result must be a VisualizationResult",
    ):
        VisualizationOutput(
            result="invalid",  # type: ignore[arg-type]
            rendered_chart=rendered_chart,
        )


def test_rejects_invalid_rendered_chart_type(
    visualization_result: VisualizationResult,
) -> None:
    with pytest.raises(
        VisualizationValidationError,
        match="rendered_chart must be a RenderedChart",
    ):
        VisualizationOutput(
            result=visualization_result,
            rendered_chart=b"invalid",  # type: ignore[arg-type]
        )


def test_output_is_immutable(
    visualization_result: VisualizationResult,
    rendered_chart: RenderedChart,
) -> None:
    output = VisualizationOutput(
        result=visualization_result,
        rendered_chart=rendered_chart,
    )

    with pytest.raises(FrozenInstanceError):
        output.result = visualization_result  # type: ignore[misc]


def test_output_uses_slots(
    visualization_result: VisualizationResult,
    rendered_chart: RenderedChart,
) -> None:
    output = VisualizationOutput(
        result=visualization_result,
        rendered_chart=rendered_chart,
    )

    assert not hasattr(output, "__dict__")
