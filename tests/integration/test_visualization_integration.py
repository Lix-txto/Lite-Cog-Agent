"""Integration tests for the complete visualization pipeline."""

from __future__ import annotations

from xml.etree import ElementTree

import pandas as pd
import pytest

from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.execution.dataset_store import DatasetReference
from lite_cog_agent.visualization.contracts import ChartType
from lite_cog_agent.visualization.renderer import OutputFormat
from lite_cog_agent.visualization.service import VisualizationService


@pytest.fixture
def source_dataframe() -> pd.DataFrame:
    """Provide a dataset supporting all chart types."""
    return pd.DataFrame(
        {
            "quarter": [1, 2, 3, 4, 5, 6],
            "category": ["A", "B", "A", "C", "B", "A"],
            "department": [
                "Sales",
                "Sales",
                "Engineering",
                "Engineering",
                "Sales",
                "Engineering",
            ],
            "region": [
                "East",
                "West",
                "East",
                "West",
                "East",
                "West",
            ],
            "amount": [100, 200, 150, 300, 250, 350],
        }
    )


@pytest.fixture
def registered_dataset(
    source_dataframe: pd.DataFrame,
):
    """Register a real DataFrame in an ExecutionContext."""
    context = ExecutionContext()

    reference = DatasetReference(
        name="sales_dataset",
        source="integration_test",
        row_count=len(source_dataframe),
        column_count=len(source_dataframe.columns),
        columns=tuple(source_dataframe.columns),
    )

    context.register_dataset(
        reference,
        source_dataframe,
    )

    yield context, reference, source_dataframe

    context.close()


@pytest.fixture
def visualization_service() -> VisualizationService:
    """Create a service with real statistical and rendering components."""
    return VisualizationService()


def create_chart(
    service: VisualizationService,
    context: ExecutionContext,
    dataset_id: str,
    chart_type: ChartType,
    output_format: OutputFormat,
):
    """Dispatch a chart request to the corresponding service method."""

    if chart_type is ChartType.BAR:
        return service.create_bar_chart(
            context,
            dataset_id,
            column="category",
            output_format=output_format,
        )

    if chart_type is ChartType.LINE:
        return service.create_line_chart(
            context,
            dataset_id,
            x_column="quarter",
            y_column="amount",
            output_format=output_format,
        )

    if chart_type is ChartType.HISTOGRAM:
        return service.create_histogram(
            context,
            dataset_id,
            column="amount",
            bins=4,
            output_format=output_format,
        )

    if chart_type is ChartType.SCATTER:
        return service.create_scatter_chart(
            context,
            dataset_id,
            x_column="quarter",
            y_column="amount",
            output_format=output_format,
        )

    if chart_type is ChartType.PIE:
        return service.create_pie_chart(
            context,
            dataset_id,
            column="category",
            output_format=output_format,
        )

    if chart_type is ChartType.BOX:
        return service.create_box_chart(
            context,
            dataset_id,
            column="amount",
            output_format=output_format,
        )

    raise AssertionError(f"Unsupported chart type: {chart_type}")


@pytest.mark.parametrize(
    "chart_type",
    [
        ChartType.BAR,
        ChartType.LINE,
        ChartType.HISTOGRAM,
        ChartType.SCATTER,
        ChartType.PIE,
        ChartType.BOX,
    ],
)
@pytest.mark.parametrize(
    "output_format",
    [
        OutputFormat.PNG,
        OutputFormat.SVG,
    ],
)
def test_chart_generation_end_to_end(
    registered_dataset,
    visualization_service: VisualizationService,
    chart_type: ChartType,
    output_format: OutputFormat,
) -> None:
    """Verify real statistical analysis and rendering."""
    context, reference, dataframe = registered_dataset

    output = create_chart(
        visualization_service,
        context,
        reference.dataset_id,
        chart_type,
        output_format,
    )

    assert output.result.dataset_id == reference.dataset_id
    assert output.result.dataset_name == reference.name

    assert (
        output.result.specification.chart_type
        is chart_type
    )

    rendered = output.rendered_chart

    assert rendered.output_format is output_format
    assert rendered.width == 1200
    assert rendered.height == 800
    assert rendered.image_data
    assert rendered.mime_type == (
        "image/png"
        if output_format is OutputFormat.PNG
        else "image/svg+xml"
    )

    assert rendered.metadata["renderer"] == "matplotlib"
    assert rendered.metadata["chart_type"] == chart_type.value

    assert context.is_closed is False
    assert context.get_dataset(reference.dataset_id) is dataframe



@pytest.mark.parametrize(
    "output_format",
    [
        OutputFormat.PNG,
        OutputFormat.SVG,
    ],
)
def test_grouped_bar_chart_end_to_end(
    registered_dataset,
    visualization_service: VisualizationService,
    output_format: OutputFormat,
) -> None:
    """Verify grouped aggregation and rendering."""
    context, reference, dataframe = registered_dataset
    original_dataframe = dataframe.copy(deep=True)

    output = visualization_service.create_grouped_bar_chart(
        context,
        reference.dataset_id,
        group_by=["department"],
        aggregations={
            "amount": ["sum"],
        },
        output_format=output_format,
    )

    # 1. Validate logical visualization result.
    assert output.result.dataset_id == reference.dataset_id
    assert output.result.dataset_name == reference.name

    specification = output.result.specification

    assert specification.chart_type is ChartType.BAR
    assert specification.title == "Grouped Analysis by department"
    assert specification.x_label == "department"
    assert specification.y_label == "Value"

    # 2. Validate grouped aggregation.
    assert len(specification.data.series) == 1

    series = specification.data.series[0]

    assert series.name == "amount_sum"
    assert len(series.x_values) == 2
    assert len(series.y_values) == 2

    assert sum(series.y_values) == 1350

    # 3. Validate rendered output.
    rendered = output.rendered_chart

    assert rendered.output_format is output_format
    assert rendered.width == 1200
    assert rendered.height == 800
    assert rendered.image_data

    assert rendered.mime_type == (
        "image/png"
        if output_format is OutputFormat.PNG
        else "image/svg+xml"
    )

    assert rendered.metadata["renderer"] == "matplotlib"
    assert rendered.metadata["chart_type"] == ChartType.BAR.value

    # 4. Validate caller-owned context lifecycle.
    assert context.is_closed is False
    assert context.get_dataset(reference.dataset_id) is dataframe

    # 5. Validate source dataset integrity.
    pd.testing.assert_frame_equal(
        dataframe,
        original_dataframe,
    )



def test_visualization_does_not_modify_source_dataframe(
    registered_dataset,
    visualization_service: VisualizationService,
) -> None:
    """Visualization must preserve the original dataset."""
    context, reference, dataframe = registered_dataset

    original_dataframe = dataframe.copy(deep=True)

    visualization_service.create_bar_chart(
        context,
        reference.dataset_id,
        column="category",
    )

    visualization_service.create_histogram(
        context,
        reference.dataset_id,
        column="amount",
    )

    visualization_service.create_scatter_chart(
        context,
        reference.dataset_id,
        x_column="quarter",
        y_column="amount",
    )

    pd.testing.assert_frame_equal(
        dataframe,
        original_dataframe,
    )


def test_visualization_does_not_close_execution_context(
    registered_dataset,
    visualization_service: VisualizationService,
) -> None:
    """Context lifecycle must remain owned by the caller."""
    context, reference, dataframe = registered_dataset

    visualization_service.create_box_chart(
        context,
        reference.dataset_id,
        column="amount",
    )

    assert context.is_closed is False

    assert context.get_dataset(reference.dataset_id) is dataframe

    assert reference.dataset_id in context.dataset_ids


def test_png_output_contains_valid_png_signature(
    registered_dataset,
    visualization_service: VisualizationService,
) -> None:
    """Verify actual PNG image bytes."""
    context, reference, _ = registered_dataset

    output = visualization_service.create_bar_chart(
        context,
        reference.dataset_id,
        column="category",
        output_format=OutputFormat.PNG,
    )

    image_data = output.rendered_chart.image_data

    assert image_data[:8] == b"\x89PNG\r\n\x1a\n"

    # PNG IHDR stores width and height as big-endian uint32.
    width = int.from_bytes(image_data[16:20], "big")
    height = int.from_bytes(image_data[20:24], "big")

    assert width == 1200
    assert height == 800


def test_svg_output_contains_valid_xml(
    registered_dataset,
    visualization_service: VisualizationService,
) -> None:
    """Verify actual SVG XML output."""
    context, reference, _ = registered_dataset

    output = visualization_service.create_line_chart(
        context,
        reference.dataset_id,
        x_column="quarter",
        y_column="amount",
        output_format=OutputFormat.SVG,
    )

    root = ElementTree.fromstring(
        output.rendered_chart.image_data
    )

    assert root.tag.endswith("svg")
    assert root.attrib["viewBox"] == "0 0 864 576"


def test_context_can_be_closed_after_visualization(
    registered_dataset,
    visualization_service: VisualizationService,
) -> None:
    """The caller can release datasets after chart generation."""
    context, reference, _ = registered_dataset

    output = visualization_service.create_pie_chart(
        context,
        reference.dataset_id,
        column="category",
    )

    assert output.rendered_chart.image_data

    context.close()

    assert context.is_closed is True
    assert context.dataset_ids == ()
    assert context.dataset_store.contains(reference.dataset_id) is False
