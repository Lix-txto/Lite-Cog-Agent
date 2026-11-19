"""Tests for visualization service."""

from datetime import date
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest

from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.execution.dataset_store import DatasetReference
from lite_cog_agent.statistics.contracts import (
    DescriptiveStatistics,
    DescriptiveStatisticsResult,
    DistributionAnalysisResult,
    FrequencyAnalysisResult,
    FrequencyItem,
    GroupedAnalysisResult,
    HistogramBin,
)
from lite_cog_agent.statistics.exceptions import (
    FrequencyAnalysisError,
    StatisticalValidationError,
)
from lite_cog_agent.statistics.service import StatisticalAnalysisService
from lite_cog_agent.visualization.contracts import ChartType
from lite_cog_agent.visualization.exceptions import (
    VisualizationExecutionError,
    VisualizationValidationError,
)
from lite_cog_agent.visualization.matplotlib_renderer import MatplotlibRenderer
from lite_cog_agent.visualization.outputs import VisualizationOutput
from lite_cog_agent.visualization.renderer import (
    ChartRenderer,
    OutputFormat,
    RenderedChart,
)
from lite_cog_agent.visualization.service import VisualizationService


def create_context() -> ExecutionContext:
    """Create a context containing a small test dataset."""
    context = ExecutionContext()

    dataframe = pd.DataFrame(
        {
            "category": ["A", "B", "A", "C"],
            "department": ["Sales", "Sales", "R&D", "R&D"],
            "region": ["East", "West", "East", "West"],
            "amount": [100, 200, 300, 400],
        }
    )

    reference = DatasetReference(
        dataset_id="dataset-1",
        name="sales-dataset",
        source="unit-test",
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        columns=tuple(dataframe.columns),
    )

    context.register_dataset(reference, dataframe)
    return context


def create_frequency_result(
    *,
    items: tuple[FrequencyItem, ...] | None = None,
    total_count: int = 4,
) -> FrequencyAnalysisResult:
    """Create a valid frequency analysis result."""
    if items is None:
        items = (
            FrequencyItem(
                value="A",
                count=2,
                percentage=50.0,
                cumulative_percentage=50.0,
            ),
            FrequencyItem(
                value="B",
                count=1,
                percentage=25.0,
                cumulative_percentage=75.0,
            ),
            FrequencyItem(
                value="C",
                count=1,
                percentage=25.0,
                cumulative_percentage=100.0,
            ),
        )

    return FrequencyAnalysisResult(
        dataset_id="dataset-1",
        dataset_name="sales-dataset",
        column="category",
        total_count=total_count,
        items=items,
    )


def create_distribution_result(
    *,
    column: str = "amount",
    count: int = 4,
    histogram: tuple[HistogramBin, ...] | None = None,
) -> DistributionAnalysisResult:
    """Create a valid distribution analysis result."""
    if histogram is None:
        histogram = (
            HistogramBin(
                lower_bound=100.0,
                upper_bound=200.0,
                count=1,
                percentage=25.0,
            ),
            HistogramBin(
                lower_bound=200.0,
                upper_bound=300.0,
                count=1,
                percentage=25.0,
            ),
            HistogramBin(
                lower_bound=300.0,
                upper_bound=400.0,
                count=1,
                percentage=25.0,
            ),
            HistogramBin(
                lower_bound=400.0,
                upper_bound=500.0,
                count=1,
                percentage=25.0,
            ),
        )

    return DistributionAnalysisResult(
        dataset_id="dataset-1",
        dataset_name="sales-dataset",
        column=column,
        count=count,
        mean=250.0,
        median=250.0,
        std=100.0,
        skewness=0.0,
        kurtosis=-1.0,
        quantiles={
            "0.25": 175.0,
            "0.5": 250.0,
            "0.75": 325.0,
        },
        histogram=histogram,
    )


def create_descriptive_result(
    *,
    column: str = "amount",
    count: int = 4,
    minimum: float | None = 100.0,
    q1: float | None = 175.0,
    median: float | None = 250.0,
    q3: float | None = 325.0,
    maximum: float | None = 400.0,
) -> DescriptiveStatisticsResult:
    """Create a valid descriptive statistics result."""
    statistics = DescriptiveStatistics(
        column=column,
        count=count,
        mean=250.0 if count > 0 else None,
        median=median,
        std=129.1 if count > 1 else None,
        variance=16666.67 if count > 1 else None,
        min=minimum,
        q1=q1,
        q3=q3,
        max=maximum,
        range=(
            maximum - minimum
            if minimum is not None and maximum is not None
            else None
        ),
    )

    return DescriptiveStatisticsResult(
        dataset_id="dataset-1",
        dataset_name="sales-dataset",
        row_count=4,
        statistics=(statistics,),
    )


def create_grouped_result() -> GroupedAnalysisResult:
    """Create a valid grouped analysis result."""
    return GroupedAnalysisResult(
        dataset_id="dataset-1",
        dataset_name="sales-dataset",
        group_by=("department",),
        aggregations={
            "amount": ("sum", "mean"),
        },
        group_count=2,
        records=(
            {
                "department": "R&D",
                "amount_sum": 700,
                "amount_mean": 350.0,
            },
            {
                "department": "Sales",
                "amount_sum": 300,
                "amount_mean": 150.0,
            },
        ),
    )


def create_renderer() -> Mock:
    """Create a renderer mock with a valid rendered result."""
    renderer = Mock(spec=ChartRenderer)

    renderer.render.return_value = RenderedChart(
        image_data=b"fake-image-data",
        output_format=OutputFormat.PNG,
        width=1200,
        height=800,
    )

    return renderer


def create_statistical_service() -> Mock:
    """Create a statistical service mock."""
    return Mock(spec=StatisticalAnalysisService)


def create_service(
    *,
    statistical_service: Mock | None = None,
    renderer: Mock | None = None,
) -> tuple[VisualizationService, Mock, Mock]:
    """Create a service with mocked dependencies."""
    statistical_service = (
        statistical_service
        if statistical_service is not None
        else create_statistical_service()
    )
    renderer = renderer if renderer is not None else create_renderer()

    service = VisualizationService(
        statistical_service=statistical_service,
        renderer=renderer,
    )

    return service, statistical_service, renderer

def create_dataframe_context(
    data: dict[str, list],
) -> ExecutionContext:
    """Create an execution context containing a test DataFrame."""
    context = ExecutionContext()

    reference = DatasetReference(
        dataset_id="dataset-1",
        name="sales-dataset",
        source="test",
        file_format="memory",
    )

    context.register_dataset(
        reference,
        pd.DataFrame(data),
    )

    return context


def create_visualization_service() -> tuple[
    VisualizationService,
    MagicMock,
]:
    """Create a visualization service with a mocked renderer."""
    renderer = MagicMock()

    renderer.render.return_value = RenderedChart(
        image_data=b"fake-image-data",
        output_format=OutputFormat.PNG,
        width=1200,
        height=800,
    )

    service = VisualizationService(
        statistical_service=MagicMock(),
        renderer=renderer,
    )

    return service, renderer


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------


def test_creates_default_dependencies() -> None:
    service = VisualizationService()

    assert isinstance(
        service._statistical_service,
        StatisticalAnalysisService,
    )
    assert isinstance(
        service._renderer,
        MatplotlibRenderer,
    )


def test_injects_custom_statistical_service() -> None:
    statistical_service = create_statistical_service()

    service = VisualizationService(
        statistical_service=statistical_service,
    )

    assert service._statistical_service is statistical_service
    assert isinstance(service._renderer, MatplotlibRenderer)


def test_injects_custom_renderer() -> None:
    renderer = create_renderer()

    service = VisualizationService(
        renderer=renderer,
    )

    assert isinstance(
        service._statistical_service,
        StatisticalAnalysisService,
    )
    assert service._renderer is renderer


def test_injects_both_dependencies() -> None:
    statistical_service = create_statistical_service()
    renderer = create_renderer()

    service = VisualizationService(
        statistical_service=statistical_service,
        renderer=renderer,
    )

    assert service._statistical_service is statistical_service
    assert service._renderer is renderer


def test_creates_independent_default_dependencies() -> None:
    first_service = VisualizationService()
    second_service = VisualizationService()

    assert first_service._statistical_service is not (
        second_service._statistical_service
    )
    assert first_service._renderer is not second_service._renderer


def test_does_not_replace_falsey_custom_dependencies() -> None:
    class FalseyRenderer:
        def __bool__(self) -> bool:
            return False

        def render(self, specification, *, output_format=None):
            raise NotImplementedError

    class FalseyStatisticalService:
        def __bool__(self) -> bool:
            return False

    statistical_service = FalseyStatisticalService()
    renderer = FalseyRenderer()

    service = VisualizationService(
        statistical_service=statistical_service,  # type: ignore[arg-type]
        renderer=renderer,  # type: ignore[arg-type]
    )

    assert service._statistical_service is statistical_service
    assert service._renderer is renderer


# ---------------------------------------------------------------------------
# Category bar chart
# ---------------------------------------------------------------------------


def test_creates_category_bar_chart() -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    statistical_service.analyze_frequency.return_value = (
        create_frequency_result()
    )

    result = service.create_bar_chart(
        context,
        "dataset-1",
        column="category",
    )

    assert isinstance(result, VisualizationOutput)
    assert result.result.dataset_id == "dataset-1"
    assert result.result.dataset_name == "sales-dataset"
    assert result.result.specification.chart_type is ChartType.BAR

    series = result.result.specification.data.series[0]

    assert series.name == "category"
    assert series.x_values == ("A", "B", "C")
    assert series.y_values == (2, 1, 1)

    statistical_service.analyze_frequency.assert_called_once_with(
        context,
        "dataset-1",
        column="category",
        top_n=None,
    )

    renderer.render.assert_called_once()
    context.close()


def test_category_bar_chart_supports_top_n() -> None:
    context = create_context()
    service, statistical_service, _ = create_service()

    statistical_service.analyze_frequency.return_value = (
        create_frequency_result(
            items=(
                FrequencyItem(
                    value="A",
                    count=2,
                    percentage=50.0,
                    cumulative_percentage=50.0,
                ),
            ),
        )
    )

    result = service.create_bar_chart(
        context,
        "dataset-1",
        column="category",
        top_n=1,
    )

    statistical_service.analyze_frequency.assert_called_once_with(
        context,
        "dataset-1",
        column="category",
        top_n=1,
    )

    metadata = result.result.specification.data.metadata

    assert metadata["top_n"] == 1
    assert metadata["total_count"] == 4
    assert metadata["returned_category_count"] == 1
    assert metadata["category_scope"] == "top_n"

    context.close()


def test_category_bar_chart_supports_custom_title_and_svg() -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    statistical_service.analyze_frequency.return_value = (
        create_frequency_result()
    )

    result = service.create_bar_chart(
        context,
        "dataset-1",
        column="category",
        title="Sales Categories",
        output_format=OutputFormat.SVG,
    )

    assert result.result.specification.title == "Sales Categories"

    renderer.render.assert_called_once_with(
        result.result.specification,
        output_format=OutputFormat.SVG,
    )

    context.close()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, "True"),
        (False, "False"),
        (None, "(Missing)"),
        (date(2026, 1, 2), "2026-01-02"),
        (1, 1),
        (1.5, 1.5),
        ("A", "A"),
    ],
)
def test_normalizes_category_values(value, expected) -> None:
    normalized = VisualizationService._normalize_category_value(value)

    assert normalized == expected


def test_rejects_unsupported_category_value() -> None:
    with pytest.raises(
        VisualizationValidationError,
        match="unsupported category value type",
    ):
        VisualizationService._normalize_category_value((1, 2))


def test_rejects_infinite_category_value() -> None:
    with pytest.raises(
        VisualizationValidationError,
        match="infinite",
    ):
        VisualizationService._normalize_category_value(float("inf"))


def test_rejects_duplicate_normalized_category_labels() -> None:
    with pytest.raises(
        VisualizationValidationError,
        match="duplicate chart labels",
    ):
        VisualizationService._normalize_category_labels(
            ("(Missing)", None)
        )


def test_rejects_empty_frequency_result() -> None:
    context = create_context()
    service, statistical_service, _ = create_service()

    statistical_service.analyze_frequency.return_value = (
        create_frequency_result(
            items=(),
            total_count=0,
        )
    )

    with pytest.raises(
        VisualizationValidationError,
        match="no categories",
    ):
        service.create_bar_chart(
            context,
            "dataset-1",
            column="category",
        )

    context.close()


@pytest.mark.parametrize(
    ("column", "top_n"),
    [
        ("", None),
        ("category", 0),
        ("category", -1),
        ("category", True),
    ],
)
def test_rejects_invalid_bar_chart_parameters(column, top_n) -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    with pytest.raises(VisualizationValidationError):
        service.create_bar_chart(
            context,
            "dataset-1",
            column=column,
            top_n=top_n,
        )

    statistical_service.analyze_frequency.assert_not_called()
    renderer.render.assert_not_called()
    context.close()


# ---------------------------------------------------------------------------
# Grouped bar chart
# ---------------------------------------------------------------------------


def test_creates_grouped_bar_chart() -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    statistical_service.analyze_grouped.return_value = (
        create_grouped_result()
    )

    result = service.create_grouped_bar_chart(
        context,
        "dataset-1",
        group_by=["department"],
        aggregations={
            "amount": ["sum", "mean"],
        },
    )

    assert isinstance(result, VisualizationOutput)
    assert result.result.specification.chart_type is ChartType.BAR

    specification = result.result.specification

    assert specification.x_label == "department"
    assert specification.y_label == "Value"

    assert len(specification.data.series) == 2

    sum_series = specification.data.series[0]
    mean_series = specification.data.series[1]

    assert sum_series.name == "amount_sum"
    assert sum_series.x_values == (
        'department="R&D"',
        'department="Sales"',
    )
    assert sum_series.y_values == (700.0, 300.0)

    assert mean_series.name == "amount_mean"
    assert mean_series.y_values == (350.0, 150.0)

    statistical_service.analyze_grouped.assert_called_once_with(
        context,
        "dataset-1",
        group_by=["department"],
        aggregations={"amount": ["sum", "mean"]},
    )

    renderer.render.assert_called_once()
    context.close()


def test_grouped_bar_chart_supports_multiple_group_columns() -> None:
    records = (
        {
            "department": "R&D",
            "region": "East",
            "amount_sum": 300,
        },
        {
            "department": "Sales",
            "region": "West",
            "amount_sum": 200,
        },
    )

    analysis = GroupedAnalysisResult(
        dataset_id="dataset-1",
        dataset_name="sales-dataset",
        group_by=("department", "region"),
        aggregations={"amount": ("sum",)},
        group_count=2,
        records=records,
    )

    context = create_context()
    service, statistical_service, _ = create_service()
    statistical_service.analyze_grouped.return_value = analysis

    result = service.create_grouped_bar_chart(
        context,
        "dataset-1",
        group_by=["department", "region"],
        aggregations={"amount": ["sum"]},
    )

    series = result.result.specification.data.series[0]

    assert series.x_values == (
        'department="R&D" | region="East"',
        'department="Sales" | region="West"',
    )

    context.close()


def test_grouped_bar_chart_supports_missing_group_values() -> None:
    analysis = GroupedAnalysisResult(
        dataset_id="dataset-1",
        dataset_name="sales-dataset",
        group_by=("department",),
        aggregations={"amount": ("sum",)},
        group_count=1,
        records=(
            {
                "department": None,
                "amount_sum": 100,
            },
        ),
    )

    context = create_context()
    service, statistical_service, _ = create_service()
    statistical_service.analyze_grouped.return_value = analysis

    result = service.create_grouped_bar_chart(
        context,
        "dataset-1",
        group_by=["department"],
        aggregations={"amount": ["sum"]},
    )

    series = result.result.specification.data.series[0]

    assert series.x_values == ("department=(Missing)",)

    context.close()


def test_grouped_bar_chart_converts_missing_aggregation_to_none() -> None:
    analysis = GroupedAnalysisResult(
        dataset_id="dataset-1",
        dataset_name="sales-dataset",
        group_by=("department",),
        aggregations={"amount": ("mean",)},
        group_count=1,
        records=(
            {
                "department": "Sales",
                "amount_mean": float("nan"),
            },
        ),
    )

    context = create_context()
    service, statistical_service, _ = create_service()
    statistical_service.analyze_grouped.return_value = analysis

    result = service.create_grouped_bar_chart(
        context,
        "dataset-1",
        group_by=["department"],
        aggregations={"amount": ["mean"]},
    )

    series = result.result.specification.data.series[0]

    assert series.y_values == (None,)

    context.close()


@pytest.mark.parametrize(
    "value",
    [
        True,
        "100",
        1 + 2j,
        float("inf"),
        float("-inf"),
    ],
)
def test_rejects_invalid_grouped_aggregation_values(value) -> None:
    analysis = GroupedAnalysisResult(
        dataset_id="dataset-1",
        dataset_name="sales-dataset",
        group_by=("department",),
        aggregations={"amount": ("sum",)},
        group_count=1,
        records=(
            {
                "department": "Sales",
                "amount_sum": value,
            },
        ),
    )

    context = create_context()
    service, statistical_service, renderer = create_service()
    statistical_service.analyze_grouped.return_value = analysis

    with pytest.raises(VisualizationValidationError):
        service.create_grouped_bar_chart(
            context,
            "dataset-1",
            group_by=["department"],
            aggregations={"amount": ["sum"]},
        )

    renderer.render.assert_not_called()
    context.close()


def test_rejects_empty_grouped_result() -> None:
    analysis = GroupedAnalysisResult(
        dataset_id="dataset-1",
        dataset_name="sales-dataset",
        group_by=("department",),
        aggregations={"amount": ("sum",)},
        group_count=0,
        records=(),
    )

    context = create_context()
    service, statistical_service, _ = create_service()
    statistical_service.analyze_grouped.return_value = analysis

    with pytest.raises(
        VisualizationValidationError,
        match="no records",
    ):
        service.create_grouped_bar_chart(
            context,
            "dataset-1",
            group_by=["department"],
            aggregations={"amount": ["sum"]},
        )

    context.close()


@pytest.mark.parametrize(
    ("group_by", "aggregations"),
    [
        ([], {"amount": ["sum"]}),
        (["department", "department"], {"amount": ["sum"]}),
        (["department"], {}),
        (["department"], {"amount": []}),
        (["department"], {"amount": ["sum", "sum"]}),
    ],
)
def test_rejects_invalid_grouped_parameters(
    group_by,
    aggregations,
) -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    with pytest.raises(VisualizationValidationError):
        service.create_grouped_bar_chart(
            context,
            "dataset-1",
            group_by=group_by,
            aggregations=aggregations,
        )

    statistical_service.analyze_grouped.assert_not_called()
    renderer.render.assert_not_called()
    context.close()


# ---------------------------------------------------------------------------
# Dataset access and error handling
# ---------------------------------------------------------------------------


def test_rejects_unknown_dataset() -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    with pytest.raises(
        VisualizationValidationError,
        match="does not exist",
    ):
        service.create_bar_chart(
            context,
            "unknown-dataset",
            column="category",
        )

    statistical_service.analyze_frequency.assert_not_called()
    renderer.render.assert_not_called()
    context.close()


def test_rejects_closed_context() -> None:
    context = create_context()
    context.close()

    service, statistical_service, renderer = create_service()

    with pytest.raises(
        VisualizationExecutionError,
        match="closed context",
    ):
        service.create_bar_chart(
            context,
            "dataset-1",
            column="category",
        )

    statistical_service.analyze_frequency.assert_not_called()
    renderer.render.assert_not_called()


def test_preserves_statistical_validation_as_visualization_validation() -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    statistical_service.analyze_frequency.side_effect = (
        StatisticalValidationError("invalid column")
    )

    with pytest.raises(VisualizationValidationError) as exc_info:
        service.create_bar_chart(
            context,
            "dataset-1",
            column="category",
        )

    assert isinstance(exc_info.value.__cause__, StatisticalValidationError)
    renderer.render.assert_not_called()
    context.close()


def test_converts_statistical_execution_error() -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    statistical_service.analyze_frequency.side_effect = (
        FrequencyAnalysisError("calculation failed")
    )

    with pytest.raises(VisualizationExecutionError) as exc_info:
        service.create_bar_chart(
            context,
            "dataset-1",
            column="category",
        )

    assert isinstance(exc_info.value.__cause__, FrequencyAnalysisError)
    renderer.render.assert_not_called()
    context.close()


def test_preserves_renderer_execution_error() -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    statistical_service.analyze_frequency.return_value = (
        create_frequency_result()
    )
    renderer.render.side_effect = VisualizationExecutionError(
        "rendering failed"
    )

    with pytest.raises(
        VisualizationExecutionError,
        match="rendering failed",
    ):
        service.create_bar_chart(
            context,
            "dataset-1",
            column="category",
        )

    context.close()


def test_wraps_unexpected_renderer_error() -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    statistical_service.analyze_frequency.return_value = (
        create_frequency_result()
    )
    renderer.render.side_effect = RuntimeError("unexpected")

    with pytest.raises(
        VisualizationExecutionError,
        match="unexpected renderer error",
    ):
        service.create_bar_chart(
            context,
            "dataset-1",
            column="category",
        )

    context.close()


def test_service_does_not_close_execution_context() -> None:
    context = create_context()
    service, statistical_service, _ = create_service()

    statistical_service.analyze_frequency.return_value = (
        create_frequency_result()
    )

    service.create_bar_chart(
        context,
        "dataset-1",
        column="category",
    )

    assert not context.is_closed
    assert context.dataset_ids == ("dataset-1",)

    context.close()


def test_rejects_invalid_output_format() -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    with pytest.raises(VisualizationValidationError):
        service.create_bar_chart(
            context,
            "dataset-1",
            column="category",
            output_format="jpg",  # type: ignore[arg-type]
        )

    statistical_service.analyze_frequency.assert_not_called()
    renderer.render.assert_not_called()
    context.close()

def test_grouped_bar_chart_distinguishes_missing_and_literal_missing_label() -> None:
    analysis = GroupedAnalysisResult(
        dataset_id="dataset-1",
        dataset_name="sales-dataset",
        group_by=("department",),
        aggregations={"amount": ("sum",)},
        group_count=2,
        records=(
            {
                "department": None,
                "amount_sum": 100,
            },
            {
                "department": "(Missing)",
                "amount_sum": 200,
            },
        ),
    )

    context = create_context()
    service, statistical_service, _ = create_service()
    statistical_service.analyze_grouped.return_value = analysis

    result = service.create_grouped_bar_chart(
        context,
        "dataset-1",
        group_by=["department"],
        aggregations={"amount": ["sum"]},
    )

    series = result.result.specification.data.series[0]

    assert series.x_values == (
        "department=(Missing)",
        'department="(Missing)"',
    )

def test_line_chart_preserves_original_observation_order() -> None:
    context = create_dataframe_context(
        {
            "day": ["Wednesday", "Monday", "Tuesday"],
            "sales": [30, 10, 20],
        }
    )

    service, renderer = create_visualization_service()

    result = service.create_line_chart(
        context,
        "dataset-1",
        x_column="day",
        y_column="sales",
    )

    specification = result.result.specification
    series = specification.data.series[0]

    assert specification.chart_type is ChartType.LINE
    assert series.x_values == (
        "Wednesday",
        "Monday",
        "Tuesday",
    )
    assert series.y_values == (30, 10, 20)

    renderer.render.assert_called_once()
    assert result.rendered_chart.image_data == b"fake-image-data"


def test_line_chart_preserves_missing_observations() -> None:
    context = create_dataframe_context(
        {
            "day": ["Monday", "Tuesday", "Wednesday", "Thursday"],
            "sales": [10, None, 30, 40],
        }
    )

    service, _ = create_visualization_service()

    result = service.create_line_chart(
        context,
        "dataset-1",
        x_column="day",
        y_column="sales",
    )

    series = result.result.specification.data.series[0]

    assert series.x_values == (
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
    )
    assert series.y_values == (10, None, 30, 40)


def test_line_chart_supports_datetime_x_values() -> None:
    context = create_dataframe_context(
        {
            "date": [
                pd.Timestamp("2026-01-01"),
                pd.Timestamp("2026-01-02"),
            ],
            "sales": [100, 150],
        }
    )

    service, _ = create_visualization_service()

    result = service.create_line_chart(
        context,
        "dataset-1",
        x_column="date",
        y_column="sales",
    )

    series = result.result.specification.data.series[0]

    assert series.x_values == (
        "2026-01-01T00:00:00",
        "2026-01-02T00:00:00",
    )


def test_line_chart_rejects_non_numeric_y_values() -> None:
    context = create_dataframe_context(
        {
            "day": ["Monday", "Tuesday"],
            "sales": ["one hundred", "two hundred"],
        }
    )

    service, renderer = create_visualization_service()

    with pytest.raises(VisualizationValidationError):
        service.create_line_chart(
            context,
            "dataset-1",
            x_column="day",
            y_column="sales",
        )

    renderer.render.assert_not_called()


def test_line_chart_rejects_no_valid_observations() -> None:
    context = create_dataframe_context(
        {
            "day": ["Monday", "Tuesday"],
            "sales": [None, np.nan],
        }
    )

    service, renderer = create_visualization_service()

    with pytest.raises(VisualizationValidationError):
        service.create_line_chart(
            context,
            "dataset-1",
            x_column="day",
            y_column="sales",
        )

    renderer.render.assert_not_called()

def test_scatter_chart_accepts_numeric_observations() -> None:
    context = create_dataframe_context(
        {
            "advertising": [100, 200, 300],
            "revenue": [500, 800, 1200],
        }
    )

    service, renderer = create_visualization_service()

    result = service.create_scatter_chart(
        context,
        "dataset-1",
        x_column="advertising",
        y_column="revenue",
    )

    specification = result.result.specification
    series = specification.data.series[0]

    assert specification.chart_type is ChartType.SCATTER
    assert series.x_values == (100, 200, 300)
    assert series.y_values == (500, 800, 1200)

    renderer.render.assert_called_once()


def test_scatter_chart_normalizes_missing_and_infinite_values() -> None:
    context = create_dataframe_context(
        {
            "x": [1, 2, np.inf, 4],
            "y": [10, np.nan, 30, 40],
        }
    )

    service, _ = create_visualization_service()

    result = service.create_scatter_chart(
        context,
        "dataset-1",
        x_column="x",
        y_column="y",
    )

    series = result.result.specification.data.series[0]

    assert series.x_values == (1, 2, None, 4)
    assert series.y_values == (10, None, 30, 40)


def test_scatter_chart_rejects_string_x_values() -> None:
    context = create_dataframe_context(
        {
            "x": ["1", "2", "3"],
            "y": [10, 20, 30],
        }
    )

    service, renderer = create_visualization_service()

    with pytest.raises(VisualizationValidationError):
        service.create_scatter_chart(
            context,
            "dataset-1",
            x_column="x",
            y_column="y",
        )

    renderer.render.assert_not_called()


def test_scatter_chart_rejects_boolean_values() -> None:
    context = create_dataframe_context(
        {
            "x": [True, False, True],
            "y": [10, 20, 30],
        }
    )

    service, renderer = create_visualization_service()

    with pytest.raises(VisualizationValidationError):
        service.create_scatter_chart(
            context,
            "dataset-1",
            x_column="x",
            y_column="y",
        )

    renderer.render.assert_not_called()


def test_scatter_chart_rejects_no_valid_observations() -> None:
    context = create_dataframe_context(
        {
            "x": [None, np.nan],
            "y": [10, 20],
        }
    )

    service, renderer = create_visualization_service()

    with pytest.raises(VisualizationValidationError):
        service.create_scatter_chart(
            context,
            "dataset-1",
            x_column="x",
            y_column="y",
        )

    renderer.render.assert_not_called()

def test_line_chart_preserves_numeric_x_values() -> None:
    context = create_dataframe_context(
        {
            "time": [1, 2, 3],
            "sales": [100, 200, 300],
        }
    )

    service, _ = create_visualization_service()

    result = service.create_line_chart(
        context,
        "dataset-1",
        x_column="time",
        y_column="sales",
    )

    series = result.result.specification.data.series[0]

    assert series.x_values == (1, 2, 3)
    assert series.y_values == (100, 200, 300)


def test_line_chart_normalizes_boolean_x_values() -> None:
    context = create_dataframe_context(
        {
            "active": [True, False, True],
            "sales": [100, 200, 300],
        }
    )

    service, _ = create_visualization_service()

    result = service.create_line_chart(
        context,
        "dataset-1",
        x_column="active",
        y_column="sales",
    )

    series = result.result.specification.data.series[0]

    assert series.x_values == ("True", "False", "True")


def test_line_chart_normalizes_infinite_y_values() -> None:
    context = create_dataframe_context(
        {
            "day": ["Monday", "Tuesday", "Wednesday"],
            "sales": [100, np.inf, -np.inf],
        }
    )

    service, _ = create_visualization_service()

    result = service.create_line_chart(
        context,
        "dataset-1",
        x_column="day",
        y_column="sales",
    )

    series = result.result.specification.data.series[0]

    assert series.y_values == (100, None, None)


@pytest.mark.parametrize(
    "y_values",
    [
        [True, False],
        [1 + 2j, 3 + 4j],
    ],
    ids=["boolean", "complex"],
)
def test_line_chart_rejects_non_real_y_values(
    y_values: list[object],
) -> None:
    context = create_dataframe_context(
        {
            "x": [1, 2],
            "y": y_values,
        }
    )

    service, renderer = create_visualization_service()

    with pytest.raises(VisualizationValidationError):
        service.create_line_chart(
            context,
            "dataset-1",
            x_column="x",
            y_column="y",
        )

    renderer.render.assert_not_called()


def test_scatter_chart_rejects_string_y_values() -> None:
    context = create_dataframe_context(
        {
            "x": [1, 2, 3],
            "y": ["10", "20", "30"],
        }
    )

    service, renderer = create_visualization_service()

    with pytest.raises(VisualizationValidationError):
        service.create_scatter_chart(
            context,
            "dataset-1",
            x_column="x",
            y_column="y",
        )

    renderer.render.assert_not_called()


def test_scatter_chart_rejects_complex_x_values() -> None:
    context = create_dataframe_context(
        {
            "x": [1 + 2j, 3 + 4j],
            "y": [10, 20],
        }
    )

    service, renderer = create_visualization_service()

    with pytest.raises(VisualizationValidationError):
        service.create_scatter_chart(
            context,
            "dataset-1",
            x_column="x",
            y_column="y",
        )

    renderer.render.assert_not_called()


def test_scatter_chart_rejects_complex_y_values() -> None:
    context = create_dataframe_context(
        {
            "x": [1, 2],
            "y": [1 + 2j, 3 + 4j],
        }
    )

    service, renderer = create_visualization_service()

    with pytest.raises(VisualizationValidationError):
        service.create_scatter_chart(
            context,
            "dataset-1",
            x_column="x",
            y_column="y",
        )

    renderer.render.assert_not_called()

def test_line_chart_does_not_call_statistical_service() -> None:
    context = create_dataframe_context(
        {
            "x": [1, 2, 3],
            "y": [10, 20, 30],
        }
    )

    statistical_service = MagicMock()
    renderer = MagicMock()

    renderer.render.return_value = RenderedChart(
        image_data=b"fake-image-data",
        output_format=OutputFormat.PNG,
        width=1200,
        height=800,
    )

    service = VisualizationService(
        statistical_service=statistical_service,
        renderer=renderer,
    )

    service.create_line_chart(
        context,
        "dataset-1",
        x_column="x",
        y_column="y",
    )

    statistical_service.assert_not_called()
    assert context.is_closed is False

def test_scatter_chart_does_not_call_statistical_service() -> None:
    context = create_dataframe_context(
        {
            "x": [1, 2, 3],
            "y": [10, 20, 30],
        }
    )

    statistical_service = MagicMock()
    renderer = MagicMock()

    renderer.render.return_value = RenderedChart(
        image_data=b"fake-image-data",
        output_format=OutputFormat.PNG,
        width=1200,
        height=800,
    )

    service = VisualizationService(
        statistical_service=statistical_service,
        renderer=renderer,
    )

    service.create_scatter_chart(
        context,
        "dataset-1",
        x_column="x",
        y_column="y",
    )

    statistical_service.assert_not_called()
    assert context.is_closed is False


# ---------------------------------------------------------------------------
# Histogram chart
# ---------------------------------------------------------------------------


def test_creates_histogram_chart() -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    statistical_service.analyze_distribution.return_value = (
        create_distribution_result()
    )

    result = service.create_histogram(
        context,
        "dataset-1",
        column="amount",
    )

    assert isinstance(result, VisualizationOutput)

    specification = result.result.specification

    assert specification.chart_type is ChartType.HISTOGRAM
    assert specification.title == "amount Distribution"
    assert specification.x_label == "amount"
    assert specification.y_label == "Frequency"

    assert specification.data.histogram_bins == (
        create_distribution_result().histogram
    )

    assert specification.data.metadata["source_analysis"] == "distribution"
    assert specification.data.metadata["requested_bins"] == 10
    assert specification.data.metadata["analysis_count"] == 4
    assert specification.data.metadata["histogram_count"] == 4
    assert specification.data.metadata["excluded_count"] == 0

    statistical_service.analyze_distribution.assert_called_once_with(
        context,
        "dataset-1",
        column="amount",
        bins=10,
    )

    renderer.render.assert_called_once_with(
        specification,
        output_format=OutputFormat.PNG,
    )

    context.close()


def test_histogram_supports_custom_bins_and_svg() -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    statistical_service.analyze_distribution.return_value = (
        create_distribution_result()
    )

    result = service.create_histogram(
        context,
        "dataset-1",
        column="amount",
        bins=5,
        output_format=OutputFormat.SVG,
        title="Amount Distribution",
    )

    assert result.result.specification.title == "Amount Distribution"

    statistical_service.analyze_distribution.assert_called_once_with(
        context,
        "dataset-1",
        column="amount",
        bins=5,
    )

    renderer.render.assert_called_once_with(
        result.result.specification,
        output_format=OutputFormat.SVG,
    )

    context.close()


@pytest.mark.parametrize(
    "bins",
    [0, -1, True, 1.5, "2", None],
)
def test_histogram_rejects_invalid_bins(bins) -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    with pytest.raises(
        VisualizationValidationError,
        match="bins must be a positive integer",
    ):
        service.create_histogram(
            context,
            "dataset-1",
            column="amount",
            bins=bins,
        )

    statistical_service.analyze_distribution.assert_not_called()
    renderer.render.assert_not_called()

    context.close()


def test_histogram_rejects_empty_distribution() -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    statistical_service.analyze_distribution.return_value = (
        create_distribution_result(
            count=0,
            histogram=(),
        )
    )

    with pytest.raises(
        VisualizationValidationError,
        match="no valid numeric values",
    ):
        service.create_histogram(
            context,
            "dataset-1",
            column="amount",
        )

    renderer.render.assert_not_called()
    context.close()


def test_histogram_rejects_zero_width_bins() -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    statistical_service.analyze_distribution.return_value = (
        create_distribution_result(
            histogram=(
                HistogramBin(
                    lower_bound=0.0,
                    upper_bound=0.0,
                    count=4,
                    percentage=100.0,
                ),
            )
        )
    )

    with pytest.raises(
        VisualizationValidationError,
        match="zero-width bins",
    ):
        service.create_histogram(
            context,
            "dataset-1",
            column="amount",
        )

    renderer.render.assert_not_called()
    context.close()


def test_histogram_accepts_non_zero_constant_distribution() -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    statistical_service.analyze_distribution.return_value = (
        create_distribution_result(
            histogram=(
                HistogramBin(
                    lower_bound=4.5,
                    upper_bound=5.5,
                    count=4,
                    percentage=100.0,
                ),
            )
        )
    )

    result = service.create_histogram(
        context,
        "dataset-1",
        column="amount",
    )

    assert result.result.specification.chart_type is ChartType.HISTOGRAM
    renderer.render.assert_called_once()

    context.close()


# ---------------------------------------------------------------------------
# Pie chart
# ---------------------------------------------------------------------------


def test_creates_pie_chart() -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    statistical_service.analyze_frequency.return_value = (
        create_frequency_result()
    )

    result = service.create_pie_chart(
        context,
        "dataset-1",
        column="category",
    )

    assert isinstance(result, VisualizationOutput)

    specification = result.result.specification

    assert specification.chart_type is ChartType.PIE
    assert specification.title == "category Composition"

    series = specification.data.series[0]

    assert series.name == "category"
    assert series.x_values == ("A", "B", "C")
    assert series.y_values == (2, 1, 1)

    metadata = specification.data.metadata

    assert metadata["source_analysis"] == "frequency"
    assert metadata["total_count"] == 4
    assert metadata["returned_category_count"] == 3
    assert metadata["displayed_count"] == 4
    assert metadata["displayed_percentage"] == 100.0
    assert metadata["category_scope"] == "all"
    assert metadata["pie_normalization"] == "displayed_categories"

    statistical_service.analyze_frequency.assert_called_once_with(
        context,
        "dataset-1",
        column="category",
        top_n=None,
    )

    renderer.render.assert_called_once_with(
        specification,
        output_format=OutputFormat.PNG,
    )

    context.close()


def test_pie_chart_supports_top_n() -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    statistical_service.analyze_frequency.return_value = (
        create_frequency_result(
            items=(
                FrequencyItem(
                    value="A",
                    count=50,
                    percentage=50.0,
                    cumulative_percentage=50.0,
                ),
                FrequencyItem(
                    value="B",
                    count=30,
                    percentage=30.0,
                    cumulative_percentage=80.0,
                ),
            ),
            total_count=100,
        )
    )

    result = service.create_pie_chart(
        context,
        "dataset-1",
        column="category",
        top_n=2,
        output_format=OutputFormat.SVG,
    )

    statistical_service.analyze_frequency.assert_called_once_with(
        context,
        "dataset-1",
        column="category",
        top_n=2,
    )

    specification = result.result.specification
    metadata = specification.data.metadata

    assert specification.data.series[0].x_values == ("A", "B")
    assert specification.data.series[0].y_values == (50, 30)

    assert metadata["top_n"] == 2
    assert metadata["total_count"] == 100
    assert metadata["displayed_count"] == 80
    assert metadata["displayed_percentage"] == 80.0
    assert metadata["category_scope"] == "top_n"

    assert result.result.metadata["category_scope"] == "top_n"
    assert result.result.metadata["displayed_percentage"] == 80.0

    renderer.render.assert_called_once_with(
        specification,
        output_format=OutputFormat.SVG,
    )

    context.close()


def test_pie_chart_rejects_empty_categories() -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    statistical_service.analyze_frequency.return_value = (
        create_frequency_result(items=(), total_count=0)
    )

    with pytest.raises(
        VisualizationValidationError,
        match="no categories",
    ):
        service.create_pie_chart(
            context,
            "dataset-1",
            column="category",
        )

    renderer.render.assert_not_called()
    context.close()


def test_pie_chart_rejects_zero_displayed_count() -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    statistical_service.analyze_frequency.return_value = (
        create_frequency_result(
            items=(
                FrequencyItem(
                    value="A",
                    count=0,
                    percentage=0.0,
                    cumulative_percentage=0.0,
                ),
            ),
            total_count=1,
        )
    )

    with pytest.raises(
        VisualizationValidationError,
        match="positive total count",
    ):
        service.create_pie_chart(
            context,
            "dataset-1",
            column="category",
        )

    renderer.render.assert_not_called()
    context.close()


@pytest.mark.parametrize(
    "top_n",
    [0, -1, True, 1.5, "2"],
)
def test_pie_chart_rejects_invalid_top_n(top_n) -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    with pytest.raises(VisualizationValidationError):
        service.create_pie_chart(
            context,
            "dataset-1",
            column="category",
            top_n=top_n,
        )

    statistical_service.analyze_frequency.assert_not_called()
    renderer.render.assert_not_called()

    context.close()


# ---------------------------------------------------------------------------
# Box chart
# ---------------------------------------------------------------------------


def test_creates_box_chart() -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    statistical_service.analyze_descriptive.return_value = (
        create_descriptive_result()
    )

    result = service.create_box_chart(
        context,
        "dataset-1",
        column="amount",
    )

    assert isinstance(result, VisualizationOutput)

    specification = result.result.specification

    assert specification.chart_type is ChartType.BOX
    assert specification.title == "amount Box Plot"
    assert specification.x_label == "Distribution"
    assert specification.y_label == "amount"

    series = specification.data.series[0]

    assert series.name == "amount"
    assert series.x_values == ()
    assert series.y_values == ()
    assert series.statistics == {
        "min": 100.0,
        "q1": 175.0,
        "median": 250.0,
        "q3": 325.0,
        "max": 400.0,
    }
    assert series.outliers == ()

    assert (
        specification.data.metadata["source_analysis"]
        == "descriptive"
    )
    assert specification.data.metadata["count"] == 4
    assert specification.data.metadata["outlier_count"] == 0
    assert (
        specification.data.metadata["outlier_method"]
        == "not_calculated"
    )

    statistical_service.analyze_descriptive.assert_called_once_with(
        context,
        "dataset-1",
        columns=["amount"],
    )

    renderer.render.assert_called_once_with(
        specification,
        output_format=OutputFormat.PNG,
    )

    context.close()


def test_box_chart_supports_custom_title_and_svg() -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    statistical_service.analyze_descriptive.return_value = (
        create_descriptive_result()
    )

    result = service.create_box_chart(
        context,
        "dataset-1",
        column="amount",
        title="Sales Amount Distribution",
        output_format=OutputFormat.SVG,
    )

    assert (
        result.result.specification.title
        == "Sales Amount Distribution"
    )

    renderer.render.assert_called_once_with(
        result.result.specification,
        output_format=OutputFormat.SVG,
    )

    context.close()


def test_box_chart_rejects_empty_column() -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    statistical_service.analyze_descriptive.return_value = (
        create_descriptive_result(
            count=0,
            minimum=None,
            q1=None,
            median=None,
            q3=None,
            maximum=None,
        )
    )

    with pytest.raises(
        VisualizationValidationError,
        match="no valid numeric values",
    ):
        service.create_box_chart(
            context,
            "dataset-1",
            column="amount",
        )

    renderer.render.assert_not_called()
    context.close()


@pytest.mark.parametrize(
    "missing_field",
    ["min", "q1", "median", "q3", "max"],
)
def test_box_chart_rejects_incomplete_statistics(
    missing_field: str,
) -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    values = {
        "minimum": 100.0,
        "q1": 175.0,
        "median": 250.0,
        "q3": 325.0,
        "maximum": 400.0,
    }

    field_mapping = {
        "min": "minimum",
        "q1": "q1",
        "median": "median",
        "q3": "q3",
        "max": "maximum",
    }

    values[field_mapping[missing_field]] = None

    statistical_service.analyze_descriptive.return_value = (
        create_descriptive_result(**values)
    )

    with pytest.raises(
        VisualizationValidationError,
        match="incomplete box plot statistics",
    ):
        service.create_box_chart(
            context,
            "dataset-1",
            column="amount",
        )

    renderer.render.assert_not_called()
    context.close()


@pytest.mark.parametrize(
    "column",
    ["", "   "],
)
def test_box_chart_rejects_invalid_column(column: str) -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    with pytest.raises(VisualizationValidationError):
        service.create_box_chart(
            context,
            "dataset-1",
            column=column,
        )

    statistical_service.analyze_descriptive.assert_not_called()
    renderer.render.assert_not_called()

    context.close()


def test_box_chart_rejects_invalid_output_format() -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    with pytest.raises(VisualizationValidationError):
        service.create_box_chart(
            context,
            "dataset-1",
            column="amount",
            output_format="jpg",  # type: ignore[arg-type]
        )

    statistical_service.analyze_descriptive.assert_not_called()
    renderer.render.assert_not_called()

    context.close()


def test_box_chart_does_not_close_execution_context() -> None:
    context = create_context()
    service, statistical_service, renderer = create_service()

    statistical_service.analyze_descriptive.return_value = (
        create_descriptive_result()
    )

    service.create_box_chart(
        context,
        "dataset-1",
        column="amount",
    )

    assert not context.is_closed
    assert context.dataset_ids == ("dataset-1",)

    context.close()