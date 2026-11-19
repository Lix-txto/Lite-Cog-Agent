"""Robustness tests for visualization service error boundaries."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from uuid import UUID
from dataclasses import replace
from unittest.mock import Mock

import pandas as pd
from pandas.testing import assert_frame_equal

import pytest

from lite_cog_agent.statistics.exceptions import (
    FrequencyAnalysisError,
    StatisticalValidationError,
)
from lite_cog_agent.statistics.service import StatisticalAnalysisService
from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.execution.dataset_store import DatasetReference
from lite_cog_agent.visualization.exceptions import (
    VisualizationException,
    VisualizationExecutionError,
    VisualizationValidationError,
)
from lite_cog_agent.visualization.renderer import OutputFormat, ChartRenderer, RenderedChart
from lite_cog_agent.visualization.service import VisualizationService
from lite_cog_agent.visualization.outputs import VisualizationOutput
from lite_cog_agent.visualization.contracts import ChartType


@pytest.fixture
def registered_dataset():
    """Create a real execution context with a registered DataFrame."""
    dataframe = pd.DataFrame(
        {
            "category": ["A", "B", "A"],
            "amount": [100, 200, 300],
        }
    )

    context = ExecutionContext()

    reference = DatasetReference(
        name="robustness_dataset",
        source="integration_test",
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        columns=tuple(dataframe.columns),
    )

    context.register_dataset(reference, dataframe)

    yield context, reference, dataframe

    context.close()


def test_missing_dataset_raises_validation_error() -> None:
    """A missing dataset should be treated as invalid input."""
    context = ExecutionContext()

    try:
        service = VisualizationService()

        with pytest.raises(
            VisualizationValidationError,
            match="does not exist",
        ):
            service.create_bar_chart(
                context,
                "missing-dataset",
                column="category",
            )
    finally:
        context.close()


def test_closed_context_raises_execution_error(
    registered_dataset,
) -> None:
    """A closed context should not be used for visualization."""
    context, reference, _ = registered_dataset

    context.close()

    service = VisualizationService()

    with pytest.raises(
        VisualizationExecutionError,
        match="closed context",
    ):
        service.create_bar_chart(
            context,
            reference.dataset_id,
            column="category",
        )


def test_invalid_context_raises_validation_error() -> None:
    """The service should reject objects that are not ExecutionContext."""
    service = VisualizationService()

    with pytest.raises(
        VisualizationValidationError,
        match="ExecutionContext",
    ):
        service.create_bar_chart(
            Mock(),
            "dataset-id",
            column="category",
        )


def test_missing_column_raises_validation_error(
    registered_dataset,
) -> None:
    """A missing DataFrame column should be rejected."""
    context, reference, _ = registered_dataset

    service = VisualizationService()

    with pytest.raises(
        VisualizationValidationError,
        match="does not exist",
    ):
        service.create_line_chart(
            context,
            reference.dataset_id,
            x_column="missing_column",
            y_column="amount",
        )


def test_missing_column_does_not_close_context(
    registered_dataset,
) -> None:
    """Validation failure must not change context lifecycle."""
    context, reference, dataframe = registered_dataset

    service = VisualizationService()

    with pytest.raises(VisualizationValidationError):
        service.create_bar_chart(
            context,
            reference.dataset_id,
            column="missing_column",
        )

    assert context.is_closed is False
    assert context.get_dataset(reference.dataset_id) is dataframe



@pytest.mark.parametrize(
    "output_format",
    [
        None,
        "png",
        "svg",
        "jpeg",
        123,
        object(),
    ],
)
def test_invalid_output_format_raises_validation_error(
    registered_dataset,
    output_format,
) -> None:
    context, reference, _ = registered_dataset
    service = VisualizationService()

    with pytest.raises(VisualizationValidationError):
        service.create_bar_chart(
            context,
            reference.dataset_id,
            column="category",
            output_format=output_format,
        )


@pytest.mark.parametrize(
    "column",
    [
        "",
        None,
        123,
        [],
    ],
)
def test_invalid_column_raises_validation_error(
    registered_dataset,
    column,
) -> None:
    context, reference, _ = registered_dataset
    service = VisualizationService()

    with pytest.raises(VisualizationValidationError):
        service.create_bar_chart(
            context,
            reference.dataset_id,
            column=column,
        )


@pytest.mark.parametrize(
    "top_n",
    [
        0,
        -1,
        1.5,
        True,
        False,
        "3",
    ],
)
def test_invalid_top_n_raises_validation_error(
    registered_dataset,
    top_n,
) -> None:
    context, reference, _ = registered_dataset
    service = VisualizationService()

    with pytest.raises(VisualizationValidationError):
        service.create_bar_chart(
            context,
            reference.dataset_id,
            column="category",
            top_n=top_n,
        )


@pytest.mark.parametrize(
    "bins",
    [
        0,
        -1,
        1.5,
        True,
        False,
        "4",
    ],
)
def test_invalid_bins_raises_validation_error(
    registered_dataset,
    bins,
) -> None:
    context, reference, _ = registered_dataset
    service = VisualizationService()

    with pytest.raises(VisualizationValidationError):
        service.create_histogram(
            context,
            reference.dataset_id,
            column="amount",
            bins=bins,
        )


@pytest.mark.parametrize(
    "group_by",
    [
        [],
        (),
        "",
        None,
        ["department", "department"],
        ["department", ""],
        ["department", None],
        ["department", 123],
    ],
)
def test_invalid_group_by_raises_validation_error(
    registered_dataset,
    group_by,
) -> None:
    context, reference, _ = registered_dataset
    service = VisualizationService()

    with pytest.raises(VisualizationValidationError):
        service.create_grouped_bar_chart(
            context,
            reference.dataset_id,
            group_by=group_by,
            aggregations={"amount": ["sum"]},
        )


@pytest.mark.parametrize(
    "aggregations",
    [
        {},
        None,
        [],
        {"": ["sum"]},
        {"amount": []},
        {"amount": ""},
        {"amount": None},
        {"amount": ["sum", "sum"]},
        {"amount": ["sum", ""]},
        {"amount": ["sum", 123]},
        {"amount": "sum"},
    ],
)
def test_invalid_aggregations_raises_validation_error(
    registered_dataset,
    aggregations,
) -> None:
    context, reference, _ = registered_dataset
    service = VisualizationService()

    with pytest.raises(VisualizationValidationError):
        service.create_grouped_bar_chart(
            context,
            reference.dataset_id,
            group_by=["category"],
            aggregations=aggregations,
        )


def test_invalid_parameters_do_not_modify_dataset_or_close_context(
    registered_dataset,
) -> None:
    context, reference, dataframe = registered_dataset
    original_dataframe = dataframe.copy(deep=True)

    service = VisualizationService()

    invalid_calls = [
        lambda: service.create_bar_chart(
            context,
            reference.dataset_id,
            column="category",
            top_n=0,
        ),
        lambda: service.create_histogram(
            context,
            reference.dataset_id,
            column="amount",
            bins=0,
        ),
        lambda: service.create_line_chart(
            context,
            reference.dataset_id,
            x_column="",
            y_column="amount",
        ),
        lambda: service.create_grouped_bar_chart(
            context,
            reference.dataset_id,
            group_by=[],
            aggregations={"amount": ["sum"]},
        ),
    ]

    for invalid_call in invalid_calls:
        with pytest.raises(VisualizationValidationError):
            invalid_call()

    assert context.is_closed is False
    assert context.get_dataset(reference.dataset_id) is dataframe
    assert_frame_equal(dataframe, original_dataframe)



def _create_statistical_boundary_service():
    """Create a service with injectable statistical and rendering mocks."""
    statistical_service = Mock(spec=StatisticalAnalysisService)
    renderer = Mock(spec=ChartRenderer)

    service = VisualizationService(
        statistical_service=statistical_service,
        renderer=renderer,
    )

    return service, statistical_service, renderer


def _assert_dataset_unchanged(
    context,
    reference,
    dataframe,
    original_dataframe,
):
    """Verify statistical failures do not damage the registered dataset."""
    assert_frame_equal(dataframe, original_dataframe)

    assert context.is_closed is False
    assert context.get_dataset(reference.dataset_id) is dataframe
    assert (
        context.get_dataset_reference(reference.dataset_id)
        == reference
    )


def test_statistical_validation_error_is_converted(
    registered_dataset,
):
    context, reference, dataframe = registered_dataset
    original_dataframe = dataframe.copy(deep=True)

    service, statistical_service, renderer = (
        _create_statistical_boundary_service()
    )

    statistical_service.analyze_frequency.side_effect = (
        StatisticalValidationError("invalid statistical input")
    )

    with pytest.raises(VisualizationValidationError) as exc_info:
        service.create_bar_chart(
            context,
            reference.dataset_id,
            column="category",
        )

    assert isinstance(
        exc_info.value.__cause__,
        StatisticalValidationError,
    )

    statistical_service.analyze_frequency.assert_called_once()
    renderer.render.assert_not_called()

    _assert_dataset_unchanged(
        context,
        reference,
        dataframe,
        original_dataframe,
    )


def test_statistical_analysis_error_is_converted(
    registered_dataset,
):
    context, reference, dataframe = registered_dataset
    original_dataframe = dataframe.copy(deep=True)

    service, statistical_service, renderer = (
        _create_statistical_boundary_service()
    )

    statistical_service.analyze_frequency.side_effect = (
        FrequencyAnalysisError("frequency calculation failed")
    )

    with pytest.raises(VisualizationExecutionError) as exc_info:
        service.create_bar_chart(
            context,
            reference.dataset_id,
            column="category",
        )

    assert isinstance(
        exc_info.value.__cause__,
        FrequencyAnalysisError,
    )

    statistical_service.analyze_frequency.assert_called_once()
    renderer.render.assert_not_called()

    _assert_dataset_unchanged(
        context,
        reference,
        dataframe,
        original_dataframe,
    )


def test_unexpected_statistical_error_is_converted(
    registered_dataset,
):
    context, reference, dataframe = registered_dataset
    original_dataframe = dataframe.copy(deep=True)

    service, statistical_service, renderer = (
        _create_statistical_boundary_service()
    )

    statistical_service.analyze_frequency.side_effect = (
        RuntimeError("unexpected statistical failure")
    )

    with pytest.raises(VisualizationExecutionError) as exc_info:
        service.create_bar_chart(
            context,
            reference.dataset_id,
            column="category",
        )

    assert isinstance(
        exc_info.value.__cause__,
        RuntimeError,
    )

    statistical_service.analyze_frequency.assert_called_once()
    renderer.render.assert_not_called()

    _assert_dataset_unchanged(
        context,
        reference,
        dataframe,
        original_dataframe,
    )


def test_invalid_statistical_result_type_is_rejected(
    registered_dataset,
):
    context, reference, dataframe = registered_dataset
    original_dataframe = dataframe.copy(deep=True)

    service, statistical_service, renderer = (
        _create_statistical_boundary_service()
    )

    statistical_service.analyze_frequency.return_value = object()

    with pytest.raises(
        VisualizationExecutionError,
        match="invalid result type",
    ):
        service.create_bar_chart(
            context,
            reference.dataset_id,
            column="category",
        )

    renderer.render.assert_not_called()

    _assert_dataset_unchanged(
        context,
        reference,
        dataframe,
        original_dataframe,
    )


@pytest.mark.parametrize(
    ("field", "value", "expected_message"),
    [
        (
            "dataset_id",
            "another-dataset",
            "does not match",
        ),
        (
            "dataset_name",
            "another-dataset-name",
            "does not match",
        ),
        (
            "column",
            "another-column",
            "unexpected column",
        ),
    ],
)
def test_statistical_result_metadata_mismatch_is_rejected(
    registered_dataset,
    field,
    value,
    expected_message,
):
    context, reference, dataframe = registered_dataset
    original_dataframe = dataframe.copy(deep=True)

    service, statistical_service, renderer = (
        _create_statistical_boundary_service()
    )

    # Obtain a valid result from the real statistical service first.
    valid_result = StatisticalAnalysisService().analyze_frequency(
        context,
        reference.dataset_id,
        column="category",
    )

    # Modify exactly one metadata field to simulate an invalid result.
    invalid_result = replace(
        valid_result,
        **{field: value},
    )

    statistical_service.analyze_frequency.return_value = invalid_result

    with pytest.raises(
        VisualizationExecutionError,
        match=expected_message,
    ):
        service.create_bar_chart(
            context,
            reference.dataset_id,
            column="category",
        )

    statistical_service.analyze_frequency.assert_called_once()
    renderer.render.assert_not_called()

    _assert_dataset_unchanged(
        context,
        reference,
        dataframe,
        original_dataframe,
    )



def _create_renderer_boundary_service(renderer):
    """Use the real statistical service and inject a renderer."""
    return VisualizationService(renderer=renderer)


def test_renderer_validation_error_is_preserved(
    registered_dataset,
):
    context, reference, dataframe = registered_dataset
    original_dataframe = dataframe.copy(deep=True)

    renderer = Mock(spec=ChartRenderer)
    renderer.render.side_effect = VisualizationValidationError(
        "invalid rendering specification"
    )

    service = _create_renderer_boundary_service(renderer)

    with pytest.raises(
        VisualizationValidationError,
        match="invalid rendering specification",
    ):
        service.create_bar_chart(
            context,
            reference.dataset_id,
            column="category",
        )

    renderer.render.assert_called_once()

    _assert_dataset_unchanged(
        context,
        reference,
        dataframe,
        original_dataframe,
    )


def test_renderer_execution_error_is_preserved(
    registered_dataset,
):
    context, reference, dataframe = registered_dataset
    original_dataframe = dataframe.copy(deep=True)

    renderer = Mock(spec=ChartRenderer)
    renderer.render.side_effect = VisualizationExecutionError(
        "renderer execution failed"
    )

    service = _create_renderer_boundary_service(renderer)

    with pytest.raises(
        VisualizationExecutionError,
        match="renderer execution failed",
    ):
        service.create_bar_chart(
            context,
            reference.dataset_id,
            column="category",
        )

    renderer.render.assert_called_once()

    _assert_dataset_unchanged(
        context,
        reference,
        dataframe,
        original_dataframe,
    )


def test_generic_visualization_exception_is_wrapped(
    registered_dataset,
):
    context, reference, dataframe = registered_dataset
    original_dataframe = dataframe.copy(deep=True)

    renderer = Mock(spec=ChartRenderer)
    renderer.render.side_effect = VisualizationException(
        "generic renderer failure"
    )

    service = _create_renderer_boundary_service(renderer)

    with pytest.raises(
        VisualizationExecutionError,
        match="renderer failed",
    ) as exc_info:
        service.create_bar_chart(
            context,
            reference.dataset_id,
            column="category",
        )

    assert isinstance(
        exc_info.value.__cause__,
        VisualizationException,
    )

    renderer.render.assert_called_once()

    _assert_dataset_unchanged(
        context,
        reference,
        dataframe,
        original_dataframe,
    )


def test_unexpected_renderer_error_is_wrapped(
    registered_dataset,
):
    context, reference, dataframe = registered_dataset
    original_dataframe = dataframe.copy(deep=True)

    renderer = Mock(spec=ChartRenderer)
    renderer.render.side_effect = RuntimeError(
        "unexpected renderer failure"
    )

    service = _create_renderer_boundary_service(renderer)

    with pytest.raises(
        VisualizationExecutionError,
        match="unexpected renderer error",
    ) as exc_info:
        service.create_bar_chart(
            context,
            reference.dataset_id,
            column="category",
        )

    assert isinstance(
        exc_info.value.__cause__,
        RuntimeError,
    )

    renderer.render.assert_called_once()

    _assert_dataset_unchanged(
        context,
        reference,
        dataframe,
        original_dataframe,
    )


@pytest.mark.parametrize(
    "invalid_result",
    [
        None,
        object(),
        b"not-a-rendered-chart",
    ],
)
def test_invalid_renderer_result_type_is_rejected(
    registered_dataset,
    invalid_result,
):
    context, reference, dataframe = registered_dataset
    original_dataframe = dataframe.copy(deep=True)

    renderer = Mock(spec=ChartRenderer)
    renderer.render.return_value = invalid_result

    service = _create_renderer_boundary_service(renderer)

    with pytest.raises(
        VisualizationExecutionError,
        match="invalid result type",
    ):
        service.create_bar_chart(
            context,
            reference.dataset_id,
            column="category",
        )

    renderer.render.assert_called_once()

    _assert_dataset_unchanged(
        context,
        reference,
        dataframe,
        original_dataframe,
    )


def test_renderer_receives_chart_specification_and_output_format(
    registered_dataset,
):
    context, reference, dataframe = registered_dataset
    original_dataframe = dataframe.copy(deep=True)

    rendered_chart = RenderedChart(
        image_data=b"test-image-data",
        output_format=OutputFormat.PNG,
        width=640,
        height=480,
    )

    renderer = Mock(spec=ChartRenderer)
    renderer.render.return_value = rendered_chart

    service = _create_renderer_boundary_service(renderer)

    output = service.create_bar_chart(
        context,
        reference.dataset_id,
        column="category",
        output_format=OutputFormat.PNG,
    )

    renderer.render.assert_called_once()

    call_args = renderer.render.call_args

    assert call_args.args[0] is output.result.specification
    assert call_args.kwargs["output_format"] is OutputFormat.PNG

    assert isinstance(output, VisualizationOutput)
    assert output.rendered_chart is rendered_chart
    assert output.result.dataset_id == reference.dataset_id
    assert output.result.dataset_name == reference.name

    _assert_dataset_unchanged(
        context,
        reference,
        dataframe,
        original_dataframe,
    )



@pytest.mark.parametrize(
    "output_format",
    [
        OutputFormat.PNG,
        OutputFormat.SVG,
    ],
)
def test_output_packaging_preserves_complete_visualization_result(
    registered_dataset,
    output_format,
):
    """The final output should preserve all visualization components."""
    context, reference, dataframe = registered_dataset
    original_dataframe = dataframe.copy(deep=True)

    rendered_chart = RenderedChart(
        image_data=b"packaged-chart-data",
        output_format=output_format,
        width=640,
        height=480,
        metadata={
            "renderer": "integration-test",
            "dpi": 72,
        },
    )

    renderer = Mock(spec=ChartRenderer)
    renderer.render.return_value = rendered_chart

    service = _create_renderer_boundary_service(renderer)

    output = service.create_bar_chart(
        context,
        reference.dataset_id,
        column="category",
        output_format=output_format,
        title="Packaging Test",
    )

    # 1. Validate final output contract.
    assert isinstance(output, VisualizationOutput)

    # 2. Validate logical result contract.
    result = output.result

    assert result.dataset_id == reference.dataset_id
    assert result.dataset_name == reference.name

    assert result.specification.chart_type is ChartType.BAR
    assert result.specification.title == "Packaging Test"
    assert result.specification.x_label == "category"
    assert result.specification.y_label == "Count"

    # 3. Validate generated visualization ID.
    visualization_uuid = UUID(result.visualization_id)

    assert visualization_uuid.version == 4
    assert str(visualization_uuid) == result.visualization_id

    # 4. Validate chart specification data.
    chart_data = result.specification.data

    assert chart_data.dataset_id == reference.dataset_id
    assert chart_data.dataset_name == reference.name

    assert len(chart_data.series) == 1

    series = chart_data.series[0]

    assert series.name == "category"
    assert series.x_values == ("A", "B")
    assert series.y_values == (2, 1)

    assert dict(chart_data.metadata) == {
        "source_analysis": "frequency",
        "column": "category",
        "top_n": None,
        "total_count": 3,
        "returned_category_count": 2,
        "category_scope": "all",
    }

    # 5. Validate result-level metadata.
    assert dict(result.metadata) == {
        "source_analysis": "frequency",
        "column": "category",
        "top_n": None,
    }

    # 6. Validate rendered chart pass-through.
    assert output.rendered_chart is rendered_chart

    assert output.rendered_chart.image_data == b"packaged-chart-data"
    assert output.rendered_chart.output_format is output_format
    assert output.rendered_chart.width == 640
    assert output.rendered_chart.height == 480

    assert output.rendered_chart.mime_type == (
        "image/png"
        if output_format is OutputFormat.PNG
        else "image/svg+xml"
    )

    assert dict(output.rendered_chart.metadata) == {
        "renderer": "integration-test",
        "dpi": 72,
    }

    # 7. Validate renderer invocation.
    renderer.render.assert_called_once_with(
        result.specification,
        output_format=output_format,
    )

    # 8. Validate dataset integrity and context lifecycle.
    _assert_dataset_unchanged(
        context,
        reference,
        dataframe,
        original_dataframe,
    )


def test_output_packaging_contracts_are_immutable(
    registered_dataset,
):
    """Packaged output and its metadata should remain immutable."""
    context, reference, dataframe = registered_dataset
    original_dataframe = dataframe.copy(deep=True)

    rendered_chart = RenderedChart(
        image_data=b"immutable-chart-data",
        output_format=OutputFormat.PNG,
        width=640,
        height=480,
        metadata={
            "renderer": "integration-test",
        },
    )

    renderer = Mock(spec=ChartRenderer)
    renderer.render.return_value = rendered_chart

    service = _create_renderer_boundary_service(renderer)

    output = service.create_bar_chart(
        context,
        reference.dataset_id,
        column="category",
    )

    # 1. VisualizationOutput is frozen.
    with pytest.raises(FrozenInstanceError):
        output.result = output.result

    with pytest.raises(FrozenInstanceError):
        output.rendered_chart = rendered_chart

    # 2. VisualizationResult is frozen.
    with pytest.raises(FrozenInstanceError):
        output.result.dataset_name = "modified-dataset"

    # 3. Result metadata is read-only.
    with pytest.raises(TypeError):
        output.result.metadata["source_analysis"] = "modified"

    # 4. ChartData metadata is read-only.
    with pytest.raises(TypeError):
        output.result.specification.data.metadata["column"] = "modified"

    # 5. RenderedChart metadata is read-only.
    with pytest.raises(TypeError):
        output.rendered_chart.metadata["renderer"] = "modified"

    # 6. Verify original values remain unchanged.
    assert output.result.dataset_name == reference.name

    assert (
        output.result.metadata["source_analysis"]
        == "frequency"
    )

    assert (
        output.result.specification.data.metadata["column"]
        == "category"
    )

    assert (
        output.rendered_chart.metadata["renderer"]
        == "integration-test"
    )

    _assert_dataset_unchanged(
        context,
        reference,
        dataframe,
        original_dataframe,
    )
