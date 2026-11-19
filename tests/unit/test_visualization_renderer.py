"""Tests for visualization renderer contracts."""

from collections.abc import Mapping

import pytest

from lite_cog_agent.visualization import (
    ChartData,
    ChartRenderer,
    ChartSeries,
    ChartSpecification,
    ChartType,
    OutputFormat,
    RenderedChart,
    VisualizationValidationError,
)


@pytest.fixture
def bar_specification():
    """Provide a valid bar chart specification."""

    data = ChartData(
        dataset_id="dataset-001",
        dataset_name="sales",
        series=(
            ChartSeries(
                name="revenue",
                x_values=("A", "B", "C"),
                y_values=(100.0, 200.0, 150.0),
            ),
        ),
    )

    return ChartSpecification(
        chart_type=ChartType.BAR,
        title="Revenue",
        data=data,
    )

class TestOutputFormat:
    """Tests for OutputFormat."""

    def test_supports_png(self):
        assert OutputFormat.PNG.value == "png"

    def test_supports_svg(self):
        assert OutputFormat.SVG.value == "svg"

    def test_output_format_is_string_enum(self):
        assert isinstance(OutputFormat.PNG, str)


class TestRenderedChart:
    """Tests for RenderedChart."""

    @pytest.fixture
    def valid_chart(self):
        return RenderedChart(
            image_data=b"image-bytes",
            output_format=OutputFormat.PNG,
            width=1200,
            height=800,
        )

    def test_creates_valid_png_result(self, valid_chart):
        assert valid_chart.image_data == b"image-bytes"
        assert valid_chart.output_format is OutputFormat.PNG
        assert valid_chart.width == 1200
        assert valid_chart.height == 800
        assert valid_chart.mime_type == "image/png"

    def test_creates_valid_svg_result(self):
        chart = RenderedChart(
            image_data=b"<svg></svg>",
            output_format=OutputFormat.SVG,
            width=1200,
            height=800,
        )

        assert chart.mime_type == "image/svg+xml"

    def test_metadata_defaults_to_empty_mapping(self, valid_chart):
        assert isinstance(valid_chart.metadata, Mapping)
        assert dict(valid_chart.metadata) == {}

    def test_metadata_is_read_only(self):
        chart = RenderedChart(
            image_data=b"image-bytes",
            output_format=OutputFormat.PNG,
            width=1200,
            height=800,
            metadata={"renderer": "matplotlib"},
        )

        with pytest.raises(TypeError):
            chart.metadata["renderer"] = "other"

    def test_metadata_is_copied(self):
        metadata = {"renderer": "matplotlib"}

        chart = RenderedChart(
            image_data=b"image-bytes",
            output_format=OutputFormat.PNG,
            width=1200,
            height=800,
            metadata=metadata,
        )

        metadata["renderer"] = "other"

        assert chart.metadata["renderer"] == "matplotlib"

    def test_rejects_non_mapping_metadata(self):
        with pytest.raises(
            VisualizationValidationError,
            match="metadata must be a mapping",
        ):
            RenderedChart(
                image_data=b"image-bytes",
                output_format=OutputFormat.PNG,
                width=1200,
                height=800,
                metadata="invalid",
            )

    def test_rejects_non_string_metadata_keys(self):
        with pytest.raises(
            VisualizationValidationError,
            match="metadata keys must be strings",
        ):
            RenderedChart(
                image_data=b"image-bytes",
                output_format=OutputFormat.PNG,
                width=1200,
                height=800,
                metadata={1: "invalid"},
            )

    def test_rejects_non_bytes_image_data(self):
        with pytest.raises(
            VisualizationValidationError,
            match="image_data must be bytes",
        ):
            RenderedChart(
                image_data="image-bytes",
                output_format=OutputFormat.PNG,
                width=1200,
                height=800,
            )

    def test_rejects_empty_image_data(self):
        with pytest.raises(
            VisualizationValidationError,
            match="image_data must not be empty",
        ):
            RenderedChart(
                image_data=b"",
                output_format=OutputFormat.PNG,
                width=1200,
                height=800,
            )

    @pytest.mark.parametrize(
        "output_format",
        ["png", "svg", None, 123],
    )
    def test_rejects_invalid_output_format(self, output_format):
        with pytest.raises(
            VisualizationValidationError,
            match="output_format must be an OutputFormat",
        ):
            RenderedChart(
                image_data=b"image-bytes",
                output_format=output_format,
                width=1200,
                height=800,
            )

    @pytest.mark.parametrize("width", [0, -1, 1.5, True, False])
    def test_rejects_invalid_width(self, width):
        with pytest.raises(
            VisualizationValidationError,
            match="width must be a positive integer",
        ):
            RenderedChart(
                image_data=b"image-bytes",
                output_format=OutputFormat.PNG,
                width=width,
                height=800,
            )

    @pytest.mark.parametrize("height", [0, -1, 1.5, True, False])
    def test_rejects_invalid_height(self, height):
        with pytest.raises(
            VisualizationValidationError,
            match="height must be a positive integer",
        ):
            RenderedChart(
                image_data=b"image-bytes",
                output_format=OutputFormat.PNG,
                width=1200,
                height=height,
            )

    def test_result_is_immutable(self, valid_chart):
        with pytest.raises(AttributeError):
            valid_chart.width = 600


class TestChartRendererProtocol:
    """Tests for ChartRenderer structural interface."""

    def test_accepts_structural_implementation(self, bar_specification):
        class FakeRenderer:
            def render(
                self,
                specification,
                *,
                output_format=OutputFormat.PNG,
            ):
                return RenderedChart(
                    image_data=b"fake-image",
                    output_format=output_format,
                    width=1200,
                    height=800,
                    metadata={"renderer": "fake"},
                )

        renderer: ChartRenderer = FakeRenderer()

        result = renderer.render(bar_specification)

        assert isinstance(result, RenderedChart)
        assert result.output_format is OutputFormat.PNG
        assert result.metadata["renderer"] == "fake"

    def test_protocol_supports_svg_output(self, bar_specification):
        class FakeRenderer:
            def render(
                self,
                specification,
                *,
                output_format=OutputFormat.PNG,
            ):
                return RenderedChart(
                    image_data=b"<svg></svg>",
                    output_format=output_format,
                    width=1200,
                    height=800,
                )

        renderer: ChartRenderer = FakeRenderer()

        result = renderer.render(
            bar_specification,
            output_format=OutputFormat.SVG,
        )

        assert result.output_format is OutputFormat.SVG
        assert result.mime_type == "image/svg+xml"
