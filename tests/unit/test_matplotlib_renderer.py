"""Unit tests for MatplotlibRenderer."""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET

import pytest
from matplotlib.figure import Figure

from lite_cog_agent.visualization import (
    ChartData,
    ChartSeries,
    ChartSpecification,
    ChartType,
    MatplotlibRenderer,
    OutputFormat,
    VisualizationExecutionError,
    VisualizationValidationError,
)


@pytest.fixture
def specification() -> ChartSpecification:
    return ChartSpecification(
        chart_type=ChartType.BAR,
        title="Renderer Test",
        data=ChartData(
            dataset_id="dataset-test",
            dataset_name="Test Dataset",
            series=(
                ChartSeries(
                    name="Sales",
                    x_values=("A", "B", "C"),
                    y_values=(10.0, 20.0, 15.0),
                ),
            ),
        ),
        x_label="Category",
        y_label="Value",
    )


@pytest.fixture
def renderer() -> MatplotlibRenderer:
    return MatplotlibRenderer()


class TestMatplotlibRendererConfiguration:
    @pytest.mark.parametrize(
        "figsize",
        [
            (0, 8),
            (12, 0),
            (-1, 8),
            (12, -1),
            (float("inf"), 8),
            (12, float("nan")),
            (True, 8),
            (12, False),
            (12,),
            (12, 8, 6),
            [12, 8],
            "12x8",
        ],
    )
    def test_invalid_figsize_raises_validation_error(
        self,
        figsize: object,
    ) -> None:
        with pytest.raises(VisualizationValidationError):
            MatplotlibRenderer(figsize=figsize)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "dpi",
        [
            0,
            -1,
            float("inf"),
            float("nan"),
            True,
            False,
        ],
    )
    def test_invalid_dpi_raises_validation_error(
        self,
        dpi: object,
    ) -> None:
        with pytest.raises(VisualizationValidationError):
            MatplotlibRenderer(dpi=dpi)  # type: ignore[arg-type]

    def test_custom_dimensions_are_applied(self) -> None:
        renderer = MatplotlibRenderer(
            figsize=(8.0, 6.0),
            dpi=150,
        )

        result = renderer.render(
            _build_specification(),
            output_format=OutputFormat.PNG,
        )

        assert result.width == 1200
        assert result.height == 900


class TestMatplotlibRendererOutput:
    def test_render_png(
        self,
        renderer: MatplotlibRenderer,
        specification: ChartSpecification,
    ) -> None:
        result = renderer.render(
            specification,
            output_format=OutputFormat.PNG,
        )

        assert result.output_format is OutputFormat.PNG
        assert result.mime_type == "image/png"
        assert result.image_data.startswith(b"\x89PNG\r\n\x1a\n")
        assert result.width == 1200
        assert result.height == 800

    def test_render_svg(
        self,
        renderer: MatplotlibRenderer,
        specification: ChartSpecification,
    ) -> None:
        result = renderer.render(
            specification,
            output_format=OutputFormat.SVG,
        )

        assert result.output_format is OutputFormat.SVG
        assert result.mime_type == "image/svg+xml"
        assert b"<svg" in result.image_data
        assert result.width == 1200
        assert result.height == 800

    def test_render_defaults_to_png(
        self,
        renderer: MatplotlibRenderer,
        specification: ChartSpecification,
    ) -> None:
        result = renderer.render(specification)

        assert result.output_format is OutputFormat.PNG
        assert result.mime_type == "image/png"

    def test_render_metadata(
        self,
        renderer: MatplotlibRenderer,
        specification: ChartSpecification,
    ) -> None:
        result = renderer.render(specification)

        assert result.metadata["renderer"] == "matplotlib"
        assert result.metadata["dpi"] == 100.0
        assert result.metadata["figsize"] == (12.0, 8.0)
        assert result.metadata["chart_type"] == "bar"

    def test_repeated_rendering_returns_independent_results(
        self,
        renderer: MatplotlibRenderer,
        specification: ChartSpecification,
    ) -> None:
        first = renderer.render(
            specification,
            output_format=OutputFormat.PNG,
        )
        second = renderer.render(
            specification,
            output_format=OutputFormat.PNG,
        )

        assert first is not second
        assert first.image_data
        assert second.image_data

    def test_png_actual_dimensions_match_rendered_chart(
        self,
        renderer: MatplotlibRenderer,
        specification: ChartSpecification,
    ) -> None:
        result = renderer.render(
            specification,
            output_format=OutputFormat.PNG,
        )

        actual_width, actual_height = struct.unpack(
            ">II",
            result.image_data[16:24],
        )

        assert actual_width == result.width
        assert actual_height == result.height


    def test_svg_is_valid_xml(
        self,
        renderer: MatplotlibRenderer,
        specification: ChartSpecification,
    ) -> None:
        result = renderer.render(
            specification,
            output_format=OutputFormat.SVG,
        )

        root = ET.fromstring(result.image_data)

        assert root.tag == "{http://www.w3.org/2000/svg}svg"


    def test_svg_has_valid_viewbox(
        self,
        renderer: MatplotlibRenderer,
        specification: ChartSpecification,
    ) -> None:
        result = renderer.render(
            specification,
            output_format=OutputFormat.SVG,
        )

        root = ET.fromstring(result.image_data)

        viewbox = root.attrib.get("viewBox")
        assert viewbox is not None

        values = [float(value) for value in viewbox.split()]

        assert len(values) == 4

        _, _, viewbox_width, viewbox_height = values

        assert viewbox_width > 0
        assert viewbox_height > 0

        assert viewbox_width / viewbox_height == pytest.approx(
            result.width / result.height
        )

    def test_mixed_format_rendering_does_not_interfere(
        self,
        renderer: MatplotlibRenderer,
        specification: ChartSpecification,
    ) -> None:
        # 1. First PNG rendering.
        first_png = renderer.render(
            specification,
            output_format=OutputFormat.PNG,
        )
        first_png_data = first_png.image_data

        # 2. SVG rendering using the same renderer.
        svg = renderer.render(
            specification,
            output_format=OutputFormat.SVG,
        )

        # 3. Second PNG rendering.
        second_png = renderer.render(
            specification,
            output_format=OutputFormat.PNG,
        )

        # Verify output formats.
        assert first_png.output_format is OutputFormat.PNG
        assert svg.output_format is OutputFormat.SVG
        assert second_png.output_format is OutputFormat.PNG

        # Verify MIME types.
        assert first_png.mime_type == "image/png"
        assert svg.mime_type == "image/svg+xml"
        assert second_png.mime_type == "image/png"

        # Verify all outputs contain valid bytes.
        assert first_png.image_data
        assert svg.image_data
        assert second_png.image_data

        # Verify PNG dimensions.
        for result in (first_png, second_png):
            width, height = struct.unpack(">II", result.image_data[16:24])
            assert (width, height) == (result.width, result.height)

        # Verify SVG XML.
        root = ET.fromstring(svg.image_data)
        assert root.tag == "{http://www.w3.org/2000/svg}svg"

        # Verify previous output was not modified.
        assert first_png.image_data == first_png_data


class TestMatplotlibRendererValidation:
    def test_invalid_specification_raises_validation_error(
        self,
        renderer: MatplotlibRenderer,
    ) -> None:
        with pytest.raises(VisualizationValidationError):
            renderer.render("invalid")  # type: ignore[arg-type]

    def test_invalid_output_format_raises_validation_error(
        self,
        renderer: MatplotlibRenderer,
        specification: ChartSpecification,
    ) -> None:
        with pytest.raises(VisualizationValidationError):
            renderer.render(
                specification,
                output_format="png",  # type: ignore[arg-type]
            )


class TestMatplotlibRendererLifecycle:
    def test_figure_is_cleared_after_successful_render(
        self,
        renderer: MatplotlibRenderer,
        specification: ChartSpecification,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        figure = Figure(figsize=(12.0, 8.0), dpi=100)

        monkeypatch.setattr(
            renderer,
            "_create_figure",
            lambda: figure,
        )

        renderer.render(specification)

        assert figure.axes == []

    def test_figure_is_cleared_after_render_failure(
        self,
        renderer: MatplotlibRenderer,
        specification: ChartSpecification,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        figure = Figure(figsize=(12.0, 8.0), dpi=100)

        monkeypatch.setattr(
            renderer,
            "_create_figure",
            lambda: figure,
        )

        def fail_render(*args: object) -> bytes:
            raise RuntimeError("simulated rendering failure")

        monkeypatch.setattr(
            renderer,
            "_render_to_bytes",
            fail_render,
        )

        with pytest.raises(VisualizationExecutionError):
            renderer.render(specification)

        assert figure.axes == []

    def test_render_failure_preserves_original_cause(
        self,
        renderer: MatplotlibRenderer,
        specification: ChartSpecification,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fail_render(*args: object) -> bytes:
            raise RuntimeError("simulated rendering failure")

        monkeypatch.setattr(
            renderer,
            "_render_to_bytes",
            fail_render,
        )

        with pytest.raises(
            VisualizationExecutionError,
        ) as exc_info:
            renderer.render(specification)

        assert isinstance(exc_info.value.__cause__, RuntimeError)


def _build_specification() -> ChartSpecification:
    return ChartSpecification(
        chart_type=ChartType.BAR,
        title="Test",
        data=ChartData(
            dataset_id="dataset-test",
            dataset_name="Test Dataset",
            series=(
                ChartSeries(
                    name="Series",
                    x_values=("A", "B"),
                    y_values=(1.0, 2.0),
                ),
            ),
        ),
    )
