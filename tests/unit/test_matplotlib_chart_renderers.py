"""Unit tests for Bar and Line chart rendering."""

from __future__ import annotations

from math import isnan

import pytest

from lite_cog_agent.visualization import (
    ChartData,
    ChartSeries,
    ChartSpecification,
    ChartType,
    MatplotlibRenderer,
    OutputFormat,
    VisualizationValidationError,
)
from lite_cog_agent.statistics.contracts import HistogramBin


@pytest.fixture
def renderer() -> MatplotlibRenderer:
    return MatplotlibRenderer()


def _make_specification(
    chart_type: ChartType,
    series: tuple[ChartSeries, ...],
) -> ChartSpecification:
    return ChartSpecification(
        chart_type=chart_type,
        title="Test Chart",
        data=ChartData(
            dataset_id="dataset-test",
            dataset_name="Test Dataset",
            series=series,
        ),
        x_label="Category",
        y_label="Value",
    )

def _make_histogram_specification(
    histogram_bins: tuple[HistogramBin, ...],
) -> ChartSpecification:
    return ChartSpecification(
        chart_type=ChartType.HISTOGRAM,
        title="Test Histogram",
        data=ChartData(
            dataset_id="dataset-test",
            dataset_name="Test Dataset",
            histogram_bins=histogram_bins,
        ),
        x_label="Value",
        y_label="Frequency",
    )

def _capture_axes(
    renderer: MatplotlibRenderer,
    specification: ChartSpecification,
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}

    # Preserve the original Figure factory.
    original_create_figure = renderer._create_figure

    def create_figure():
        figure = original_create_figure()

        # Prevent cleanup only for this test Figure.
        # Figure initialization has already completed.
        monkeypatch.setattr(
            figure,
            "clear",
            lambda: None,
        )

        return figure

    monkeypatch.setattr(
        renderer,
        "_create_figure",
        create_figure,
    )

    def capture(canvas, output_format):
        captured["axes"] = canvas.figure.axes[0]
        return b"test-image"

    monkeypatch.setattr(
        renderer,
        "_render_to_bytes",
        capture,
    )

    renderer.render(
        specification,
        output_format=OutputFormat.PNG,
    )

    return captured["axes"]


class TestBarRenderer:
    def test_renders_single_series(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.BAR,
            (
                ChartSeries(
                    name="Sales",
                    x_values=("A", "B", "C"),
                    y_values=(10, 20, 15),
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        assert len(axes.patches) == 3
        assert [
            tick.get_text()
            for tick in axes.get_xticklabels()
        ] == ["A", "B", "C"]

    def test_renders_grouped_multiple_series(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.BAR,
            (
                ChartSeries(
                    name="2025",
                    x_values=("Q1", "Q2"),
                    y_values=(100, 150),
                ),
                ChartSeries(
                    name="2026",
                    x_values=("Q1", "Q2"),
                    y_values=(120, 170),
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        assert len(axes.patches) == 4
        assert axes.get_legend() is not None

        legend_labels = [
            text.get_text()
            for text in axes.get_legend().get_texts()
        ]

        assert legend_labels == ["2025", "2026"]

    def test_aligns_different_categories_across_series(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.BAR,
            (
                ChartSeries(
                    name="A",
                    x_values=("Q1", "Q2"),
                    y_values=(10, 20),
                ),
                ChartSeries(
                    name="B",
                    x_values=("Q2", "Q3"),
                    y_values=(30, 40),
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        assert len(axes.patches) == 4

        assert [
            tick.get_text()
            for tick in axes.get_xticklabels()
        ] == ["Q1", "Q2", "Q3"]

    def test_skips_none_values(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.BAR,
            (
                ChartSeries(
                    name="Sales",
                    x_values=("A", "B", None, "D"),
                    y_values=(10, None, 30, 40),
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        assert len(axes.patches) == 2

    def test_supports_numeric_categories(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.BAR,
            (
                ChartSeries(
                    name="Sales",
                    x_values=(1, 2, 3),
                    y_values=(10, 20, 30),
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        assert len(axes.patches) == 3

    def test_rejects_duplicate_categories(
        self,
        renderer: MatplotlibRenderer,
    ) -> None:
        specification = _make_specification(
            ChartType.BAR,
            (
                ChartSeries(
                    name="Sales",
                    x_values=("A", "A"),
                    y_values=(10, 20),
                ),
            ),
        )

        with pytest.raises(VisualizationValidationError):
            renderer.render(specification)

    def test_rejects_all_null_x_values(
        self,
        renderer: MatplotlibRenderer,
    ) -> None:
        specification = _make_specification(
            ChartType.BAR,
            (
                ChartSeries(
                    name="Sales",
                    x_values=(None, None),
                    y_values=(10, 20),
                ),
            ),
        )

        with pytest.raises(VisualizationValidationError):
            renderer.render(specification)


class TestLineRenderer:
    def test_renders_numeric_line(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.LINE,
            (
                ChartSeries(
                    name="Sales",
                    x_values=(1, 2, 3),
                    y_values=(10, 20, 15),
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        assert len(axes.lines) == 1

        line = axes.lines[0]

        assert list(line.get_xdata()) == [1.0, 2.0, 3.0]
        assert list(line.get_ydata()) == [10.0, 20.0, 15.0]

    def test_renders_categorical_line(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.LINE,
            (
                ChartSeries(
                    name="Sales",
                    x_values=("Jan", "Feb", "Mar"),
                    y_values=(10, 20, 15),
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        assert len(axes.lines) == 1

        assert [
            tick.get_text()
            for tick in axes.get_xticklabels()
        ] == ["Jan", "Feb", "Mar"]

    def test_renders_multiple_series(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.LINE,
            (
                ChartSeries(
                    name="2025",
                    x_values=(1, 2, 3),
                    y_values=(10, 20, 15),
                ),
                ChartSeries(
                    name="2026",
                    x_values=(1, 2, 3),
                    y_values=(15, 25, 20),
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        assert len(axes.lines) == 2
        assert axes.get_legend() is not None

    def test_none_y_value_creates_line_gap(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.LINE,
            (
                ChartSeries(
                    name="Sales",
                    x_values=(1, 2, 3),
                    y_values=(10, None, 30),
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        y_values = list(axes.lines[0].get_ydata())

        assert y_values[0] == 10.0
        assert isnan(y_values[1])
        assert y_values[2] == 30.0

    def test_none_x_value_creates_line_gap(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.LINE,
            (
                ChartSeries(
                    name="Sales",
                    x_values=(1, None, 3),
                    y_values=(10, 20, 30),
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        line = axes.lines[0]

        assert isnan(line.get_xdata()[1])
        assert isnan(line.get_ydata()[1])

    def test_rejects_all_null_x_values(
        self,
        renderer: MatplotlibRenderer,
    ) -> None:
        specification = _make_specification(
            ChartType.LINE,
            (
                ChartSeries(
                    name="Sales",
                    x_values=(None, None),
                    y_values=(10, 20),
                ),
            ),
        )

        with pytest.raises(VisualizationValidationError):
            renderer.render(specification)

    def test_applies_title_and_axis_labels(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.LINE,
            (
                ChartSeries(
                    name="Sales",
                    x_values=(1, 2),
                    y_values=(10, 20),
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        assert axes.get_title() == "Test Chart"
        assert axes.get_xlabel() == "Category"
        assert axes.get_ylabel() == "Value"


class TestHistogramRenderer:
    def test_renders_histogram_bins(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_histogram_specification(
            (
                HistogramBin(
                    lower_bound=0.0,
                    upper_bound=10.0,
                    count=5,
                    percentage=20.0,
                ),
                HistogramBin(
                    lower_bound=10.0,
                    upper_bound=30.0,
                    count=12,
                    percentage=48.0,
                ),
                HistogramBin(
                    lower_bound=30.0,
                    upper_bound=50.0,
                    count=8,
                    percentage=32.0,
                ),
            )
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        assert len(axes.patches) == 3

        assert [
            patch.get_x()
            for patch in axes.patches
        ] == pytest.approx([0.0, 10.0, 30.0])

        assert [
            patch.get_width()
            for patch in axes.patches
        ] == pytest.approx([10.0, 20.0, 20.0])

        assert [
            patch.get_height()
            for patch in axes.patches
        ] == pytest.approx([5.0, 12.0, 8.0])

    def test_preserves_zero_count_bins(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_histogram_specification(
            (
                HistogramBin(
                    lower_bound=0.0,
                    upper_bound=10.0,
                    count=5,
                    percentage=100.0,
                ),
                HistogramBin(
                    lower_bound=10.0,
                    upper_bound=20.0,
                    count=0,
                    percentage=0.0,
                ),
            )
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        assert len(axes.patches) == 2
        assert axes.patches[1].get_height() == 0.0

    def test_applies_title_and_axis_labels(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_histogram_specification(
            (
                HistogramBin(
                    lower_bound=0.0,
                    upper_bound=10.0,
                    count=5,
                    percentage=100.0,
                ),
            )
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        assert axes.get_title() == "Test Histogram"
        assert axes.get_xlabel() == "Value"
        assert axes.get_ylabel() == "Frequency"


class TestScatterRenderer:
    def test_renders_single_series(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.SCATTER,
            (
                ChartSeries(
                    name="Observations",
                    x_values=(1, 2, 3),
                    y_values=(10, 20, 15),
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        assert len(axes.collections) == 1

        offsets = axes.collections[0].get_offsets()

        assert offsets.tolist() == [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 15.0],
        ]

    def test_preserves_numeric_x_coordinates(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.SCATTER,
            (
                ChartSeries(
                    name="Observations",
                    x_values=(1, 10, 100),
                    y_values=(2, 20, 200),
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        offsets = axes.collections[0].get_offsets()

        assert offsets[:, 0].tolist() == [1.0, 10.0, 100.0]

    def test_skips_none_coordinates(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.SCATTER,
            (
                ChartSeries(
                    name="Observations",
                    x_values=(1, 2, None, 4),
                    y_values=(10, None, 30, 40),
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        assert len(axes.collections) == 1

        offsets = axes.collections[0].get_offsets()

        assert offsets.tolist() == [
            [1.0, 10.0],
            [4.0, 40.0],
        ]

    def test_renders_multiple_series_with_legend(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.SCATTER,
            (
                ChartSeries(
                    name="Group A",
                    x_values=(1, 2),
                    y_values=(10, 20),
                ),
                ChartSeries(
                    name="Group B",
                    x_values=(3, 4),
                    y_values=(30, 40),
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        assert len(axes.collections) == 2
        assert axes.get_legend() is not None

        legend_labels = [
            text.get_text()
            for text in axes.get_legend().get_texts()
        ]

        assert legend_labels == ["Group A", "Group B"]

    def test_skips_series_without_valid_points(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.SCATTER,
            (
                ChartSeries(
                    name="Empty",
                    x_values=(1, 2),
                    y_values=(None, None),
                ),
                ChartSeries(
                    name="Valid",
                    x_values=(3, 4),
                    y_values=(30, 40),
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        assert len(axes.collections) == 1

    def test_rejects_all_missing_points(
        self,
        renderer: MatplotlibRenderer,
    ) -> None:
        specification = _make_specification(
            ChartType.SCATTER,
            (
                ChartSeries(
                    name="Empty",
                    x_values=(1, 2),
                    y_values=(None, None),
                ),
            ),
        )

        with pytest.raises(
            VisualizationValidationError,
            match="at least one valid data point",
        ):
            renderer.render(specification)

    def test_applies_title_and_axis_labels(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.SCATTER,
            (
                ChartSeries(
                    name="Observations",
                    x_values=(1, 2),
                    y_values=(10, 20),
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        assert axes.get_title() == "Test Chart"
        assert axes.get_xlabel() == "Category"
        assert axes.get_ylabel() == "Value"


class TestPieRenderer:
    def test_renders_single_series(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.PIE,
            (
                ChartSeries(
                    name="Market Share",
                    x_values=("A", "B", "C"),
                    y_values=(50, 30, 20),
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        assert len(axes.patches) == 3

    def test_preserves_category_labels(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.PIE,
            (
                ChartSeries(
                    name="Market Share",
                    x_values=("Product A", "Product B", "Product C"),
                    y_values=(50, 30, 20),
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        labels = [
            text.get_text()
            for text in axes.texts
            if text.get_text() in {
                "Product A",
                "Product B",
                "Product C",
            }
        ]

        assert labels == [
            "Product A",
            "Product B",
            "Product C",
        ]

    def test_displays_percentages(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.PIE,
            (
                ChartSeries(
                    name="Market Share",
                    x_values=("A", "B", "C"),
                    y_values=(50, 30, 20),
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        texts = [
            text.get_text()
            for text in axes.texts
        ]

        assert any("50.0%" in text for text in texts)
        assert any("30.0%" in text for text in texts)
        assert any("20.0%" in text for text in texts)

    def test_rejects_zero_total_values(
        self,
        renderer: MatplotlibRenderer,
    ) -> None:
        specification = _make_specification(
            ChartType.PIE,
            (
                ChartSeries(
                    name="Market Share",
                    x_values=("A", "B", "C"),
                    y_values=(0, 0, 0),
                ),
            ),
        )

        with pytest.raises(
            VisualizationValidationError,
            match="positive total",
        ):
            renderer.render(specification)

    def test_applies_title_and_axis_labels(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.PIE,
            (
                ChartSeries(
                    name="Market Share",
                    x_values=("A", "B"),
                    y_values=(60, 40),
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        assert axes.get_title() == "Test Chart"
        assert axes.get_xlabel() == "Category"
        assert axes.get_ylabel() == "Value"


class TestBoxRenderer:
    def test_renders_single_series(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.BOX,
            (
                ChartSeries(
                    name="Sales",
                    statistics={
                        "min": 10,
                        "q1": 20,
                        "median": 30,
                        "q3": 40,
                        "max": 50,
                    },
                    outliers=(5, 60),
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        assert len(axes.lines) > 0

    def test_renders_multiple_series(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.BOX,
            (
                ChartSeries(
                    name="2025",
                    statistics={
                        "min": 10,
                        "q1": 20,
                        "median": 30,
                        "q3": 40,
                        "max": 50,
                    },
                ),
                ChartSeries(
                    name="2026",
                    statistics={
                        "min": 15,
                        "q1": 25,
                        "median": 35,
                        "q3": 45,
                        "max": 55,
                    },
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        assert [
            tick.get_text()
            for tick in axes.get_xticklabels()
        ] == ["2025", "2026"]

    def test_preserves_box_statistics(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.BOX,
            (
                ChartSeries(
                    name="Sales",
                    statistics={
                        "min": 10,
                        "q1": 20,
                        "median": 30,
                        "q3": 40,
                        "max": 50,
                    },
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        assert axes.get_ylim()[0] <= 10
        assert axes.get_ylim()[1] >= 50

        median_lines = [
            line
            for line in axes.lines
            if list(line.get_ydata()) == [30, 30]
        ]

        assert len(median_lines) == 1

    def test_preserves_outliers(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.BOX,
            (
                ChartSeries(
                    name="Sales",
                    statistics={
                        "min": 10,
                        "q1": 20,
                        "median": 30,
                        "q3": 40,
                        "max": 50,
                    },
                    outliers=(5, 60),
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        flier_lines = [
            line
            for line in axes.lines
            if line.get_marker() not in (None, "", "None")
        ]

        assert len(flier_lines) == 1
        assert list(flier_lines[0].get_ydata()) == [5, 60]

    def test_supports_empty_outliers(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.BOX,
            (
                ChartSeries(
                    name="Sales",
                    statistics={
                        "min": 10,
                        "q1": 20,
                        "median": 30,
                        "q3": 40,
                        "max": 50,
                    },
                    outliers=(),
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        assert len(axes.lines) > 0

        flier_lines = [
            line
            for line in axes.lines
            if line.get_marker() not in (None, "", "None")
        ]

        assert len(flier_lines) == 1
        assert len(flier_lines[0].get_ydata()) == 0

    def test_applies_title_and_axis_labels(
        self,
        renderer: MatplotlibRenderer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        specification = _make_specification(
            ChartType.BOX,
            (
                ChartSeries(
                    name="Sales",
                    statistics={
                        "min": 10,
                        "q1": 20,
                        "median": 30,
                        "q3": 40,
                        "max": 50,
                    },
                ),
            ),
        )

        axes = _capture_axes(
            renderer,
            specification,
            monkeypatch,
        )

        assert axes.get_title() == "Test Chart"
        assert axes.get_xlabel() == "Category"
        assert axes.get_ylabel() == "Value"