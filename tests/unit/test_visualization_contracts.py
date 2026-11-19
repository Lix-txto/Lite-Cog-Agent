"""Tests for visualization contracts."""

from dataclasses import FrozenInstanceError
from math import inf, nan

import pytest

from lite_cog_agent.statistics.contracts import HistogramBin
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


@pytest.fixture
def bar_data():
    return ChartData(
        dataset_id="dataset-001",
        dataset_name="sales",
        series=(
            ChartSeries(
                name="revenue",
                x_values=("Jan", "Feb", "Mar"),
                y_values=(100.0, 120.0, 150.0),
            ),
        ),
    )


@pytest.fixture
def histogram_data():
    return ChartData(
        dataset_id="dataset-001",
        dataset_name="sales",
        histogram_bins=(
            HistogramBin(
                lower_bound=0.0,
                upper_bound=10.0,
                count=2,
                percentage=40.0,
            ),
            HistogramBin(
                lower_bound=10.0,
                upper_bound=20.0,
                count=3,
                percentage=60.0,
            ),
        ),
    )


class TestChartType:
    @pytest.mark.parametrize(
        ("chart_type", "value"),
        [
            (ChartType.BAR, "bar"),
            (ChartType.LINE, "line"),
            (ChartType.HISTOGRAM, "histogram"),
            (ChartType.SCATTER, "scatter"),
            (ChartType.PIE, "pie"),
            (ChartType.BOX, "box"),
        ],
    )
    def test_chart_type_values(self, chart_type, value):
        assert chart_type.value == value

    def test_chart_type_is_string_enum(self):
        assert isinstance(ChartType.BAR, str)
        assert ChartType.BAR == "bar"


class TestChartSeries:
    def test_creates_valid_series(self):
        series = ChartSeries(
            name="revenue",
            x_values=("Jan", "Feb"),
            y_values=(100.0, 120.0),
        )

        assert series.name == "revenue"
        assert series.x_values == ("Jan", "Feb")
        assert series.y_values == (100.0, 120.0)
        assert dict(series.statistics) == {}
        assert series.outliers == ()

    def test_accepts_numeric_x_values(self):
        series = ChartSeries(
            name="observations",
            x_values=(1, 2.5, None),
            y_values=(10.0, 20.0, None),
        )

        assert series.x_values == (1, 2.5, None)

    def test_rejects_empty_name(self):
        with pytest.raises(
            VisualizationValidationError,
            match="name must be a non-empty string",
        ):
            ChartSeries(name=" ", y_values=())

    def test_rejects_non_tuple_x_values(self):
        with pytest.raises(
            VisualizationValidationError,
            match="x_values must be a tuple",
        ):
            ChartSeries(
                name="series",
                x_values=["A", "B"],
                y_values=(1.0, 2.0),
            )

    def test_rejects_non_tuple_y_values(self):
        with pytest.raises(
            VisualizationValidationError,
            match="y_values must be a tuple",
        ):
            ChartSeries(
                name="series",
                x_values=("A", "B"),
                y_values=[1.0, 2.0],
            )

    def test_rejects_mismatched_value_lengths(self):
        with pytest.raises(
            VisualizationValidationError,
            match="same length",
        ):
            ChartSeries(
                name="series",
                x_values=("A", "B"),
                y_values=(1.0,),
            )

    @pytest.mark.parametrize("value", [True, nan, inf])
    def test_rejects_invalid_numeric_x_values(self, value):
        with pytest.raises(VisualizationValidationError):
            ChartSeries(
                name="series",
                x_values=(value,),
                y_values=(1.0,),
            )

    @pytest.mark.parametrize("value", [True, nan, inf])
    def test_rejects_invalid_y_values(self, value):
        with pytest.raises(VisualizationValidationError):
            ChartSeries(
                name="series",
                x_values=("A",),
                y_values=(value,),
            )

    def test_accepts_none_y_value(self):
        series = ChartSeries(
            name="series",
            x_values=("A",),
            y_values=(None,),
        )

        assert series.y_values == (None,)

    def test_statistics_are_read_only(self):
        statistics = {"mean": 10.0}

        series = ChartSeries(
            name="series",
            statistics=statistics,
        )

        statistics["mean"] = 20.0

        assert series.statistics["mean"] == 10.0

        with pytest.raises(TypeError):
            series.statistics["mean"] = 30.0

    def test_rejects_invalid_statistics(self):
        with pytest.raises(VisualizationValidationError):
            ChartSeries(
                name="series",
                statistics={"mean": nan},
            )

    def test_outliers_are_validated(self):
        with pytest.raises(VisualizationValidationError):
            ChartSeries(
                name="series",
                outliers=(1.0, inf),
            )

    def test_rejects_non_tuple_outliers(self):
        with pytest.raises(
            VisualizationValidationError,
            match="outliers must be a tuple",
        ):
            ChartSeries(
                name="series",
                outliers=[1.0, 2.0],
            )

    def test_series_is_frozen(self):
        series = ChartSeries(name="series")

        with pytest.raises(FrozenInstanceError):
            series.name = "changed"


class TestChartData:
    def test_creates_valid_chart_data(self, bar_data):
        assert bar_data.dataset_id == "dataset-001"
        assert len(bar_data.series) == 1

    def test_rejects_empty_dataset_id(self):
        with pytest.raises(VisualizationValidationError):
            ChartData(
                dataset_id=" ",
                dataset_name="sales",
                series=(ChartSeries(name="series"),),
            )

    def test_rejects_empty_dataset_name(self):
        with pytest.raises(VisualizationValidationError):
            ChartData(
                dataset_id="dataset-001",
                dataset_name="",
                series=(ChartSeries(name="series"),),
            )

    def test_rejects_non_tuple_series(self):
        with pytest.raises(
            VisualizationValidationError,
            match="series must be a tuple",
        ):
            ChartData(
                dataset_id="dataset-001",
                dataset_name="sales",
                series=[ChartSeries(name="series")],
            )

    def test_rejects_invalid_series_item(self):
        with pytest.raises(
            VisualizationValidationError,
            match="ChartSeries instances",
        ):
            ChartData(
                dataset_id="dataset-001",
                dataset_name="sales",
                series=("invalid",),
            )

    def test_rejects_duplicate_series_names(self):
        with pytest.raises(
            VisualizationValidationError,
            match="unique",
        ):
            ChartData(
                dataset_id="dataset-001",
                dataset_name="sales",
                series=(
                    ChartSeries(name="revenue"),
                    ChartSeries(name="revenue"),
                ),
            )

    def test_rejects_empty_chart_data(self):
        with pytest.raises(
            VisualizationValidationError,
            match="at least one series or histogram bin",
        ):
            ChartData(
                dataset_id="dataset-001",
                dataset_name="sales",
            )

    def test_rejects_series_and_histogram_bins_together(self):
        with pytest.raises(
            VisualizationValidationError,
            match="cannot both contain data",
        ):
            ChartData(
                dataset_id="dataset-001",
                dataset_name="sales",
                series=(ChartSeries(name="series"),),
                histogram_bins=(
                    HistogramBin(0.0, 1.0, 1, 100.0),
                ),
            )

    def test_rejects_non_tuple_histogram_bins(self):
        with pytest.raises(
            VisualizationValidationError,
            match="histogram_bins must be a tuple",
        ):
            ChartData(
                dataset_id="dataset-001",
                dataset_name="sales",
                histogram_bins=[
                    HistogramBin(0.0, 1.0, 1, 100.0),
                ],
            )

    def test_rejects_invalid_histogram_item(self):
        with pytest.raises(
            VisualizationValidationError,
            match="HistogramBin instances",
        ):
            ChartData(
                dataset_id="dataset-001",
                dataset_name="sales",
                histogram_bins=("invalid",),
            )

    def test_rejects_histogram_percentages_not_summing_to_100(self):
        with pytest.raises(
            VisualizationValidationError,
            match="sum to 100",
        ):
            ChartData(
                dataset_id="dataset-001",
                dataset_name="sales",
                histogram_bins=(
                    HistogramBin(0.0, 1.0, 1, 50.0),
                    HistogramBin(1.0, 2.0, 1, 40.0),
                ),
            )

    def test_accepts_empty_histogram_counts(self):
        data = ChartData(
            dataset_id="dataset-001",
            dataset_name="empty",
            histogram_bins=(
                HistogramBin(0.0, 1.0, 0, 0.0),
                HistogramBin(1.0, 2.0, 0, 0.0),
            ),
        )

        assert len(data.histogram_bins) == 2

    def test_metadata_is_read_only(self):
        metadata = {"source": "csv"}

        data = ChartData(
            dataset_id="dataset-001",
            dataset_name="sales",
            series=(ChartSeries(name="series"),),
            metadata=metadata,
        )

        metadata["source"] = "excel"

        assert data.metadata["source"] == "csv"

        with pytest.raises(TypeError):
            data.metadata["source"] = "json"


class TestChartSpecification:
    def test_creates_bar_specification(self, bar_data):
        specification = ChartSpecification(
            chart_type=ChartType.BAR,
            title="Monthly Revenue",
            data=bar_data,
            x_label="Month",
            y_label="Revenue",
            options={"show_legend": True},
        )

        assert specification.chart_type is ChartType.BAR
        assert specification.title == "Monthly Revenue"
        assert specification.options["show_legend"] is True

    def test_creates_histogram_specification(self, histogram_data):
        specification = ChartSpecification(
            chart_type=ChartType.HISTOGRAM,
            title="Distribution",
            data=histogram_data,
        )

        assert specification.chart_type is ChartType.HISTOGRAM

    def test_rejects_string_chart_type(self, bar_data):
        with pytest.raises(
            VisualizationValidationError,
            match="ChartType",
        ):
            ChartSpecification(
                chart_type="bar",
                title="Revenue",
                data=bar_data,
            )

    def test_rejects_empty_title(self, bar_data):
        with pytest.raises(VisualizationValidationError):
            ChartSpecification(
                chart_type=ChartType.BAR,
                title=" ",
                data=bar_data,
            )

    def test_rejects_invalid_data_type(self):
        with pytest.raises(
            VisualizationValidationError,
            match="ChartData",
        ):
            ChartSpecification(
                chart_type=ChartType.BAR,
                title="Revenue",
                data=None,
            )

    def test_rejects_empty_axis_label(self, bar_data):
        with pytest.raises(VisualizationValidationError):
            ChartSpecification(
                chart_type=ChartType.BAR,
                title="Revenue",
                data=bar_data,
                x_label=" ",
            )

    def test_options_are_read_only(self, bar_data):
        options = {"show_grid": True}

        specification = ChartSpecification(
            chart_type=ChartType.BAR,
            title="Revenue",
            data=bar_data,
            options=options,
        )

        options["show_grid"] = False

        assert specification.options["show_grid"] is True

        with pytest.raises(TypeError):
            specification.options["show_grid"] = False

    def test_rejects_histogram_without_bins(self, bar_data):
        with pytest.raises(
            VisualizationValidationError,
            match="require histogram_bins",
        ):
            ChartSpecification(
                chart_type=ChartType.HISTOGRAM,
                title="Distribution",
                data=bar_data,
            )

    def test_rejects_non_histogram_with_bins(self, histogram_data):
        with pytest.raises(
            VisualizationValidationError,
            match="only histogram charts",
        ):
            ChartSpecification(
                chart_type=ChartType.BAR,
                title="Distribution",
                data=histogram_data,
            )

    def test_rejects_box_chart_without_statistics(self):
        data = ChartData(
            dataset_id="dataset-001",
            dataset_name="sales",
            series=(ChartSeries(name="revenue"),),
        )

        with pytest.raises(
            VisualizationValidationError,
            match="exactly",
        ):
            ChartSpecification(
                chart_type=ChartType.BOX,
                title="Revenue Distribution",
                data=data,
            )

    def test_creates_valid_box_chart(self):
        data = ChartData(
            dataset_id="dataset-001",
            dataset_name="sales",
            series=(
                ChartSeries(
                    name="revenue",
                    statistics={
                        "min": 10.0,
                        "q1": 20.0,
                        "median": 30.0,
                        "q3": 40.0,
                        "max": 50.0,
                    },
                    outliers=(100.0,),
                ),
            ),
        )

        specification = ChartSpecification(
            chart_type=ChartType.BOX,
            title="Revenue Distribution",
            data=data,
        )

        assert specification.chart_type is ChartType.BOX

    def test_rejects_unordered_box_statistics(self):
        data = ChartData(
            dataset_id="dataset-001",
            dataset_name="sales",
            series=(
                ChartSeries(
                    name="revenue",
                    statistics={
                        "min": 10.0,
                        "q1": 30.0,
                        "median": 20.0,
                        "q3": 40.0,
                        "max": 50.0,
                    },
                ),
            ),
        )

        with pytest.raises(
            VisualizationValidationError,
            match="must be ordered",
        ):
            ChartSpecification(
                chart_type=ChartType.BOX,
                title="Revenue Distribution",
                data=data,
            )

    def test_rejects_xy_values_for_box_chart(self):
        data = ChartData(
            dataset_id="dataset-001",
            dataset_name="sales",
            series=(
                ChartSeries(
                    name="revenue",
                    x_values=("A",),
                    y_values=(10.0,),
                    statistics={
                        "min": 1.0,
                        "q1": 2.0,
                        "median": 3.0,
                        "q3": 4.0,
                        "max": 5.0,
                    },
                ),
            ),
        )

        with pytest.raises(
            VisualizationValidationError,
            match="cannot contain x_values or y_values",
        ):
            ChartSpecification(
                chart_type=ChartType.BOX,
                title="Box",
                data=data,
            )

    def test_rejects_statistics_for_bar_chart(self):
        data = ChartData(
            dataset_id="dataset-001",
            dataset_name="sales",
            series=(
                ChartSeries(
                    name="revenue",
                    x_values=("A",),
                    y_values=(10.0,),
                    statistics={"mean": 10.0},
                ),
            ),
        )

        with pytest.raises(
            VisualizationValidationError,
            match="only box charts",
        ):
            ChartSpecification(
                chart_type=ChartType.BAR,
                title="Revenue",
                data=data,
            )

    def test_rejects_non_numeric_scatter_x_values(self):
        data = ChartData(
            dataset_id="dataset-001",
            dataset_name="sales",
            series=(
                ChartSeries(
                    name="observations",
                    x_values=("A",),
                    y_values=(10.0,),
                ),
            ),
        )

        with pytest.raises(
            VisualizationValidationError,
            match="scatter chart x_values must be numeric",
        ):
            ChartSpecification(
                chart_type=ChartType.SCATTER,
                title="Scatter",
                data=data,
            )

    def test_rejects_negative_pie_values(self):
        data = ChartData(
            dataset_id="dataset-001",
            dataset_name="sales",
            series=(
                ChartSeries(
                    name="revenue",
                    x_values=("A", "B"),
                    y_values=(10.0, -1.0),
                ),
            ),
        )

        with pytest.raises(
            VisualizationValidationError,
            match="non-negative",
        ):
            ChartSpecification(
                chart_type=ChartType.PIE,
                title="Revenue Share",
                data=data,
            )

    def test_rejects_empty_xy_values_for_bar_chart(self):
        data = ChartData(
            dataset_id="dataset-001",
            dataset_name="sales",
            series=(ChartSeries(name="revenue"),),
        )

        with pytest.raises(
            VisualizationValidationError,
            match="non-empty x_values and y_values",
        ):
            ChartSpecification(
                chart_type=ChartType.BAR,
                title="Revenue",
                data=data,
            )


class TestVisualizationResult:
    def test_creates_valid_result(self, bar_data):
        specification = ChartSpecification(
            chart_type=ChartType.BAR,
            title="Monthly Revenue",
            data=bar_data,
        )

        result = VisualizationResult(
            visualization_id="viz-001",
            dataset_id="dataset-001",
            dataset_name="sales",
            specification=specification,
        )

        assert result.visualization_id == "viz-001"
        assert result.specification is specification

    def test_rejects_empty_visualization_id(self, bar_data):
        specification = ChartSpecification(
            chart_type=ChartType.BAR,
            title="Revenue",
            data=bar_data,
        )

        with pytest.raises(VisualizationValidationError):
            VisualizationResult(
                visualization_id=" ",
                dataset_id="dataset-001",
                dataset_name="sales",
                specification=specification,
            )

    def test_rejects_dataset_id_mismatch(self, bar_data):
        specification = ChartSpecification(
            chart_type=ChartType.BAR,
            title="Revenue",
            data=bar_data,
        )

        with pytest.raises(
            VisualizationValidationError,
            match="dataset_id must match",
        ):
            VisualizationResult(
                visualization_id="viz-001",
                dataset_id="another-dataset",
                dataset_name="sales",
                specification=specification,
            )

    def test_rejects_dataset_name_mismatch(self, bar_data):
        specification = ChartSpecification(
            chart_type=ChartType.BAR,
            title="Revenue",
            data=bar_data,
        )

        with pytest.raises(
            VisualizationValidationError,
            match="dataset_name must match",
        ):
            VisualizationResult(
                visualization_id="viz-001",
                dataset_id="dataset-001",
                dataset_name="another-name",
                specification=specification,
            )

    def test_rejects_invalid_specification(self):
        with pytest.raises(
            VisualizationValidationError,
            match="ChartSpecification",
        ):
            VisualizationResult(
                visualization_id="viz-001",
                dataset_id="dataset-001",
                dataset_name="sales",
                specification=None,
            )

    def test_metadata_is_read_only(self, bar_data):
        specification = ChartSpecification(
            chart_type=ChartType.BAR,
            title="Revenue",
            data=bar_data,
        )
        metadata = {"created_by": "test"}

        result = VisualizationResult(
            visualization_id="viz-001",
            dataset_id="dataset-001",
            dataset_name="sales",
            specification=specification,
            metadata=metadata,
        )

        metadata["created_by"] = "changed"

        assert result.metadata["created_by"] == "test"

        with pytest.raises(TypeError):
            result.metadata["created_by"] = "changed"

    def test_result_is_frozen(self, bar_data):
        specification = ChartSpecification(
            chart_type=ChartType.BAR,
            title="Revenue",
            data=bar_data,
        )
        result = VisualizationResult(
            visualization_id="viz-001",
            dataset_id="dataset-001",
            dataset_name="sales",
            specification=specification,
        )

        with pytest.raises(FrozenInstanceError):
            result.visualization_id = "changed"
