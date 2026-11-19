"""Matplotlib-based chart renderer."""

from __future__ import annotations

from io import BytesIO
from math import isfinite

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure

from lite_cog_agent.visualization.contracts import (
    ChartSpecification,
    ChartType,
)
from lite_cog_agent.visualization.exceptions import (
    VisualizationExecutionError,
    VisualizationValidationError,
)
from lite_cog_agent.visualization.renderer import (
    OutputFormat,
    RenderedChart,
)


class MatplotlibRenderer:
    """Render chart specifications into PNG or SVG bytes.

    Each render call creates an independent Figure and Canvas.
    The renderer does not use pyplot or modify the global backend.
    """

    def __init__(
        self,
        *,
        figsize: tuple[float, float] = (12.0, 8.0),
        dpi: float = 100.0,
    ) -> None:
        self._figsize = self._validate_figsize(figsize)
        self._dpi = self._validate_positive_number(dpi, "dpi")

    @staticmethod
    def _validate_positive_number(value: object, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise VisualizationValidationError(
                f"{name} must be a positive finite number."
            )

        numeric_value = float(value)

        if not isfinite(numeric_value) or numeric_value <= 0:
            raise VisualizationValidationError(
                f"{name} must be a positive finite number."
            )

        return numeric_value

    @classmethod
    def _validate_figsize(
        cls,
        figsize: object,
    ) -> tuple[float, float]:
        if not isinstance(figsize, tuple) or len(figsize) != 2:
            raise VisualizationValidationError(
                "figsize must be a tuple containing width and height."
            )

        width = cls._validate_positive_number(figsize[0], "figsize width")
        height = cls._validate_positive_number(figsize[1], "figsize height")

        return width, height

    def render(
        self,
        specification: ChartSpecification,
        *,
        output_format: OutputFormat = OutputFormat.PNG,
    ) -> RenderedChart:
        """Render a specification and return its image bytes."""

        if not isinstance(specification, ChartSpecification):
            raise VisualizationValidationError(
                "specification must be a ChartSpecification."
            )

        if not isinstance(output_format, OutputFormat):
            raise VisualizationValidationError(
                "output_format must be an OutputFormat."
            )

        figure: Figure | None = None

        try:
            figure = self._create_figure()
            canvas = self._create_canvas(figure, output_format)

            self._configure_axes(figure, specification)

            image_data = self._render_to_bytes(canvas, output_format)

            return RenderedChart(
                image_data=image_data,
                output_format=output_format,
                width=round(self._figsize[0] * self._dpi),
                height=round(self._figsize[1] * self._dpi),
                metadata={
                    "renderer": "matplotlib",
                    "dpi": self._dpi,
                    "figsize": self._figsize,
                    "chart_type": specification.chart_type.value,
                },
            )

        except VisualizationValidationError:
            raise

        except Exception as exc:
            raise VisualizationExecutionError(
                f"Failed to render chart: {exc}"
            ) from exc

        finally:
            if figure is not None:
                figure.clear()

    def _create_figure(self) -> Figure:
        """Create a fresh Figure for one render operation."""

        return Figure(
            figsize=self._figsize,
            dpi=self._dpi,
        )

    @staticmethod
    def _create_canvas(
        figure: Figure,
        output_format: OutputFormat,
    ) -> FigureCanvasAgg | FigureCanvasSVG:
        """Create the canvas corresponding to the requested format."""

        if output_format is OutputFormat.PNG:
            return FigureCanvasAgg(figure)

        if output_format is OutputFormat.SVG:
            return FigureCanvasSVG(figure)

        raise VisualizationValidationError(
            f"Unsupported output format: {output_format!r}"
        )

    def _configure_axes(
        self,
        figure: Figure,
        specification: ChartSpecification,
    ) -> None:
        """Create axes and dispatch to the chart-specific renderer."""

        axes = figure.add_subplot(1, 1, 1)

        axes.set_title(specification.title)

        if specification.x_label is not None:
            axes.set_xlabel(specification.x_label)

        if specification.y_label is not None:
            axes.set_ylabel(specification.y_label)

        if specification.chart_type is ChartType.BAR:
            self._render_bar(axes, specification)
            return

        if specification.chart_type is ChartType.LINE:
            self._render_line(axes, specification)
            return

        if specification.chart_type is ChartType.HISTOGRAM:
            self._render_histogram(axes, specification)
            return

        if specification.chart_type is ChartType.SCATTER:
            self._render_scatter(axes, specification)
            return
        
        if specification.chart_type is ChartType.PIE:
            self._render_pie(axes, specification)
            return

        if specification.chart_type is ChartType.BOX:
            self._render_box(axes, specification)
            return
                    
        raise VisualizationExecutionError(
            f"Chart type {specification.chart_type.value!r} "
            "is not implemented yet."
        )


    @staticmethod
    def _render_to_bytes(
        canvas: FigureCanvasAgg | FigureCanvasSVG,
        output_format: OutputFormat,
    ) -> bytes:
        """Render the canvas into an in-memory byte buffer."""

        buffer = BytesIO()

        try:
            if output_format is OutputFormat.PNG:
                if not isinstance(canvas, FigureCanvasAgg):
                    raise VisualizationExecutionError(
                        "PNG output requires FigureCanvasAgg."
                    )

                canvas.print_png(buffer)

            elif output_format is OutputFormat.SVG:
                if not isinstance(canvas, FigureCanvasSVG):
                    raise VisualizationExecutionError(
                        "SVG output requires FigureCanvasSVG."
                    )

                canvas.print_svg(buffer)

            else:
                raise VisualizationValidationError(
                    f"Unsupported output format: {output_format!r}"
                )

            image_data = buffer.getvalue()

            if not image_data:
                raise VisualizationExecutionError(
                    "Matplotlib returned empty image data."
                )

            return image_data

        finally:
            buffer.close()


    @staticmethod
    def _render_bar(
        axes,
        specification: ChartSpecification,
    ) -> None:
        """Render grouped categorical or numeric bar series."""

        series_list = specification.data.series

        categories: list[str | int | float] = []
        seen_categories: set[str | int | float] = set()

        for series in series_list:
            series_categories: set[str | int | float] = set()

            for value in series.x_values:
                if value is None:
                    continue

                if value in series_categories:
                    raise VisualizationValidationError(
                        f"Bar series {series.name!r} "
                        f"contains duplicate category {value!r}."
                    )

                series_categories.add(value)

                if value not in seen_categories:
                    categories.append(value)
                    seen_categories.add(value)

        if not categories:
            raise VisualizationValidationError(
                "Bar chart requires at least one non-null x value."
            )

        category_positions = {
            category: index
            for index, category in enumerate(categories)
        }

        series_count = len(series_list)
        bar_width = 0.8 / series_count

        for series_index, series in enumerate(series_list):
            offset = (
                series_index - (series_count - 1) / 2
            ) * bar_width

            positions: list[float] = []
            values: list[float] = []

            for x_value, y_value in zip(
                series.x_values,
                series.y_values,
                strict=True,
            ):
                if x_value is None or y_value is None:
                    continue

                positions.append(
                    category_positions[x_value] + offset
                )
                values.append(float(y_value))

            axes.bar(
                positions,
                values,
                width=bar_width,
                label=series.name,
            )

        axes.set_xticks(
            list(range(len(categories))),
            labels=[str(category) for category in categories],
        )

        if series_count > 1:
            axes.legend()


    @staticmethod
    def _render_line(
        axes,
        specification: ChartSpecification,
    ) -> None:
        """Render one or multiple line series."""

        series_list = specification.data.series

        all_x_values = [
            value
            for series in series_list
            for value in series.x_values
            if value is not None
        ]

        if not all_x_values:
            raise VisualizationValidationError(
                "Line chart requires at least one non-null x value."
            )

        numeric_x_axis = all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            for value in all_x_values
        )

        categories: list[str | int | float] = []
        category_positions: dict[str | int | float, int] = {}

        if not numeric_x_axis:
            for value in all_x_values:
                if value not in category_positions:
                    category_positions[value] = len(categories)
                    categories.append(value)

        for series in series_list:
            x_values: list[float] = []
            y_values: list[float] = []

            for x_value, y_value in zip(
                series.x_values,
                series.y_values,
                strict=True,
            ):
                if x_value is None:
                    x_values.append(float("nan"))
                    y_values.append(float("nan"))
                    continue

                if y_value is None:
                    y_values.append(float("nan"))
                else:
                    y_values.append(float(y_value))

                if numeric_x_axis:
                    x_values.append(float(x_value))
                else:
                    x_values.append(
                        float(category_positions[x_value])
                    )

            axes.plot(
                x_values,
                y_values,
                marker="o",
                label=series.name,
            )

        if not numeric_x_axis:
            axes.set_xticks(
                list(range(len(categories))),
                labels=[str(category) for category in categories],
            )

        if len(series_list) > 1:
            axes.legend()

    @staticmethod
    def _render_histogram(
        axes,
        specification: ChartSpecification,
    ) -> None:
        """Render a histogram using precomputed histogram bins."""

        histogram_bins = specification.data.histogram_bins

        if not histogram_bins:
            raise VisualizationValidationError(
                "Histogram chart requires at least one bin."
            )

        left_edges = [
            item.lower_bound
            for item in histogram_bins
        ]

        counts = [
            item.count
            for item in histogram_bins
        ]

        widths = [
            item.upper_bound - item.lower_bound
            for item in histogram_bins
        ]

        axes.bar(
            left_edges,
            counts,
            width=widths,
            align="edge",
        )

    @staticmethod
    def _render_scatter(
        axes,
        specification: ChartSpecification,
    ) -> None:
        """Render one or multiple scatter series."""

        series_list = specification.data.series
        has_points = False

        for series in series_list:
            x_values: list[float] = []
            y_values: list[float] = []

            for x_value, y_value in zip(
                series.x_values,
                series.y_values,
                strict=True,
            ):
                if x_value is None or y_value is None:
                    continue

                x_values.append(float(x_value))
                y_values.append(float(y_value))

            if not x_values:
                continue

            axes.scatter(
                x_values,
                y_values,
                label=series.name,
            )

            has_points = True

        if not has_points:
            raise VisualizationValidationError(
                "Scatter chart requires at least one valid data point."
            )

        if len(series_list) > 1:
            axes.legend()

    @staticmethod
    def _render_pie(
        axes,
        specification: ChartSpecification,
    ) -> None:
        """Render a pie chart from a single series."""

        series = specification.data.series[0]

        labels = [
            str(value) if value is not None else ""
            for value in series.x_values
        ]

        values = list(series.y_values)

        total = sum(values)

        if total <= 0:
            raise VisualizationValidationError(
                "Pie chart requires a positive total value."
            )

        axes.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
        )

    @staticmethod
    def _render_box(
        axes,
        specification: ChartSpecification,
    ) -> None:
        """Render one or multiple box plots."""

        boxplot_stats = []

        for series in specification.data.series:
            statistics = series.statistics

            boxplot_stats.append(
                {
                    "label": series.name,
                    "whislo": statistics["min"],
                    "q1": statistics["q1"],
                    "med": statistics["median"],
                    "q3": statistics["q3"],
                    "whishi": statistics["max"],
                    "fliers": list(series.outliers),
                }
            )

        axes.bxp(
            boxplot_stats,
            showfliers=True,
        )