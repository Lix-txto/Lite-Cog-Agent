
"""Visualization service for creating charts from analyzed datasets."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import date, datetime
from math import isfinite
from numbers import Integral, Real
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.execution.dataset_store import DatasetReference
from lite_cog_agent.statistics.contracts import (
    DescriptiveStatisticsResult,
    DistributionAnalysisResult,
    FrequencyAnalysisResult,
    GroupedAnalysisResult,
)
from lite_cog_agent.statistics.exceptions import (
    StatisticalAnalysisException,
    StatisticalValidationError,
)
from lite_cog_agent.statistics.service import StatisticalAnalysisService
from lite_cog_agent.visualization.contracts import (
    ChartData,
    ChartSeries,
    ChartSpecification,
    ChartType,
    VisualizationResult,
)
from lite_cog_agent.visualization.exceptions import (
    VisualizationExecutionError,
    VisualizationException,
    VisualizationValidationError,
)
from lite_cog_agent.visualization.matplotlib_renderer import MatplotlibRenderer
from lite_cog_agent.visualization.outputs import VisualizationOutput
from lite_cog_agent.visualization.renderer import (
    ChartRenderer,
    OutputFormat,
    RenderedChart,
)


_MISSING_LABEL = "(Missing)"


class VisualizationService:
    """Coordinate statistical analysis and chart rendering.

    The service does not own the ExecutionContext lifecycle.
    It retrieves dataset metadata through the context's public
    interface, delegates statistical calculations, constructs
    chart specifications, and invokes a renderer.
    """

    def __init__(
        self,
        *,
        statistical_service: StatisticalAnalysisService | None = None,
        renderer: ChartRenderer | None = None,
    ) -> None:
        """Initialize the visualization service dependencies."""
        self._statistical_service = (
            statistical_service
            if statistical_service is not None
            else StatisticalAnalysisService()
        )

        self._renderer = (
            renderer
            if renderer is not None
            else MatplotlibRenderer()
        )

    def create_bar_chart(
        self,
        context: ExecutionContext,
        dataset_id: str,
        *,
        column: str,
        top_n: int | None = None,
        output_format: OutputFormat = OutputFormat.PNG,
        title: str | None = None,
    ) -> VisualizationOutput:
        """Create a category frequency bar chart."""
        reference = self._get_dataset_reference(
            context,
            dataset_id,
        )
        self._validate_output_format(output_format)
        self._validate_column(column)
        self._validate_top_n(top_n)

        analysis = self._run_statistical_analysis(
            lambda: self._statistical_service.analyze_frequency(
                context,
                dataset_id,
                column=column,
                top_n=top_n,
            )
        )

        if not isinstance(analysis, FrequencyAnalysisResult):
            raise VisualizationExecutionError(
                "frequency analysis returned an invalid result type"
            )

        self._validate_analysis_dataset(
            analysis.dataset_id,
            analysis.dataset_name,
            reference,
        )

        if analysis.column != column:
            raise VisualizationExecutionError(
                "frequency analysis returned an unexpected column"
            )

        if not analysis.items:
            raise VisualizationValidationError(
                f"column '{column}' has no categories to visualize"
            )

        labels = self._normalize_category_labels(
            tuple(item.value for item in analysis.items)
        )
        counts = tuple(item.count for item in analysis.items)

        series = ChartSeries(
            name=column,
            x_values=labels,
            y_values=counts,
        )

        chart_data = ChartData(
            dataset_id=reference.dataset_id,
            dataset_name=reference.name,
            series=(series,),
            metadata={
                "source_analysis": "frequency",
                "column": column,
                "top_n": top_n,
                "total_count": analysis.total_count,
                "returned_category_count": len(analysis.items),
                "category_scope": (
                    "top_n" if top_n is not None else "all"
                ),
            },
        )

        specification = ChartSpecification(
            chart_type=ChartType.BAR,
            title=title or f"{column} Frequency",
            data=chart_data,
            x_label=column,
            y_label="Count",
        )

        return self._render_and_package(
            reference=reference,
            specification=specification,
            output_format=output_format,
            metadata={
                "source_analysis": "frequency",
                "column": column,
                "top_n": top_n,
            },
        )

    def create_grouped_bar_chart(
        self,
        context: ExecutionContext,
        dataset_id: str,
        *,
        group_by: list[str],
        aggregations: Mapping[str, list[str] | tuple[str, ...]],
        output_format: OutputFormat = OutputFormat.PNG,
        title: str | None = None,
    ) -> VisualizationOutput:
        """Create a grouped bar chart from grouped aggregations."""
        reference = self._get_dataset_reference(
            context,
            dataset_id,
        )
        self._validate_output_format(output_format)
        self._validate_group_by(group_by)
        self._validate_aggregations(aggregations)

        analysis = self._run_statistical_analysis(
            lambda: self._statistical_service.analyze_grouped(
                context,
                dataset_id,
                group_by=group_by,
                aggregations=aggregations,
            )
        )

        if not isinstance(analysis, GroupedAnalysisResult):
            raise VisualizationExecutionError(
                "grouped analysis returned an invalid result type"
            )

        self._validate_analysis_dataset(
            analysis.dataset_id,
            analysis.dataset_name,
            reference,
        )

        expected_group_by = tuple(group_by)

        if analysis.group_by != expected_group_by:
            raise VisualizationExecutionError(
                "grouped analysis returned unexpected grouping columns"
            )

        if not analysis.records:
            raise VisualizationValidationError(
                "grouped analysis returned no records to visualize"
            )

        normalized_aggregations = {
            column: tuple(operations)
            for column, operations in aggregations.items()
        }

        if dict(analysis.aggregations) != normalized_aggregations:
            raise VisualizationExecutionError(
                "grouped analysis returned unexpected aggregations"
            )

        group_labels = self._build_group_labels(
            analysis.records,
            expected_group_by,
        )

        series_items: list[ChartSeries] = []

        for column, operations in normalized_aggregations.items():
            for operation in operations:
                field_name = f"{column}_{operation}"

                values: list[float | None] = []

                for index, record in enumerate(analysis.records):
                    if field_name not in record:
                        raise VisualizationExecutionError(
                            f"grouped analysis record {index} "
                            f"is missing field '{field_name}'"
                        )

                    values.append(
                        self._normalize_aggregation_value(
                            record[field_name],
                            field_name=field_name,
                            record_index=index,
                        )
                    )

                series_items.append(
                    ChartSeries(
                        name=field_name,
                        x_values=group_labels,
                        y_values=tuple(values),
                    )
                )

        if not series_items:
            raise VisualizationValidationError(
                "grouped bar chart requires at least one aggregation"
            )

        chart_data = ChartData(
            dataset_id=reference.dataset_id,
            dataset_name=reference.name,
            series=tuple(series_items),
            metadata={
                "source_analysis": "grouped",
                "group_by": expected_group_by,
                "aggregations": normalized_aggregations,
                "group_count": analysis.group_count,
            },
        )

        default_title = (
            f"Grouped Analysis by {', '.join(expected_group_by)}"
        )

        specification = ChartSpecification(
            chart_type=ChartType.BAR,
            title=title or default_title,
            data=chart_data,
            x_label=" | ".join(expected_group_by),
            y_label="Value",
        )

        return self._render_and_package(
            reference=reference,
            specification=specification,
            output_format=output_format,
            metadata={
                "source_analysis": "grouped",
                "group_by": expected_group_by,
                "aggregations": normalized_aggregations,
            },
        )

    def create_line_chart(
        self,
        context: ExecutionContext,
        dataset_id: str,
        *,
        x_column: str,
        y_column: str,
        output_format: OutputFormat = OutputFormat.PNG,
        title: str | None = None,
    ) -> VisualizationOutput:
        """Create a line chart from raw dataset observations."""
        reference = self._get_dataset_reference(
            context,
            dataset_id,
        )
        self._validate_output_format(output_format)
        self._validate_column(x_column)
        self._validate_column(y_column)

        dataframe = self._get_dataframe(
            context,
            dataset_id,
        )

        x_raw_values = self._get_column_values(
            dataframe,
            x_column,
        )
        y_raw_values = self._get_column_values(
            dataframe,
            y_column,
        )

        if len(x_raw_values) != len(y_raw_values):
            raise VisualizationExecutionError(
                "X and Y columns have inconsistent lengths"
            )

        x_values = tuple(
            self._normalize_line_x_value(
                value,
                column=x_column,
                row_index=index,
            )
            for index, value in enumerate(x_raw_values)
        )

        y_values = tuple(
            self._normalize_numeric_value(
                value,
                column=y_column,
                row_index=index,
            )
            for index, value in enumerate(y_raw_values)
        )

        if not any(
            x_value is not None and y_value is not None
            for x_value, y_value in zip(x_values, y_values)
        ):
            raise VisualizationValidationError(
                "line chart has no valid X/Y observations"
            )

        series = ChartSeries(
            name=y_column,
            x_values=x_values,
            y_values=y_values,
        )

        chart_data = ChartData(
            dataset_id=reference.dataset_id,
            dataset_name=reference.name,
            series=(series,),
            metadata={
                "source_analysis": "raw_dataset",
                "chart_source": "line",
                "x_column": x_column,
                "y_column": y_column,
                "observation_count": len(x_values),
            },
        )

        specification = ChartSpecification(
            chart_type=ChartType.LINE,
            title=self._resolve_title(
                title,
                f"{y_column} over {x_column}",
            ),
            data=chart_data,
            x_label=x_column,
            y_label=y_column,
        )

        return self._render_and_package(
            reference=reference,
            specification=specification,
            output_format=output_format,
            metadata={
                "source_analysis": "raw_dataset",
                "chart_type": "line",
                "x_column": x_column,
                "y_column": y_column,
            },
        )

    def create_scatter_chart(
        self,
        context: ExecutionContext,
        dataset_id: str,
        *,
        x_column: str,
        y_column: str,
        output_format: OutputFormat = OutputFormat.PNG,
        title: str | None = None,
    ) -> VisualizationOutput:
        """Create a scatter chart from raw numeric observations."""
        reference = self._get_dataset_reference(
            context,
            dataset_id,
        )
        self._validate_output_format(output_format)
        self._validate_column(x_column)
        self._validate_column(y_column)

        dataframe = self._get_dataframe(
            context,
            dataset_id,
        )

        x_raw_values = self._get_column_values(
            dataframe,
            x_column,
        )
        y_raw_values = self._get_column_values(
            dataframe,
            y_column,
        )

        if len(x_raw_values) != len(y_raw_values):
            raise VisualizationExecutionError(
                "X and Y columns have inconsistent lengths"
            )

        x_values = tuple(
            self._normalize_numeric_value(
                value,
                column=x_column,
                row_index=index,
            )
            for index, value in enumerate(x_raw_values)
        )

        y_values = tuple(
            self._normalize_numeric_value(
                value,
                column=y_column,
                row_index=index,
            )
            for index, value in enumerate(y_raw_values)
        )

        if not any(
            x_value is not None and y_value is not None
            for x_value, y_value in zip(x_values, y_values)
        ):
            raise VisualizationValidationError(
                "scatter chart has no valid X/Y observations"
            )

        series = ChartSeries(
            name=f"{x_column} vs {y_column}",
            x_values=x_values,
            y_values=y_values,
        )

        chart_data = ChartData(
            dataset_id=reference.dataset_id,
            dataset_name=reference.name,
            series=(series,),
            metadata={
                "source_analysis": "raw_dataset",
                "chart_source": "scatter",
                "x_column": x_column,
                "y_column": y_column,
                "observation_count": len(x_values),
            },
        )

        specification = ChartSpecification(
            chart_type=ChartType.SCATTER,
            title=self._resolve_title(
                title,
                f"{y_column} vs {x_column}",
            ),
            data=chart_data,
            x_label=x_column,
            y_label=y_column,
        )

        return self._render_and_package(
            reference=reference,
            specification=specification,
            output_format=output_format,
            metadata={
                "source_analysis": "raw_dataset",
                "chart_type": "scatter",
                "x_column": x_column,
                "y_column": y_column,
            },
        )

    def create_histogram(
        self,
        context: ExecutionContext,
        dataset_id: str,
        *,
        column: str,
        bins: int = 10,
        output_format: OutputFormat = OutputFormat.PNG,
        title: str | None = None,
    ) -> VisualizationOutput:
        """Create a histogram using an existing distribution analysis."""
        reference = self._get_dataset_reference(
            context,
            dataset_id,
        )
        self._validate_output_format(output_format)
        self._validate_column(column)
        self._validate_bins(bins)

        analysis = self._run_statistical_analysis(
            lambda: self._statistical_service.analyze_distribution(
                context,
                dataset_id,
                column=column,
                bins=bins,
            )
        )

        if not isinstance(analysis, DistributionAnalysisResult):
            raise VisualizationExecutionError(
                "distribution analysis returned an invalid result type"
            )

        self._validate_analysis_dataset(
            analysis.dataset_id,
            analysis.dataset_name,
            reference,
        )

        if analysis.column != column:
            raise VisualizationExecutionError(
                "distribution analysis returned an unexpected column"
            )

        if analysis.count <= 0 or not analysis.histogram:
            raise VisualizationValidationError(
                f"column '{column}' has no valid numeric values to visualize"
            )

        if any(
            item.upper_bound <= item.lower_bound
            for item in analysis.histogram
        ):
            raise VisualizationValidationError(
                "histogram contains one or more zero-width bins"
            )

        histogram_count = sum(
            item.count for item in analysis.histogram
        )

        chart_data = ChartData(
            dataset_id=reference.dataset_id,
            dataset_name=reference.name,
            histogram_bins=analysis.histogram,
            metadata={
                "source_analysis": "distribution",
                "column": column,
                "requested_bins": bins,
                "analysis_count": analysis.count,
                "histogram_count": histogram_count,
                "excluded_count": analysis.count - histogram_count,
            },
        )

        specification = ChartSpecification(
            chart_type=ChartType.HISTOGRAM,
            title=self._resolve_title(
                title,
                f"{column} Distribution",
            ),
            data=chart_data,
            x_label=column,
            y_label="Frequency",
        )

        return self._render_and_package(
            reference=reference,
            specification=specification,
            output_format=output_format,
            metadata={
                "source_analysis": "distribution",
                "column": column,
                "bins": bins,
                "analysis_count": analysis.count,
                "histogram_count": histogram_count,
            },
        )

    def create_pie_chart(
        self,
        context: ExecutionContext,
        dataset_id: str,
        *,
        column: str,
        top_n: int | None = None,
        output_format: OutputFormat = OutputFormat.PNG,
        title: str | None = None,
    ) -> VisualizationOutput:
        """Create a pie chart using an existing frequency analysis."""
        reference = self._get_dataset_reference(
            context,
            dataset_id,
        )
        self._validate_output_format(output_format)
        self._validate_column(column)
        self._validate_top_n(top_n)

        analysis = self._run_statistical_analysis(
            lambda: self._statistical_service.analyze_frequency(
                context,
                dataset_id,
                column=column,
                top_n=top_n,
            )
        )

        if not isinstance(analysis, FrequencyAnalysisResult):
            raise VisualizationExecutionError(
                "frequency analysis returned an invalid result type"
            )

        self._validate_analysis_dataset(
            analysis.dataset_id,
            analysis.dataset_name,
            reference,
        )

        if analysis.column != column:
            raise VisualizationExecutionError(
                "frequency analysis returned an unexpected column"
            )

        if not analysis.items:
            raise VisualizationValidationError(
                f"column '{column}' has no categories to visualize"
            )

        labels = self._normalize_category_labels(
            tuple(item.value for item in analysis.items)
        )
        counts = tuple(item.count for item in analysis.items)

        displayed_count = sum(counts)

        if displayed_count <= 0:
            raise VisualizationValidationError(
                "pie chart requires a positive total count"
            )

        displayed_percentage = sum(
            item.percentage for item in analysis.items
        )

        series = ChartSeries(
            name=column,
            x_values=labels,
            y_values=counts,
        )

        category_scope = (
            "top_n" if top_n is not None else "all"
        )

        chart_data = ChartData(
            dataset_id=reference.dataset_id,
            dataset_name=reference.name,
            series=(series,),
            metadata={
                "source_analysis": "frequency",
                "column": column,
                "top_n": top_n,
                "total_count": analysis.total_count,
                "returned_category_count": len(analysis.items),
                "displayed_count": displayed_count,
                "displayed_percentage": displayed_percentage,
                "category_scope": category_scope,
                "pie_normalization": "displayed_categories",
            },
        )

        specification = ChartSpecification(
            chart_type=ChartType.PIE,
            title=self._resolve_title(
                title,
                f"{column} Composition",
            ),
            data=chart_data,
            x_label=column,
            y_label="Percentage",
        )

        return self._render_and_package(
            reference=reference,
            specification=specification,
            output_format=output_format,
            metadata={
                "source_analysis": "frequency",
                "column": column,
                "top_n": top_n,
                "category_scope": category_scope,
                "total_count": analysis.total_count,
                "displayed_count": displayed_count,
                "displayed_percentage": displayed_percentage,
            },
        )

    def create_box_chart(
        self,
        context: ExecutionContext,
        dataset_id: str,
        *,
        column: str,
        output_format: OutputFormat = OutputFormat.PNG,
        title: str | None = None,
    ) -> VisualizationOutput:
        """Create a box plot using existing descriptive statistics."""
        reference = self._get_dataset_reference(
            context,
            dataset_id,
        )
        self._validate_output_format(output_format)
        self._validate_column(column)

        analysis = self._run_statistical_analysis(
            lambda: self._statistical_service.analyze_descriptive(
                context,
                dataset_id,
                columns=[column],
            )
        )

        if not isinstance(analysis, DescriptiveStatisticsResult):
            raise VisualizationExecutionError(
                "descriptive analysis returned an invalid result type"
            )

        self._validate_analysis_dataset(
            analysis.dataset_id,
            analysis.dataset_name,
            reference,
        )

        column_statistics = next(
            (
                item
                for item in analysis.statistics
                if item.column == column
            ),
            None,
        )

        if column_statistics is None:
            raise VisualizationExecutionError(
                "descriptive analysis did not return the requested column"
            )

        if column_statistics.count <= 0:
            raise VisualizationValidationError(
                f"column '{column}' has no valid numeric values to visualize"
            )

        required_statistics = {
            "min": column_statistics.min,
            "q1": column_statistics.q1,
            "median": column_statistics.median,
            "q3": column_statistics.q3,
            "max": column_statistics.max,
        }

        if any(
            value is None
            for value in required_statistics.values()
        ):
            raise VisualizationValidationError(
                f"column '{column}' has incomplete box plot statistics"
            )

        minimum = column_statistics.min
        q1 = column_statistics.q1
        median = column_statistics.median
        q3 = column_statistics.q3
        maximum = column_statistics.max

        if not (
            minimum <= q1 <= median <= q3 <= maximum
        ):
            raise VisualizationExecutionError(
                "descriptive analysis returned invalid box plot statistics"
            )

        series = ChartSeries(
            name=column,
            statistics=required_statistics,
            outliers=(),
        )

        chart_data = ChartData(
            dataset_id=reference.dataset_id,
            dataset_name=reference.name,
            series=(series,),
            metadata={
                "source_analysis": "descriptive",
                "column": column,
                "count": column_statistics.count,
                "outlier_count": 0,
                "outlier_method": "not_calculated",
            },
        )

        specification = ChartSpecification(
            chart_type=ChartType.BOX,
            title=self._resolve_title(
                title,
                f"{column} Box Plot",
            ),
            data=chart_data,
            x_label="Distribution",
            y_label=column,
        )

        return self._render_and_package(
            reference=reference,
            specification=specification,
            output_format=output_format,
            metadata={
                "source_analysis": "descriptive",
                "column": column,
                "count": column_statistics.count,
                "outlier_count": 0,
                "outlier_method": "not_calculated",
            },
        )

    @staticmethod
    def _get_dataframe(
        context: ExecutionContext,
        dataset_id: str,
    ) -> pd.DataFrame:
        """Retrieve a DataFrame from the execution context."""
        try:
            dataframe = context.get_dataset(dataset_id)
        except KeyError as exc:
            raise VisualizationValidationError(
                f"dataset '{dataset_id}' does not exist in this context"
            ) from exc
        except RuntimeError as exc:
            raise VisualizationExecutionError(
                f"unable to access dataset: {exc}"
            ) from exc
        except Exception as exc:
            raise VisualizationExecutionError(
                f"unexpected dataset access error: {exc}"
            ) from exc

        if not isinstance(dataframe, pd.DataFrame):
            raise VisualizationExecutionError(
                "dataset object must be a pandas DataFrame"
            )

        return dataframe

    @staticmethod
    def _get_column_values(
        dataframe: pd.DataFrame,
        column: str,
    ) -> tuple[Any, ...]:
        """Retrieve values from one unambiguous DataFrame column."""
        if column not in dataframe.columns:
            raise VisualizationValidationError(
                f"column '{column}' does not exist in the dataset"
            )

        values = dataframe[column]

        if not isinstance(values, pd.Series):
            raise VisualizationValidationError(
                f"column '{column}' is ambiguous"
            )

        return tuple(values.tolist())

    @staticmethod
    def _resolve_title(
        title: str | None,
        default_title: str,
    ) -> str:
        """Validate a custom title or return the default title."""
        if title is None:
            return default_title

        if not isinstance(title, str) or not title.strip():
            raise VisualizationValidationError(
                "title must be a non-empty string or None"
            )

        return title

    @staticmethod
    def _get_dataset_reference(
        context: ExecutionContext,
        dataset_id: str,
    ) -> DatasetReference:
        """Retrieve a dataset reference through the public context API."""
        if not isinstance(context, ExecutionContext):
            raise VisualizationValidationError(
                "context must be an ExecutionContext"
            )

        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise VisualizationValidationError(
                "dataset_id must be a non-empty string"
            )

        if context.is_closed:
            raise VisualizationExecutionError(
                "cannot visualize a dataset from a closed context"
            )

        try:
            reference = context.get_dataset_reference(dataset_id)
        except KeyError as exc:
            raise VisualizationValidationError(
                f"dataset '{dataset_id}' does not exist in this context"
            ) from exc
        except RuntimeError as exc:
            raise VisualizationExecutionError(
                f"unable to access dataset reference: {exc}"
            ) from exc
        except Exception as exc:
            raise VisualizationExecutionError(
                f"unexpected error accessing dataset reference: {exc}"
            ) from exc

        if not isinstance(reference, DatasetReference):
            raise VisualizationExecutionError(
                "dataset reference has an invalid type"
            )

        if reference.dataset_id != dataset_id:
            raise VisualizationExecutionError(
                "dataset reference does not match the requested dataset"
            )

        return reference

    @staticmethod
    def _validate_analysis_dataset(
        dataset_id: str,
        dataset_name: str,
        reference: DatasetReference,
    ) -> None:
        """Ensure an analysis result belongs to the requested dataset."""
        if (
            dataset_id != reference.dataset_id
            or dataset_name != reference.name
        ):
            raise VisualizationExecutionError(
                "statistical analysis result does not match "
                "the requested dataset"
            )

    @staticmethod
    def _validate_output_format(
        output_format: OutputFormat,
    ) -> None:
        if not isinstance(output_format, OutputFormat):
            raise VisualizationValidationError(
                "output_format must be an OutputFormat"
            )

    @staticmethod
    def _validate_column(column: str) -> None:
        if not isinstance(column, str) or not column.strip():
            raise VisualizationValidationError(
                "column must be a non-empty string"
            )

    @staticmethod
    def _validate_top_n(top_n: int | None) -> None:
        if top_n is not None and (
            isinstance(top_n, bool)
            or not isinstance(top_n, int)
            or top_n <= 0
        ):
            raise VisualizationValidationError(
                "top_n must be a positive integer or None"
            )

    @staticmethod
    def _validate_bins(bins: int) -> None:
        """Validate the requested histogram bin count."""
        if (
            isinstance(bins, bool)
            or not isinstance(bins, int)
            or bins <= 0
        ):
            raise VisualizationValidationError(
                "bins must be a positive integer"
            )

    @staticmethod
    def _validate_group_by(group_by: list[str]) -> None:
        if not isinstance(group_by, list) or not group_by:
            raise VisualizationValidationError(
                "group_by must be a non-empty list"
            )

        if any(
            not isinstance(column, str) or not column.strip()
            for column in group_by
        ):
            raise VisualizationValidationError(
                "group_by must contain non-empty strings"
            )

        if len(group_by) != len(set(group_by)):
            raise VisualizationValidationError(
                "group_by cannot contain duplicate columns"
            )

    @staticmethod
    def _validate_aggregations(
        aggregations: Mapping[str, list[str] | tuple[str, ...]],
    ) -> None:
        if not isinstance(aggregations, Mapping) or not aggregations:
            raise VisualizationValidationError(
                "aggregations must be a non-empty mapping"
            )

        for column, operations in aggregations.items():
            if not isinstance(column, str) or not column.strip():
                raise VisualizationValidationError(
                    "aggregation column names must be non-empty strings"
                )

            if not isinstance(operations, (list, tuple)) or not operations:
                raise VisualizationValidationError(
                    f"aggregations for '{column}' must be non-empty"
                )

            if any(
                not isinstance(operation, str) or not operation.strip()
                for operation in operations
            ):
                raise VisualizationValidationError(
                    f"aggregations for '{column}' must contain strings"
                )

            if len(operations) != len(set(operations)):
                raise VisualizationValidationError(
                    f"aggregations for '{column}' cannot contain duplicates"
                )

    @staticmethod
    def _run_statistical_analysis(
        operation: Callable[[], Any],
    ) -> Any:
        """Execute statistical analysis and normalize its exceptions."""
        try:
            return operation()
        except StatisticalValidationError as exc:
            raise VisualizationValidationError(
                f"statistical analysis validation failed: {exc}"
            ) from exc
        except StatisticalAnalysisException as exc:
            raise VisualizationExecutionError(
                f"statistical analysis failed: {exc}"
            ) from exc
        except Exception as exc:
            raise VisualizationExecutionError(
                f"unexpected statistical analysis error: {exc}"
            ) from exc

    def _render_and_package(
        self,
        *,
        reference: DatasetReference,
        specification: ChartSpecification,
        output_format: OutputFormat,
        metadata: Mapping[str, Any],
    ) -> VisualizationOutput:
        """Render a specification and assemble its public output."""
        try:
            rendered_chart = self._renderer.render(
                specification,
                output_format=output_format,
            )
        except VisualizationValidationError:
            raise
        except VisualizationExecutionError:
            raise
        except VisualizationException as exc:
            raise VisualizationExecutionError(
                f"renderer failed: {exc}"
            ) from exc
        except Exception as exc:
            raise VisualizationExecutionError(
                f"unexpected renderer error: {exc}"
            ) from exc

        if not isinstance(rendered_chart, RenderedChart):
            raise VisualizationExecutionError(
                "renderer returned an invalid result type"
            )

        result = VisualizationResult(
            visualization_id=str(uuid4()),
            dataset_id=reference.dataset_id,
            dataset_name=reference.name,
            specification=specification,
            metadata=metadata,
        )

        return VisualizationOutput(
            result=result,
            rendered_chart=rendered_chart,
        )

    @classmethod
    def _normalize_category_labels(
        cls,
        values: tuple[Any, ...],
    ) -> tuple[str | int | float, ...]:
        """Normalize category values into supported chart labels."""
        labels = tuple(
            cls._normalize_category_value(value)
            for value in values
        )

        if len(labels) != len(set(labels)):
            raise VisualizationValidationError(
                "category values produce duplicate chart labels"
            )

        return labels

    @staticmethod
    def _normalize_category_value(
        value: Any,
    ) -> str | int | float:
        """Convert one category value into a supported chart label."""
        if pd.api.types.is_scalar(value) and pd.isna(value):
            return _MISSING_LABEL

        if isinstance(value, np.datetime64):
            try:
                return pd.Timestamp(value).isoformat()
            except (TypeError, ValueError, OverflowError) as exc:
                raise VisualizationValidationError(
                    "unsupported NumPy datetime category"
                ) from exc

        if isinstance(value, np.generic):
            try:
                value = value.item()
            except (TypeError, ValueError) as exc:
                raise VisualizationValidationError(
                    "unable to normalize NumPy category value"
                ) from exc

        if isinstance(value, bool):
            return "True" if value else "False"

        if isinstance(value, (datetime, date)):
            return value.isoformat()

        if isinstance(value, str):
            return value

        if isinstance(value, Integral):
            return int(value)

        if isinstance(value, Real):
            normalized = float(value)

            if not isfinite(normalized):
                raise VisualizationValidationError(
                    "category values cannot contain infinite numbers"
                )

            return normalized

        raise VisualizationValidationError(
            f"unsupported category value type: {type(value).__name__}"
        )

    @classmethod
    def _build_group_labels(
        cls,
        records: tuple[Mapping[str, Any], ...],
        group_by: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Build stable, human-readable labels for grouped records."""
        labels: list[str] = []

        for index, record in enumerate(records):
            parts: list[str] = []

            for column in group_by:
                if column not in record:
                    raise VisualizationExecutionError(
                        f"grouped analysis record {index} "
                        f"is missing grouping field '{column}'"
                    )

                raw_value = record[column]

                # Handle missing values before normalizing them.
                if cls._is_missing_value(raw_value):
                    display_value = _MISSING_LABEL
                else:
                    value = cls._normalize_category_value(raw_value)

                    if isinstance(value, str):
                        display_value = json.dumps(
                            value,
                            ensure_ascii=False,
                        )
                    else:
                        display_value = str(value)

                parts.append(f"{column}={display_value}")

            labels.append(" | ".join(parts))

        if len(labels) != len(set(labels)):
            raise VisualizationValidationError(
                "grouping values produce duplicate chart labels"
            )

        return tuple(labels)

    @staticmethod
    def _normalize_aggregation_value(
        value: Any,
        *,
        field_name: str,
        record_index: int,
    ) -> float | None:
        """Validate and normalize a grouped bar Y-axis value."""
        if value is None:
            return None

        if pd.api.types.is_scalar(value) and pd.isna(value):
            return None

        if isinstance(value, np.generic):
            try:
                value = value.item()
            except (TypeError, ValueError) as exc:
                raise VisualizationValidationError(
                    f"invalid value in '{field_name}' at record {record_index}"
                ) from exc

        if isinstance(value, bool) or not isinstance(value, Real):
            raise VisualizationValidationError(
                f"'{field_name}' at record {record_index} "
                "must be a numeric value or None"
            )

        normalized = float(value)

        if not isfinite(normalized):
            raise VisualizationValidationError(
                f"'{field_name}' at record {record_index} "
                "must be finite"
            )

        return normalized

    @staticmethod
    def _is_missing_value(value: Any) -> bool:
        """Check whether a scalar value represents missing data."""
        return (
            pd.api.types.is_scalar(value)
            and bool(pd.isna(value))
        )

    @classmethod
    def _normalize_line_x_value(
        cls,
        value: Any,
        *,
        column: str,
        row_index: int,
    ) -> str | int | float | None:
        """Normalize a line chart X-axis value."""
        if cls._is_missing_value(value):
            return None

        if isinstance(value, np.datetime64):
            try:
                return pd.Timestamp(value).isoformat()
            except (TypeError, ValueError, OverflowError) as exc:
                raise VisualizationValidationError(
                    f"invalid datetime in '{column}' at row {row_index}"
                ) from exc

        if isinstance(value, np.generic):
            try:
                value = value.item()
            except (TypeError, ValueError) as exc:
                raise VisualizationValidationError(
                    f"unable to normalize '{column}' at row {row_index}"
                ) from exc

        if cls._is_missing_value(value):
            return None

        if isinstance(value, bool):
            return "True" if value else "False"

        if isinstance(value, (datetime, date)):
            return value.isoformat()

        if isinstance(value, str):
            return value

        if isinstance(value, Integral):
            return int(value)

        if isinstance(value, Real):
            normalized = float(value)

            if not isfinite(normalized):
                return None

            return normalized

        raise VisualizationValidationError(
            f"unsupported line X value type in '{column}' "
            f"at row {row_index}: {type(value).__name__}"
        )

    @classmethod
    def _normalize_numeric_value(
        cls,
        value: Any,
        *,
        column: str,
        row_index: int,
    ) -> int | float | None:
        """Normalize a numeric chart value, preserving missing observations."""
        if cls._is_missing_value(value):
            return None

        if isinstance(value, np.generic):
            try:
                value = value.item()
            except (TypeError, ValueError) as exc:
                raise VisualizationValidationError(
                    f"unable to normalize '{column}' at row {row_index}"
                ) from exc

        if cls._is_missing_value(value):
            return None

        if isinstance(value, bool) or not isinstance(value, Real):
            raise VisualizationValidationError(
                f"'{column}' at row {row_index} "
                "must contain numeric values or missing values"
            )

        if isinstance(value, Integral):
            return int(value)

        normalized = float(value)

        if not isfinite(normalized):
            return None

        return normalized
