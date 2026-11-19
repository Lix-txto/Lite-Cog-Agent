"""Service layer for statistical analysis."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_numeric_dtype,
)

from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.execution.dataset_store import DatasetReference
from lite_cog_agent.statistics.contracts import (
    CorrelationAnalysisResult,
    DescriptiveStatisticsResult,
    DistributionAnalysisResult,
    FrequencyAnalysisResult,
    GroupedAnalysisResult,
)
from lite_cog_agent.statistics.correlation import CorrelationAnalyzer
from lite_cog_agent.statistics.descriptive import DescriptiveAnalyzer
from lite_cog_agent.statistics.distribution import DistributionAnalyzer
from lite_cog_agent.statistics.exceptions import (
    StatisticalAnalysisException,
    StatisticalExecutionError,
    StatisticalValidationError,
)
from lite_cog_agent.statistics.frequency import FrequencyAnalyzer
from lite_cog_agent.statistics.grouped import GroupedAnalyzer


class StatisticalAnalysisService:
    """Coordinate dataset access, analyzers, and result assembly.

    The service does not own the ExecutionContext lifecycle.
    It retrieves datasets through the context's public interface,
    delegates calculations to analyzers, and assembles result contracts.
    """

    def __init__(
        self,
        *,
        descriptive_analyzer: DescriptiveAnalyzer | None = None,
        frequency_analyzer: FrequencyAnalyzer | None = None,
        correlation_analyzer: CorrelationAnalyzer | None = None,
        grouped_analyzer: GroupedAnalyzer | None = None,
        distribution_analyzer: DistributionAnalyzer | None = None,
    ) -> None:
        self._descriptive_analyzer = (
            descriptive_analyzer
            if descriptive_analyzer is not None
            else DescriptiveAnalyzer()
        )
        self._frequency_analyzer = (
            frequency_analyzer
            if frequency_analyzer is not None
            else FrequencyAnalyzer()
        )
        self._correlation_analyzer = (
            correlation_analyzer
            if correlation_analyzer is not None
            else CorrelationAnalyzer()
        )
        self._grouped_analyzer = (
            grouped_analyzer
            if grouped_analyzer is not None
            else GroupedAnalyzer()
        )
        self._distribution_analyzer = (
            distribution_analyzer
            if distribution_analyzer is not None
            else DistributionAnalyzer()
        )

    def analyze_descriptive(
        self,
        context: ExecutionContext,
        dataset_id: str,
        *,
        columns: list[str] | None = None,
    ) -> DescriptiveStatisticsResult:
        """Calculate descriptive statistics for a dataset."""
        dataframe, reference = self._get_dataset(
            context,
            dataset_id,
        )

        statistics = self._run_analyzer(
            operation="descriptive",
            analyzer=self._descriptive_analyzer,
            dataframe=dataframe,
            columns=columns,
        )

        return self._build_result(
            operation="descriptive",
            result_type=DescriptiveStatisticsResult,
            dataset_id=reference.dataset_id,
            dataset_name=reference.name,
            row_count=len(dataframe),
            statistics=statistics,
        )

    def analyze_frequency(
        self,
        context: ExecutionContext,
        dataset_id: str,
        *,
        column: str,
        top_n: int | None = None,
        dropna: bool = True,
    ) -> FrequencyAnalysisResult:
        """Calculate category frequencies for one dataset column."""
        dataframe, reference = self._get_dataset(
            context,
            dataset_id,
        )

        items = self._run_analyzer(
            operation="frequency",
            analyzer=self._frequency_analyzer,
            dataframe=dataframe,
            column=column,
            top_n=top_n,
            dropna=dropna,
        )

        def process_result() -> FrequencyAnalysisResult:
            # Keep the denominator consistent with FrequencyAnalyzer:
            # top_n limits returned categories, not the statistical population.
            total_count = (
                int(dataframe[column].notna().sum())
                if dropna
                else len(dataframe)
            )

            return self._build_result(
                operation="frequency",
                result_type=FrequencyAnalysisResult,
                dataset_id=reference.dataset_id,
                dataset_name=reference.name,
                column=column,
                total_count=total_count,
                items=items,
            )

        return self._process_result(
            operation="frequency",
            processor=process_result,
        )
    
    def analyze_correlation(
        self,
        context: ExecutionContext,
        dataset_id: str,
        *,
        columns: list[str] | None = None,
        method: str = "pearson",
        min_periods: int = 2,
    ) -> CorrelationAnalysisResult:
        """Calculate a correlation matrix and pairwise observation counts."""
        dataframe, reference = self._get_dataset(
            context,
            dataset_id,
        )

        def process_result() -> CorrelationAnalysisResult:
            matrix, pair_counts = self._run_analyzer(
                operation="correlation",
                analyzer=self._correlation_analyzer,
                dataframe=dataframe,
                columns=columns,
                method=method,
                min_periods=min_periods,
            )

            selected_columns = self._resolve_correlation_columns(
                dataframe,
                columns,
            )

            return self._build_result(
                operation="correlation",
                result_type=CorrelationAnalysisResult,
                dataset_id=reference.dataset_id,
                dataset_name=reference.name,
                method=method,
                columns=selected_columns,
                matrix=matrix,
                pair_counts=pair_counts,
            )

        return self._process_result(
            operation="correlation",
            processor=process_result,
        )

    def analyze_grouped(
        self,
        context: ExecutionContext,
        dataset_id: str,
        *,
        group_by: list[str],
        aggregations: Mapping[str, list[str] | tuple[str, ...]],
        dropna: bool = True,
        sort: bool = True,
    ) -> GroupedAnalysisResult:
        """Calculate configured aggregations for dataset groups."""
        dataframe, reference = self._get_dataset(
            context,
            dataset_id,
        )

        def process_result() -> GroupedAnalysisResult:
            records = self._run_analyzer(
                operation="grouped",
                analyzer=self._grouped_analyzer,
                dataframe=dataframe,
                group_by=group_by,
                aggregations=aggregations,
                dropna=dropna,
                sort=sort,
            )

            normalized_aggregations = {
                column: tuple(operations)
                for column, operations in aggregations.items()
            }

            return self._build_result(
                operation="grouped",
                result_type=GroupedAnalysisResult,
                dataset_id=reference.dataset_id,
                dataset_name=reference.name,
                group_by=tuple(group_by),
                aggregations=normalized_aggregations,
                group_count=len(records),
                records=records,
            )

        return self._process_result(
            operation="grouped",
            processor=process_result,
        )

    def analyze_distribution(
        self,
        context: ExecutionContext,
        dataset_id: str,
        *,
        column: str,
        bins: int = 10,
        quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
    ) -> DistributionAnalysisResult:
        """Analyze the distribution of one numeric dataset column."""
        dataframe, reference = self._get_dataset(
            context,
            dataset_id,
        )

        def process_result() -> DistributionAnalysisResult:
            analysis = self._run_analyzer(
                operation="distribution",
                analyzer=self._distribution_analyzer,
                dataframe=dataframe,
                column=column,
                bins=bins,
                quantiles=quantiles,
            )

            return self._build_result(
                operation="distribution",
                result_type=DistributionAnalysisResult,
                dataset_id=reference.dataset_id,
                dataset_name=reference.name,
                column=column,
                count=analysis.count,
                mean=analysis.mean,
                median=analysis.median,
                std=analysis.std,
                skewness=analysis.skewness,
                kurtosis=analysis.kurtosis,
                quantiles=analysis.quantiles,
                histogram=analysis.histogram,
            )

        return self._process_result(
            operation="distribution",
            processor=process_result,
        )

    @staticmethod
    def _get_dataset(
        context: ExecutionContext,
        dataset_id: str,
    ) -> tuple[pd.DataFrame, DatasetReference]:
        """Retrieve a DataFrame and its reference through the context."""
        if not isinstance(context, ExecutionContext):
            raise StatisticalValidationError(
                "context must be an ExecutionContext"
            )

        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise StatisticalValidationError(
                "dataset_id must be a non-empty string"
            )

        try:
            dataframe = context.get_dataset(dataset_id)
            reference = context.get_dataset_reference(dataset_id)
        except (KeyError, RuntimeError) as exc:
            raise StatisticalExecutionError(
                f"Unable to access dataset '{dataset_id}': {exc}"
            ) from exc
        except Exception as exc:
            raise StatisticalExecutionError(
                f"Unexpected error accessing dataset '{dataset_id}': {exc}"
            ) from exc

        if not isinstance(dataframe, pd.DataFrame):
            raise StatisticalValidationError(
                "dataset must be a pandas DataFrame"
            )

        if not isinstance(reference, DatasetReference):
            raise StatisticalExecutionError(
                "dataset reference has an invalid type"
            )

        if reference.dataset_id != dataset_id:
            raise StatisticalExecutionError(
                "dataset reference does not match the requested dataset"
            )

        return dataframe, reference

    @staticmethod
    def _run_analyzer(
        *,
        operation: str,
        analyzer: Any,
        dataframe: pd.DataFrame,
        **parameters: Any,
    ) -> Any:
        """Run an analyzer and normalize execution failures."""
        try:
            return analyzer.analyze(
                dataframe,
                **parameters,
            )
        except StatisticalValidationError:
            # Preserve user-facing input validation errors.
            raise
        except StatisticalAnalysisException as exc:
            raise StatisticalExecutionError(
                f"{operation} analysis failed: {exc}"
            ) from exc
        except Exception as exc:
            raise StatisticalExecutionError(
                f"Unexpected error during {operation} analysis: {exc}"
            ) from exc

    @staticmethod
    def _process_result(
        *,
        operation: str,
        processor: Callable[[], Any],
    ) -> Any:
        """Process analyzer output and normalize unexpected failures."""
        try:
            return processor()
        except StatisticalAnalysisException:
            # Preserve already normalized statistical exceptions.
            raise
        except Exception as exc:
            raise StatisticalExecutionError(
                f"Unexpected error processing {operation} analysis result: "
                f"{exc}"
            ) from exc

    @staticmethod
    def _resolve_correlation_columns(
        dataframe: pd.DataFrame,
        columns: list[str] | None,
    ) -> tuple[str, ...]:
        """Resolve columns in the same order used by CorrelationAnalyzer.

        Explicit selections retain caller order. When omitted, supported
        numeric columns retain their original DataFrame column order.
        """
        if columns is not None:
            return tuple(columns)

        selected: list[str] = []

        for column in dataframe.columns:
            series = dataframe[column]
            dtype = series.dtype

            if is_bool_dtype(dtype) or is_complex_dtype(dtype):
                continue

            if is_numeric_dtype(dtype):
                selected.append(column)

        return tuple(selected)

    @staticmethod
    def _build_result(
        *,
        operation: str,
        result_type: type,
        **values: Any,
    ) -> Any:
        """Construct a result contract and normalize assembly failures."""
        try:
            return result_type(**values)
        except Exception as exc:
            raise StatisticalExecutionError(
                f"Failed to assemble {operation} analysis result: {exc}"
            ) from exc
