"""Integration tests for cross-module data pipeline consistency."""

import pandas as pd
import pytest

from lite_cog_agent.cleaning.contracts import (
    CleaningRequest,
    CleaningStatus,
    CleaningStep,
)
from lite_cog_agent.cleaning.service import CleaningService
from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.loading.contracts import DatasetLoadRequest
from lite_cog_agent.loading.service import DatasetLoadingService
from lite_cog_agent.profiling.service import ProfilingService
from lite_cog_agent.statistics.service import StatisticalAnalysisService
from lite_cog_agent.visualization.contracts import ChartType
from lite_cog_agent.visualization.service import VisualizationService


def _create_sales_csv(tmp_path):
    """Create a dataset with one duplicate row and one missing value."""
    csv_path = tmp_path / "sales_consistency.csv"

    csv_path.write_text(
        "department,product,quantity,amount\n"
        "Sales,A,2,100\n"
        "Sales,B,3,150\n"
        "Support,A,1,80\n"
        "Support,C,4,200\n"
        "Sales,A,2,100\n"
        "Sales,B,,120\n",
        encoding="utf-8",
    )

    return csv_path


def _load_source_dataset(tmp_path, context):
    """Load the source dataset into the supplied context."""
    csv_path = _create_sales_csv(tmp_path)

    return DatasetLoadingService().load_into_context(
        DatasetLoadRequest(
            file_path=csv_path,
            name="sales_consistency",
        ),
        context,
    )


def _clean_source_dataset(context, dataset_id):
    """Fill missing quantity values and remove duplicate rows."""
    return CleaningService(context).clean(
        CleaningRequest(
            dataset_id=dataset_id,
            steps=(
                CleaningStep(
                    operation="missing.fill",
                    parameters={
                        "value": 0,
                        "columns": ["quantity"],
                    },
                ),
                CleaningStep(
                    operation="duplicates.drop",
                    parameters={
                        "keep": "first",
                        "ignore_index": True,
                    },
                ),
            ),
            output_dataset_name="sales_consistency_cleaned",
        )
    )


def _frequency_counts(result):
    """Convert a frequency result into a category-to-count mapping."""
    return {
        item.value: item.count
        for item in result.items
    }


def test_cleaned_dataset_profile_matches_cleaning_quality_report(
    tmp_path,
):
    """Independent profiling must agree with the cleaning quality report."""
    context = ExecutionContext()

    try:
        source_reference = _load_source_dataset(
            tmp_path,
            context,
        )

        cleaning_result = _clean_source_dataset(
            context,
            source_reference.dataset_id,
        )

        assert cleaning_result.status is CleaningStatus.SUCCESS
        assert cleaning_result.output_dataset_id is not None
        assert cleaning_result.quality_comparison is not None

        cleaned_dataset_id = cleaning_result.output_dataset_id
        cleaned_dataframe = context.get_dataset(cleaned_dataset_id)
        cleaned_reference = context.get_dataset_reference(
            cleaned_dataset_id
        )

        # Run Profiling independently after cleaning.
        profiling_result = ProfilingService().profile(
            cleaned_dataframe,
            cleaned_reference,
        )

        cleaning_quality = (
            cleaning_result.quality_comparison.after
        )

        # The independently calculated quality report must match
        # the report produced inside CleaningService.
        assert profiling_result.quality_report == cleaning_quality

        # Verify the dataset-level profile independently as well.
        dataset_profile = profiling_result.dataset_profile

        assert dataset_profile.dataset_id == cleaned_dataset_id
        assert dataset_profile.row_count == 5
        assert dataset_profile.column_count == 4
        assert dataset_profile.duplicate_row_count == 0

        assert cleaning_quality.total_cells == 20
        assert cleaning_quality.missing_cells == 0
        assert cleaning_quality.completeness == 100.0
        assert cleaning_quality.missing_rate == 0.0
        assert cleaning_quality.duplicate_row_count == 0
        assert cleaning_quality.duplicate_rate == 0.0

    finally:
        context.close()


def test_cleaning_changes_frequency_analysis_consistently(
    tmp_path,
):
    """Frequency results must reflect filling missing values and deduplication."""
    context = ExecutionContext()

    try:
        source_reference = _load_source_dataset(
            tmp_path,
            context,
        )

        cleaning_result = _clean_source_dataset(
            context,
            source_reference.dataset_id,
        )

        assert cleaning_result.status is CleaningStatus.SUCCESS
        assert cleaning_result.output_dataset_id is not None

        cleaned_dataset_id = cleaning_result.output_dataset_id

        statistical_service = StatisticalAnalysisService()

        # Analyze the source and cleaned datasets independently.
        source_frequency = statistical_service.analyze_frequency(
            context,
            source_reference.dataset_id,
            column="department",
        )

        cleaned_frequency = statistical_service.analyze_frequency(
            context,
            cleaned_dataset_id,
            column="department",
        )

        assert source_frequency.dataset_id == (
            source_reference.dataset_id
        )
        assert cleaned_frequency.dataset_id == cleaned_dataset_id

        assert source_frequency.total_count == 6
        assert cleaned_frequency.total_count == 5

        assert _frequency_counts(source_frequency) == {
            "Sales": 4,
            "Support": 2,
        }

        assert _frequency_counts(cleaned_frequency) == {
            "Sales": 3,
            "Support": 2,
        }

        # The duplicate Sales row is removed by cleaning.
        assert (
            _frequency_counts(source_frequency)["Sales"]
            - _frequency_counts(cleaned_frequency)["Sales"]
            == 1
        )

        # The Support category is unaffected.
        assert (
            _frequency_counts(source_frequency)["Support"]
            == _frequency_counts(cleaned_frequency)["Support"]
        )

    finally:
        context.close()


def test_bar_chart_data_matches_frequency_analysis(
    tmp_path,
):
    """Bar chart specification data must match frequency analysis exactly."""
    context = ExecutionContext()

    try:
        source_reference = _load_source_dataset(
            tmp_path,
            context,
        )

        cleaning_result = _clean_source_dataset(
            context,
            source_reference.dataset_id,
        )

        assert cleaning_result.status is CleaningStatus.SUCCESS
        assert cleaning_result.output_dataset_id is not None

        cleaned_dataset_id = cleaning_result.output_dataset_id

        statistical_service = StatisticalAnalysisService()

        frequency_result = statistical_service.analyze_frequency(
            context,
            cleaned_dataset_id,
            column="product",
        )

        visualization_output = VisualizationService().create_bar_chart(
            context,
            cleaned_dataset_id,
            column="product",
        )

        visualization_result = visualization_output.result
        specification = visualization_result.specification

        assert visualization_result.dataset_id == cleaned_dataset_id
        assert specification.chart_type is ChartType.BAR
        assert specification.data.dataset_id == cleaned_dataset_id

        assert len(specification.data.series) == 1

        chart_series = specification.data.series[0]

        # Compare the actual structured chart data with the
        # independently calculated frequency analysis.
        chart_counts = dict(
            zip(
                chart_series.x_values,
                chart_series.y_values,
                strict=True,
            )
        )

        assert chart_counts == _frequency_counts(frequency_result)
        assert sum(chart_series.y_values) == frequency_result.total_count

        assert chart_series.name == "product"

    finally:
        context.close()


def test_histogram_data_matches_distribution_analysis_and_context_lifecycle(
    tmp_path,
):
    """Histogram bins must match statistical output without closing the context."""
    context = ExecutionContext()

    try:
        source_reference = _load_source_dataset(
            tmp_path,
            context,
        )

        cleaning_result = _clean_source_dataset(
            context,
            source_reference.dataset_id,
        )

        assert cleaning_result.status is CleaningStatus.SUCCESS
        assert cleaning_result.output_dataset_id is not None

        cleaned_dataset_id = cleaning_result.output_dataset_id

        statistical_service = StatisticalAnalysisService()
        visualization_service = VisualizationService()

        distribution_result = (
            statistical_service.analyze_distribution(
                context,
                cleaned_dataset_id,
                column="quantity",
                bins=5,
            )
        )

        visualization_output = visualization_service.create_histogram(
            context,
            cleaned_dataset_id,
            column="quantity",
            bins=5,
        )

        visualization_result = visualization_output.result
        specification = visualization_result.specification

        assert visualization_result.dataset_id == cleaned_dataset_id
        assert specification.chart_type is ChartType.HISTOGRAM
        assert specification.data.dataset_id == cleaned_dataset_id

        # Visualization must reuse the statistical histogram bins,
        # not independently calculate a different distribution.
        assert (
            specification.data.histogram_bins
            == distribution_result.histogram
        )

        assert sum(
            item.count
            for item in specification.data.histogram_bins
        ) == distribution_result.count

        assert distribution_result.count == 5
        assert distribution_result.mean == pytest.approx(2.0)

        # Loading, cleaning, statistics, and visualization all
        # operate on the same context. Services must not close it.
        assert context.is_closed is False

        assert context.dataset_store.contains(
            source_reference.dataset_id
        )
        assert context.dataset_store.contains(cleaned_dataset_id)

        assert set(context.dataset_ids) == {
            source_reference.dataset_id,
            cleaned_dataset_id,
        }

        assert len(context.dataset_store) == 2

        # Both source and cleaned datasets remain accessible.
        assert len(
            context.get_dataset(source_reference.dataset_id)
        ) == 6

        assert len(context.get_dataset(cleaned_dataset_id)) == 5

    finally:
        context.close()

    # Context lifecycle is owned by the caller, not by any service.
    assert context.is_closed is True
    assert len(context.dataset_store) == 0
