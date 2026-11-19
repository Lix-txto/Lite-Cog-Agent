"""Integration tests for loading, cleaning, statistics, and visualization."""

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
from lite_cog_agent.visualization.renderer import OutputFormat
from lite_cog_agent.visualization.service import VisualizationService


def _create_sales_csv(tmp_path):
    """Create a CSV containing one duplicate row and one missing value."""
    csv_path = tmp_path / "sales_pipeline.csv"

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
    """Load the integration-test CSV into the execution context."""
    csv_path = _create_sales_csv(tmp_path)

    loading_service = DatasetLoadingService()

    reference = loading_service.load_into_context(
        DatasetLoadRequest(
            file_path=csv_path,
            name="sales_pipeline",
        ),
        context,
    )

    return reference


def _clean_sales_dataset(context, source_dataset_id):
    """Fill missing quantity values and remove duplicate rows."""
    cleaning_service = CleaningService(context)

    return cleaning_service.clean(
        CleaningRequest(
            dataset_id=source_dataset_id,
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
            output_dataset_name="sales_pipeline_cleaned",
        )
    )


def test_loading_cleaning_statistics_visualization_end_to_end(
    tmp_path,
):
    """Verify the complete data-analysis pipeline."""
    context = ExecutionContext()

    try:
        # -----------------------------------------------------
        # 1. Load source data.
        # -----------------------------------------------------
        source_reference = _load_source_dataset(
            tmp_path,
            context,
        )

        source_dataframe = context.get_dataset(
            source_reference.dataset_id
        )

        original_dataframe = source_dataframe.copy(deep=True)

        assert source_reference.name == "sales_pipeline"
        assert source_reference.row_count == 6
        assert source_reference.column_count == 4

        assert source_dataframe.isna().sum().sum() == 1
        assert source_dataframe.duplicated().sum() == 1

        # -----------------------------------------------------
        # 2. Profile source data.
        # -----------------------------------------------------
        profiling_service = ProfilingService()

        before_profile = profiling_service.profile(
            source_dataframe,
            source_reference,
        )

        before_quality = before_profile.quality_report

        assert before_quality.total_cells == 24
        assert before_quality.missing_cells == 1
        assert before_quality.duplicate_row_count == 1

        # -----------------------------------------------------
        # 3. Clean source data.
        # -----------------------------------------------------
        cleaning_result = _clean_sales_dataset(
            context,
            source_reference.dataset_id,
        )

        assert cleaning_result.status is CleaningStatus.SUCCESS
        assert cleaning_result.error is None
        assert len(cleaning_result.operations) == 2

        assert all(
            operation.status is CleaningStatus.SUCCESS
            for operation in cleaning_result.operations
        )

        assert cleaning_result.input_row_count == 6
        assert cleaning_result.input_column_count == 4
        assert cleaning_result.output_row_count == 5
        assert cleaning_result.output_column_count == 4

        assert cleaning_result.output_dataset_id is not None

        # -----------------------------------------------------
        # 4. Retrieve cleaned data and validate its integrity.
        # -----------------------------------------------------
        cleaned_dataset_id = cleaning_result.output_dataset_id

        cleaned_dataframe = context.get_dataset(
            cleaned_dataset_id
        )

        cleaned_reference = context.get_dataset_reference(
            cleaned_dataset_id
        )

        assert cleaned_reference.name == "sales_pipeline_cleaned"
        assert cleaned_reference.source == "cleaning"

        assert len(cleaned_dataframe) == 5
        assert list(cleaned_dataframe.columns) == [
            "department",
            "product",
            "quantity",
            "amount",
        ]

        assert cleaned_dataframe["quantity"].isna().sum() == 0
        assert cleaned_dataframe.duplicated().sum() == 0

        assert cleaned_dataframe["quantity"].tolist() == [
            2,
            3,
            1,
            4,
            0,
        ]

        # -----------------------------------------------------
        # 5. Validate cleaning quality comparison.
        # -----------------------------------------------------
        comparison = cleaning_result.quality_comparison

        assert comparison is not None
        assert comparison.before == before_quality

        after_quality = comparison.after

        assert after_quality.total_cells == 20
        assert after_quality.missing_cells == 0
        assert after_quality.completeness == 100.0
        assert after_quality.missing_rate == 0.0
        assert after_quality.duplicate_row_count == 0

        assert after_quality.completeness > (
            before_quality.completeness
        )

        # -----------------------------------------------------
        # 6. Run statistical analysis on the cleaned dataset.
        # -----------------------------------------------------
        statistical_service = StatisticalAnalysisService()

        descriptive_result = (
            statistical_service.analyze_descriptive(
                context,
                cleaned_dataset_id,
            )
        )

        descriptive_by_column = {
            item.column: item
            for item in descriptive_result.statistics
        }

        assert descriptive_result.dataset_id == cleaned_dataset_id
        assert descriptive_result.row_count == 5

        assert descriptive_by_column["quantity"].count == 5
        assert descriptive_by_column["quantity"].mean == pytest.approx(
            2.0
        )

        assert descriptive_by_column["amount"].count == 5
        assert descriptive_by_column["amount"].mean == pytest.approx(
            130.0
        )

        grouped_result = statistical_service.analyze_grouped(
            context,
            cleaned_dataset_id,
            group_by=["department"],
            aggregations={
                "amount": ["sum"],
                "quantity": ["sum"],
            },
        )

        assert grouped_result.dataset_id == cleaned_dataset_id
        assert grouped_result.group_count == 2

        grouped_by_department = {
            record["department"]: record
            for record in grouped_result.records
        }

        assert grouped_by_department["Sales"]["amount_sum"] == 370
        assert grouped_by_department["Support"]["amount_sum"] == 280

        assert grouped_by_department["Sales"]["quantity_sum"] == 5
        assert grouped_by_department["Support"]["quantity_sum"] == 5

        distribution_result = (
            statistical_service.analyze_distribution(
                context,
                cleaned_dataset_id,
                column="quantity",
                bins=5,
            )
        )

        assert distribution_result.dataset_id == cleaned_dataset_id
        assert distribution_result.count == 5
        assert distribution_result.mean == pytest.approx(2.0)

        assert sum(
            item.count
            for item in distribution_result.histogram
        ) == 5

        # -----------------------------------------------------
        # 7. Generate visualizations from the cleaned dataset.
        # -----------------------------------------------------
        visualization_service = VisualizationService()

        grouped_chart = (
            visualization_service.create_grouped_bar_chart(
                context,
                cleaned_dataset_id,
                group_by=["department"],
                aggregations={
                    "amount": ["sum"],
                },
                output_format=OutputFormat.PNG,
            )
        )

        grouped_chart_result = grouped_chart.result
        grouped_specification = grouped_chart_result.specification

        assert grouped_chart_result.dataset_id == cleaned_dataset_id
        assert grouped_specification.chart_type is ChartType.BAR
        assert (
            grouped_specification.data.dataset_id
            == cleaned_dataset_id
        )

        assert grouped_chart.rendered_chart.output_format is (
            OutputFormat.PNG
        )
        assert grouped_chart.rendered_chart.mime_type == "image/png"
        assert grouped_chart.rendered_chart.image_data.startswith(
            b"\x89PNG\r\n\x1a\n"
        )

        histogram_chart = visualization_service.create_histogram(
            context,
            cleaned_dataset_id,
            column="quantity",
            bins=5,
            output_format=OutputFormat.SVG,
        )

        histogram_result = histogram_chart.result
        histogram_specification = histogram_result.specification

        assert histogram_result.dataset_id == cleaned_dataset_id
        assert histogram_specification.chart_type is ChartType.HISTOGRAM

        assert (
            histogram_specification.data.dataset_id
            == cleaned_dataset_id
        )

        assert len(
            histogram_specification.data.histogram_bins
        ) == 5

        assert sum(
            item.count
            for item in histogram_specification.data.histogram_bins
        ) == 5

        assert histogram_chart.rendered_chart.output_format is (
            OutputFormat.SVG
        )
        assert histogram_chart.rendered_chart.mime_type == (
            "image/svg+xml"
        )
        assert b"<svg" in histogram_chart.rendered_chart.image_data

        # -----------------------------------------------------
        # 8. Verify the source dataset was not modified.
        # -----------------------------------------------------
        pd.testing.assert_frame_equal(
            context.get_dataset(source_reference.dataset_id),
            original_dataframe,
        )

        assert context.dataset_store.contains(
            source_reference.dataset_id
        )

        assert context.dataset_store.contains(
            cleaned_dataset_id
        )

        assert len(context.dataset_store) == 2

    finally:
        context.close()


def test_cleaning_failure_does_not_publish_output_dataset(
    tmp_path,
):
    """A failed cleaning workflow must not register an output dataset."""
    context = ExecutionContext()

    try:
        source_reference = _load_source_dataset(
            tmp_path,
            context,
        )

        source_dataframe = context.get_dataset(
            source_reference.dataset_id
        )

        original_dataframe = source_dataframe.copy(deep=True)

        cleaning_service = CleaningService(context)

        result = cleaning_service.clean(
            CleaningRequest(
                dataset_id=source_reference.dataset_id,
                steps=(
                    CleaningStep(
                        operation="unknown.operation",
                        parameters={},
                    ),
                ),
            )
        )

        assert result.status is CleaningStatus.FAILED
        assert result.error is not None
        assert result.output_dataset_id is None
        assert result.output_row_count is None
        assert result.output_column_count is None

        assert len(result.operations) == 1
        assert result.operations[0].status is CleaningStatus.FAILED

        assert result.error.code == "cleaning_validation_error"

        assert context.dataset_store.contains(
            source_reference.dataset_id
        )

        assert len(context.dataset_store) == 1

        pd.testing.assert_frame_equal(
            context.get_dataset(source_reference.dataset_id),
            original_dataframe,
        )

    finally:
        context.close()


@pytest.mark.parametrize(
    "output_format",
    [
        OutputFormat.PNG,
        OutputFormat.SVG,
    ],
)
def test_cleaned_dataset_can_generate_both_chart_formats(
    tmp_path,
    output_format,
):
    """Both supported renderer formats work with cleaned datasets."""
    context = ExecutionContext()

    try:
        source_reference = _load_source_dataset(
            tmp_path,
            context,
        )

        cleaning_result = _clean_sales_dataset(
            context,
            source_reference.dataset_id,
        )

        assert cleaning_result.status is CleaningStatus.SUCCESS

        cleaned_dataset_id = cleaning_result.output_dataset_id

        visualization_service = VisualizationService()

        output = visualization_service.create_histogram(
            context,
            cleaned_dataset_id,
            column="amount",
            bins=5,
            output_format=output_format,
        )

        assert output.result.dataset_id == cleaned_dataset_id
        assert output.rendered_chart.output_format is output_format
        assert output.rendered_chart.image_data

        if output_format is OutputFormat.PNG:
            assert output.rendered_chart.mime_type == "image/png"
            assert output.rendered_chart.image_data.startswith(
                b"\x89PNG\r\n\x1a\n"
            )
        else:
            assert output.rendered_chart.mime_type == "image/svg+xml"
            assert b"<svg" in output.rendered_chart.image_data

    finally:
        context.close()