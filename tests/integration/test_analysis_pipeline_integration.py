"""Integration tests for the loading, profiling, and statistics pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.loading import (
    DatasetLoadRequest,
    DatasetLoadingService,
)
from lite_cog_agent.profiling import ProfilingService
from lite_cog_agent.statistics import StatisticalAnalysisService


@pytest.fixture
def source_dataframe() -> pd.DataFrame:
    """Create a representative business dataset."""
    return pd.DataFrame(
        {
            "department": [
                "Sales",
                "Sales",
                "Support",
                "Support",
                "Sales",
                "Sales",
            ],
            "product": [
                "A",
                "B",
                "A",
                "C",
                "A",
                "B",
            ],
            "quantity": [
                2,
                3,
                1,
                4,
                2,
                None,
            ],
            "amount": [
                100,
                150,
                80,
                200,
                100,
                120,
            ],
        }
    )


@pytest.fixture
def context():
    """Provide an isolated execution context."""
    execution_context = ExecutionContext()

    yield execution_context

    execution_context.close()


@pytest.fixture
def loading_service() -> DatasetLoadingService:
    return DatasetLoadingService()


@pytest.fixture
def profiling_service() -> ProfilingService:
    return ProfilingService()


@pytest.fixture
def statistical_service() -> StatisticalAnalysisService:
    return StatisticalAnalysisService()


@pytest.fixture
def loaded_dataset(
    tmp_path,
    source_dataframe: pd.DataFrame,
    loading_service: DatasetLoadingService,
    context: ExecutionContext,
):
    """Write a CSV file and load it into the execution context."""
    file_path = tmp_path / "sales_data.csv"

    source_dataframe.to_csv(
        file_path,
        index=False,
    )

    reference = loading_service.load_into_context(
        DatasetLoadRequest(
            file_path=file_path,
        ),
        context,
    )

    return reference


class TestAnalysisPipelineIntegration:
    """Test the complete loading -> profiling -> statistics pipeline."""

    def test_loaded_dataset_is_registered_in_context(
        self,
        context: ExecutionContext,
        loaded_dataset,        
    ):
        reference = loaded_dataset

        dataframe = context.get_dataset(
            reference.dataset_id,
        )

        assert isinstance(dataframe, pd.DataFrame)

        assert reference.row_count == 6
        assert reference.column_count == 4

        assert tuple(dataframe.columns) == (
            "department",
            "product",
            "quantity",
            "amount",
        )

        assert len(dataframe) == 6

        assert context.get_dataset_reference(
            reference.dataset_id,
        ) is reference

        assert dataframe["quantity"].isna().sum() == 1

    def test_loaded_dataset_can_be_profiled(
        self,
        context: ExecutionContext,
        loaded_dataset,
        profiling_service: ProfilingService,
    ):
        reference = loaded_dataset

        dataframe = context.get_dataset(
            reference.dataset_id,
        )

        report = profiling_service.profile(
            dataframe,
            reference,
        )

        dataset_profile = report.dataset_profile
        quality_report = report.quality_report

        assert dataset_profile.dataset_id == reference.dataset_id
        assert dataset_profile.dataset_name == reference.name

        assert dataset_profile.row_count == 6
        assert dataset_profile.column_count == 4

        assert dataset_profile.duplicate_row_count == 1

        assert quality_report.total_cells == 24
        assert quality_report.missing_cells == 1

        assert quality_report.completeness == pytest.approx(
            95.8333333333,
        )

        assert quality_report.missing_rate == pytest.approx(
            4.1666666667,
        )

        assert quality_report.duplicate_rate == pytest.approx(
            16.6666666667,
        )

    def test_descriptive_analysis_uses_loaded_dataset(
        self,
        context: ExecutionContext,
        loaded_dataset,
        statistical_service: StatisticalAnalysisService,
    ):
        reference = loaded_dataset

        result = statistical_service.analyze_descriptive(
            context,
            reference.dataset_id,
            columns=[
                "quantity",
                "amount",
            ],
        )

        assert result.dataset_id == reference.dataset_id
        assert result.dataset_name == reference.name
        assert result.row_count == 6

        statistics = {
            item.column: item
            for item in result.statistics
        }

        assert set(statistics) == {
            "quantity",
            "amount",
        }

        assert statistics["quantity"].count == 5
        assert statistics["quantity"].mean == pytest.approx(2.4)

        assert statistics["amount"].count == 6
        assert statistics["amount"].mean == pytest.approx(125.0)

    def test_frequency_analysis_returns_correct_department_counts(
        self,
        context: ExecutionContext,
        loaded_dataset,
        statistical_service: StatisticalAnalysisService,
    ):
        reference = loaded_dataset

        result = statistical_service.analyze_frequency(
            context,
            reference.dataset_id,
            column="department",
        )

        assert result.dataset_id == reference.dataset_id
        assert result.column == "department"
        assert result.total_count == 6

        frequencies = {
            item.value: item
            for item in result.items
        }

        assert frequencies["Sales"].count == 4
        assert frequencies["Support"].count == 2

        assert frequencies["Sales"].percentage == pytest.approx(
            400 / 6,
        )

        assert frequencies["Support"].percentage == pytest.approx(
            200 / 6,
        )

    def test_frequency_analysis_returns_correct_product_counts(
        self,
        context: ExecutionContext,
        loaded_dataset,
        statistical_service: StatisticalAnalysisService,
    ):
        reference = loaded_dataset

        result = statistical_service.analyze_frequency(
            context,
            reference.dataset_id,
            column="product",
        )

        frequencies = {
            item.value: item.count
            for item in result.items
        }

        assert result.total_count == 6

        assert frequencies == {
            "A": 3,
            "B": 2,
            "C": 1,
        }

    def test_grouped_analysis_returns_correct_aggregations(
        self,
        context: ExecutionContext,
        loaded_dataset,
        statistical_service: StatisticalAnalysisService,
    ):
        reference = loaded_dataset

        result = statistical_service.analyze_grouped(
            context,
            reference.dataset_id,
            group_by=["department"],
            aggregations={
                "amount": ["sum"],
                "quantity": ["sum"],
            },
        )

        assert result.dataset_id == reference.dataset_id

        assert result.group_by == ("department",)
        assert result.group_count == 2

        records = {
            record["department"]: record
            for record in result.records
        }

        assert set(records) == {
            "Sales",
            "Support",
        }

        assert records["Sales"]["amount_sum"] == pytest.approx(470.0)
        assert records["Support"]["amount_sum"] == pytest.approx(280.0)

        assert records["Sales"]["quantity_sum"] == pytest.approx(7.0)
        assert records["Support"]["quantity_sum"] == pytest.approx(5.0)

    def test_distribution_analysis_returns_correct_numeric_summary(
        self,
        context: ExecutionContext,
        loaded_dataset,
        statistical_service: StatisticalAnalysisService,
    ):
        reference = loaded_dataset

        result = statistical_service.analyze_distribution(
            context,
            reference.dataset_id,
            column="quantity",
            bins=5,
        )

        assert result.dataset_id == reference.dataset_id
        assert result.column == "quantity"

        assert result.count == 5
        assert result.mean == pytest.approx(2.4)
        assert result.median == pytest.approx(2.0)

        assert sum(
            item.count
            for item in result.histogram
        ) == result.count

    def test_analysis_pipeline_does_not_modify_loaded_dataset(
        self,
        context: ExecutionContext,
        loaded_dataset,
        profiling_service: ProfilingService,
        statistical_service: StatisticalAnalysisService,
    ):
        reference = loaded_dataset

        dataframe = context.get_dataset(
            reference.dataset_id,
        )

        original_dataframe = dataframe.copy(deep=True)

        profiling_service.profile(
            dataframe,
            reference,
        )

        statistical_service.analyze_descriptive(
            context,
            reference.dataset_id,
        )

        statistical_service.analyze_frequency(
            context,
            reference.dataset_id,
            column="department",
        )

        statistical_service.analyze_grouped(
            context,
            reference.dataset_id,
            group_by=["department"],
            aggregations={
                "amount": ["sum"],
            },
        )

        statistical_service.analyze_distribution(
            context,
            reference.dataset_id,
            column="quantity",
        )

        pd.testing.assert_frame_equal(
            dataframe,
            original_dataframe,
        )
