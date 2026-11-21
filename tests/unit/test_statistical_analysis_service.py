"""Unit tests for StatisticalAnalysisService."""

from __future__ import annotations

from collections.abc import Mapping
from unittest.mock import patch

import pandas as pd
import pytest

from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.execution.dataset_store import DatasetReference
from lite_cog_agent.statistics import (
    CorrelationAnalysisError,
    CorrelationAnalysisResult,
    CorrelationAnalyzer,
    DescriptiveAnalysisError,
    DescriptiveAnalyzer,
    DescriptiveStatistics,
    DescriptiveStatisticsResult,
    DistributionAnalysisData,
    DistributionAnalysisError,
    DistributionAnalysisResult,
    DistributionAnalyzer,
    FrequencyAnalysisError,
    FrequencyAnalyzer,
    FrequencyItem,
    GroupedAnalysisError,
    GroupedAnalyzer,
    HistogramBin,
    StatisticalAnalysisService,
    StatisticalExecutionError,
    StatisticalValidationError,
)
from lite_cog_agent.statistics.contracts import (
    DerivedMetricAnalysisResult,
)
from lite_cog_agent.statistics.derived_metric import DerivedMetricAnalyzer
from lite_cog_agent.statistics.exceptions import (
    DerivedMetricAnalysisError,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Provide a representative dataset for service tests."""
    return pd.DataFrame(
        {
            "age": [22, 25, 30, 35],
            "salary": [5000.0, 6000.0, 8000.0, 10000.0],
            "department": ["A", "A", "B", "B"],
            "active": [True, False, True, False],
        }
    )


@pytest.fixture
def dataset_context(
    sample_dataframe: pd.DataFrame,
) -> tuple[ExecutionContext, DatasetReference]:
    """Register a dataset in a real ExecutionContext."""
    context = ExecutionContext()

    reference = DatasetReference(
        name="employee_dataset",
        source="unit_test",
        row_count=len(sample_dataframe),
        column_count=len(sample_dataframe.columns),
        columns=tuple(sample_dataframe.columns),
    )

    context.register_dataset(reference, sample_dataframe)

    yield context, reference

    context.close()


class StubAnalyzer:
    """A controllable analyzer used to isolate service orchestration."""

    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[pd.DataFrame, dict]] = []

    def analyze(self, dataframe: pd.DataFrame, **parameters):
        self.calls.append((dataframe, parameters))

        if self.error is not None:
            raise self.error

        return self.result


class InvalidDataContext(ExecutionContext):
    """ExecutionContext returning an invalid dataset object."""

    def get_dataset(self, dataset_id: str):
        return ["not", "a", "dataframe"]


class InvalidReferenceContext(ExecutionContext):
    """ExecutionContext returning an invalid dataset reference."""

    def get_dataset_reference(self, dataset_id: str):
        return object()


class FailingDataContext(ExecutionContext):
    """ExecutionContext raising an unexpected dataset access error."""

    def get_dataset(self, dataset_id: str):
        raise ValueError("simulated storage failure")


def make_context(
    dataframe: pd.DataFrame | None = None,
) -> tuple[ExecutionContext, DatasetReference]:
    """Create a context containing a registered test dataset."""
    if dataframe is None:
        dataframe = pd.DataFrame(
            {
                "age": [20, 30, 40],
                "salary": [3000.0, 5000.0, 7000.0],
                "department": ["A", "B", "B"],
            }
        )

    context = ExecutionContext()

    reference = DatasetReference(
        name="test_dataset",
        source="unit_test",
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        columns=tuple(dataframe.columns),
    )

    context.register_dataset(reference, dataframe)

    return context, reference


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestStatisticalAnalysisServiceInitialization:
    def test_creates_all_default_analyzers(self):
        service = StatisticalAnalysisService()

        assert isinstance(service._descriptive_analyzer, DescriptiveAnalyzer)
        assert isinstance(service._frequency_analyzer, FrequencyAnalyzer)
        assert isinstance(service._correlation_analyzer, CorrelationAnalyzer)
        assert isinstance(service._grouped_analyzer, GroupedAnalyzer)
        assert isinstance(service._distribution_analyzer, DistributionAnalyzer)
        assert isinstance(service._derived_metric_analyzer, DerivedMetricAnalyzer,)

    def test_supports_analyzer_dependency_injection(self):
        analyzers = {
            "descriptive_analyzer": StubAnalyzer(),
            "frequency_analyzer": StubAnalyzer(),
            "correlation_analyzer": StubAnalyzer(),
            "grouped_analyzer": StubAnalyzer(),
            "distribution_analyzer": StubAnalyzer(),
            "derived_metric_analyzer": StubAnalyzer(),
        }

        service = StatisticalAnalysisService(**analyzers)

        assert service._descriptive_analyzer is analyzers["descriptive_analyzer"]
        assert service._frequency_analyzer is analyzers["frequency_analyzer"]
        assert service._correlation_analyzer is analyzers["correlation_analyzer"]
        assert service._grouped_analyzer is analyzers["grouped_analyzer"]
        assert service._distribution_analyzer is analyzers["distribution_analyzer"]
        assert service._derived_metric_analyzer is analyzers["derived_metric_analyzer"]


# ---------------------------------------------------------------------------
# Descriptive analysis
# ---------------------------------------------------------------------------


class TestDescriptiveAnalysis:
    def test_assembles_descriptive_result(self, dataset_context):
        context, reference = dataset_context

        statistics = (
            DescriptiveStatistics(
                column="age",
                count=4,
                mean=28.0,
                median=27.5,
                std=5.507570547286102,
                variance=30.333333333333332,
                min=22.0,
                q1=24.25,
                q3=31.25,
                max=35.0,
                range=13.0,
            ),
        )

        analyzer = StubAnalyzer(result=statistics)
        service = StatisticalAnalysisService(
            descriptive_analyzer=analyzer,
        )

        result = service.analyze_descriptive(
            context,
            reference.dataset_id,
            columns=["age"],
        )

        assert isinstance(result, DescriptiveStatisticsResult)
        assert result.dataset_id == reference.dataset_id
        assert result.dataset_name == reference.name
        assert result.row_count == 4
        assert result.statistics == statistics

        called_dataframe, parameters = analyzer.calls[0]

        assert called_dataframe is context.get_dataset(reference.dataset_id)
        assert parameters == {"columns": ["age"]}

    def test_passes_none_columns_to_analyzer(self, dataset_context):
        context, reference = dataset_context

        analyzer = StubAnalyzer(result=())
        service = StatisticalAnalysisService(
            descriptive_analyzer=analyzer,
        )

        result = service.analyze_descriptive(
            context,
            reference.dataset_id,
        )

        assert result.statistics == ()
        assert analyzer.calls[0][1] == {"columns": None}


# ---------------------------------------------------------------------------
# Frequency analysis
# ---------------------------------------------------------------------------


class TestFrequencyAnalysis:
    def test_assembles_frequency_result(self, dataset_context):
        context, reference = dataset_context

        items = (
            FrequencyItem(
                value="A",
                count=2,
                percentage=50.0,
                cumulative_percentage=50.0,
            ),
            FrequencyItem(
                value="B",
                count=2,
                percentage=50.0,
                cumulative_percentage=100.0,
            ),
        )

        analyzer = StubAnalyzer(result=items)
        service = StatisticalAnalysisService(
            frequency_analyzer=analyzer,
        )

        result = service.analyze_frequency(
            context,
            reference.dataset_id,
            column="department",
            top_n=2,
            dropna=True,
        )

        assert result.dataset_id == reference.dataset_id
        assert result.dataset_name == reference.name
        assert result.column == "department"
        assert result.total_count == 4
        assert result.items == items

        assert analyzer.calls[0][1] == {
            "column": "department",
            "top_n": 2,
            "dropna": True,
        }

    def test_total_count_is_not_limited_by_top_n(self):
        dataframe = pd.DataFrame(
            {
                "category": ["A", "A", "A", "B", "B", "C"],
            }
        )
        context, reference = make_context(dataframe)

        analyzer = StubAnalyzer(
            result=(
                FrequencyItem(
                    value="A",
                    count=3,
                    percentage=50.0,
                    cumulative_percentage=50.0,
                ),
            )
        )

        service = StatisticalAnalysisService(
            frequency_analyzer=analyzer,
        )

        result = service.analyze_frequency(
            context,
            reference.dataset_id,
            column="category",
            top_n=1,
        )

        assert len(result.items) == 1
        assert result.items[0].value == "A"
        assert result.total_count == 6

        context.close()

    def test_dropna_false_includes_missing_values_in_total_count(self):
        dataframe = pd.DataFrame(
            {
                "category": ["A", "B", None, "A"],
            }
        )
        context, reference = make_context(dataframe)

        analyzer = StubAnalyzer(result=())
        service = StatisticalAnalysisService(
            frequency_analyzer=analyzer,
        )

        result = service.analyze_frequency(
            context,
            reference.dataset_id,
            column="category",
            dropna=False,
        )

        assert result.total_count == 4

        context.close()

    def test_dropna_true_excludes_missing_values_from_total_count(self):
        dataframe = pd.DataFrame(
            {
                "category": ["A", "B", None, "A"],
            }
        )
        context, reference = make_context(dataframe)

        analyzer = StubAnalyzer(result=())
        service = StatisticalAnalysisService(
            frequency_analyzer=analyzer,
        )

        result = service.analyze_frequency(
            context,
            reference.dataset_id,
            column="category",
            dropna=True,
        )

        assert result.total_count == 3

        context.close()


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------


class TestCorrelationAnalysis:
    def test_assembles_correlation_result(self, dataset_context):
        context, reference = dataset_context

        matrix = (
            (1.0, 0.8),
            (0.8, 1.0),
        )
        pair_counts = (
            (4, 4),
            (4, 4),
        )

        analyzer = StubAnalyzer(
            result=(matrix, pair_counts),
        )

        service = StatisticalAnalysisService(
            correlation_analyzer=analyzer,
        )

        result = service.analyze_correlation(
            context,
            reference.dataset_id,
            columns=["salary", "age"],
            method="spearman",
            min_periods=3,
        )

        assert isinstance(result, CorrelationAnalysisResult)
        assert result.dataset_id == reference.dataset_id
        assert result.dataset_name == reference.name
        assert result.method == "spearman"
        assert result.columns == ("salary", "age")
        assert result.matrix == matrix
        assert result.pair_counts == pair_counts

        assert analyzer.calls[0][1] == {
            "columns": ["salary", "age"],
            "method": "spearman",
            "min_periods": 3,
        }

    def test_preserves_explicit_column_order(self, dataset_context):
        context, reference = dataset_context

        analyzer = StubAnalyzer(
            result=(
                ((1.0, 0.5), (0.5, 1.0)),
                ((4, 4), (4, 4)),
            )
        )

        service = StatisticalAnalysisService(
            correlation_analyzer=analyzer,
        )

        result = service.analyze_correlation(
            context,
            reference.dataset_id,
            columns=["salary", "age"],
        )

        assert result.columns == ("salary", "age")

    def test_resolves_default_numeric_columns_in_dataframe_order(self):
        dataframe = pd.DataFrame(
            {
                "department": ["A", "B"],
                "salary": [3000.0, 5000.0],
                "active": [True, False],
                "age": [20, 30],
            }
        )

        context, reference = make_context(dataframe)

        analyzer = StubAnalyzer(
            result=(
                ((1.0, 0.5), (0.5, 1.0)),
                ((2, 2), (2, 2)),
            )
        )

        service = StatisticalAnalysisService(
            correlation_analyzer=analyzer,
        )

        result = service.analyze_correlation(
            context,
            reference.dataset_id,
        )

        assert result.columns == ("salary", "age")

        context.close()

    def test_empty_default_numeric_columns_are_supported(self):
        dataframe = pd.DataFrame(
            {
                "department": ["A", "B"],
                "active": [True, False],
            }
        )

        context, reference = make_context(dataframe)

        analyzer = StubAnalyzer(result=((), ()))
        service = StatisticalAnalysisService(
            correlation_analyzer=analyzer,
        )

        result = service.analyze_correlation(
            context,
            reference.dataset_id,
        )

        assert result.columns == ()
        assert result.matrix == ()
        assert result.pair_counts == ()

        context.close()


# ---------------------------------------------------------------------------
# Grouped analysis
# ---------------------------------------------------------------------------


class TestGroupedAnalysis:
    def test_assembles_grouped_result(self, dataset_context):
        context, reference = dataset_context

        records = (
            {
                "department": "A",
                "salary_count": 2,
                "salary_mean": 5500.0,
            },
            {
                "department": "B",
                "salary_count": 2,
                "salary_mean": 9000.0,
            },
        )

        analyzer = StubAnalyzer(result=records)
        service = StatisticalAnalysisService(
            grouped_analyzer=analyzer,
        )

        aggregations = {
            "salary": ["count", "mean"],
        }

        result = service.analyze_grouped(
            context,
            reference.dataset_id,
            group_by=["department"],
            aggregations=aggregations,
            dropna=False,
            sort=False,
        )

        assert result.dataset_id == reference.dataset_id
        assert result.dataset_name == reference.name
        assert result.group_by == ("department",)
        assert result.aggregations == {
            "salary": ("count", "mean"),
        }
        assert result.group_count == 2
        assert len(result.records) == 2

        assert analyzer.calls[0][1] == {
            "group_by": ["department"],
            "aggregations": aggregations,
            "dropna": False,
            "sort": False,
        }

        assert isinstance(result.aggregations, Mapping)
        assert isinstance(result.aggregations["salary"], tuple)

    def test_accepts_tuple_aggregation_operations(self, dataset_context):
        context, reference = dataset_context

        analyzer = StubAnalyzer(result=())
        service = StatisticalAnalysisService(
            grouped_analyzer=analyzer,
        )

        result = service.analyze_grouped(
            context,
            reference.dataset_id,
            group_by=["department"],
            aggregations={"salary": ("count", "mean")},
        )

        assert result.aggregations["salary"] == ("count", "mean")

    def test_grouped_records_are_immutable(self, dataset_context):
        context, reference = dataset_context

        analyzer = StubAnalyzer(
            result=(
                {
                    "department": "A",
                    "salary_mean": 5500.0,
                },
            )
        )
        service = StatisticalAnalysisService(
            grouped_analyzer=analyzer,
        )

        result = service.analyze_grouped(
            context,
            reference.dataset_id,
            group_by=["department"],
            aggregations={"salary": ["mean"]},
        )

        assert isinstance(result.records, tuple)
        assert isinstance(result.records[0], Mapping)

        with pytest.raises(TypeError):
            result.records[0]["salary_mean"] = 9999.0

# ---------------------------------------------------------------------------
# Derived metric analysis
# ---------------------------------------------------------------------------

class TestDerivedMetricAnalysis:
    def test_assembles_ungrouped_result(self, dataset_context):
        context, reference = dataset_context

        analyzer = StubAnalyzer(
            result=(
                {
                    "valid_row_count": 4,
                    "result": 29000.0,
                },
            )
        )
        service = StatisticalAnalysisService(
            derived_metric_analyzer=analyzer,
        )

        result = service.analyze_derived_metric(
            context,
            reference.dataset_id,
            left_column="age",
            right_column="salary",
        )

        assert isinstance(result, DerivedMetricAnalysisResult)
        assert result.dataset_id == reference.dataset_id
        assert result.dataset_name == reference.name
        assert result.row_count == 4
        assert result.valid_row_count == 4
        assert result.excluded_row_count == 0
        assert result.left_column == "age"
        assert result.right_column == "salary"
        assert result.operator == "multiply"
        assert result.aggregation == "sum"
        assert result.group_by == ()
        assert result.records[0]["valid_row_count"] == 4
        assert result.records[0]["result"] == 29000.0

        assert analyzer.calls[0][1] == {
            "left_column": "age",
            "right_column": "salary",
            "operator": "multiply",
            "aggregation": "sum",
            "group_by": (),
            "dropna": True,
        }

    def test_assembles_grouped_result(self, dataset_context):
        context, reference = dataset_context

        analyzer = StubAnalyzer(
            result=(
                {
                    "department": "A",
                    "valid_row_count": 2,
                    "result": 275000.0,
                },
                {
                    "department": "B",
                    "valid_row_count": 2,
                    "result": 660000.0,
                },
            )
        )
        service = StatisticalAnalysisService(
            derived_metric_analyzer=analyzer,
        )

        result = service.analyze_derived_metric(
            context,
            reference.dataset_id,
            left_column="age",
            right_column="salary",
            group_by=["department"],
            dropna=False,
        )

        assert result.group_by == ("department",)
        assert result.row_count == 4
        assert result.valid_row_count == 4
        assert result.excluded_row_count == 0
        assert len(result.records) == 2

        assert analyzer.calls[0][1] == {
            "left_column": "age",
            "right_column": "salary",
            "operator": "multiply",
            "aggregation": "sum",
            "group_by": ["department"],
            "dropna": False,
        }

    def test_counts_missing_group_keys_as_excluded_rows(self):
        dataframe = pd.DataFrame(
            {
                "group": ["A", None, "B", None],
                "left": [2.0, 3.0, 4.0, 5.0],
                "right": [10.0, 10.0, 10.0, 10.0],
            }
        )
        context, reference = make_context(dataframe)

        analyzer = StubAnalyzer(
            result=(
                {
                    "group": "A",
                    "valid_row_count": 1,
                    "result": 20.0,
                },
                {
                    "group": "B",
                    "valid_row_count": 1,
                    "result": 40.0,
                },
            )
        )
        service = StatisticalAnalysisService(
            derived_metric_analyzer=analyzer,
        )

        result = service.analyze_derived_metric(
            context,
            reference.dataset_id,
            left_column="left",
            right_column="right",
            group_by=["group"],
            dropna=True,
        )

        assert result.row_count == 4
        assert result.valid_row_count == 2
        assert result.excluded_row_count == 2
        assert sum(
            record["valid_row_count"]
            for record in result.records
        ) == result.valid_row_count

    def test_preserves_zero_valid_row_result(self, dataset_context):
        context, reference = dataset_context

        analyzer = StubAnalyzer(
            result=(
                {
                    "valid_row_count": 0,
                    "result": None,
                },
            )
        )
        service = StatisticalAnalysisService(
            derived_metric_analyzer=analyzer,
        )

        result = service.analyze_derived_metric(
            context,
            reference.dataset_id,
            left_column="age",
            right_column="salary",
        )

        assert result.row_count == 4
        assert result.valid_row_count == 0
        assert result.excluded_row_count == 4
        assert result.records[0]["result"] is None

    def test_wraps_analyzer_execution_error(self, dataset_context):
        context, reference = dataset_context

        analyzer = StubAnalyzer(
            error=DerivedMetricAnalysisError(
                "simulated derived metric failure"
            )
        )
        service = StatisticalAnalysisService(
            derived_metric_analyzer=analyzer,
        )

        with pytest.raises(
            StatisticalExecutionError,
            match="derived_metric analysis failed",
        ):
            service.analyze_derived_metric(
                context,
                reference.dataset_id,
                left_column="age",
                right_column="salary",
            )

    def test_wraps_invalid_result_contract(self, dataset_context):
        context, reference = dataset_context

        analyzer = StubAnalyzer(
            result=(
                {
                    "valid_row_count": 4,
                    # Deliberately missing "result".
                },
            )
        )
        service = StatisticalAnalysisService(
            derived_metric_analyzer=analyzer,
        )

        with pytest.raises(
            StatisticalExecutionError,
            match="Failed to assemble derived_metric analysis result",
        ):
            service.analyze_derived_metric(
                context,
                reference.dataset_id,
                left_column="age",
                right_column="salary",
            )


# ---------------------------------------------------------------------------
# Distribution analysis
# ---------------------------------------------------------------------------


class TestDistributionAnalysis:
    def test_assembles_distribution_result(self, dataset_context):
        context, reference = dataset_context

        analysis = DistributionAnalysisData(
            count=4,
            mean=7250.0,
            median=7000.0,
            std=2217.355782608345,
            skewness=0.0,
            kurtosis=-1.0,
            quantiles={
                "0.25": 5750.0,
                "0.5": 7000.0,
                "0.75": 8500.0,
            },
            histogram=(
                HistogramBin(
                    lower_bound=5000.0,
                    upper_bound=10000.0,
                    count=4,
                    percentage=100.0,
                ),
            ),
        )

        analyzer = StubAnalyzer(result=analysis)
        service = StatisticalAnalysisService(
            distribution_analyzer=analyzer,
        )

        result = service.analyze_distribution(
            context,
            reference.dataset_id,
            column="salary",
            bins=1,
            quantiles=(0.25, 0.5, 0.75),
        )

        assert isinstance(result, DistributionAnalysisResult)
        assert result.dataset_id == reference.dataset_id
        assert result.dataset_name == reference.name
        assert result.column == "salary"
        assert result.count == 4
        assert result.mean == 7250.0
        assert result.median == 7000.0
        assert result.std == 2217.355782608345
        assert result.skewness == 0.0
        assert result.kurtosis == -1.0
        assert dict(result.quantiles) == {
            "0.25": 5750.0,
            "0.5": 7000.0,
            "0.75": 8500.0,
        }
        assert result.histogram == analysis.histogram

        assert analyzer.calls[0][1] == {
            "column": "salary",
            "bins": 1,
            "quantiles": (0.25, 0.5, 0.75),
        }


# ---------------------------------------------------------------------------
# Dataset access and validation
# ---------------------------------------------------------------------------


class TestDatasetAccess:
    def test_rejects_invalid_context_type(self):
        service = StatisticalAnalysisService()

        with pytest.raises(
            StatisticalValidationError,
            match="context must be an ExecutionContext",
        ):
            service.analyze_descriptive(
                object(),
                "dataset-id",
            )

    @pytest.mark.parametrize(
        "dataset_id",
        [
            "",
            " ",
            "\t",
            None,
            123,
        ],
    )
    def test_rejects_invalid_dataset_id(self, dataset_id):
        service = StatisticalAnalysisService()
        context = ExecutionContext()

        with pytest.raises(
            StatisticalValidationError,
            match="dataset_id must be a non-empty string",
        ):
            service.analyze_descriptive(
                context,
                dataset_id,
            )

        context.close()

    def test_missing_dataset_becomes_execution_error(self):
        context = ExecutionContext()
        service = StatisticalAnalysisService()

        with pytest.raises(
            StatisticalExecutionError,
            match="Unable to access dataset",
        ) as exc_info:
            service.analyze_descriptive(
                context,
                "missing-dataset",
            )

        assert isinstance(exc_info.value.__cause__, KeyError)

        context.close()

    def test_closed_context_becomes_execution_error(self, dataset_context):
        context, reference = dataset_context
        context.close()

        service = StatisticalAnalysisService()

        with pytest.raises(StatisticalExecutionError):
            service.analyze_descriptive(
                context,
                reference.dataset_id,
            )

    def test_rejects_non_dataframe_dataset(self):
        context, reference = make_context()
        dataframe = context.get_dataset(reference.dataset_id)

        invalid_context = InvalidDataContext()
        invalid_context.register_dataset(reference, dataframe)

        with pytest.raises(
            StatisticalValidationError,
            match="dataset must be a pandas DataFrame",
        ):
            StatisticalAnalysisService().analyze_descriptive(
                invalid_context,
                reference.dataset_id,
            )

        context.close()
        invalid_context.close()

    def test_rejects_invalid_dataset_reference(self):
        context, reference = make_context()
        dataframe = context.get_dataset(reference.dataset_id)

        invalid_context = InvalidReferenceContext()
        invalid_context.register_dataset(reference, dataframe)

        with pytest.raises(
            StatisticalExecutionError,
            match="dataset reference has an invalid type",
        ):
            StatisticalAnalysisService().analyze_descriptive(
                invalid_context,
                reference.dataset_id,
            )

        context.close()
        invalid_context.close()

    def test_wraps_unexpected_context_access_error(self):
        context, reference = make_context()

        failing_context = FailingDataContext()

        with pytest.raises(
            StatisticalExecutionError,
            match="Unexpected error accessing dataset",
        ) as exc_info:
            StatisticalAnalysisService().analyze_descriptive(
                failing_context,
                reference.dataset_id,
            )

        assert isinstance(exc_info.value.__cause__, ValueError)

        context.close()
        failing_context.close()

    def test_rejects_mismatched_dataset_reference(self):
        context = ReferenceOverrideContext(
            reference=DatasetReference(
                dataset_id="different-dataset-id",
                name="different-dataset",
            )
        )

        reference = DatasetReference(
            dataset_id="requested-dataset-id",
            name="requested-dataset",
        )

        dataframe = pd.DataFrame({"value": [1, 2, 3]})
        context.register_dataset(reference, dataframe)

        service = StatisticalAnalysisService()

        with pytest.raises(
            StatisticalExecutionError,
            match="does not match the requested dataset",
        ):
            service.analyze_descriptive(
                context,
                reference.dataset_id,
            )

        context.close()

    def test_wraps_unexpected_reference_access_error(self):
        context = ReferenceOverrideContext(
            error=ValueError("reference storage failure"),
        )

        reference = DatasetReference(
            name="test-dataset",
        )

        dataframe = pd.DataFrame({"value": [1, 2, 3]})
        context.register_dataset(reference, dataframe)

        service = StatisticalAnalysisService()

        with pytest.raises(
            StatisticalExecutionError,
            match="Unexpected error accessing dataset",
        ) as exc_info:
            service.analyze_descriptive(
                context,
                reference.dataset_id,
            )

        assert isinstance(exc_info.value.__cause__, ValueError)
        assert str(exc_info.value.__cause__) == "reference storage failure"

        context.close()


# ---------------------------------------------------------------------------
# Analyzer exception handling
# ---------------------------------------------------------------------------


class TestAnalyzerErrorHandling:
    def test_preserves_statistical_validation_error(self, dataset_context):
        context, reference = dataset_context

        error = StatisticalValidationError("invalid columns")
        analyzer = StubAnalyzer(error=error)

        service = StatisticalAnalysisService(
            descriptive_analyzer=analyzer,
        )

        with pytest.raises(StatisticalValidationError) as exc_info:
            service.analyze_descriptive(
                context,
                reference.dataset_id,
            )

        assert exc_info.value is error

    @pytest.mark.parametrize(
        "error",
        [
            DescriptiveAnalysisError("descriptive failure"),
            FrequencyAnalysisError("frequency failure"),
            CorrelationAnalysisError("correlation failure"),
            GroupedAnalysisError("grouped failure"),
            DistributionAnalysisError("distribution failure"),
        ],
    )
    def test_wraps_analyzer_specific_errors(
        self,
        dataset_context,
        error,
    ):
        context, reference = dataset_context

        analyzer = StubAnalyzer(error=error)
        service = StatisticalAnalysisService(
            descriptive_analyzer=analyzer,
        )

        with pytest.raises(StatisticalExecutionError) as exc_info:
            service.analyze_descriptive(
                context,
                reference.dataset_id,
            )

        assert exc_info.value.__cause__ is error
        assert "descriptive analysis failed" in str(exc_info.value)

    def test_wraps_unexpected_analyzer_error(self, dataset_context):
        context, reference = dataset_context

        error = RuntimeError("unexpected analyzer failure")
        analyzer = StubAnalyzer(error=error)

        service = StatisticalAnalysisService(
            descriptive_analyzer=analyzer,
        )

        with pytest.raises(
            StatisticalExecutionError,
            match="Unexpected error during descriptive analysis",
        ) as exc_info:
            service.analyze_descriptive(
                context,
                reference.dataset_id,
            )

        assert exc_info.value.__cause__ is error


# ---------------------------------------------------------------------------
# Result assembly and lifecycle
# ---------------------------------------------------------------------------


class TestResultAssembly:
    def test_wraps_result_contract_construction_error(self, dataset_context):
        context, reference = dataset_context

        analyzer = StubAnalyzer(result=())
        service = StatisticalAnalysisService(
            descriptive_analyzer=analyzer,
        )

        with patch(
            "lite_cog_agent.statistics.service.DescriptiveStatisticsResult",
            side_effect=ValueError("invalid result"),
        ), pytest.raises(
            StatisticalExecutionError,
            match="Failed to assemble descriptive analysis result",
        ) as exc_info:
            service.analyze_descriptive(
                context,
                reference.dataset_id,
            )

        assert isinstance(exc_info.value.__cause__, ValueError)


class TestServiceLifecycle:
    def test_does_not_close_execution_context(self):
        dataframe = pd.DataFrame(
            {
                "age": [20, 30, 40],
            }
        )

        context, reference = make_context(dataframe)

        analyzer = StubAnalyzer(result=())
        service = StatisticalAnalysisService(
            descriptive_analyzer=analyzer,
        )

        with patch.object(
            context,
            "close",
            wraps=context.close,
        ) as close_mock:
            service.analyze_descriptive(
                context,
                reference.dataset_id,
            )

            close_mock.assert_not_called()

        context.close()

    def test_does_not_modify_original_dataframe(self):
        dataframe = pd.DataFrame(
            {
                "age": [20, 30, 40],
                "salary": [3000.0, 5000.0, 7000.0],
            }
        )

        original = dataframe.copy(deep=True)
        context, reference = make_context(dataframe)

        analyzer = StubAnalyzer(result=())
        service = StatisticalAnalysisService(
            descriptive_analyzer=analyzer,
        )

        service.analyze_descriptive(
            context,
            reference.dataset_id,
        )

        pd.testing.assert_frame_equal(dataframe, original)

        context.close()

class ReferenceOverrideContext(ExecutionContext):
    """ExecutionContext with a controlled reference lookup."""

    def __init__(
        self,
        *,
        reference=None,
        error: Exception | None = None,
    ):
        super().__init__()
        self._reference_override = reference
        self._reference_error = error

    def get_dataset_reference(self, dataset_id: str):
        if self._reference_error is not None:
            raise self._reference_error

        return self._reference_override

# ---------------------------------------------------------------------------
# Analyzer result processing robustness
# ---------------------------------------------------------------------------


class TestAnalyzerResultProcessing:
    def test_wraps_frequency_total_count_error(self, dataset_context):
        context, reference = dataset_context

        analyzer = StubAnalyzer(result=())
        service = StatisticalAnalysisService(
            frequency_analyzer=analyzer,
        )

        with pytest.raises(
            StatisticalExecutionError,
        ) as exc_info:
            service.analyze_frequency(
                context,
                reference.dataset_id,
                column="nonexistent_column",
            )

        assert isinstance(exc_info.value.__cause__, KeyError)

    def test_wraps_correlation_result_unpacking_error(
        self,
        dataset_context,
    ):
        context, reference = dataset_context

        # Correlation Analyzer should return:
        # (matrix, pair_counts)
        # Deliberately return an invalid structure.
        analyzer = StubAnalyzer(result=((),))

        service = StatisticalAnalysisService(
            correlation_analyzer=analyzer,
        )

        with pytest.raises(
            StatisticalExecutionError,
        ) as exc_info:
            service.analyze_correlation(
                context,
                reference.dataset_id,
            )

        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_wraps_grouped_aggregation_normalization_error(
        self,
        dataset_context,
    ):
        context, reference = dataset_context

        analyzer = StubAnalyzer(result=())

        service = StatisticalAnalysisService(
            grouped_analyzer=analyzer,
        )

        with pytest.raises(
            StatisticalExecutionError,
        ) as exc_info:
            service.analyze_grouped(
                context,
                reference.dataset_id,
                group_by=["department"],
                aggregations={
                    "salary": None,
                },
            )

        assert isinstance(exc_info.value.__cause__, TypeError)

    def test_wraps_distribution_result_access_error(
        self,
        dataset_context,
    ):
        context, reference = dataset_context

        analyzer = StubAnalyzer(result=None)

        service = StatisticalAnalysisService(
            distribution_analyzer=analyzer,
        )

        with pytest.raises(
            StatisticalExecutionError,
        ) as exc_info:
            service.analyze_distribution(
                context,
                reference.dataset_id,
                column="salary",
            )

        assert isinstance(exc_info.value.__cause__, AttributeError)