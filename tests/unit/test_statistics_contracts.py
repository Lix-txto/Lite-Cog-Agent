"""Tests for statistical analysis contracts."""

from dataclasses import FrozenInstanceError

import pytest

from lite_cog_agent.statistics.contracts import (
    CorrelationAnalysisResult,
    DescriptiveStatistics,
    DescriptiveStatisticsResult,
    DistributionAnalysisResult,
    FrequencyAnalysisResult,
    FrequencyItem,
    GroupedAnalysisResult,
    HistogramBin,
)
from lite_cog_agent.statistics.exceptions import (
    StatisticalValidationError,
)


def make_descriptive(**overrides):
    values = {
        "column": "age",
        "count": 4,
        "mean": 30.0,
        "median": 30.0,
        "std": 10.0,
        "variance": 100.0,
        "min": 20.0,
        "q1": 22.5,
        "q3": 37.5,
        "max": 40.0,
        "range": 20.0,
    }
    values.update(overrides)
    return DescriptiveStatistics(**values)


def make_correlation(**overrides):
    values = {
        "dataset_id": "dataset-1",
        "dataset_name": "employees",
        "method": "pearson",
        "columns": ("age", "salary"),
        "matrix": (
            (1.0, 0.8),
            (0.8, 1.0),
        ),
        "pair_counts": (
            (10, 9),
            (9, 10),
        ),
    }
    values.update(overrides)
    return CorrelationAnalysisResult(**values)


def test_descriptive_statistics_creation():
    result = make_descriptive()

    assert result.column == "age"
    assert result.count == 4
    assert result.mean == 30.0
    assert result.range == 20.0


def test_descriptive_statistics_allows_none_values_for_empty_column():
    result = make_descriptive(
        count=0,
        mean=None,
        median=None,
        std=None,
        variance=None,
        min=None,
        q1=None,
        q3=None,
        max=None,
        range=None,
    )

    assert result.count == 0
    assert result.mean is None
    assert result.std is None


def test_descriptive_statistics_single_value_has_no_sample_std():
    result = make_descriptive(
        count=1,
        mean=30.0,
        median=30.0,
        std=None,
        variance=None,
        min=30.0,
        q1=30.0,
        q3=30.0,
        max=30.0,
        range=0.0,
    )

    assert result.std is None
    assert result.variance is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"column": ""},
        {"count": -1},
        {"count": True},
        {"mean": float("nan")},
        {"median": float("inf")},
        {"std": -1.0},
        {"variance": -1.0},
        {"range": -1.0},
        {"min": 50.0, "max": 40.0},
        {"q1": 40.0, "q3": 30.0},
        {"count": 0},
        {"count": 1, "std": 1.0},
    ],
)
def test_descriptive_statistics_rejects_invalid_values(overrides):
    with pytest.raises(StatisticalValidationError):
        make_descriptive(**overrides)


def test_descriptive_statistics_is_immutable():
    result = make_descriptive()

    with pytest.raises(FrozenInstanceError):
        result.mean = 100.0


def test_descriptive_result_creation():
    result = DescriptiveStatisticsResult(
        dataset_id="dataset-1",
        dataset_name="employees",
        row_count=4,
        statistics=(make_descriptive(),),
    )

    assert result.dataset_id == "dataset-1"
    assert result.row_count == 4
    assert len(result.statistics) == 1


def test_descriptive_result_rejects_duplicate_columns():
    item = make_descriptive()

    with pytest.raises(StatisticalValidationError):
        DescriptiveStatisticsResult(
            dataset_id="dataset-1",
            dataset_name="employees",
            row_count=4,
            statistics=(item, item),
        )


def test_descriptive_result_rejects_count_greater_than_rows():
    with pytest.raises(StatisticalValidationError):
        DescriptiveStatisticsResult(
            dataset_id="dataset-1",
            dataset_name="employees",
            row_count=2,
            statistics=(make_descriptive(),),
        )


def test_frequency_item_creation():
    item = FrequencyItem(
        value="Engineering",
        count=4,
        percentage=40.0,
        cumulative_percentage=40.0,
    )

    assert item.value == "Engineering"
    assert item.count == 4


@pytest.mark.parametrize(
    "overrides",
    [
        {"count": -1},
        {"count": True},
        {"percentage": -1.0},
        {"percentage": 101.0},
        {"percentage": float("nan")},
        {"cumulative_percentage": 20.0},
    ],
)
def test_frequency_item_rejects_invalid_values(overrides):
    values = {
        "value": "Engineering",
        "count": 4,
        "percentage": 40.0,
        "cumulative_percentage": 40.0,
    }
    values.update(overrides)

    with pytest.raises(StatisticalValidationError):
        FrequencyItem(**values)


def test_frequency_result_creation():
    result = FrequencyAnalysisResult(
        dataset_id="dataset-1",
        dataset_name="employees",
        column="department",
        total_count=10,
        items=(
            FrequencyItem("Engineering", 4, 40.0, 40.0),
            FrequencyItem("Sales", 3, 30.0, 70.0),
        ),
    )

    assert result.total_count == 10
    assert len(result.items) == 2


def test_frequency_result_allows_empty_items():
    result = FrequencyAnalysisResult(
        dataset_id="dataset-1",
        dataset_name="employees",
        column="department",
        total_count=0,
        items=(),
    )

    assert result.items == ()


def test_frequency_result_rejects_count_exceeding_total():
    with pytest.raises(StatisticalValidationError):
        FrequencyAnalysisResult(
            dataset_id="dataset-1",
            dataset_name="employees",
            column="department",
            total_count=3,
            items=(
                FrequencyItem("Engineering", 4, 100.0, 100.0),
            ),
        )


def test_frequency_result_rejects_inconsistent_cumulative_percentage():
    with pytest.raises(StatisticalValidationError):
        FrequencyAnalysisResult(
            dataset_id="dataset-1",
            dataset_name="employees",
            column="department",
            total_count=10,
            items=(
                FrequencyItem("Engineering", 4, 40.0, 40.0),
                FrequencyItem("Sales", 3, 30.0, 80.0),
            ),
        )


def test_correlation_result_creation():
    result = make_correlation()

    assert result.method == "pearson"
    assert result.matrix[0][1] == 0.8
    assert result.pair_counts[0][1] == 9


@pytest.mark.parametrize(
    "overrides",
    [
        {"method": "kendall"},
        {"columns": ("age", "age")},
        {"matrix": ((1.0,),)},
        {"matrix": ((1.0, 1.5), (1.5, 1.0))},
        {"matrix": ((1.0, 0.8), (0.7, 1.0))},
        {"matrix": ((1.0, float("nan")), (float("nan"), 1.0))},
        {"pair_counts": ((10,), (9,))},
        {"pair_counts": ((10, 9), (8, 10))},
        {"pair_counts": ((10, True), (True, 10))},
    ],
)
def test_correlation_result_rejects_invalid_values(overrides):
    with pytest.raises(StatisticalValidationError):
        make_correlation(**overrides)


def test_correlation_result_allows_uncomputable_coefficients():
    result = make_correlation(
        matrix=(
            (1.0, None),
            (None, 1.0),
        ),
    )

    assert result.matrix[0][1] is None


def test_grouped_result_creation():
    result = GroupedAnalysisResult(
        dataset_id="dataset-1",
        dataset_name="employees",
        group_by=("department",),
        aggregations={
            "salary": ("mean", "max"),
        },
        group_count=1,
        records=(
            {
                "department": "Engineering",
                "salary_mean": 12000.0,
                "salary_max": 18000.0,
            },
        ),
    )

    assert result.group_count == 1
    assert result.records[0]["department"] == "Engineering"
    assert result.aggregations["salary"] == ("mean", "max")


def test_grouped_result_mapping_is_read_only():
    result = GroupedAnalysisResult(
        dataset_id="dataset-1",
        dataset_name="employees",
        group_by=("department",),
        aggregations={"salary": ("mean",)},
        group_count=1,
        records=({"department": "Engineering"},),
    )

    with pytest.raises(TypeError):
        result.aggregations["salary"] = ("max",)

    with pytest.raises(TypeError):
        result.records[0]["department"] = "Sales"


def test_grouped_result_rejects_invalid_aggregation():
    with pytest.raises(StatisticalValidationError):
        GroupedAnalysisResult(
            dataset_id="dataset-1",
            dataset_name="employees",
            group_by=("department",),
            aggregations={"salary": []},
            group_count=0,
            records=(),
        )


def test_grouped_result_rejects_mismatched_group_count():
    with pytest.raises(StatisticalValidationError):
        GroupedAnalysisResult(
            dataset_id="dataset-1",
            dataset_name="employees",
            group_by=("department",),
            aggregations={"salary": ("mean",)},
            group_count=2,
            records=(
                {"department": "Engineering"},
            ),
        )


def test_histogram_bin_creation():
    item = HistogramBin(
        lower_bound=0.0,
        upper_bound=10.0,
        count=5,
        percentage=50.0,
    )

    assert item.count == 5
    assert item.upper_bound == 10.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"lower_bound": 10.0, "upper_bound": 0.0},
        {"count": -1},
        {"count": True},
        {"percentage": 101.0},
        {"upper_bound": float("inf")},
    ],
)
def test_histogram_bin_rejects_invalid_values(overrides):
    values = {
        "lower_bound": 0.0,
        "upper_bound": 10.0,
        "count": 5,
        "percentage": 50.0,
    }
    values.update(overrides)

    with pytest.raises(StatisticalValidationError):
        HistogramBin(**values)


def test_distribution_result_creation():
    result = DistributionAnalysisResult(
        dataset_id="dataset-1",
        dataset_name="employees",
        column="salary",
        count=4,
        mean=10000.0,
        median=9500.0,
        std=1000.0,
        skewness=0.2,
        kurtosis=-0.5,
        quantiles={
            "0.25": 9000.0,
            "0.5": 9500.0,
            "0.75": 11000.0,
        },
        histogram=(
            HistogramBin(8000.0, 10000.0, 2, 50.0),
            HistogramBin(10000.0, 12000.0, 2, 50.0),
        ),
    )

    assert result.count == 4
    assert result.quantiles["0.5"] == 9500.0
    assert len(result.histogram) == 2


def test_distribution_result_allows_empty_distribution():
    result = DistributionAnalysisResult(
        dataset_id="dataset-1",
        dataset_name="employees",
        column="salary",
        count=0,
        mean=None,
        median=None,
        std=None,
        skewness=None,
        kurtosis=None,
        quantiles={"0.5": None},
        histogram=(),
    )

    assert result.count == 0
    assert result.mean is None
    assert result.histogram == ()


def test_distribution_quantiles_are_read_only():
    result = DistributionAnalysisResult(
        dataset_id="dataset-1",
        dataset_name="employees",
        column="salary",
        count=1,
        mean=10000.0,
        median=10000.0,
        std=None,
        skewness=None,
        kurtosis=None,
        quantiles={"0.5": 10000.0},
        histogram=(),
    )

    with pytest.raises(TypeError):
        result.quantiles["0.5"] = 20000.0


def test_distribution_rejects_histogram_count_exceeding_count():
    with pytest.raises(StatisticalValidationError):
        DistributionAnalysisResult(
            dataset_id="dataset-1",
            dataset_name="employees",
            column="salary",
            count=2,
            mean=10000.0,
            median=10000.0,
            std=0.0,
            skewness=None,
            kurtosis=None,
            quantiles={},
            histogram=(
                HistogramBin(0.0, 10.0, 3, 100.0),
            ),
        )


def test_distribution_rejects_non_finite_statistics():
    with pytest.raises(StatisticalValidationError):
        DistributionAnalysisResult(
            dataset_id="dataset-1",
            dataset_name="employees",
            column="salary",
            count=2,
            mean=float("inf"),
            median=10000.0,
            std=1.0,
            skewness=None,
            kurtosis=None,
            quantiles={},
            histogram=(),
        )


def test_contracts_are_frozen():
    result = FrequencyItem(
        value="Engineering",
        count=4,
        percentage=40.0,
        cumulative_percentage=40.0,
    )

    with pytest.raises(FrozenInstanceError):
        result.count = 5
