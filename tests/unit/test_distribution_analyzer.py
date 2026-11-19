"""Unit tests for DistributionAnalyzer."""

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from lite_cog_agent.statistics.contracts import HistogramBin
from lite_cog_agent.statistics.distribution import (
    DistributionAnalysisData,
    DistributionAnalyzer,
)
from lite_cog_agent.statistics.exceptions import (
    DistributionAnalysisError,
    StatisticalValidationError,
)


@pytest.fixture
def analyzer() -> DistributionAnalyzer:
    return DistributionAnalyzer()


def test_basic_distribution_statistics(
    analyzer: DistributionAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {"value": [10, 20, 30, 40, 50]}
    )

    result = analyzer.analyze(
        dataframe,
        column="value",
        bins=5,
    )

    assert result.count == 5
    assert result.mean == 30.0
    assert result.median == 30.0
    assert result.std == pytest.approx(math.sqrt(250))
    assert result.skewness == pytest.approx(0.0, abs=1e-12)
    assert result.kurtosis == pytest.approx(-1.2)


def test_default_quantiles(
    analyzer: DistributionAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {"value": [10, 20, 30, 40, 50]}
    )

    result = analyzer.analyze(
        dataframe,
        column="value",
    )

    assert dict(result.quantiles) == {
        "0.25": 20.0,
        "0.5": 30.0,
        "0.75": 40.0,
    }


def test_custom_quantiles(
    analyzer: DistributionAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {"value": [10, 20, 30, 40, 50]}
    )

    result = analyzer.analyze(
        dataframe,
        column="value",
        quantiles=(0.1, 0.9),
    )

    assert result.quantiles["0.1"] == pytest.approx(14.0)
    assert result.quantiles["0.9"] == pytest.approx(46.0)

    # Median is independent of the requested quantiles.
    assert result.median == 30.0


@pytest.mark.parametrize(
    ("quantiles", "expected"),
    [
        ((0.0,), 10.0),
        ((1.0,), 50.0),
    ],
)
def test_quantile_boundary_probabilities(
    analyzer: DistributionAnalyzer,
    quantiles: tuple[float, ...],
    expected: float,
) -> None:
    dataframe = pd.DataFrame(
        {"value": [10, 20, 30, 40, 50]}
    )

    result = analyzer.analyze(
        dataframe,
        column="value",
        quantiles=quantiles,
    )

    assert next(iter(result.quantiles.values())) == expected


def test_histogram_bin_count_and_percentages(
    analyzer: DistributionAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {"value": [10, 20, 30, 40, 50]}
    )

    result = analyzer.analyze(
        dataframe,
        column="value",
        bins=5,
    )

    assert len(result.histogram) == 5
    assert sum(item.count for item in result.histogram) == 5

    assert sum(
        item.percentage for item in result.histogram
    ) == pytest.approx(100.0)

    assert all(
        isinstance(item, HistogramBin)
        for item in result.histogram
    )


def test_histogram_boundaries_cover_min_and_max(
    analyzer: DistributionAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {"value": [10, 20, 30, 40, 50]}
    )

    result = analyzer.analyze(
        dataframe,
        column="value",
        bins=5,
    )

    assert result.histogram[0].lower_bound == pytest.approx(10.0)
    assert result.histogram[-1].upper_bound == pytest.approx(50.0)


def test_last_histogram_bin_includes_maximum_value(
    analyzer: DistributionAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {"value": [0, 1, 2, 3, 4, 5]}
    )

    result = analyzer.analyze(
        dataframe,
        column="value",
        bins=5,
    )

    assert result.histogram[-1].count >= 1
    assert sum(item.count for item in result.histogram) == 6


def test_histogram_with_one_bin(
    analyzer: DistributionAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {"value": [1, 2, 3, 4, 5]}
    )

    result = analyzer.analyze(
        dataframe,
        column="value",
        bins=1,
    )

    assert len(result.histogram) == 1
    assert result.histogram[0].count == 5
    assert result.histogram[0].percentage == 100.0


def test_nan_and_infinity_are_excluded(
    analyzer: DistributionAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "value": [
                1.0,
                2.0,
                float("nan"),
                float("inf"),
                float("-inf"),
            ]
        }
    )

    result = analyzer.analyze(
        dataframe,
        column="value",
    )

    assert result.count == 2
    assert result.mean == 1.5
    assert result.median == 1.5
    assert sum(item.count for item in result.histogram) == 2


def test_all_invalid_values_return_empty_result(
    analyzer: DistributionAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "value": [
                float("nan"),
                float("inf"),
                float("-inf"),
            ]
        }
    )

    result = analyzer.analyze(
        dataframe,
        column="value",
    )

    assert result.count == 0
    assert result.mean is None
    assert result.median is None
    assert result.std is None
    assert result.skewness is None
    assert result.kurtosis is None
    assert all(
        value is None
        for value in result.quantiles.values()
    )
    assert result.histogram == ()


def test_single_observation(
    analyzer: DistributionAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {"value": [42.0]}
    )

    result = analyzer.analyze(
        dataframe,
        column="value",
    )

    assert result.count == 1
    assert result.mean == 42.0
    assert result.median == 42.0
    assert result.std is None
    assert result.skewness is None
    assert result.kurtosis is None
    assert sum(item.count for item in result.histogram) == 1


def test_constant_values(
    analyzer: DistributionAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {"value": [5.0, 5.0, 5.0, 5.0]}
    )

    result = analyzer.analyze(
        dataframe,
        column="value",
        bins=4,
    )

    assert result.count == 4
    assert result.mean == 5.0
    assert result.median == 5.0
    assert result.std == 0.0
    assert result.skewness is None
    assert result.kurtosis is None
    assert sum(item.count for item in result.histogram) == 4


def test_original_dataframe_is_not_modified(
    analyzer: DistributionAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {"value": [1.0, 2.0, float("nan"), float("inf")]}
    )

    original = dataframe.copy(deep=True)

    analyzer.analyze(
        dataframe,
        column="value",
    )

    pd.testing.assert_frame_equal(dataframe, original)


def test_result_is_immutable(
    analyzer: DistributionAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {"value": [1, 2, 3]}
    )

    result = analyzer.analyze(
        dataframe,
        column="value",
    )

    with pytest.raises(TypeError):
        result.quantiles["0.5"] = 999.0

    with pytest.raises(FrozenInstanceError):
        result.count = 100

@pytest.mark.parametrize(
    "dataframe",
    [
        None,
        [],
        {"value": [1, 2, 3]},
    ],
)
def test_rejects_non_dataframe_input(
    analyzer: DistributionAnalyzer,
    dataframe,
) -> None:
    with pytest.raises(StatisticalValidationError):
        analyzer.analyze(dataframe, column="value")


@pytest.mark.parametrize(
    "bins",
    [
        0,
        -1,
        True,
        1.5,
        "2",
        None,
    ],
)
def test_rejects_invalid_bins(
    analyzer: DistributionAnalyzer,
    bins,
) -> None:
    dataframe = pd.DataFrame({"value": [1, 2, 3]})

    with pytest.raises(StatisticalValidationError):
        analyzer.analyze(dataframe, column="value", bins=bins)


@pytest.mark.parametrize(
    "quantiles",
    [
        [0.25, 0.5, 0.75],
        (0.25, 0.5, 1.1),
        (-0.1, 0.5),
        (float("nan"),),
        (float("inf"),),
        (True,),
        (0.5, 0.5),
    ],
)
def test_rejects_invalid_quantiles(
    analyzer: DistributionAnalyzer,
    quantiles,
) -> None:
    dataframe = pd.DataFrame({"value": [1, 2, 3]})

    with pytest.raises(StatisticalValidationError):
        analyzer.analyze(
            dataframe,
            column="value",
            quantiles=quantiles,
        )


def test_rejects_missing_column(
    analyzer: DistributionAnalyzer,
) -> None:
    dataframe = pd.DataFrame({"value": [1, 2, 3]})

    with pytest.raises(StatisticalValidationError):
        analyzer.analyze(dataframe, column="missing")


def test_rejects_empty_column_name(
    analyzer: DistributionAnalyzer,
) -> None:
    dataframe = pd.DataFrame({"value": [1, 2, 3]})

    with pytest.raises(StatisticalValidationError):
        analyzer.analyze(dataframe, column="")


def test_rejects_duplicate_dataframe_columns(
    analyzer: DistributionAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        [[1, 10], [2, 20]],
        columns=["value", "value"],
    )

    with pytest.raises(StatisticalValidationError):
        analyzer.analyze(dataframe, column="value")

@pytest.mark.parametrize(
    "values",
    [
        ["a", "b", "c"],
        [True, False, True],
        [1 + 2j, 3 + 4j, 5 + 6j],
    ],
)
def test_rejects_unsupported_column_types(
    analyzer: DistributionAnalyzer,
    values,
) -> None:
    dataframe = pd.DataFrame({"value": values})

    with pytest.raises(StatisticalValidationError):
        analyzer.analyze(dataframe, column="value")

def test_supports_nullable_integer_dtype(
    analyzer: DistributionAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "value": pd.Series(
                [10, 20, pd.NA, 40],
                dtype="Int64",
            )
        }
    )

    result = analyzer.analyze(dataframe, column="value")

    assert result.count == 3
    assert result.mean == pytest.approx(70 / 3)
    assert result.median == pytest.approx(20)


def test_supports_nullable_float_dtype(
    analyzer: DistributionAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "value": pd.Series(
                [1.5, 2.5, pd.NA, 4.5],
                dtype="Float64",
            )
        }
    )

    result = analyzer.analyze(dataframe, column="value")

    assert result.count == 3
    assert result.mean == pytest.approx(8.5 / 3)
    assert result.median == pytest.approx(2.5)


def test_empty_quantiles_are_allowed(
    analyzer: DistributionAnalyzer,
) -> None:
    dataframe = pd.DataFrame({"value": [10, 20, 30]})

    result = analyzer.analyze(
        dataframe,
        column="value",
        quantiles=(),
    )

    assert result.count == 3
    assert result.quantiles == {}
    assert result.median == pytest.approx(20)

def test_handles_large_values(
    analyzer: DistributionAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {"value": [1e200, 2e200, 3e200]}
    )

    result = analyzer.analyze(dataframe, column="value")

    assert result.count == 3
    assert result.mean == pytest.approx(2e200)
    assert result.median == pytest.approx(2e200)
    assert result.std is not None
    assert math.isfinite(result.std)
    assert result.std == pytest.approx(1e200)


def test_handles_large_positive_and_negative_values(
    analyzer: DistributionAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {"value": [-1e308, 1e308]}
    )

    result = analyzer.analyze(dataframe, column="value")

    assert result.count == 2
    assert result.mean == pytest.approx(0.0)
    assert result.median == pytest.approx(0.0)
    assert result.std is not None
    assert math.isfinite(result.std)
    assert result.std == pytest.approx(math.sqrt(2) * 1e308)


def test_handles_large_values_near_float_limit(
    analyzer: DistributionAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {"value": [1e308, 1.7e308]}
    )

    result = analyzer.analyze(dataframe, column="value")

    assert result.count == 2
    assert result.mean is not None
    assert math.isfinite(result.mean)
    assert result.std is not None
    assert math.isfinite(result.std)


def test_handles_all_zero_values(
    analyzer: DistributionAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {"value": [0.0, 0.0, 0.0, 0.0]}
    )

    result = analyzer.analyze(dataframe, column="value")

    assert result.count == 4
    assert result.mean == 0.0
    assert result.median == 0.0
    assert result.std == 0.0
    assert sum(item.count for item in result.histogram) == 4

    # 每个分箱的边界应合法，不能出现反向区间。
    assert all(
        item.lower_bound <= item.upper_bound
        for item in result.histogram
    )

def test_supports_duplicate_dataframe_index(
    analyzer: DistributionAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {"value": [10, 20, 30, 40]},
        index=[0, 0, 1, 1],
    )

    result = analyzer.analyze(dataframe, column="value")

    assert result.count == 4
    assert result.mean == pytest.approx(25.0)


def test_supports_large_integer_values(
    analyzer: DistributionAnalyzer,
) -> None:
    dataframe = pd.DataFrame(
        {
            "value": [
                2**60,
                2**60 + 2,
                2**60 + 4,
            ]
        }
    )

    result = analyzer.analyze(dataframe, column="value")

    assert result.count == 3
    assert result.mean is not None
    assert math.isfinite(result.mean)

def test_wraps_histogram_runtime_error(
    analyzer: DistributionAnalyzer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataframe = pd.DataFrame({"value": [1, 2, 3]})

    def raise_runtime_error(*args, **kwargs):
        raise RuntimeError("simulated histogram failure")

    monkeypatch.setattr(np, "histogram", raise_runtime_error)

    with pytest.raises(DistributionAnalysisError) as exc_info:
        analyzer.analyze(dataframe, column="value")

    assert "simulated histogram failure" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_wraps_quantile_runtime_error(
    analyzer: DistributionAnalyzer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataframe = pd.DataFrame({"value": [1, 2, 3]})

    def raise_runtime_error(*args, **kwargs):
        raise RuntimeError("simulated quantile failure")

    monkeypatch.setattr(np, "quantile", raise_runtime_error)

    with pytest.raises(DistributionAnalysisError) as exc_info:
        analyzer.analyze(dataframe, column="value")

    assert "simulated quantile failure" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)

def make_distribution_data(
    *,
    count: int = 3,
    mean: float | None = 2.0,
    median: float | None = 2.0,
    std: float | None = 1.0,
    skewness: float | None = 0.0,
    kurtosis: float | None = 0.0,
    quantiles: dict[str, float | None] | None = None,
    histogram: tuple[HistogramBin, ...] | None = None,
) -> DistributionAnalysisData:
    if quantiles is None:
        quantiles = {
            "0.25": 1.5,
            "0.5": 2.0,
            "0.75": 2.5,
        }

    if histogram is None:
        histogram = (
            HistogramBin(
                lower_bound=1.0,
                upper_bound=2.0,
                count=1,
                percentage=100 / 3,
            ),
            HistogramBin(
                lower_bound=2.0,
                upper_bound=3.0,
                count=2,
                percentage=200 / 3,
            ),
        )

    return DistributionAnalysisData(
        count=count,
        mean=mean,
        median=median,
        std=std,
        skewness=skewness,
        kurtosis=kurtosis,
        quantiles=quantiles,
        histogram=histogram,
    )

def test_distribution_data_accepts_valid_values() -> None:
    result = make_distribution_data()

    assert result.count == 3
    assert result.mean == 2.0
    assert result.median == 2.0
    assert result.std == 1.0
    assert result.quantiles["0.5"] == 2.0
    assert len(result.histogram) == 2

@pytest.mark.parametrize(
    "field,value",
    [
        ("mean", float("nan")),
        ("mean", float("inf")),
        ("median", float("-inf")),
        ("std", float("nan")),
        ("skewness", float("inf")),
        ("kurtosis", float("-inf")),
    ],
)
def test_rejects_non_finite_statistics(
    field: str,
    value: float,
) -> None:
    with pytest.raises(StatisticalValidationError):
        make_distribution_data(**{field: value})


@pytest.mark.parametrize(
    "count",
    [-1, True, 1.5],
)
def test_rejects_invalid_count(count) -> None:
    with pytest.raises(StatisticalValidationError):
        make_distribution_data(count=count)


def test_rejects_negative_std() -> None:
    with pytest.raises(StatisticalValidationError):
        make_distribution_data(std=-1.0)


def test_accepts_none_for_unavailable_statistics() -> None:
    result = make_distribution_data(
        mean=None,
        median=None,
        std=None,
        skewness=None,
        kurtosis=None,
    )

    assert result.mean is None
    assert result.median is None
    assert result.std is None

def test_quantiles_are_read_only() -> None:
    source = {
        "0.25": 1.5,
        "0.5": 2.0,
        "0.75": 2.5,
    }

    result = make_distribution_data(quantiles=source)

    source["0.5"] = 999.0

    assert result.quantiles["0.5"] == 2.0

    with pytest.raises(TypeError):
        result.quantiles["0.5"] = 999.0


@pytest.mark.parametrize(
    "quantiles",
    [
        {"": 1.0},
        {"0.5": float("nan")},
        {"0.5": float("inf")},
        {"0.5": True},
    ],
)
def test_rejects_invalid_quantiles_mapping(quantiles) -> None:
    with pytest.raises(StatisticalValidationError):
        make_distribution_data(quantiles=quantiles)

def test_histogram_counts_must_equal_total_count() -> None:
    histogram = (
        HistogramBin(
            lower_bound=1.0,
            upper_bound=2.0,
            count=1,
            percentage=100.0,
        ),
    )

    with pytest.raises(StatisticalValidationError):
        make_distribution_data(
            count=3,
            histogram=histogram,
        )


def test_histogram_percentages_must_sum_to_100() -> None:
    histogram = (
        HistogramBin(
            lower_bound=1.0,
            upper_bound=2.0,
            count=1,
            percentage=20.0,
        ),
        HistogramBin(
            lower_bound=2.0,
            upper_bound=3.0,
            count=2,
            percentage=70.0,
        ),
    )

    with pytest.raises(StatisticalValidationError):
        make_distribution_data(histogram=histogram)


def test_histogram_must_contain_histogram_bins() -> None:
    with pytest.raises(StatisticalValidationError):
        make_distribution_data(
            histogram=(object(),)
        )

def test_empty_distribution_data_contract() -> None:
    result = make_distribution_data(
        count=0,
        mean=None,
        median=None,
        std=None,
        skewness=None,
        kurtosis=None,
        quantiles={
            "0.25": None,
            "0.5": None,
            "0.75": None,
        },
        histogram=(),
    )

    assert result.count == 0
    assert result.mean is None
    assert result.median is None
    assert result.std is None
    assert result.histogram == ()


def test_empty_distribution_rejects_non_empty_histogram() -> None:
    histogram = (
        HistogramBin(
            lower_bound=0.0,
            upper_bound=1.0,
            count=1,
            percentage=100.0,
        ),
    )

    with pytest.raises(StatisticalValidationError):
        make_distribution_data(
            count=0,
            mean=None,
            median=None,
            std=None,
            skewness=None,
            kurtosis=None,
            quantiles={"0.5": None},
            histogram=histogram,
        )

def test_histogram_percentages_must_match_bin_counts() -> None:
    histogram = (
        HistogramBin(
            lower_bound=1.0,
            upper_bound=2.0,
            count=1,
            percentage=20.0,
        ),
        HistogramBin(
            lower_bound=2.0,
            upper_bound=3.0,
            count=2,
            percentage=80.0,
        ),
    )

    with pytest.raises(StatisticalValidationError):
        make_distribution_data(histogram=histogram)
