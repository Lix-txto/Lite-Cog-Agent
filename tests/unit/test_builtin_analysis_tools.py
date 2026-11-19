"""Tests for builtin analysis tools."""

from __future__ import annotations

import asyncio
import json

import pandas as pd
import pytest
from jsonschema import Draft202012Validator

from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.execution.dataset_store import DatasetReference
from lite_cog_agent.profiling.service import ProfilingService
from lite_cog_agent.statistics.service import StatisticalAnalysisService
from lite_cog_agent.tools.builtin.analysis_tools import (
    CorrelationAnalysisTool,
    DatasetProfileTool,
    DescriptiveAnalysisTool,
    DistributionAnalysisTool,
    FrequencyAnalysisTool,
    GroupedAnalysisTool,
)


@pytest.fixture
def analysis_context() -> ExecutionContext:
    """Create a context containing a representative sales dataset."""
    context = ExecutionContext()

    dataframe = pd.DataFrame(
        {
            "department": [
                "Sales",
                "Sales",
                "Support",
                "Support",
                "Sales",
                "Sales",
            ],
            "product": ["A", "B", "A", "C", "A", "B"],
            "quantity": [2, 3, 1, 4, 2, None],
            "amount": [100, 150, 80, 200, 100, 120],
        }
    )

    reference = DatasetReference(
        dataset_id="analysis-test-dataset",
        name="sales",
        source="sales.csv",
        file_format="csv",
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        columns=tuple(dataframe.columns),
    )

    context.register_dataset(reference, dataframe)

    yield context

    context.close()


@pytest.fixture
def profiling_service() -> ProfilingService:
    return ProfilingService()


@pytest.fixture
def statistics_service() -> StatisticalAnalysisService:
    return StatisticalAnalysisService()


def run_tool(tool, arguments: dict):
    """Run an async tool in the synchronous pytest test."""
    return asyncio.run(tool.execute(arguments))


def assert_json_compatible(value) -> None:
    """Ensure a result can be serialized by the standard JSON encoder."""
    json.dumps(value, ensure_ascii=False, allow_nan=False)


def test_profile_tool_returns_dataset_profile(
    analysis_context,
    profiling_service,
):
    tool = DatasetProfileTool(
        analysis_context,
        profiling_service,
    )

    result = run_tool(
        tool,
        {"dataset_id": "analysis-test-dataset"},
    )

    assert result.success is True
    assert result.error is None
    assert result.output["dataset_profile"]["row_count"] == 6
    assert result.output["dataset_profile"]["column_count"] == 4
    assert result.output["quality_report"]["missing_cells"] == 1
    assert result.metadata["tool"] == "analysis.profile"

    assert_json_compatible(result.output)


def test_descriptive_tool_returns_numeric_statistics(
    analysis_context,
    statistics_service,
):
    tool = DescriptiveAnalysisTool(
        analysis_context,
        statistics_service,
    )

    result = run_tool(
        tool,
        {
            "dataset_id": "analysis-test-dataset",
            "columns": ["amount"],
        },
    )

    assert result.success is True
    assert result.output["dataset_id"] == "analysis-test-dataset"
    assert result.output["statistics"][0]["column"] == "amount"
    assert result.output["statistics"][0]["count"] == 6
    assert result.output["statistics"][0]["mean"] == pytest.approx(125.0)

    assert_json_compatible(result.output)


def test_descriptive_tool_can_analyze_all_numeric_columns(
    analysis_context,
    statistics_service,
):
    tool = DescriptiveAnalysisTool(
        analysis_context,
        statistics_service,
    )

    result = run_tool(
        tool,
        {"dataset_id": "analysis-test-dataset"},
    )

    assert result.success is True

    columns = {
        item["column"]
        for item in result.output["statistics"]
    }

    assert columns == {"quantity", "amount"}


def test_frequency_tool_returns_category_counts(
    analysis_context,
    statistics_service,
):
    tool = FrequencyAnalysisTool(
        analysis_context,
        statistics_service,
    )

    result = run_tool(
        tool,
        {
            "dataset_id": "analysis-test-dataset",
            "column": "department",
        },
    )

    assert result.success is True
    assert result.output["column"] == "department"
    assert result.output["total_count"] == 6

    items = {
        item["value"]: item["count"]
        for item in result.output["items"]
    }

    assert items == {"Sales": 4, "Support": 2}

    assert_json_compatible(result.output)


def test_frequency_tool_supports_top_n_and_dropna(
    analysis_context,
    statistics_service,
):
    tool = FrequencyAnalysisTool(
        analysis_context,
        statistics_service,
    )

    result = run_tool(
        tool,
        {
            "dataset_id": "analysis-test-dataset",
            "column": "quantity",
            "top_n": 2,
            "dropna": True,
        },
    )

    assert result.success is True
    assert result.output["total_count"] == 5
    assert len(result.output["items"]) == 2


def test_correlation_tool_returns_matrix_and_pair_counts(
    analysis_context,
    statistics_service,
):
    tool = CorrelationAnalysisTool(
        analysis_context,
        statistics_service,
    )

    result = run_tool(
        tool,
        {
            "dataset_id": "analysis-test-dataset",
            "columns": ["quantity", "amount"],
            "method": "pearson",
            "min_periods": 2,
        },
    )

    assert result.success is True
    assert result.output["method"] == "pearson"
    assert result.output["columns"] == ["quantity", "amount"]
    assert len(result.output["matrix"]) == 2
    assert len(result.output["pair_counts"]) == 2
    assert result.output["pair_counts"][0][0] == 5

    assert_json_compatible(result.output)


def test_correlation_tool_supports_spearman(
    analysis_context,
    statistics_service,
):
    tool = CorrelationAnalysisTool(
        analysis_context,
        statistics_service,
    )

    result = run_tool(
        tool,
        {
            "dataset_id": "analysis-test-dataset",
            "method": "spearman",
        },
    )

    assert result.success is True
    assert result.output["method"] == "spearman"


def test_grouped_tool_returns_group_aggregations(
    analysis_context,
    statistics_service,
):
    tool = GroupedAnalysisTool(
        analysis_context,
        statistics_service,
    )

    result = run_tool(
        tool,
        {
            "dataset_id": "analysis-test-dataset",
            "group_by": ["department"],
            "aggregations": {
                "amount": ["sum", "mean"],
                "quantity": ["sum", "count"],
            },
        },
    )

    assert result.success is True
    assert result.output["group_count"] == 2

    records = {
        record["department"]: record
        for record in result.output["records"]
    }

    assert records["Sales"]["amount_sum"] == pytest.approx(470.0)
    assert records["Support"]["amount_sum"] == pytest.approx(280.0)
    assert records["Sales"]["quantity_sum"] == pytest.approx(7.0)
    assert records["Support"]["quantity_sum"] == pytest.approx(5.0)

    assert_json_compatible(result.output)


def test_grouped_tool_supports_multiple_group_columns(
    analysis_context,
    statistics_service,
):
    tool = GroupedAnalysisTool(
        analysis_context,
        statistics_service,
    )

    result = run_tool(
        tool,
        {
            "dataset_id": "analysis-test-dataset",
            "group_by": ["department", "product"],
            "aggregations": {
                "amount": ["sum"],
            },
        },
    )

    assert result.success is True
    assert result.output["group_by"] == ["department", "product"]
    assert result.output["group_count"] == 4


def test_distribution_tool_returns_statistics_and_histogram(
    analysis_context,
    statistics_service,
):
    tool = DistributionAnalysisTool(
        analysis_context,
        statistics_service,
    )

    result = run_tool(
        tool,
        {
            "dataset_id": "analysis-test-dataset",
            "column": "amount",
            "bins": 4,
            "quantiles": [0.25, 0.5, 0.75],
        },
    )

    assert result.success is True
    assert result.output["column"] == "amount"
    assert result.output["count"] == 6
    assert result.output["mean"] == pytest.approx(125.0)
    assert sum(
        item["count"]
        for item in result.output["histogram"]
    ) == 6

    assert_json_compatible(result.output)


def test_distribution_tool_converts_quantiles_list_to_tuple(
    analysis_context,
    statistics_service,
):
    tool = DistributionAnalysisTool(
        analysis_context,
        statistics_service,
    )

    result = run_tool(
        tool,
        {
            "dataset_id": "analysis-test-dataset",
            "column": "amount",
            "quantiles": [0.1, 0.5, 0.9],
        },
    )

    assert result.success is True
    assert set(result.output["quantiles"]) == {
        "0.1",
        "0.5",
        "0.9",
    }


@pytest.mark.parametrize(
    ("tool_class", "arguments"),
    [
        (DatasetProfileTool, {}),
        (DescriptiveAnalysisTool, {}),
        (FrequencyAnalysisTool, {"column": "amount"}),
        (CorrelationAnalysisTool, {}),
        (GroupedAnalysisTool, {}),
        (DistributionAnalysisTool, {"column": "amount"}),
    ],
)
def test_analysis_tools_reject_missing_dataset_id(
    analysis_context,
    profiling_service,
    statistics_service,
    tool_class,
    arguments,
):
    if tool_class is DatasetProfileTool:
        tool = tool_class(analysis_context, profiling_service)
    else:
        tool = tool_class(analysis_context, statistics_service)

    result = run_tool(tool, arguments)

    assert result.success is False
    assert result.error is not None


def test_frequency_tool_rejects_invalid_top_n(
    analysis_context,
    statistics_service,
):
    tool = FrequencyAnalysisTool(
        analysis_context,
        statistics_service,
    )

    result = run_tool(
        tool,
        {
            "dataset_id": "analysis-test-dataset",
            "column": "department",
            "top_n": True,
        },
    )

    assert result.success is False
    assert "top_n" in result.error


def test_correlation_tool_rejects_unsupported_method(
    analysis_context,
    statistics_service,
):
    tool = CorrelationAnalysisTool(
        analysis_context,
        statistics_service,
    )

    result = run_tool(
        tool,
        {
            "dataset_id": "analysis-test-dataset",
            "method": "kendall",
        },
    )

    assert result.success is False
    assert "method" in result.error


def test_grouped_tool_rejects_unsupported_aggregation(
    analysis_context,
    statistics_service,
):
    tool = GroupedAnalysisTool(
        analysis_context,
        statistics_service,
    )

    result = run_tool(
        tool,
        {
            "dataset_id": "analysis-test-dataset",
            "group_by": ["department"],
            "aggregations": {
                "amount": ["median_absolute_deviation"],
            },
        },
    )

    assert result.success is False
    assert "aggregation" in result.error


def test_distribution_tool_rejects_invalid_bins(
    analysis_context,
    statistics_service,
):
    tool = DistributionAnalysisTool(
        analysis_context,
        statistics_service,
    )

    result = run_tool(
        tool,
        {
            "dataset_id": "analysis-test-dataset",
            "column": "amount",
            "bins": 0,
        },
    )

    assert result.success is False
    assert "bins" in result.error


def test_distribution_tool_rejects_duplicate_quantiles(
    analysis_context,
    statistics_service,
):
    tool = DistributionAnalysisTool(
        analysis_context,
        statistics_service,
    )

    result = run_tool(
        tool,
        {
            "dataset_id": "analysis-test-dataset",
            "column": "amount",
            "quantiles": [0.25, 0.5, 0.5],
        },
    )

    assert result.success is False
    assert "quantiles" in result.error


@pytest.mark.parametrize(
    ("tool_class", "arguments"),
    [
        (DatasetProfileTool, {"dataset_id": "missing-dataset"}),
        (
            DescriptiveAnalysisTool,
            {"dataset_id": "missing-dataset"},
        ),
        (
            FrequencyAnalysisTool,
            {
                "dataset_id": "missing-dataset",
                "column": "amount",
            },
        ),
        (
            CorrelationAnalysisTool,
            {"dataset_id": "missing-dataset"},
        ),
        (
            GroupedAnalysisTool,
            {
                "dataset_id": "missing-dataset",
                "group_by": ["department"],
                "aggregations": {"amount": ["sum"]},
            },
        ),
        (
            DistributionAnalysisTool,
            {
                "dataset_id": "missing-dataset",
                "column": "amount",
            },
        ),
    ],
)
def test_analysis_tools_return_failure_for_unknown_dataset(
    analysis_context,
    profiling_service,
    statistics_service,
    tool_class,
    arguments,
):
    if tool_class is DatasetProfileTool:
        tool = tool_class(analysis_context, profiling_service)
    else:
        tool = tool_class(analysis_context, statistics_service)

    result = run_tool(tool, arguments)

    assert result.success is False
    assert result.error == (
        "Dataset is not available in the current execution context."
    )


def test_analysis_tools_do_not_close_execution_context(
    analysis_context,
    statistics_service,
):
    tool = FrequencyAnalysisTool(
        analysis_context,
        statistics_service,
    )

    result = run_tool(
        tool,
        {
            "dataset_id": "analysis-test-dataset",
            "column": "department",
        },
    )

    assert result.success is True
    assert analysis_context.is_closed is False
    assert "analysis-test-dataset" in analysis_context.dataset_ids


def test_analysis_tool_schema_is_valid_json_schema(
    analysis_context,
    profiling_service,
    statistics_service,
):
    tools = [
        DatasetProfileTool(analysis_context, profiling_service),
        DescriptiveAnalysisTool(analysis_context, statistics_service),
        FrequencyAnalysisTool(analysis_context, statistics_service),
        CorrelationAnalysisTool(analysis_context, statistics_service),
        GroupedAnalysisTool(analysis_context, statistics_service),
        DistributionAnalysisTool(analysis_context, statistics_service),
    ]

    for tool in tools:
        Draft202012Validator.check_schema(tool.parameters)


def test_analysis_tool_schema_rejects_unknown_parameters(
    analysis_context,
    statistics_service,
):
    tool = FrequencyAnalysisTool(
        analysis_context,
        statistics_service,
    )

    validator = Draft202012Validator(tool.parameters)

    errors = list(
        validator.iter_errors(
            {
                "dataset_id": "analysis-test-dataset",
                "column": "department",
                "unexpected": True,
            }
        )
    )

    assert errors


def test_analysis_tool_schema_rejects_boolean_top_n(
    analysis_context,
    statistics_service,
):
    tool = FrequencyAnalysisTool(
        analysis_context,
        statistics_service,
    )

    validator = Draft202012Validator(tool.parameters)

    errors = list(
        validator.iter_errors(
            {
                "dataset_id": "analysis-test-dataset",
                "column": "department",
                "top_n": True,
            }
        )
    )

    assert errors


def test_analysis_tool_rejects_non_object_arguments(
    analysis_context,
    statistics_service,
):
    tool = DescriptiveAnalysisTool(
        analysis_context,
        statistics_service,
    )

    result = run_tool(tool, [])

    assert result.success is False
    assert result.error == "Tool arguments must be an object."


def test_analysis_tool_rejects_closed_context(
    analysis_context,
    statistics_service,
):
    tool = DescriptiveAnalysisTool(
        analysis_context,
        statistics_service,
    )

    analysis_context.close()

    result = run_tool(
        tool,
        {"dataset_id": "analysis-test-dataset"},
    )

    assert result.success is False
    assert result.error == "ExecutionContext is closed."