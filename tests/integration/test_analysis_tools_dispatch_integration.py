"""Integration tests for builtin analysis tools and the tool dispatch pipeline."""

from __future__ import annotations

import asyncio
import json

import pandas as pd
import pytest

from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.execution.dataset_store import DatasetReference
from lite_cog_agent.llm.schemas import ToolCall
from lite_cog_agent.tools.builtin.analysis_tools import (
    CorrelationAnalysisTool,
    DatasetProfileTool,
    DescriptiveAnalysisTool,
    DistributionAnalysisTool,
    FrequencyAnalysisTool,
    GroupedAnalysisTool,
)
from lite_cog_agent.tools.builtin.factory import create_builtin_tool_services
from lite_cog_agent.tools.dispatcher import ToolDispatcher
from lite_cog_agent.tools.message_converter import (
    convert_tool_results_to_messages,
)
from lite_cog_agent.tools.registry import ToolRegistry


DATASET_ID = "analysis-dispatch-integration-dataset"


@pytest.fixture
def integration_context():
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
        dataset_id=DATASET_ID,
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


def create_analysis_registry(context: ExecutionContext) -> ToolRegistry:
    """Build and register all six builtin analysis tools."""
    services = create_builtin_tool_services(context)

    tools = [
        DatasetProfileTool(context, services.profiling),
        DescriptiveAnalysisTool(context, services.statistics),
        FrequencyAnalysisTool(context, services.statistics),
        CorrelationAnalysisTool(context, services.statistics),
        GroupedAnalysisTool(context, services.statistics),
        DistributionAnalysisTool(context, services.statistics),
    ]

    registry = ToolRegistry()

    for tool in tools:
        registry.register(tool)

    return registry


def create_analysis_calls() -> list[ToolCall]:
    """Create a representative batch of LLM tool calls."""
    return [
        ToolCall(
            id="call_profile",
            name="analysis.profile",
            arguments={"dataset_id": DATASET_ID},
        ),
        ToolCall(
            id="call_descriptive",
            name="analysis.descriptive",
            arguments={
                "dataset_id": DATASET_ID,
                "columns": ["quantity", "amount"],
            },
        ),
        ToolCall(
            id="call_frequency",
            name="analysis.frequency",
            arguments={
                "dataset_id": DATASET_ID,
                "column": "department",
                "top_n": 2,
            },
        ),
        ToolCall(
            id="call_correlation",
            name="analysis.correlation",
            arguments={
                "dataset_id": DATASET_ID,
                "columns": ["quantity", "amount"],
                "method": "pearson",
            },
        ),
        ToolCall(
            id="call_grouped",
            name="analysis.grouped",
            arguments={
                "dataset_id": DATASET_ID,
                "group_by": ["department"],
                "aggregations": {
                    "amount": ["sum"],
                    "quantity": ["sum"],
                },
            },
        ),
        ToolCall(
            id="call_distribution",
            name="analysis.distribution",
            arguments={
                "dataset_id": DATASET_ID,
                "column": "amount",
                "bins": 3,
                "quantiles": [0.25, 0.5, 0.75],
            },
        ),
    ]


def test_analysis_tools_register_and_generate_definitions(
    integration_context,
):
    registry = create_analysis_registry(integration_context)

    expected_names = {
        "analysis.profile",
        "analysis.descriptive",
        "analysis.frequency",
        "analysis.correlation",
        "analysis.grouped",
        "analysis.distribution",
    }

    assert len(registry) == 6

    registered_names = {
        tool.name for tool in registry.list_tools()
    }

    assert registered_names == expected_names

    definitions = registry.get_definitions()

    assert len(definitions) == 6
    assert {definition.name for definition in definitions} == expected_names

    for definition in definitions:
        assert definition.description
        assert definition.parameters["type"] == "object"
        assert definition.parameters["additionalProperties"] is False


def test_analysis_tools_dispatch_and_convert_to_llm_messages(
    integration_context,
):
    registry = create_analysis_registry(integration_context)
    dispatcher = ToolDispatcher(registry=registry)

    tool_calls = create_analysis_calls()

    async def run_test():
        return await dispatcher.dispatch_many(tool_calls)

    results = asyncio.run(run_test())

    # Every analysis tool should execute successfully.
    assert len(results) == 6
    assert all(result.success for result in results)

    # Verify the dispatcher added execution metadata.
    for result in results:
        assert result.metadata["status"] == "success"
        assert result.metadata["tool_name"].startswith("analysis.")
        assert result.metadata["execution_time_seconds"] >= 0
        assert result.metadata["queue_wait_seconds"] >= 0

    # Verify tool-specific outputs.
    profile_output = results[0].output
    assert "dataset_profile" in profile_output
    assert "quality_report" in profile_output

    descriptive_output = results[1].output
    assert descriptive_output["row_count"] == 6
    assert len(descriptive_output["statistics"]) == 2

    frequency_output = results[2].output
    assert frequency_output["total_count"] == 6
    assert len(frequency_output["items"]) == 2

    correlation_output = results[3].output
    assert correlation_output["method"] == "pearson"
    assert len(correlation_output["matrix"]) == 2

    grouped_output = results[4].output
    assert grouped_output["group_count"] == 2

    grouped_records = {
        record["department"]: record
        for record in grouped_output["records"]
    }

    assert grouped_records["Sales"]["amount_sum"] == 470
    assert grouped_records["Support"]["amount_sum"] == 280
    assert grouped_records["Sales"]["quantity_sum"] == 7
    assert grouped_records["Support"]["quantity_sum"] == 5

    distribution_output = results[5].output
    assert distribution_output["count"] == 6
    assert len(distribution_output["histogram"]) == 3
    assert sum(
        item["count"]
        for item in distribution_output["histogram"]
    ) == 6

    # Convert the dispatcher results into LLM tool messages.
    messages = convert_tool_results_to_messages(
        tool_calls,
        results,
    )

    assert len(messages) == 6

    for tool_call, message in zip(
        tool_calls,
        messages,
        strict=True,
    ):
        assert message.role == "tool"
        assert message.tool_call_id == tool_call.id

        payload = json.loads(message.content)

        assert payload["success"] is True
        assert payload["error"] is None
        assert isinstance(payload["output"], dict)
        assert payload["metadata"]["tool_name"] == tool_call.name
        assert payload["metadata"]["dataset_id"] == DATASET_ID

    # Analysis tools must not own or close the execution context.
    assert integration_context.is_closed is False

    assert (
        integration_context.get_dataset(DATASET_ID).shape
        == (6, 4)
    )


def test_dispatcher_validates_analysis_tool_arguments(
    integration_context,
):
    registry = create_analysis_registry(integration_context)
    dispatcher = ToolDispatcher(registry=registry)

    invalid_call = ToolCall(
        id="call_invalid_frequency",
        name="analysis.frequency",
        arguments={
            "dataset_id": DATASET_ID,
            # Required column argument is intentionally missing.
        },
    )

    async def run_test():
        return await dispatcher.dispatch(invalid_call)

    result = asyncio.run(run_test())

    assert result.success is False
    assert result.error
    assert result.metadata["status"] == "failed"
    assert result.metadata["error_type"] == (
        "ToolParameterValidationError"
    )

    messages = convert_tool_results_to_messages(
        [invalid_call],
        [result],
    )

    payload = json.loads(messages[0].content)

    assert payload["success"] is False
    assert payload["output"] is None
    assert payload["error"] == result.error


def test_dispatcher_preserves_analysis_tool_failure(
    integration_context,
):
    registry = create_analysis_registry(integration_context)
    dispatcher = ToolDispatcher(registry=registry)

    missing_dataset_call = ToolCall(
        id="call_missing_dataset",
        name="analysis.descriptive",
        arguments={
            "dataset_id": "missing-dataset",
        },
    )

    async def run_test():
        return await dispatcher.dispatch(missing_dataset_call)

    result = asyncio.run(run_test())

    assert result.success is False
    assert result.error == (
        "Dataset is not available in the current execution context."
    )

    assert result.metadata["status"] == "failed"
    assert result.metadata["error_type"] == "ToolReportedFailure"

    messages = convert_tool_results_to_messages(
        [missing_dataset_call],
        [result],
    )

    payload = json.loads(messages[0].content)

    assert payload["success"] is False
    assert payload["error"] == result.error

    # A failed tool call must not destroy the shared context.
    assert integration_context.is_closed is False
    assert (
        integration_context.get_dataset(DATASET_ID).shape
        == (6, 4)
    )
