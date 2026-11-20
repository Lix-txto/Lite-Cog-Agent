"""Integration tests for the dataset loading, cleaning, and analysis workflow."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.tools.builtin import (
    DatasetCleaningTool,
    create_builtin_tool_services,
)
from lite_cog_agent.tools.builtin.analysis_tools import (
    DatasetProfileTool,
    DescriptiveAnalysisTool,
)
from lite_cog_agent.tools.builtin.dataset_tools import DatasetLoadTool


@pytest.fixture
def execution_context():
    """Create one execution context shared by all workflow tools."""
    context = ExecutionContext()

    yield context

    context.close()


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """Create a CSV dataset containing one row with a missing value."""
    root = tmp_path / "datasets"
    root.mkdir()

    csv_content = (
        "name,age,score\n"
        "Alice,20,80\n"
        "Bob,30,90\n"
        ",40,85\n"
        "David,50,95\n"
    )

    (root / "users.csv").write_text(
        csv_content,
        encoding="utf-8",
    )

    return root


def run_tool(tool: Any, arguments: dict[str, Any]):
    """Execute an asynchronous builtin tool in a synchronous test."""
    return asyncio.run(tool.execute(arguments))


def assert_json_compatible(value: Any) -> None:
    """Ensure tool output can be encoded by strict JSON."""
    json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
    )


def test_load_clean_profile_and_descriptive_analysis_workflow(
    execution_context: ExecutionContext,
    dataset_root: Path,
    tmp_path: Path,
) -> None:
    """Verify that loaded and cleaned datasets flow through analysis tools."""

    # ---------------------------------------------------------
    # 1. Create shared builtin services and tools
    # ---------------------------------------------------------

    services = create_builtin_tool_services(
        execution_context,
        artifact_root=tmp_path / "artifacts",
    )

    load_tool = DatasetLoadTool(
        context=execution_context,
        loading_service=services.loading,
        dataset_root=dataset_root,
    )

    cleaning_tool = DatasetCleaningTool(
        context=execution_context,
        cleaning_service=services.cleaning,
    )

    profile_tool = DatasetProfileTool(
        context=execution_context,
        profiling_service=services.profiling,
    )

    descriptive_tool = DescriptiveAnalysisTool(
        context=execution_context,
        statistics_service=services.statistics,
    )

    # ---------------------------------------------------------
    # 2. Load the source dataset
    # ---------------------------------------------------------

    load_result = run_tool(
        load_tool,
        {
            "file_path": "users.csv",
        },
    )

    assert load_result.success is True
    assert load_result.error is None

    source_dataset_id = load_result.output["dataset_id"]

    assert source_dataset_id
    assert load_result.output["row_count"] == 4
    assert load_result.output["column_count"] == 3
    assert load_result.output["columns"] == [
        "name",
        "age",
        "score",
    ]

    source_dataframe = execution_context.get_dataset(
        source_dataset_id
    )

    assert isinstance(source_dataframe, pd.DataFrame)
    assert source_dataframe.shape == (4, 3)
    assert source_dataframe["name"].isna().sum() == 1

    original_dataframe = source_dataframe.copy(deep=True)

    # ---------------------------------------------------------
    # 3. Clean the source dataset
    # ---------------------------------------------------------

    cleaning_result = run_tool(
        cleaning_tool,
        {
            "dataset_id": source_dataset_id,
            "steps": [
                {
                    "operation": "missing.drop",
                    "parameters": {
                        "how": "any",
                    },
                }
            ],
        },
    )

    assert cleaning_result.success is True
    assert cleaning_result.error is None

    cleaning_output = cleaning_result.output

    assert cleaning_output["status"] == "success"
    assert cleaning_output["source_dataset_id"] == source_dataset_id
    assert cleaning_output["input_row_count"] == 4
    assert cleaning_output["output_row_count"] == 3
    assert cleaning_output["output_column_count"] == 3

    assert len(cleaning_output["operations"]) == 1
    assert cleaning_output["operations"][0]["operation"] == (
        "missing.drop"
    )
    assert cleaning_output["operations"][0]["status"] == "success"

    cleaned_dataset_id = cleaning_output["output_dataset_id"]

    assert cleaned_dataset_id
    assert cleaned_dataset_id != source_dataset_id

    # ---------------------------------------------------------
    # 4. Verify dataset registration and source isolation
    # ---------------------------------------------------------

    cleaned_dataframe = execution_context.get_dataset(
        cleaned_dataset_id
    )

    cleaned_reference = execution_context.get_dataset_reference(
        cleaned_dataset_id
    )

    assert isinstance(cleaned_dataframe, pd.DataFrame)
    assert cleaned_dataframe.shape == (3, 3)
    assert cleaned_dataframe.isna().sum().sum() == 0

    assert cleaned_reference.source == "cleaning"
    assert cleaned_reference.metadata["source_dataset_id"] == (
        source_dataset_id
    )

    # Cleaning must not modify the original dataset.
    pd.testing.assert_frame_equal(
        execution_context.get_dataset(source_dataset_id),
        original_dataframe,
    )

    # ---------------------------------------------------------
    # 5. Profile the cleaned dataset
    # ---------------------------------------------------------

    profile_result = run_tool(
        profile_tool,
        {
            "dataset_id": cleaned_dataset_id,
        },
    )

    assert profile_result.success is True
    assert profile_result.error is None

    profile_output = profile_result.output

    dataset_profile = profile_output["dataset_profile"]
    quality_report = profile_output["quality_report"]

    assert dataset_profile["dataset_id"] == cleaned_dataset_id
    assert dataset_profile["row_count"] == 3
    assert dataset_profile["column_count"] == 3

    assert quality_report["missing_cells"] == 0
    assert quality_report["total_cells"] == 9
    assert quality_report["completeness"] == 100.0

    # ---------------------------------------------------------
    # 6. Calculate descriptive statistics
    # ---------------------------------------------------------

    statistics_result = run_tool(
        descriptive_tool,
        {
            "dataset_id": cleaned_dataset_id,
            "columns": ["age", "score"],
        },
    )

    assert statistics_result.success is True
    assert statistics_result.error is None

    statistics_output = statistics_result.output

    assert statistics_output["dataset_id"] == cleaned_dataset_id
    assert statistics_output["row_count"] == 3

    statistics_by_column = {
        item["column"]: item
        for item in statistics_output["statistics"]
    }

    assert set(statistics_by_column) == {"age", "score"}

    age_statistics = statistics_by_column["age"]
    assert age_statistics["count"] == 3
    assert age_statistics["mean"] == pytest.approx(100 / 3)
    assert age_statistics["min"] == 20
    assert age_statistics["max"] == 50

    score_statistics = statistics_by_column["score"]
    assert score_statistics["count"] == 3
    assert score_statistics["mean"] == pytest.approx(265 / 3)
    assert score_statistics["min"] == 80
    assert score_statistics["max"] == 95

    # ---------------------------------------------------------
    # 7. Verify strict JSON compatibility
    # ---------------------------------------------------------

    assert_json_compatible(load_result.output)
    assert_json_compatible(load_result.metadata)

    assert_json_compatible(cleaning_result.output)
    assert_json_compatible(cleaning_result.metadata)

    assert_json_compatible(profile_result.output)
    assert_json_compatible(profile_result.metadata)

    assert_json_compatible(statistics_result.output)
    assert_json_compatible(statistics_result.metadata)

    # ---------------------------------------------------------
    # 8. Verify the shared execution context
    # ---------------------------------------------------------

    assert source_dataset_id in execution_context.dataset_ids
    assert cleaned_dataset_id in execution_context.dataset_ids

    assert len(execution_context.dataset_ids) == 2
