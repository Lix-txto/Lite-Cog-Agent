"""Tests for builtin visualization tools."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest
from jsonschema import Draft202012Validator

from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.execution.dataset_store import DatasetReference
from lite_cog_agent.storage.file_manager import FileManager, FileStorageError
from lite_cog_agent.tools.builtin.visualization_tools import (
    BarChartTool,
    BoxChartTool,
    GroupedBarChartTool,
    HistogramTool,
    LineChartTool,
    PieChartTool,
    ScatterChartTool,
    create_visualization_tools,
)
from lite_cog_agent.visualization.renderer import OutputFormat
from lite_cog_agent.visualization.service import VisualizationService


DATASET_ID = "visualization-test-dataset"

TOOL_CASES = [
    (
        BarChartTool,
        "visualization.bar",
        {"column": "department"},
    ),
    (
        GroupedBarChartTool,
        "visualization.grouped_bar",
        {
            "group_by": ["department"],
            "aggregations": {"amount": ["sum", "mean"]},
        },
    ),
    (
        LineChartTool,
        "visualization.line",
        {"x_column": "month", "y_column": "amount"},
    ),
    (
        HistogramTool,
        "visualization.histogram",
        {"column": "amount"},
    ),
    (
        ScatterChartTool,
        "visualization.scatter",
        {"x_column": "quantity", "y_column": "amount"},
    ),
    (
        PieChartTool,
        "visualization.pie",
        {"column": "product"},
    ),
    (
        BoxChartTool,
        "visualization.box",
        {"column": "amount"},
    ),
]


@pytest.fixture
def context() -> ExecutionContext:
    """Create a context with a representative dataset."""
    execution_context = ExecutionContext()

    dataframe = pd.DataFrame(
        {
            "department": [
                "Sales",
                "Sales",
                "Support",
                "Support",
                "Sales",
                "Support",
            ],
            "product": ["A", "B", "A", "C", "A", "B"],
            "month": [1, 2, 3, 4, 5, 6],
            "quantity": [2, 3, 1, 4, 5, 6],
            "amount": [100, 150, 80, 200, 120, 180],
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

    execution_context.register_dataset(reference, dataframe)

    yield execution_context

    execution_context.close()


@pytest.fixture
def visualization_service() -> VisualizationService:
    return VisualizationService()


@pytest.fixture
def file_manager(tmp_path: Path) -> FileManager:
    return FileManager(tmp_path / "artifacts")


@pytest.fixture
def tools(
    context: ExecutionContext,
    visualization_service: VisualizationService,
    file_manager: FileManager,
):
    return create_visualization_tools(
        context,
        visualization_service,
        file_manager,
    )


def run_tool(tool, arguments: dict):
    """Execute an asynchronous tool from a synchronous test."""
    return asyncio.run(tool.execute(arguments))


def assert_json_compatible(value) -> None:
    json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
    )


def test_visualization_tool_factory_creates_seven_tools(tools):
    assert len(tools) == 7

    assert [tool.name for tool in tools] == [
        "visualization.bar",
        "visualization.grouped_bar",
        "visualization.line",
        "visualization.histogram",
        "visualization.scatter",
        "visualization.pie",
        "visualization.box",
    ]


@pytest.mark.parametrize(
    ("tool_class", "expected_name", "specific_arguments"),
    TOOL_CASES,
)
def test_tool_metadata_and_schema(
    context,
    visualization_service,
    file_manager,
    tool_class,
    expected_name,
    specific_arguments,
):
    tool = tool_class(
        context,
        visualization_service,
        file_manager,
    )

    assert tool.name == expected_name
    assert isinstance(tool.description, str)
    assert tool.description

    schema = tool.parameters

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert "dataset_id" in schema["required"]

    Draft202012Validator.check_schema(schema)

    valid_arguments = {
        "dataset_id": DATASET_ID,
        **specific_arguments,
    }

    Draft202012Validator(schema).validate(valid_arguments)


@pytest.mark.parametrize(
    ("tool_class", "expected_name", "specific_arguments"),
    TOOL_CASES,
)
def test_each_tool_creates_and_stores_artifact(
    context,
    visualization_service,
    file_manager,
    tool_class,
    expected_name,
    specific_arguments,
):
    tool = tool_class(
        context,
        visualization_service,
        file_manager,
    )

    result = run_tool(
        tool,
        {
            "dataset_id": DATASET_ID,
            **specific_arguments,
        },
    )

    assert result.success is True
    assert result.error is None

    assert result.output is not None
    assert "visualization" in result.output
    assert "artifact" in result.output

    visualization = result.output["visualization"]
    artifact = result.output["artifact"]

    assert visualization["dataset_id"] == DATASET_ID
    assert visualization["visualization_id"]

    assert artifact["artifact_id"]
    assert artifact["filename"]
    assert artifact["size_bytes"] > 0
    assert artifact["format"] == "png"
    assert artifact["media_type"] == "image/png"

    assert result.metadata["tool"] == expected_name
    assert result.metadata["dataset_id"] == DATASET_ID
    assert result.metadata["artifact_id"] == artifact["artifact_id"]

    stored_path = file_manager.root / artifact["filename"]

    assert stored_path.is_file()
    assert stored_path.stat().st_size == artifact["size_bytes"]

    image_data = stored_path.read_bytes()

    assert image_data.startswith(b"\x89PNG\r\n\x1a\n")

    # Binary image data and absolute filesystem paths must not leak.
    serialized = json.dumps(
        result.output,
        ensure_ascii=False,
        allow_nan=False,
    )

    assert file_manager.root.as_posix() not in serialized
    assert "image_data" not in serialized

    assert_json_compatible(result.output)
    assert_json_compatible(result.metadata)


@pytest.mark.parametrize(
    ("tool_class", "expected_name", "specific_arguments"),
    TOOL_CASES,
)
def test_each_tool_supports_svg_output(
    context,
    visualization_service,
    file_manager,
    tool_class,
    expected_name,
    specific_arguments,
):
    tool = tool_class(
        context,
        visualization_service,
        file_manager,
    )

    result = run_tool(
        tool,
        {
            "dataset_id": DATASET_ID,
            "output_format": "svg",
            **specific_arguments,
        },
    )

    assert result.success is True

    artifact = result.output["artifact"]

    assert artifact["format"] == "svg"
    assert artifact["media_type"] == "image/svg+xml"
    assert artifact["filename"].endswith(".svg")

    stored_path = file_manager.root / artifact["filename"]

    assert stored_path.is_file()

    content = stored_path.read_bytes()

    assert b"<svg" in content[:1000]



def test_bar_tool_passes_optional_parameters(
    context,
    visualization_service,
    file_manager,
    monkeypatch,
):
    tool = BarChartTool(
        context,
        visualization_service,
        file_manager,
    )

    captured = {}

    # 必须先保存原始方法，避免 monkeypatch 后递归调用自己。
    original_create_bar_chart = visualization_service.create_bar_chart

    def fake_create_bar_chart(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

        return original_create_bar_chart(*args, **kwargs)

    monkeypatch.setattr(
        visualization_service,
        "create_bar_chart",
        fake_create_bar_chart,
    )

    result = run_tool(
        tool,
        {
            "dataset_id": DATASET_ID,
            "column": "department",
            "top_n": 2,
            "output_format": "png",
            "title": "Department Distribution",
        },
    )

    assert result.success is True
    assert captured["args"][0] is context
    assert captured["args"][1] == DATASET_ID
    assert captured["kwargs"]["column"] == "department"
    assert captured["kwargs"]["top_n"] == 2
    assert captured["kwargs"]["title"] == "Department Distribution"


def test_grouped_bar_tool_passes_grouping_configuration(
    context,
    visualization_service,
    file_manager,
    monkeypatch,
):
    tool = GroupedBarChartTool(
        context,
        visualization_service,
        file_manager,
    )

    captured = {}
    original = visualization_service.create_grouped_bar_chart

    def spy(*args, **kwargs):
        captured["kwargs"] = kwargs
        return original(*args, **kwargs)

    monkeypatch.setattr(
        visualization_service,
        "create_grouped_bar_chart",
        spy,
    )

    result = run_tool(
        tool,
        {
            "dataset_id": DATASET_ID,
            "group_by": ["department"],
            "aggregations": {"amount": ["sum", "mean"]},
        },
    )

    assert result.success is True
    assert captured["kwargs"]["group_by"] == ["department"]
    assert captured["kwargs"]["aggregations"] == {
        "amount": ["sum", "mean"],
    }


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"column": "department"},
        {"dataset_id": DATASET_ID},
        {"dataset_id": "", "column": "department"},
        {
            "dataset_id": DATASET_ID,
            "column": "department",
            "output_format": "pdf",
        },
        {
            "dataset_id": DATASET_ID,
            "column": "department",
            "top_n": 0,
        },
        {
            "dataset_id": DATASET_ID,
            "column": "department",
            "top_n": True,
        },
        {
            "dataset_id": DATASET_ID,
            "column": "department",
            "title": "",
        },
    ],
)
def test_bar_tool_rejects_invalid_arguments(
    context,
    visualization_service,
    file_manager,
    arguments,
):
    tool = BarChartTool(
        context,
        visualization_service,
        file_manager,
    )

    result = run_tool(tool, arguments)

    assert result.success is False
    assert result.error


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "dataset_id": DATASET_ID,
            "group_by": [],
            "aggregations": {"amount": ["sum"]},
        },
        {
            "dataset_id": DATASET_ID,
            "group_by": ["department", "department"],
            "aggregations": {"amount": ["sum"]},
        },
        {
            "dataset_id": DATASET_ID,
            "group_by": ["department"],
            "aggregations": {},
        },
        {
            "dataset_id": DATASET_ID,
            "group_by": ["department"],
            "aggregations": {"amount": ["invalid"]},
        },
        {
            "dataset_id": DATASET_ID,
            "group_by": ["department"],
            "aggregations": {"amount": ["sum", "sum"]},
        },
    ],
)
def test_grouped_bar_tool_rejects_invalid_arguments(
    context,
    visualization_service,
    file_manager,
    arguments,
):
    tool = GroupedBarChartTool(
        context,
        visualization_service,
        file_manager,
    )

    result = run_tool(tool, arguments)

    assert result.success is False
    assert result.error


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "dataset_id": DATASET_ID,
            "column": "amount",
            "bins": 0,
        },
        {
            "dataset_id": DATASET_ID,
            "column": "amount",
            "bins": True,
        },
        {
            "dataset_id": DATASET_ID,
            "column": "amount",
            "bins": 1.5,
        },
    ],
)
def test_histogram_tool_rejects_invalid_bins(
    context,
    visualization_service,
    file_manager,
    arguments,
):
    tool = HistogramTool(
        context,
        visualization_service,
        file_manager,
    )

    result = run_tool(tool, arguments)

    assert result.success is False
    assert "bins" in result.error


def test_tool_rejects_non_object_arguments(
    context,
    visualization_service,
    file_manager,
):
    tool = BarChartTool(
        context,
        visualization_service,
        file_manager,
    )

    result = run_tool(tool, [])

    assert result.success is False
    assert "object" in result.error


def test_tool_returns_failure_for_unknown_dataset(
    context,
    visualization_service,
    file_manager,
):
    tool = BarChartTool(
        context,
        visualization_service,
        file_manager,
    )

    result = run_tool(
        tool,
        {
            "dataset_id": "unknown-dataset",
            "column": "department",
        },
    )

    assert result.success is False
    assert result.error
    assert result.metadata["dataset_id"] == "unknown-dataset"

    assert list(file_manager.root.iterdir()) == []



def test_tool_returns_failure_when_visualization_fails(
    context,
    visualization_service,
    file_manager,
    monkeypatch,
):
    tool = BarChartTool(
        context,
        visualization_service,
        file_manager,
    )

    def fail(*args, **kwargs):
        raise RuntimeError("internal rendering failure")

    monkeypatch.setattr(
        visualization_service,
        "create_bar_chart",
        fail,
    )

    result = run_tool(
        tool,
        {
            "dataset_id": DATASET_ID,
            "column": "department",
        },
    )

    assert result.success is False
    assert result.error == "internal rendering failure"
    assert result.metadata["tool"] == "visualization.bar"
    assert result.metadata["dataset_id"] == DATASET_ID



def test_tool_returns_failure_when_artifact_storage_fails(
    context,
    visualization_service,
    file_manager,
    monkeypatch,
):
    tool = BarChartTool(
        context,
        visualization_service,
        file_manager,
    )

    def fail(*args, **kwargs):
        raise FileStorageError("disk unavailable")

    monkeypatch.setattr(file_manager, "save_bytes", fail)

    result = run_tool(
        tool,
        {
            "dataset_id": DATASET_ID,
            "column": "department",
        },
    )

    assert result.success is False
    assert "disk unavailable" in result.error
    assert list(file_manager.root.iterdir()) == []


def test_tool_does_not_close_shared_context(
    context,
    visualization_service,
    file_manager,
):
    tool = BarChartTool(
        context,
        visualization_service,
        file_manager,
    )

    result = run_tool(
        tool,
        {
            "dataset_id": DATASET_ID,
            "column": "department",
        },
    )

    assert result.success is True
    assert context.is_closed is False

    dataframe = context.get_dataset(DATASET_ID)

    assert dataframe.shape == (6, 5)


def test_tool_rejects_execution_after_context_is_closed(
    context,
    visualization_service,
    file_manager,
):
    tool = BarChartTool(
        context,
        visualization_service,
        file_manager,
    )

    context.close()

    result = run_tool(
        tool,
        {
            "dataset_id": DATASET_ID,
            "column": "department",
        },
    )

    assert result.success is False
    assert "closed" in result.error.lower()


def test_tool_constructor_rejects_closed_context(
    context,
    visualization_service,
    file_manager,
):
    context.close()

    with pytest.raises(ValueError, match="closed"):
        BarChartTool(
            context,
            visualization_service,
            file_manager,
        )


def test_factory_tools_share_the_same_dependencies(
    context,
    visualization_service,
    file_manager,
):
    tools = create_visualization_tools(
        context,
        visualization_service,
        file_manager,
    )

    assert len(tools) == 7

    for tool in tools:
        assert tool._context is context
        assert tool._visualization_service is visualization_service
        assert tool._file_manager is file_manager
