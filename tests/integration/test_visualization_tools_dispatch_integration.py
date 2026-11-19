"""Integration tests for visualization tools through ToolDispatcher."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.execution.dataset_store import DatasetReference
from lite_cog_agent.llm.schemas import ToolCall
from lite_cog_agent.storage.file_manager import FileStorageError
from lite_cog_agent.tools.builtin.factory import (
    create_builtin_tool_services,
)
from lite_cog_agent.tools.builtin.visualization_tools import (
    create_visualization_tools,
)
from lite_cog_agent.tools.dispatcher import ToolDispatcher
from lite_cog_agent.tools.message_converter import (
    convert_tool_results_to_messages,
)
from lite_cog_agent.tools.registry import ToolRegistry
from lite_cog_agent.visualization.exceptions import (
    VisualizationExecutionError,
)


DATASET_ID = "visualization-dispatch-integration-dataset"


@pytest.fixture
def integration_environment(
    tmp_path: Path,
) -> dict[str, Any]:
    """Build a real visualization tool execution environment."""

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

    reference = DatasetReference(
        dataset_id=DATASET_ID,
        name="sales_dataset",
        source="integration_test.csv",
        file_format="csv",
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        columns=tuple(dataframe.columns),
    )

    context.register_dataset(reference, dataframe)

    artifact_root = tmp_path / "visualization-artifacts"

    services = create_builtin_tool_services(
        context,
        artifact_root=artifact_root,
    )

    visualization_tools = create_visualization_tools(
        context,
        services.visualization,
        services.file_manager,
    )

    registry = ToolRegistry()

    for tool in visualization_tools:
        registry.register(tool)

    dispatcher = ToolDispatcher(registry)

    environment = {
        "context": context,
        "dataframe": dataframe,
        "services": services,
        "tools": visualization_tools,
        "registry": registry,
        "dispatcher": dispatcher,
        "artifact_root": artifact_root,
    }

    try:
        yield environment
    finally:
        context.close()


def make_call(
    call_id: str,
    name: str,
    arguments: dict[str, Any],
) -> ToolCall:
    """Create a visualization ToolCall."""

    return ToolCall(
        id=call_id,
        name=name,
        arguments=arguments,
    )


def assert_dispatch_success(result: Any) -> None:
    """Validate common successful Dispatcher metadata."""

    assert result.success is True
    assert result.error is None

    assert result.metadata["status"] == "success"
    assert result.metadata["error_type"] is None
    assert isinstance(
        result.metadata["execution_time_seconds"],
        (int, float),
    )
    assert result.metadata["execution_time_seconds"] >= 0

    assert isinstance(
        result.metadata["queue_wait_seconds"],
        (int, float),
    )
    assert result.metadata["queue_wait_seconds"] >= 0


# ---------------------------------------------------------------------
# Registry and tool definitions
# ---------------------------------------------------------------------


def test_visualization_tools_register_and_generate_definitions(
    integration_environment: dict[str, Any],
) -> None:
    """All seven visualization tools should be registered."""

    registry = integration_environment["registry"]

    expected_names = {
        "visualization.bar",
        "visualization.grouped_bar",
        "visualization.line",
        "visualization.histogram",
        "visualization.scatter",
        "visualization.pie",
        "visualization.box",
    }

    registered_tools = registry.list_tools()

    assert len(registry) == 7
    assert {
        tool.name
        for tool in registered_tools
    } == expected_names

    definitions = registry.get_definitions()

    assert len(definitions) == 7
    assert {
        definition.name
        for definition in definitions
    } == expected_names

    for definition in definitions:
        assert definition.description
        assert definition.parameters["type"] == "object"
        assert "dataset_id" in definition.parameters["required"]
        assert definition.parameters["additionalProperties"] is False


# ---------------------------------------------------------------------
# Dispatcher execution and artifact persistence
# ---------------------------------------------------------------------


def test_dispatch_bar_chart_creates_png_artifact(
    integration_environment: dict[str, Any],
) -> None:
    """A dispatched bar chart should create a valid PNG artifact."""

    dispatcher = integration_environment["dispatcher"]
    artifact_root = integration_environment["artifact_root"]

    call = make_call(
        "call-bar-png",
        "visualization.bar",
        {
            "dataset_id": DATASET_ID,
            "column": "department",
            "output_format": "png",
        },
    )

    result = asyncio.run(dispatcher.dispatch(call))

    assert_dispatch_success(result)

    assert result.metadata["tool_name"] == "visualization.bar"
    assert result.metadata["dataset_id"] == DATASET_ID

    visualization = result.output["visualization"]
    artifact = result.output["artifact"]

    assert visualization["dataset_id"] == DATASET_ID

    assert artifact["format"] == "png"
    assert artifact["media_type"] == "image/png"
    assert artifact["size_bytes"] > 0
    assert artifact["filename"].endswith(".png")

    stored_path = artifact_root / artifact["filename"]

    assert stored_path.is_file()

    image_data = stored_path.read_bytes()

    assert image_data.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(image_data) == artifact["size_bytes"]

    assert result.metadata["artifact_id"] == artifact["artifact_id"]
    assert result.metadata["visualization_id"] == (
        visualization["visualization_id"]
    )


def test_dispatch_histogram_creates_svg_artifact(
    integration_environment: dict[str, Any],
) -> None:
    """A dispatched histogram should create a valid SVG artifact."""

    dispatcher = integration_environment["dispatcher"]
    artifact_root = integration_environment["artifact_root"]

    call = make_call(
        "call-histogram-svg",
        "visualization.histogram",
        {
            "dataset_id": DATASET_ID,
            "column": "amount",
            "bins": 3,
            "output_format": "svg",
        },
    )

    result = asyncio.run(dispatcher.dispatch(call))

    assert_dispatch_success(result)

    artifact = result.output["artifact"]

    assert artifact["format"] == "svg"
    assert artifact["media_type"] == "image/svg+xml"
    assert artifact["filename"].endswith(".svg")

    stored_path = artifact_root / artifact["filename"]

    assert stored_path.is_file()

    svg_content = stored_path.read_text(encoding="utf-8")

    assert "<svg" in svg_content
    assert "</svg>" in svg_content


# ---------------------------------------------------------------------
# Tool Message conversion
# ---------------------------------------------------------------------


def test_visualization_dispatch_result_converts_to_llm_message(
    integration_environment: dict[str, Any],
) -> None:
    """Visualization results should become serializable tool messages."""

    dispatcher = integration_environment["dispatcher"]

    calls = [
        make_call(
            "call-message-bar",
            "visualization.bar",
            {
                "dataset_id": DATASET_ID,
                "column": "department",
            },
        ),
        make_call(
            "call-message-pie",
            "visualization.pie",
            {
                "dataset_id": DATASET_ID,
                "column": "product",
            },
        ),
    ]

    results = asyncio.run(dispatcher.dispatch_many(calls))

    assert len(results) == 2
    assert all(result.success for result in results)

    messages = convert_tool_results_to_messages(
        calls,
        results,
    )

    assert len(messages) == 2

    for call, message, result in zip(
        calls,
        messages,
        results,
        strict=True,
    ):
        assert message.role == "tool"
        assert message.tool_call_id == call.id

        payload = json.loads(message.content)

        assert payload["success"] is True
        assert payload["error"] is None
        assert payload["output"] == result.output

        assert payload["metadata"]["tool_name"] == call.name
        assert payload["metadata"]["status"] == "success"

        assert isinstance(payload["output"]["artifact"], dict)


def test_tool_message_does_not_expose_binary_data_or_storage_path(
    integration_environment: dict[str, Any],
) -> None:
    """LLM messages should contain metadata, never raw image bytes or paths."""

    dispatcher = integration_environment["dispatcher"]
    artifact_root = integration_environment["artifact_root"]

    call = make_call(
        "call-no-binary",
        "visualization.bar",
        {
            "dataset_id": DATASET_ID,
            "column": "department",
        },
    )

    result = asyncio.run(dispatcher.dispatch(call))

    assert_dispatch_success(result)

    messages = convert_tool_results_to_messages(
        [call],
        [result],
    )

    message_content = messages[0].content

    assert isinstance(message_content, str)
    assert str(artifact_root) not in message_content

    assert "image_data" not in message_content
    assert "image/png" in message_content

    assert isinstance(result.output["artifact"]["size_bytes"], int)
    assert "path" not in result.output["artifact"]


# ---------------------------------------------------------------------
# Parameter validation and failure handling
# ---------------------------------------------------------------------


def test_dispatch_rejects_invalid_visualization_arguments(
    integration_environment: dict[str, Any],
) -> None:
    """Dispatcher should reject arguments that violate the tool schema."""

    dispatcher = integration_environment["dispatcher"]

    call = make_call(
        "call-invalid-arguments",
        "visualization.bar",
        {
            "dataset_id": DATASET_ID,
            # Missing required column.
            "unexpected_parameter": "value",
        },
    )

    result = asyncio.run(dispatcher.dispatch(call))

    assert result.success is False
    assert result.error

    assert result.metadata["status"] == "failed"
    assert result.metadata["error_type"] == (
        "ToolParameterValidationError"
    )


def test_dispatch_preserves_unknown_dataset_failure(
    integration_environment: dict[str, Any],
) -> None:
    """Unknown dataset failures should remain controlled tool failures."""

    dispatcher = integration_environment["dispatcher"]

    call = make_call(
        "call-unknown-dataset",
        "visualization.bar",
        {
            "dataset_id": "missing-dataset",
            "column": "department",
        },
    )

    result = asyncio.run(dispatcher.dispatch(call))

    assert result.success is False
    assert result.error

    assert result.metadata["tool_name"] == "visualization.bar"
    assert result.metadata["status"] == "failed"
    assert result.metadata["error_type"] == "ToolReportedFailure"


def test_dispatch_preserves_visualization_service_failure(
    integration_environment: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Visualization service failures should not escape the Tool boundary."""

    dispatcher = integration_environment["dispatcher"]
    services = integration_environment["services"]

    def fail_visualization(*args: Any, **kwargs: Any) -> None:
        raise VisualizationExecutionError(
            "Simulated visualization failure"
        )

    monkeypatch.setattr(
        services.visualization,
        "create_bar_chart",
        fail_visualization,
    )

    call = make_call(
        "call-visualization-failure",
        "visualization.bar",
        {
            "dataset_id": DATASET_ID,
            "column": "department",
        },
    )

    result = asyncio.run(dispatcher.dispatch(call))

    assert result.success is False
    assert "Simulated visualization failure" in result.error

    assert result.metadata["status"] == "failed"
    assert result.metadata["error_type"] == "ToolReportedFailure"

    assert list(
        integration_environment["artifact_root"].iterdir()
    ) == []


def test_dispatch_handles_artifact_storage_failure(
    integration_environment: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FileManager failures should produce a failed ToolResult."""

    dispatcher = integration_environment["dispatcher"]
    services = integration_environment["services"]

    def fail_storage(*args: Any, **kwargs: Any) -> None:
        raise FileStorageError("Simulated storage failure")

    monkeypatch.setattr(
        services.file_manager,
        "save_bytes",
        fail_storage,
    )

    call = make_call(
        "call-storage-failure",
        "visualization.bar",
        {
            "dataset_id": DATASET_ID,
            "column": "department",
        },
    )

    result = asyncio.run(dispatcher.dispatch(call))

    assert result.success is False
    assert "Simulated storage failure" in result.error

    assert result.metadata["status"] == "failed"
    assert result.metadata["error_type"] == "ToolReportedFailure"

    assert list(
        integration_environment["artifact_root"].iterdir()
    ) == []


# ---------------------------------------------------------------------
# Concurrent dispatch
# ---------------------------------------------------------------------


def test_dispatch_many_visualization_tools_preserves_result_order(
    integration_environment: dict[str, Any],
) -> None:
    """Concurrent chart generation should preserve input result order."""

    dispatcher = integration_environment["dispatcher"]
    artifact_root = integration_environment["artifact_root"]

    calls = [
        make_call(
            "call-concurrent-bar",
            "visualization.bar",
            {
                "dataset_id": DATASET_ID,
                "column": "department",
            },
        ),
        make_call(
            "call-concurrent-histogram",
            "visualization.histogram",
            {
                "dataset_id": DATASET_ID,
                "column": "amount",
                "bins": 4,
            },
        ),
        make_call(
            "call-concurrent-scatter",
            "visualization.scatter",
            {
                "dataset_id": DATASET_ID,
                "x_column": "quantity",
                "y_column": "amount",
            },
        ),
    ]

    results = asyncio.run(dispatcher.dispatch_many(calls))

    assert len(results) == len(calls)
    assert all(result.success for result in results)

    expected_tool_names = [
        call.name
        for call in calls
    ]

    actual_tool_names = [
        result.metadata["tool_name"]
        for result in results
    ]

    assert actual_tool_names == expected_tool_names

    artifact_ids = [
        result.output["artifact"]["artifact_id"]
        for result in results
    ]

    assert len(set(artifact_ids)) == len(artifact_ids)

    filenames = [
        result.output["artifact"]["filename"]
        for result in results
    ]

    for filename in filenames:
        assert (artifact_root / filename).is_file()

    assert len(list(artifact_root.iterdir())) == len(calls)


def test_dispatch_many_messages_keep_matching_tool_call_ids(
    integration_environment: dict[str, Any],
) -> None:
    """Concurrent results should map back to their original ToolCall IDs."""

    dispatcher = integration_environment["dispatcher"]

    calls = [
        make_call(
            "call-concurrent-pie",
            "visualization.pie",
            {
                "dataset_id": DATASET_ID,
                "column": "product",
            },
        ),
        make_call(
            "call-concurrent-box",
            "visualization.box",
            {
                "dataset_id": DATASET_ID,
                "column": "amount",
            },
        ),
    ]

    results = asyncio.run(dispatcher.dispatch_many(calls))

    messages = convert_tool_results_to_messages(
        calls,
        results,
    )

    assert len(messages) == 2

    assert [
        message.tool_call_id
        for message in messages
    ] == [
        "call-concurrent-pie",
        "call-concurrent-box",
    ]

    for message in messages:
        payload = json.loads(message.content)
        assert payload["success"] is True


# ---------------------------------------------------------------------
# Context lifecycle
# ---------------------------------------------------------------------


def test_visualization_dispatch_does_not_close_execution_context(
    integration_environment: dict[str, Any],
) -> None:
    """Visualization tools must not own the ExecutionContext lifecycle."""

    context = integration_environment["context"]
    dispatcher = integration_environment["dispatcher"]

    call = make_call(
        "call-context-lifecycle",
        "visualization.bar",
        {
            "dataset_id": DATASET_ID,
            "column": "department",
        },
    )

    result = asyncio.run(dispatcher.dispatch(call))

    assert_dispatch_success(result)

    assert context.is_closed is False

    dataframe = context.get_dataset(DATASET_ID)

    assert dataframe.shape == (6, 4)

def test_artifact_survives_execution_context_close(
    integration_environment: dict[str, Any],
) -> None:
    """Verify persisted artifacts survive execution context cleanup."""

    context = integration_environment["context"]
    dispatcher = integration_environment["dispatcher"]
    artifact_root = integration_environment["artifact_root"]

    # Generate a real PNG artifact through the complete tool pipeline.
    call = make_call(
        "call-artifact-lifecycle",
        "visualization.bar",
        {
            "dataset_id": DATASET_ID,
            "column": "department",
            "output_format": "png",
        },
    )

    result = asyncio.run(dispatcher.dispatch(call))

    assert_dispatch_success(result)

    artifact = result.output["artifact"]

    assert artifact["format"] == "png"
    assert artifact["media_type"] == "image/png"

    artifact_path = artifact_root / artifact["filename"]

    assert artifact_path.is_file()

    # Preserve the original artifact content for post-cleanup verification.
    original_content = artifact_path.read_bytes()

    assert original_content.startswith(b"\x89PNG\r\n\x1a\n")

    # Closing the execution context should release datasets only.
    context.close()

    assert context.is_closed is True
    assert context.dataset_ids == ()

    with pytest.raises(RuntimeError):
        context.get_dataset(DATASET_ID)

    # Persisted visualization artifacts must remain available.
    assert artifact_path.is_file()
    assert artifact_path.read_bytes() == original_content
    assert artifact_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

def test_dispatch_many_isolates_visualization_failure_and_recovers(
    integration_environment: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed visualization must not affect other batch calls or recovery."""

    context = integration_environment["context"]
    dataframe = integration_environment["dataframe"]
    services = integration_environment["services"]
    dispatcher = integration_environment["dispatcher"]
    artifact_root = integration_environment["artifact_root"]

    original_dataframe = dataframe.copy(deep=True)
    original_create_bar_chart = services.visualization.create_bar_chart

    def fail_bar_chart(*args: Any, **kwargs: Any) -> None:
        raise VisualizationExecutionError(
            "Simulated batch visualization failure"
        )

    # Inject a failure into the synchronous visualization service method.
    # The real Tool.execute() and asyncio.to_thread() paths remain unchanged.
    monkeypatch.setattr(
        services.visualization,
        "create_bar_chart",
        fail_bar_chart,
    )

    calls = [
        make_call(
            "call-batch-failing-bar",
            "visualization.bar",
            {
                "dataset_id": DATASET_ID,
                "column": "department",
                "output_format": "png",
            },
        ),
        make_call(
            "call-batch-success-histogram",
            "visualization.histogram",
            {
                "dataset_id": DATASET_ID,
                "column": "amount",
                "bins": 3,
                "output_format": "svg",
            },
        ),
    ]

    # One failed Tool must not prevent the other batch call from completing.
    results = asyncio.run(dispatcher.dispatch_many(calls))

    assert len(results) == 2

    # Results must preserve the original input order.
    assert results[0].metadata["tool_name"] == "visualization.bar"
    assert results[1].metadata["tool_name"] == "visualization.histogram"

    # The bar chart fails at the VisualizationService boundary.
    failed_result = results[0]

    assert failed_result.success is False
    assert failed_result.output is None
    assert "Simulated batch visualization failure" in failed_result.error
    assert failed_result.metadata["status"] == "failed"
    assert failed_result.metadata["error_type"] == "ToolReportedFailure"

    # The histogram must complete and persist its SVG artifact.
    successful_result = results[1]

    assert_dispatch_success(successful_result)

    histogram_output = successful_result.output
    histogram_artifact = histogram_output["artifact"]

    assert histogram_output["visualization"]["dataset_id"] == DATASET_ID
    assert histogram_artifact["format"] == "svg"
    assert histogram_artifact["media_type"] == "image/svg+xml"
    assert histogram_artifact["size_bytes"] > 0

    histogram_path = artifact_root / histogram_artifact["filename"]

    assert histogram_path.is_file()

    svg_content = histogram_path.read_text(encoding="utf-8")

    assert "<svg" in svg_content
    assert "</svg>" in svg_content

    # The failed bar chart must not leave a partial artifact behind.
    artifact_files = list(artifact_root.iterdir())

    assert artifact_files == [histogram_path]

    # A failed visualization must not damage the shared execution context.
    assert context.is_closed is False
    assert context.get_dataset(DATASET_ID) is dataframe

    pd.testing.assert_frame_equal(
        context.get_dataset(DATASET_ID),
        original_dataframe,
    )

    # Restore the original service method and retry the failed operation.
    monkeypatch.setattr(
        services.visualization,
        "create_bar_chart",
        original_create_bar_chart,
    )

    recovery_call = make_call(
        "call-recovered-bar",
        "visualization.bar",
        {
            "dataset_id": DATASET_ID,
            "column": "department",
            "output_format": "png",
        },
    )

    recovery_result = asyncio.run(
        dispatcher.dispatch(recovery_call)
    )

    # The same Tool should work again after the injected failure is removed.
    assert_dispatch_success(recovery_result)

    recovery_artifact = recovery_result.output["artifact"]

    assert recovery_artifact["format"] == "png"
    assert recovery_artifact["media_type"] == "image/png"

    recovery_path = artifact_root / recovery_artifact["filename"]

    assert recovery_path.is_file()
    assert recovery_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    # Both successful operations should have persisted exactly one artifact.
    assert len(list(artifact_root.iterdir())) == 2

    assert (
        histogram_artifact["artifact_id"]
        != recovery_artifact["artifact_id"]
    )

    # Recovery must preserve the original dataset and context lifecycle.
    assert context.is_closed is False

    pd.testing.assert_frame_equal(
        context.get_dataset(DATASET_ID),
        original_dataframe,
    )
