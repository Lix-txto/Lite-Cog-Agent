"""Tests for the builtin dataset cleaning tool."""

from __future__ import annotations

import asyncio
import json

import pandas as pd
import pytest
from jsonschema import Draft202012Validator

from lite_cog_agent.cleaning.service import CleaningService
from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.execution.dataset_store import DatasetReference
from lite_cog_agent.tools.builtin.cleaning_tools import DatasetCleaningTool


@pytest.fixture
def cleaning_context() -> ExecutionContext:
    """Create a context containing a dataset with missing values."""
    context = ExecutionContext()

    dataframe = pd.DataFrame(
        {
            "name": ["Alice", "Bob", None, "David"],
            "age": [20, 30, 40, 50],
            "city": ["NY", "LA", "SF", "NY"],
        }
    )

    reference = DatasetReference(
        dataset_id="cleaning-test-dataset",
        name="users",
        source="test",
        file_format="csv",
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        columns=tuple(dataframe.columns),
    )

    context.register_dataset(reference, dataframe)

    yield context

    context.close()


@pytest.fixture
def cleaning_service(
    cleaning_context: ExecutionContext,
) -> CleaningService:
    return CleaningService(cleaning_context)


@pytest.fixture
def cleaning_tool(
    cleaning_context: ExecutionContext,
    cleaning_service: CleaningService,
) -> DatasetCleaningTool:
    return DatasetCleaningTool(
        context=cleaning_context,
        cleaning_service=cleaning_service,
    )


def run_tool(tool: DatasetCleaningTool, arguments: dict):
    """Run an async tool from a synchronous pytest test."""
    return asyncio.run(tool.execute(arguments))


def valid_arguments() -> dict:
    """Return a minimal valid cleaning request."""
    return {
        "dataset_id": "cleaning-test-dataset",
        "steps": [
            {
                "operation": "missing.drop",
                "parameters": {"how": "any"},
            }
        ],
    }


def assert_json_compatible(value) -> None:
    """Ensure the result can be serialized by the standard JSON encoder."""
    json.dumps(value, ensure_ascii=False, allow_nan=False)


# ---------------------------------------------------------------------------
# Constructor and metadata
# ---------------------------------------------------------------------------


def test_tool_initializes_with_matching_context(
    cleaning_context: ExecutionContext,
    cleaning_service: CleaningService,
) -> None:
    tool = DatasetCleaningTool(
        cleaning_context,
        cleaning_service,
    )

    assert tool.name == "dataset.clean"


def test_tool_rejects_invalid_context(
    cleaning_service: CleaningService,
) -> None:
    with pytest.raises(
        TypeError,
        match="context must be an ExecutionContext",
    ):
        DatasetCleaningTool(
            context=object(),  # type: ignore[arg-type]
            cleaning_service=cleaning_service,
        )


def test_tool_rejects_invalid_cleaning_service(
    cleaning_context: ExecutionContext,
) -> None:
    with pytest.raises(
        TypeError,
        match="cleaning_service must be a CleaningService",
    ):
        DatasetCleaningTool(
            context=cleaning_context,
            cleaning_service=object(),  # type: ignore[arg-type]
        )


def test_tool_rejects_service_using_another_context() -> None:
    first_context = ExecutionContext()
    second_context = ExecutionContext()

    service = CleaningService(second_context)

    with pytest.raises(
        ValueError,
        match="same ExecutionContext",
    ):
        DatasetCleaningTool(
            context=first_context,
            cleaning_service=service,
        )

    first_context.close()
    second_context.close()


def test_tool_rejects_closed_context() -> None:
    context = ExecutionContext()
    service = CleaningService(context)
    context.close()

    with pytest.raises(
        RuntimeError,
        match="ExecutionContext is closed",
    ):
        DatasetCleaningTool(
            context=context,
            cleaning_service=service,
        )


def test_tool_metadata(cleaning_tool: DatasetCleaningTool) -> None:
    assert cleaning_tool.name == "dataset.clean"
    assert "ordered sequence" in cleaning_tool.description
    assert "output_dataset_id" in cleaning_tool.description

    schema = cleaning_tool.parameters

    assert schema["type"] == "object"
    assert schema["required"] == ["dataset_id", "steps"]
    assert schema["additionalProperties"] is False

    assert schema["properties"]["dataset_id"]["type"] == "string"
    assert schema["properties"]["steps"]["type"] == "array"
    assert schema["properties"]["steps"]["minItems"] == 1

    operation_schema = (
        schema["properties"]["steps"]["items"]
        ["properties"]["operation"]
    )

    assert set(operation_schema["enum"]) == {
        "missing.drop",
        "missing.fill",
        "duplicates.drop",
        "columns.select",
        "columns.drop",
        "columns.rename",
        "types.convert",
    }


def test_tool_schema_is_valid_json_schema(
    cleaning_tool: DatasetCleaningTool,
) -> None:
    Draft202012Validator.check_schema(cleaning_tool.parameters)


def test_tool_definition_can_be_generated(
    cleaning_tool: DatasetCleaningTool,
) -> None:
    definition = cleaning_tool.to_definition()

    assert definition.name == "dataset.clean"
    assert definition.description == cleaning_tool.description
    assert definition.parameters == cleaning_tool.parameters


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"dataset_id": "cleaning-test-dataset"},
        {"steps": [{"operation": "missing.drop", "parameters": {}}]},
        {"dataset_id": "", "steps": []},
        {"dataset_id": "   ", "steps": []},
        {"dataset_id": 123, "steps": []},
        {"dataset_id": "cleaning-test-dataset", "steps": []},
        {
            "dataset_id": "cleaning-test-dataset",
            "steps": "missing.drop",
        },
        {
            "dataset_id": "cleaning-test-dataset",
            "steps": [None],
        },
        {
            "dataset_id": "cleaning-test-dataset",
            "steps": [{"operation": "unknown.operation", "parameters": {}}],
        },
        {
            "dataset_id": "cleaning-test-dataset",
            "steps": [{"operation": "missing.drop"}],
        },
        {
            "dataset_id": "cleaning-test-dataset",
            "steps": [
                {
                    "operation": "missing.drop",
                    "parameters": [],
                }
            ],
        },
        {
            "dataset_id": "cleaning-test-dataset",
            "steps": [
                {
                    "operation": "missing.drop",
                    "parameters": {},
                    "unexpected": True,
                }
            ],
        },
        {
            "dataset_id": "cleaning-test-dataset",
            "steps": [
                {
                    "operation": "missing.drop",
                    "parameters": {},
                }
            ],
            "unexpected": True,
        },
        {
            "dataset_id": "cleaning-test-dataset",
            "steps": [
                {
                    "operation": "missing.drop",
                    "parameters": {},
                }
            ],
            "output_dataset_name": "",
        },
        {
            "dataset_id": "cleaning-test-dataset",
            "steps": [
                {
                    "operation": "missing.drop",
                    "parameters": {},
                }
            ],
            "output_dataset_name": 123,
        },
    ],
)
def test_rejects_invalid_arguments(
    cleaning_tool: DatasetCleaningTool,
    arguments: dict,
) -> None:
    result = run_tool(cleaning_tool, arguments)

    assert result.success is False
    assert result.error


def test_rejects_non_object_arguments(
    cleaning_tool: DatasetCleaningTool,
) -> None:
    result = run_tool(
        cleaning_tool,
        [],  # type: ignore[arg-type]
    )

    assert result.success is False
    assert "object" in result.error


def test_rejects_dataset_not_in_context(
    cleaning_tool: DatasetCleaningTool,
) -> None:
    arguments = valid_arguments()
    arguments["dataset_id"] = "unknown-dataset"

    result = run_tool(cleaning_tool, arguments)

    assert result.success is False
    assert result.error
    assert result.metadata["tool"] == "dataset.clean"


def test_rejects_execution_after_context_is_closed(
    cleaning_context: ExecutionContext,
    cleaning_service: CleaningService,
) -> None:
    tool = DatasetCleaningTool(
        cleaning_context,
        cleaning_service,
    )

    cleaning_context.close()

    result = run_tool(tool, valid_arguments())

    assert result.success is False
    assert result.error == "ExecutionContext is closed."


# ---------------------------------------------------------------------------
# Successful cleaning
# ---------------------------------------------------------------------------


def test_clean_tool_returns_cleaning_result(
    cleaning_tool: DatasetCleaningTool,
    cleaning_context: ExecutionContext,
) -> None:
    source = cleaning_context.get_dataset("cleaning-test-dataset")
    original = source.copy(deep=True)

    result = run_tool(cleaning_tool, valid_arguments())

    assert result.success is True
    assert result.error is None

    output = result.output

    assert output["status"] == "success"
    assert output["source_dataset_id"] == "cleaning-test-dataset"
    assert output["input_row_count"] == 4
    assert output["input_column_count"] == 3
    assert output["output_row_count"] == 3
    assert output["output_column_count"] == 3
    assert output["output_dataset_id"]

    assert len(output["operations"]) == 1
    assert output["operations"][0]["operation"] == "missing.drop"
    assert output["operations"][0]["status"] == "success"

    assert result.metadata["tool"] == "dataset.clean"
    assert result.metadata["dataset_id"] == "cleaning-test-dataset"
    assert (
        result.metadata["output_dataset_id"]
        == output["output_dataset_id"]
    )
    assert result.metadata["status"] == "success"

    assert_json_compatible(output)


def test_clean_tool_registers_output_in_same_context(
    cleaning_tool: DatasetCleaningTool,
    cleaning_context: ExecutionContext,
) -> None:
    result = run_tool(cleaning_tool, valid_arguments())

    assert result.success is True

    output_dataset_id = result.output["output_dataset_id"]

    cleaned_dataframe = cleaning_context.get_dataset(
        output_dataset_id
    )
    output_reference = cleaning_context.get_dataset_reference(
        output_dataset_id
    )

    assert isinstance(cleaned_dataframe, pd.DataFrame)
    assert cleaned_dataframe.shape == (3, 3)
    assert output_reference.source == "cleaning"
    assert output_reference.metadata["source_dataset_id"] == (
        "cleaning-test-dataset"
    )


def test_clean_tool_does_not_modify_source_dataset(
    cleaning_tool: DatasetCleaningTool,
    cleaning_context: ExecutionContext,
) -> None:
    source = cleaning_context.get_dataset("cleaning-test-dataset")
    original = source.copy(deep=True)

    result = run_tool(cleaning_tool, valid_arguments())

    assert result.success is True

    pd.testing.assert_frame_equal(source, original)


def test_clean_tool_supports_custom_output_name(
    cleaning_tool: DatasetCleaningTool,
    cleaning_context: ExecutionContext,
) -> None:
    arguments = valid_arguments()
    arguments["output_dataset_name"] = "users_without_missing_values"

    result = run_tool(cleaning_tool, arguments)

    assert result.success is True

    output_reference = cleaning_context.get_dataset_reference(
        result.output["output_dataset_id"]
    )

    assert output_reference.name == "users_without_missing_values"


def test_clean_tool_supports_multiple_ordered_steps(
    cleaning_tool: DatasetCleaningTool,
    cleaning_context: ExecutionContext,
) -> None:
    arguments = {
        "dataset_id": "cleaning-test-dataset",
        "steps": [
            {
                "operation": "missing.drop",
                "parameters": {"how": "any"},
            },
            {
                "operation": "columns.select",
                "parameters": {"columns": ["name", "age"]},
            },
        ],
    }

    result = run_tool(cleaning_tool, arguments)

    assert result.success is True
    assert len(result.output["operations"]) == 2
    assert result.output["output_row_count"] == 3
    assert result.output["output_column_count"] == 2

    cleaned = cleaning_context.get_dataset(
        result.output["output_dataset_id"]
    )

    assert list(cleaned.columns) == ["name", "age"]


# ---------------------------------------------------------------------------
# Cleaning failures
# ---------------------------------------------------------------------------


def test_returns_failure_when_cleaning_operation_fails(
    cleaning_tool: DatasetCleaningTool,
    cleaning_context: ExecutionContext,
) -> None:
    original_ids = cleaning_context.dataset_ids
    original_count = len(cleaning_context.dataset_store)

    arguments = {
        "dataset_id": "cleaning-test-dataset",
        "steps": [
            {
                "operation": "missing.drop",
                "parameters": {"how": "invalid"},
            }
        ],
    }

    result = run_tool(cleaning_tool, arguments)

    assert result.success is False
    assert result.error
    assert result.metadata["tool"] == "dataset.clean"

    cleaning_result = result.metadata["cleaning_result"]

    assert cleaning_result["status"] == "failed"
    assert cleaning_result["output_dataset_id"] is None
    assert cleaning_result["error"] is not None

    assert cleaning_context.dataset_ids == original_ids
    assert len(cleaning_context.dataset_store) == original_count


def test_returns_failure_when_cleaning_parameters_are_invalid(
    cleaning_tool: DatasetCleaningTool,
) -> None:
    arguments = {
        "dataset_id": "cleaning-test-dataset",
        "steps": [
            {
                "operation": "missing.fill",
                "parameters": {},
            }
        ],
    }

    result = run_tool(cleaning_tool, arguments)

    assert result.success is False
    assert result.metadata["cleaning_result"]["status"] == "failed"


def test_failed_cleaning_does_not_publish_output_dataset(
    cleaning_tool: DatasetCleaningTool,
    cleaning_context: ExecutionContext,
) -> None:
    original_ids = cleaning_context.dataset_ids

    arguments = {
        "dataset_id": "cleaning-test-dataset",
        "steps": [
            {
                "operation": "missing.drop",
                "parameters": {"how": "invalid"},
            }
        ],
    }

    result = run_tool(cleaning_tool, arguments)

    assert result.success is False
    assert cleaning_context.dataset_ids == original_ids


# ---------------------------------------------------------------------------
# Async execution
# ---------------------------------------------------------------------------


def test_cleaning_service_runs_through_asyncio_to_thread(
    cleaning_tool: DatasetCleaningTool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lite_cog_agent.tools.builtin.cleaning_tools as cleaning_tools_module

    original_to_thread = asyncio.to_thread
    calls: list[tuple[object, tuple, dict]] = []

    async def tracking_to_thread(func, /, *args, **kwargs):
        calls.append((func, args, kwargs))
        return await original_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(
        cleaning_tools_module.asyncio,
        "to_thread",
        tracking_to_thread,
    )

    result = run_tool(cleaning_tool, valid_arguments())

    assert result.success is True
    assert len(calls) == 1
    assert calls[0][0] == cleaning_tool._cleaning_service.clean
