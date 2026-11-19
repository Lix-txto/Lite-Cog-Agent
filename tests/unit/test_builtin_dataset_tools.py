"""Tests for builtin dataset tools."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest

from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.loading.service import DatasetLoadingService
from lite_cog_agent.tools.builtin.dataset_tools import DatasetLoadTool


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "datasets"
    root.mkdir()

    (root / "sales.csv").write_text(
        "department,amount\nSales,100\nSupport,200\n",
        encoding="utf-8",
    )

    (root / "records.json").write_text(
        json.dumps(
            [
                {"department": "Sales", "amount": 100},
                {"department": "Support", "amount": 200},
            ]
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "department": ["Sales", "Support"],
            "amount": [100, 200],
        }
    ).to_excel(
        root / "sales.xlsx",
        index=False,
    )    

    return root


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext()


@pytest.fixture
def tool(
    context: ExecutionContext,
    dataset_root: Path,
) -> DatasetLoadTool:
    return DatasetLoadTool(
        context=context,
        loading_service=DatasetLoadingService(),
        dataset_root=dataset_root,
    )


def test_tool_metadata() -> None:
    context = ExecutionContext()
    tool = DatasetLoadTool(
        context,
        DatasetLoadingService(),
        Path("data/datasets"),
    )

    assert tool.name == "dataset.load"
    assert "CSV" in tool.description
    assert tool.parameters["type"] == "object"
    assert tool.parameters["required"] == ["file_path"]
    assert tool.parameters["additionalProperties"] is False


def test_loads_csv_into_context(
    tool: DatasetLoadTool,
    context: ExecutionContext,
) -> None:
    result = asyncio.run(
        tool.execute({"file_path": "sales.csv"})
    )

    assert result.success is True
    assert result.error is None

    output = result.output
    assert output["name"] == "sales"
    assert output["file_format"] == "csv"
    assert output["row_count"] == 2
    assert output["column_count"] == 2
    assert output["columns"] == ["department", "amount"]
    assert "dataset_id" in output
    assert "source" in output

    dataset = context.get_dataset(output["dataset_id"])
    assert isinstance(dataset, pd.DataFrame)
    assert dataset.shape == (2, 2)


def test_loads_json_into_context(
    tool: DatasetLoadTool,
    context: ExecutionContext,
) -> None:
    result = asyncio.run(
        tool.execute({"file_path": "records.json"})
    )

    assert result.success is True
    assert result.output["file_format"] == "json"
    assert result.output["row_count"] == 2
    assert context.get_dataset(result.output["dataset_id"]).shape == (2, 2)


def test_loads_file_with_custom_name(
    tool: DatasetLoadTool,
) -> None:
    result = asyncio.run(
        tool.execute(
            {
                "file_path": "sales.csv",
                "name": "Quarterly Sales",
            }
        )
    )

    assert result.success is True
    assert result.output["name"] == "Quarterly Sales"


def test_loads_file_with_explicit_format(
    tool: DatasetLoadTool,
) -> None:
    result = asyncio.run(
        tool.execute(
            {
                "file_path": "sales.csv",
                "file_format": "CSV",
            }
        )
    )

    assert result.success is True
    assert result.output["file_format"] == "csv"


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"file_path": ""},
        {"file_path": "   "},
        {"file_path": 123},
        {"file_path": "sales.csv", "unexpected": True},
    ],
)
def test_rejects_invalid_arguments(
    tool: DatasetLoadTool,
    arguments: dict,
) -> None:
    result = asyncio.run(tool.execute(arguments))

    assert result.success is False
    assert result.error


def test_rejects_non_object_arguments(
    tool: DatasetLoadTool,
) -> None:
    result = asyncio.run(tool.execute([]))  # type: ignore[arg-type]

    assert result.success is False
    assert "object" in result.error


def test_rejects_invalid_options(
    tool: DatasetLoadTool,
) -> None:
    result = asyncio.run(
        tool.execute(
            {
                "file_path": "sales.csv",
                "options": [],
            }
        )
    )

    assert result.success is False
    assert "options" in result.error


@pytest.mark.parametrize(
    "file_format",
    ["xls", "parquet", "exe", "unknown"],
)
def test_rejects_unsupported_explicit_format(
    tool: DatasetLoadTool,
    file_format: str,
) -> None:
    result = asyncio.run(
        tool.execute(
            {
                "file_path": "sales.csv",
                "file_format": file_format,
            }
        )
    )

    assert result.success is False
    assert "file_format" in result.error


def test_rejects_nonexistent_file(
    tool: DatasetLoadTool,
) -> None:
    result = asyncio.run(
        tool.execute({"file_path": "missing.csv"})
    )

    assert result.success is False
    assert result.error == "Dataset file does not exist"


def test_rejects_path_traversal(
    tool: DatasetLoadTool,
    tmp_path: Path,
) -> None:
    outside_file = tmp_path / "outside.csv"
    outside_file.write_text("value\n1\n", encoding="utf-8")

    result = asyncio.run(
        tool.execute({"file_path": "../outside.csv"})
    )

    assert result.success is False
    assert "outside the allowed" in result.error


def test_rejects_absolute_path_outside_root(
    tool: DatasetLoadTool,
    tmp_path: Path,
) -> None:
    outside_file = tmp_path / "outside.csv"
    outside_file.write_text("value\n1\n", encoding="utf-8")

    result = asyncio.run(
        tool.execute({"file_path": str(outside_file)})
    )

    assert result.success is False
    assert "outside the allowed" in result.error


def test_rejects_symlink_pointing_outside_root(
    tool: DatasetLoadTool,
    dataset_root: Path,
    tmp_path: Path,
) -> None:
    outside_file = tmp_path / "outside.csv"
    outside_file.write_text("value\n1\n", encoding="utf-8")

    link = dataset_root / "outside-link.csv"

    try:
        link.symlink_to(outside_file)
    except (OSError, NotImplementedError):
        pytest.skip("Symbolic links are not supported in this environment")

    result = asyncio.run(
        tool.execute({"file_path": "outside-link.csv"})
    )

    assert result.success is False
    assert "outside the allowed" in result.error


def test_rejects_directory(
    tool: DatasetLoadTool,
    dataset_root: Path,
) -> None:
    (dataset_root / "folder").mkdir()

    result = asyncio.run(
        tool.execute({"file_path": "folder"})
    )

    assert result.success is False
    assert "regular file" in result.error


def test_rejects_unsupported_extension(
    tool: DatasetLoadTool,
    dataset_root: Path,
) -> None:
    (dataset_root / "notes.txt").write_text(
        "not a dataset",
        encoding="utf-8",
    )

    result = asyncio.run(
        tool.execute({"file_path": "notes.txt"})
    )

    assert result.success is False
    assert "Unsupported dataset extension" in result.error


def test_failed_load_does_not_register_dataset(
    tool: DatasetLoadTool,
    context: ExecutionContext,
) -> None:
    result = asyncio.run(
        tool.execute({"file_path": "invalid.csv"})
    )

    assert result.success is False
    assert context.dataset_ids == ()


def test_result_does_not_expose_absolute_path(
    tool: DatasetLoadTool,
) -> None:
    result = asyncio.run(
        tool.execute({"file_path": "sales.csv"})
    )

    assert result.success is True

    serialized = json.dumps(result.output)
    assert str(tool._dataset_root) not in serialized


def test_result_is_strict_json_compatible(
    tool: DatasetLoadTool,
) -> None:
    result = asyncio.run(
        tool.execute({"file_path": "sales.csv"})
    )

    assert result.success is True

    encoded = json.dumps(
        result.output,
        ensure_ascii=False,
        allow_nan=False,
    )

    assert json.loads(encoded) == result.output


def test_rejects_closed_context(
    dataset_root: Path,
) -> None:
    context = ExecutionContext()
    context.close()

    with pytest.raises(RuntimeError, match="closed"):
        DatasetLoadTool(
            context,
            DatasetLoadingService(),
            dataset_root,
        )


def test_rejects_invalid_service(
    context: ExecutionContext,
    dataset_root: Path,
) -> None:
    with pytest.raises(TypeError, match="DatasetLoadingService"):
        DatasetLoadTool(
            context,
            object(),  # type: ignore[arg-type]
            dataset_root,
        )

def test_loads_xlsx_with_automatic_format_detection(
    tool: DatasetLoadTool,
    context: ExecutionContext,
) -> None:
    result = asyncio.run(
        tool.execute({"file_path": "sales.xlsx"})
    )

    assert result.success is True
    assert result.output["file_format"] == "xlsx"
    assert result.output["row_count"] == 2
    assert result.output["column_count"] == 2

    dataset = context.get_dataset(result.output["dataset_id"])
    assert isinstance(dataset, pd.DataFrame)
    assert dataset.shape == (2, 2)

    reference = context.get_dataset_reference(
        result.output["dataset_id"]
    )
    assert reference.file_format == "excel"


def test_loads_xlsx_with_explicit_public_format(
    tool: DatasetLoadTool,
) -> None:
    result = asyncio.run(
        tool.execute(
            {
                "file_path": "sales.xlsx",
                "file_format": "xlsx",
            }
        )
    )

    assert result.success is True
    assert result.output["file_format"] == "xlsx"


@pytest.mark.parametrize(
    ("file_path", "file_format"),
    [
        ("sales.csv", "json"),
        ("sales.csv", "xlsx"),
        ("sales.xlsx", "csv"),
        ("sales.xlsx", "json"),
        ("records.json", "csv"),
        ("records.json", "xlsx"),
    ],
)
def test_rejects_format_extension_mismatch(
    tool: DatasetLoadTool,
    context: ExecutionContext,
    file_path: str,
    file_format: str,
) -> None:
    result = asyncio.run(
        tool.execute(
            {
                "file_path": file_path,
                "file_format": file_format,
            }
        )
    )

    assert result.success is False
    assert "does not match" in result.error
    assert context.dataset_ids == ()


def test_rejects_internal_excel_format_from_tool_arguments(
    tool: DatasetLoadTool,
) -> None:
    result = asyncio.run(
        tool.execute(
            {
                "file_path": "sales.xlsx",
                "file_format": "excel",
            }
        )
    )

    assert result.success is False
    assert "file_format must be one of" in result.error


def test_loads_csv_with_supported_options(
    tool: DatasetLoadTool,
) -> None:
    result = asyncio.run(
        tool.execute(
            {
                "file_path": "sales.csv",
                "options": {
                    "delimiter": ",",
                    "encoding": "utf-8",
                    "nrows": 1,
                },
            }
        )
    )

    assert result.success is True
    assert result.output["row_count"] == 1


def test_loads_xlsx_with_supported_options(
    tool: DatasetLoadTool,
) -> None:
    result = asyncio.run(
        tool.execute(
            {
                "file_path": "sales.xlsx",
                "options": {
                    "sheet_name": 0,
                    "header": 0,
                    "nrows": 1,
                },
            }
        )
    )

    assert result.success is True
    assert result.output["row_count"] == 1


def test_loads_json_with_supported_options(
    tool: DatasetLoadTool,
) -> None:
    result = asyncio.run(
        tool.execute(
            {
                "file_path": "records.json",
                "options": {},
            }
        )
    )

    assert result.success is True
    assert result.output["row_count"] == 2