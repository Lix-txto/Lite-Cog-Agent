"""Builtin tools for dataset operations."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.loading.contracts import DatasetLoadRequest
from lite_cog_agent.loading.service import DatasetLoadingService
from lite_cog_agent.tools.base import BaseTool
from lite_cog_agent.tools.schemas import ToolResult


class DatasetLoadTool(BaseTool):
    """Load a local dataset into the current execution context."""

    PUBLIC_TO_INTERNAL_FORMAT = {
        "csv": "csv",
        "xlsx": "excel",
        "json": "json",
    }

    EXTENSION_TO_PUBLIC_FORMAT = {
        ".csv": "csv",
        ".xlsx": "xlsx",
        ".json": "json",
    }

    def __init__(
        self,
        context: ExecutionContext,
        loading_service: DatasetLoadingService,
        dataset_root: Path | str,
    ) -> None:
        if not isinstance(context, ExecutionContext):
            raise TypeError("context must be an ExecutionContext")

        if context.is_closed:
            raise RuntimeError("ExecutionContext is closed")

        if not isinstance(loading_service, DatasetLoadingService):
            raise TypeError(
                "loading_service must be a DatasetLoadingService"
            )

        if not isinstance(dataset_root, (str, Path)):
            raise TypeError("dataset_root must be a Path or string")

        root = Path(dataset_root).expanduser().resolve()

        if root.exists() and not root.is_dir():
            raise ValueError("dataset_root must be a directory")

        self._context = context
        self._loading_service = loading_service
        self._dataset_root = root

    @property
    def name(self) -> str:
        return "dataset.load"

    @property
    def description(self) -> str:
        return (
            "Load a CSV, XLSX, or JSON dataset from the allowed datasets "
            "directory into the current analysis session. Returns dataset "
            "metadata and a dataset_id for subsequent analysis tools."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "Path relative to the datasets directory, "
                        "for example sales.csv."
                    ),
                },
                "name": {
                    "type": ["string", "null"],
                    "description": "Optional display name for the dataset.",
                },
                "file_format": {
                    "type": ["string", "null"],
                    "enum": ["csv", "xlsx", "json", None],
                    "description": (
                        "Optional public file format. Must match the "
                        "file extension. If omitted, inferred from "
                        "the extension."
                    ),
                },
                "options": {
                    "type": "object",
                    "description": (
                        "Loader-specific options. CSV: encoding, delimiter, "
                        "nrows. XLSX: sheet_name, header, nrows. "
                        "JSON: records_path."
                    ),
                    "properties": {
                        "encoding": {"type": "string"},
                        "delimiter": {"type": "string"},
                        "nrows": {
                            "type": ["integer", "null"],
                            "minimum": 1,
                        },
                        "sheet_name": {
                            "type": ["string", "integer"],
                        },
                        "header": {
                            "type": ["integer", "null"],
                            "minimum": 0,
                        },
                        "records_path": {
                            "type": ["string", "null"],
                        },
                    },
                    "additionalProperties": False,
                },
            },
            "required": ["file_path"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Validate the requested file and load it asynchronously."""

        if not isinstance(arguments, dict):
            return ToolResult.fail("Tool arguments must be an object.")

        try:
            request = self._build_request(arguments)
            resolved_path = self._resolve_dataset_path(request.file_path)

            public_format = self.EXTENSION_TO_PUBLIC_FORMAT[
                resolved_path.suffix.lower()
            ]

            if (
                request.file_format is not None
                and request.file_format != public_format
            ):
                return ToolResult.fail(
                    "file_format does not match the file extension."
                )

            internal_format = self.PUBLIC_TO_INTERNAL_FORMAT[
                public_format
            ]

            safe_request = DatasetLoadRequest(
                file_path=resolved_path,
                name=request.name,
                file_format=internal_format,
                options=request.options,
            )

            reference = await asyncio.to_thread(
                self._loading_service.load_into_context,
                safe_request,
                self._context,
            )

            output = {
                "dataset_id": reference.dataset_id,
                "name": reference.name,
                "file_format": public_format,
                "row_count": reference.row_count,
                "column_count": reference.column_count,
                "columns": list(reference.columns),
                "source": Path(reference.source).name,
            }

            return ToolResult.ok(
                output=output,
                metadata={
                    "tool": self.name,
                    "context_id": self._context.context_id,
                },
            )

        except (TypeError, ValueError) as exc:
            return ToolResult.fail(str(exc))

        except Exception as exc:
            return ToolResult.fail(
                f"Dataset loading failed: {type(exc).__name__}."
            )

    def _build_request(
        self,
        arguments: dict[str, Any],
    ) -> DatasetLoadRequest:
        allowed_keys = {
            "file_path",
            "name",
            "file_format",
            "options",
        }

        unknown_keys = set(arguments) - allowed_keys

        if unknown_keys:
            raise ValueError(
                "Unknown arguments: "
                + ", ".join(sorted(unknown_keys))
            )

        file_path = arguments.get("file_path")

        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path must be a non-empty string")

        name = arguments.get("name")

        if name is not None and not isinstance(name, str):
            raise TypeError("name must be a string or None")

        file_format = arguments.get("file_format")

        if file_format is not None:
            if not isinstance(file_format, str):
                raise TypeError("file_format must be a string or None")

            file_format = file_format.strip().lower().lstrip(".")

            if file_format not in self.PUBLIC_TO_INTERNAL_FORMAT:
                raise ValueError(
                    "file_format must be one of: csv, xlsx, json"
                )

        options = arguments.get("options", {})

        if not isinstance(options, dict):
            raise TypeError("options must be an object")

        return DatasetLoadRequest(
            file_path=file_path,
            name=name,
            file_format=file_format,
            options=options,
        )

    def _resolve_dataset_path(self, file_path: Path) -> Path:
        """Resolve a requested path and enforce the allowed root."""

        if not isinstance(file_path, Path):
            raise TypeError("file_path must be a Path")

        candidate = (
            file_path
            if file_path.is_absolute()
            else self._dataset_root / file_path
        )

        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError("Dataset file does not exist") from exc
        except (OSError, RuntimeError) as exc:
            raise ValueError("Dataset path cannot be resolved") from exc

        try:
            resolved.relative_to(self._dataset_root)
        except ValueError as exc:
            raise ValueError(
                "Dataset path is outside the allowed datasets directory"
            ) from exc

        if not resolved.is_file():
            raise ValueError("Dataset path must reference a regular file")

        if resolved.suffix.lower() not in self.EXTENSION_TO_PUBLIC_FORMAT:
            raise ValueError(
                "Unsupported dataset extension. "
                "Supported extensions: .csv, .xlsx, .json"
            )

        return resolved
