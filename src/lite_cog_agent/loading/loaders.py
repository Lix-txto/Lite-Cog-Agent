"""Dataset file loaders."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from lite_cog_agent.execution.dataset_store import DatasetReference
from lite_cog_agent.loading.config import DatasetLoadingConfig
from lite_cog_agent.loading.contracts import (
    DatasetLoadRequest,
    LoadedDataset,
)
from lite_cog_agent.loading.exceptions import (
    DatasetFileNotFoundError,
    DatasetFileTooLargeError,
    DatasetParseError,
    DatasetValidationError,
    UnsupportedFileFormatError,
)


class CSVLoader:
    """Load CSV files into pandas DataFrames."""

    def __init__(
        self,
        config: DatasetLoadingConfig | None = None,
    ) -> None:
        self.config = config or DatasetLoadingConfig()

    def load(
        self,
        request: DatasetLoadRequest,
    ) -> LoadedDataset:
        """Load a CSV file and return its data and metadata."""

        if not isinstance(request, DatasetLoadRequest):
            raise TypeError("request must be a DatasetLoadRequest")

        file_path = request.file_path

        self._validate_file(file_path)

        file_format = request.file_format or "csv"

        if file_format.lower() != "csv":
            raise UnsupportedFileFormatError(
                f"CSVLoader cannot load format: {file_format}"
            )

        options = dict(request.options)

        encoding = options.pop(
            "encoding",
            self.config.default_encoding,
        )

        delimiter = options.pop(
            "delimiter",
            self.config.default_delimiter,
        )

        requested_nrows = options.pop("nrows", None)

        if not isinstance(encoding, str) or not encoding.strip():
            raise DatasetValidationError(
                "encoding must be a non-empty string"
            )

        if not isinstance(delimiter, str) or len(delimiter) != 1:
            raise DatasetValidationError(
                "delimiter must contain exactly one character"
            )

        if requested_nrows is not None:
            if (
                isinstance(requested_nrows, bool)
                or not isinstance(requested_nrows, int)
                or requested_nrows <= 0
            ):
                raise DatasetValidationError(
                    "nrows must be a positive integer"
                )

            nrows = min(
                requested_nrows,
                self.config.max_rows,
            )
        else:
            nrows = self.config.max_rows

        if options:
            unsupported_options = ", ".join(sorted(options))

            raise DatasetValidationError(
                f"Unsupported CSV options: {unsupported_options}"
            )

        try:
            dataframe = pd.read_csv(
                file_path,
                encoding=encoding,
                sep=delimiter,
                nrows=nrows,
            )

        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            raise DatasetParseError(
                f"Failed to parse CSV file: {file_path.name}"
            ) from exc

        if dataframe.empty:
            raise DatasetValidationError(
                "CSV file does not contain any data rows"
            )

        if len(dataframe.columns) > self.config.max_columns:
            raise DatasetValidationError(
                f"CSV contains too many columns: "
                f"{len(dataframe.columns)}"
            )

        columns = tuple(str(column) for column in dataframe.columns)

        reference = DatasetReference(
            name=request.name or file_path.stem,
            source=str(file_path.resolve()),
            file_format="csv",
            row_count=len(dataframe),
            column_count=len(dataframe.columns),
            columns=columns,
            metadata={
                "encoding": encoding,
                "delimiter": delimiter,
                "file_size_bytes": file_path.stat().st_size,
            },
        )

        return LoadedDataset(
            reference=reference,
            data=dataframe,
        )

    def _validate_file(self, file_path: Path) -> None:
        """Validate the local file before parsing."""

        if not file_path.exists():
            raise DatasetFileNotFoundError(
                f"Dataset file does not exist: {file_path}"
            )

        if not file_path.is_file():
            raise DatasetValidationError(
                f"Dataset path is not a regular file: {file_path}"
            )

        file_size = file_path.stat().st_size

        if file_size > self.config.max_file_size_bytes:
            raise DatasetFileTooLargeError(
                f"Dataset file exceeds maximum size of "
                f"{self.config.max_file_size_bytes} bytes"
            )


class ExcelLoader:
    """Load .xlsx files into pandas DataFrames."""

    def __init__(
        self,
        config: DatasetLoadingConfig | None = None,
    ) -> None:
        self.config = config or DatasetLoadingConfig()

    def load(
        self,
        request: DatasetLoadRequest,
    ) -> LoadedDataset:
        """Load an Excel worksheet and return its data and metadata."""

        if not isinstance(request, DatasetLoadRequest):
            raise TypeError("request must be a DatasetLoadRequest")

        file_path = request.file_path
        self._validate_file(file_path)

        file_format = request.file_format or "excel"

        if file_format.lower() != "excel":
            raise UnsupportedFileFormatError(
                f"ExcelLoader cannot load format: {file_format}"
            )

        options = dict(request.options)

        sheet_name = options.pop("sheet_name", 0)
        header = options.pop("header", 0)
        requested_nrows = options.pop("nrows", None)

        self._validate_sheet_name(sheet_name)
        self._validate_header(header)

        if requested_nrows is not None:
            if (
                isinstance(requested_nrows, bool)
                or not isinstance(requested_nrows, int)
                or requested_nrows <= 0
            ):
                raise DatasetValidationError(
                    "nrows must be a positive integer"
                )

            nrows = min(requested_nrows, self.config.max_rows)
        else:
            nrows = self.config.max_rows

        if options:
            unsupported_options = ", ".join(sorted(options))
            raise DatasetValidationError(
                f"Unsupported Excel options: {unsupported_options}"
            )

        try:
            dataframe = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
                header=header,
                nrows=nrows,
                engine="openpyxl",
            )
        except (ValueError, OSError, ImportError) as exc:
            raise DatasetParseError(
                f"Failed to parse Excel file: {file_path.name}"
            ) from exc
        except Exception as exc:
            # openpyxl may raise format-specific exceptions for
            # malformed or corrupted workbook files.
            raise DatasetParseError(
                f"Failed to parse Excel file: {file_path.name}"
            ) from exc

        if dataframe.empty:
            raise DatasetValidationError(
                "Excel worksheet does not contain any data rows"
            )

        if len(dataframe.columns) > self.config.max_columns:
            raise DatasetValidationError(
                f"Excel worksheet contains too many columns: "
                f"{len(dataframe.columns)}"
            )

        columns = tuple(str(column) for column in dataframe.columns)

        reference = DatasetReference(
            name=request.name or file_path.stem,
            source=str(file_path.resolve()),
            file_format="excel",
            row_count=len(dataframe),
            column_count=len(dataframe.columns),
            columns=columns,
            metadata={
                "engine": "openpyxl",
                "sheet_name": sheet_name,
                "header": header,
                "file_size_bytes": file_path.stat().st_size,
            },
        )

        return LoadedDataset(
            reference=reference,
            data=dataframe,
        )

    @staticmethod
    def _validate_sheet_name(sheet_name: Any) -> None:
        """Validate a worksheet name or zero-based worksheet index."""

        if isinstance(sheet_name, bool):
            raise DatasetValidationError(
                "sheet_name must be a string or non-negative integer"
            )

        if isinstance(sheet_name, int):
            if sheet_name < 0:
                raise DatasetValidationError(
                    "sheet_name index cannot be negative"
                )
            return

        if isinstance(sheet_name, str):
            if not sheet_name.strip():
                raise DatasetValidationError(
                    "sheet_name cannot be empty"
                )
            return

        raise DatasetValidationError(
            "sheet_name must be a string or non-negative integer"
        )

    @staticmethod
    def _validate_header(header: Any) -> None:
        """Validate the header row index."""

        if header is None:
            return

        if (
            isinstance(header, bool)
            or not isinstance(header, int)
            or header < 0
        ):
            raise DatasetValidationError(
                "header must be a non-negative integer or None"
            )

    def _validate_file(self, file_path: Path) -> None:
        """Validate the local Excel file before parsing."""

        if not file_path.exists():
            raise DatasetFileNotFoundError(
                f"Dataset file does not exist: {file_path}"
            )

        if not file_path.is_file():
            raise DatasetValidationError(
                f"Dataset path is not a regular file: {file_path}"
            )

        file_size = file_path.stat().st_size

        if file_size > self.config.max_file_size_bytes:
            raise DatasetFileTooLargeError(
                f"Dataset file exceeds maximum size of "
                f"{self.config.max_file_size_bytes} bytes"
            )


class JSONLoader:
    """Load tabular JSON files into pandas DataFrames."""

    def __init__(
        self,
        config: DatasetLoadingConfig | None = None,
    ) -> None:
        self.config = config or DatasetLoadingConfig()

    def load(
        self,
        request: DatasetLoadRequest,
    ) -> LoadedDataset:
        """Load a JSON records array and return its dataset."""

        if not isinstance(request, DatasetLoadRequest):
            raise TypeError("request must be a DatasetLoadRequest")

        file_path = request.file_path
        self._validate_file(file_path)

        file_format = request.file_format or "json"

        if file_format.lower() != "json":
            raise UnsupportedFileFormatError(
                f"JSONLoader cannot load format: {file_format}"
            )

        options = dict(request.options)
        records_path = options.pop("records_path", None)

        if options:
            unsupported_options = ", ".join(sorted(options))
            raise DatasetValidationError(
                f"Unsupported JSON options: {unsupported_options}"
            )

        try:
            with file_path.open("r", encoding="utf-8-sig") as file:
                payload = json.load(file)
        except (UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
            raise DatasetParseError(
                f"Failed to parse JSON file: {file_path.name}"
            ) from exc

        records = self._extract_records(
            payload=payload,
            records_path=records_path,
        )

        if not records:
            raise DatasetValidationError(
                "JSON records array does not contain any records"
            )

        if len(records) > self.config.max_rows:
            raise DatasetValidationError(
                f"JSON contains too many rows: {len(records)}"
            )

        dataframe = pd.DataFrame.from_records(records)

        if dataframe.empty:
            raise DatasetValidationError(
                "JSON records do not contain tabular data"
            )

        if len(dataframe.columns) > self.config.max_columns:
            raise DatasetValidationError(
                f"JSON contains too many columns: "
                f"{len(dataframe.columns)}"
            )

        columns = tuple(str(column) for column in dataframe.columns)

        reference = DatasetReference(
            name=request.name or file_path.stem,
            source=str(file_path.resolve()),
            file_format="json",
            row_count=len(dataframe),
            column_count=len(dataframe.columns),
            columns=columns,
            metadata={
                "engine": "python-json",
                "records_path": records_path,
                "file_size_bytes": file_path.stat().st_size,
            },
        )

        return LoadedDataset(
            reference=reference,
            data=dataframe,
        )

    def _extract_records(
        self,
        payload: Any,
        records_path: str | None,
    ) -> list[dict[str, Any]]:
        """Extract a list of JSON objects from the payload."""

        if records_path is not None:
            if (
                not isinstance(records_path, str)
                or not records_path.strip()
            ):
                raise DatasetValidationError(
                    "records_path must be a non-empty string"
                )

            current = payload

            for key in records_path.split("."):
                if not key:
                    raise DatasetValidationError(
                        "records_path contains an empty path segment"
                    )

                if not isinstance(current, dict):
                    raise DatasetValidationError(
                        f"Cannot resolve records_path segment: {key}"
                    )

                if key not in current:
                    raise DatasetValidationError(
                        f"JSON records_path does not exist: {records_path}"
                    )

                current = current[key]

            payload = current

        if not isinstance(payload, list):
            raise DatasetValidationError(
                "JSON root or selected records_path must be an array"
            )

        if not all(isinstance(record, dict) for record in payload):
            raise DatasetValidationError(
                "Every JSON record must be an object"
            )

        return payload

    def _validate_file(self, file_path: Path) -> None:
        """Validate the local JSON file before parsing."""

        if not file_path.exists():
            raise DatasetFileNotFoundError(
                f"Dataset file does not exist: {file_path}"
            )

        if not file_path.is_file():
            raise DatasetValidationError(
                f"Dataset path is not a regular file: {file_path}"
            )

        file_size = file_path.stat().st_size

        if file_size > self.config.max_file_size_bytes:
            raise DatasetFileTooLargeError(
                f"Dataset file exceeds maximum size of "
                f"{self.config.max_file_size_bytes} bytes"
            )