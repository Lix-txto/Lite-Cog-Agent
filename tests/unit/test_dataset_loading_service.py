"""Unit tests for dataset loading service."""

import json
from pathlib import Path

import pandas as pd
import pytest

from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.loading import (
    DatasetLoadRequest,
    DatasetLoadingConfig,
    DatasetLoadingService,
    UnsupportedFileFormatError,
    detect_file_format,
    DatasetFileNotFoundError
)


@pytest.fixture
def service() -> DatasetLoadingService:
    return DatasetLoadingService()


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext()


class TestDatasetLoadingService:
    @pytest.mark.parametrize(
        ("filename", "file_format"),
        [
            ("sales.csv", "csv"),
            ("sales.xlsx", "excel"),
            ("sales.json", "json"),
        ],
    )
    def test_detects_supported_file_formats(
        self,
        tmp_path: Path,
        filename: str,
        file_format: str,
    ) -> None:
        file_path = tmp_path / filename

        assert detect_file_format(file_path) == file_format

    def test_loads_csv_and_registers_dataset(
        self,
        tmp_path: Path,
        service: DatasetLoadingService,
        context: ExecutionContext,
    ) -> None:
        file_path = tmp_path / "sales.csv"
        pd.DataFrame(
            {
                "product": ["A", "B"],
                "price": [100, 200],
            }
        ).to_csv(file_path, index=False)

        reference = service.load_into_context(
            DatasetLoadRequest(file_path=file_path),
            context,
        )

        loaded_data = context.get_dataset(reference.dataset_id)

        assert reference.file_format == "csv"
        assert reference.name == "sales"
        assert reference.row_count == 2
        assert reference.column_count == 2
        assert list(loaded_data["product"]) == ["A", "B"]

    def test_loads_excel_and_registers_dataset(
        self,
        tmp_path: Path,
        service: DatasetLoadingService,
        context: ExecutionContext,
    ) -> None:
        file_path = tmp_path / "sales.xlsx"
        pd.DataFrame(
            {
                "product": ["A", "B"],
                "price": [100, 200],
            }
        ).to_excel(file_path, index=False)

        reference = service.load_into_context(
            DatasetLoadRequest(file_path=file_path),
            context,
        )

        loaded_data = context.get_dataset(reference.dataset_id)

        assert reference.file_format == "excel"
        assert reference.row_count == 2
        assert list(loaded_data["product"]) == ["A", "B"]

    def test_loads_json_and_registers_dataset(
        self,
        tmp_path: Path,
        service: DatasetLoadingService,
        context: ExecutionContext,
    ) -> None:
        file_path = tmp_path / "sales.json"
        file_path.write_text(
            json.dumps(
                [
                    {"product": "A", "price": 100},
                    {"product": "B", "price": 200},
                ]
            ),
            encoding="utf-8",
        )

        reference = service.load_into_context(
            DatasetLoadRequest(file_path=file_path),
            context,
        )

        loaded_data = context.get_dataset(reference.dataset_id)

        assert reference.file_format == "json"
        assert reference.row_count == 2
        assert list(loaded_data["product"]) == ["A", "B"]

    def test_returns_dataset_reference(
        self,
        tmp_path: Path,
        service: DatasetLoadingService,
        context: ExecutionContext,
    ) -> None:
        file_path = tmp_path / "data.csv"
        pd.DataFrame({"value": [1, 2]}).to_csv(
            file_path,
            index=False,
        )

        reference = service.load_into_context(
            DatasetLoadRequest(file_path=file_path),
            context,
        )

        assert reference is context.get_dataset_reference(
            reference.dataset_id
        )

    def test_preserves_custom_dataset_name(
        self,
        tmp_path: Path,
        service: DatasetLoadingService,
        context: ExecutionContext,
    ) -> None:
        file_path = tmp_path / "data.csv"
        pd.DataFrame({"value": [1]}).to_csv(
            file_path,
            index=False,
        )

        reference = service.load_into_context(
            DatasetLoadRequest(
                file_path=file_path,
                name="custom dataset",
            ),
            context,
        )

        assert reference.name == "custom dataset"

    def test_uses_explicit_file_format(
        self,
        tmp_path: Path,
        service: DatasetLoadingService,
        context: ExecutionContext,
    ) -> None:
        file_path = tmp_path / "data.csv"
        file_path.write_text(
            json.dumps([{"value": 1}]),
            encoding="utf-8",
        )

        reference = service.load_into_context(
            DatasetLoadRequest(
                file_path=file_path,
                file_format="json",
            ),
            context,
        )

        assert reference.file_format == "json"
        assert context.get_dataset(reference.dataset_id).iloc[0]["value"] == 1

    def test_raises_for_unsupported_extension(
        self,
        tmp_path: Path,
        service: DatasetLoadingService,
        context: ExecutionContext,
    ) -> None:
        file_path = tmp_path / "data.parquet"
        file_path.write_bytes(b"placeholder")

        with pytest.raises(UnsupportedFileFormatError):
            service.load_into_context(
                DatasetLoadRequest(file_path=file_path),
                context,
            )

        assert context.dataset_ids == ()

    def test_raises_for_unsupported_explicit_format(
        self,
        tmp_path: Path,
        service: DatasetLoadingService,
        context: ExecutionContext,
    ) -> None:
        file_path = tmp_path / "data.csv"
        file_path.write_text("value\n1\n", encoding="utf-8")

        with pytest.raises(UnsupportedFileFormatError):
            service.load_into_context(
                DatasetLoadRequest(
                    file_path=file_path,
                    file_format="parquet",
                ),
                context,
            )

        assert context.dataset_ids == ()

    def test_propagates_loader_errors(
        self,
        tmp_path: Path,
        service: DatasetLoadingService,
        context: ExecutionContext,
    ) -> None:
        file_path = tmp_path / "missing.csv"

        with pytest.raises(DatasetFileNotFoundError):
            service.load_into_context(
                DatasetLoadRequest(file_path=file_path),
                context,
            )

        assert context.dataset_ids == ()

    def test_rejects_invalid_request(
        self,
        service: DatasetLoadingService,
        context: ExecutionContext,
    ) -> None:
        with pytest.raises(TypeError, match="DatasetLoadRequest"):
            service.load_into_context(
                "invalid",  # type: ignore[arg-type]
                context,
            )

    def test_rejects_invalid_context(
        self,
        tmp_path: Path,
        service: DatasetLoadingService,
    ) -> None:
        file_path = tmp_path / "data.csv"
        pd.DataFrame({"value": [1]}).to_csv(
            file_path,
            index=False,
        )

        with pytest.raises(TypeError, match="ExecutionContext"):
            service.load_into_context(
                DatasetLoadRequest(file_path=file_path),
                "invalid",  # type: ignore[arg-type]
            )

    def test_rejects_invalid_config(self) -> None:
        with pytest.raises(TypeError, match="DatasetLoadingConfig"):
            DatasetLoadingService(config="invalid")  # type: ignore[arg-type]

    def test_uses_shared_loading_config(self) -> None:
        config = DatasetLoadingConfig(max_rows=10)

        service = DatasetLoadingService(config=config)

        assert service.config is config
        assert service._loaders["csv"].config is config
        assert service._loaders["excel"].config is config
        assert service._loaders["json"].config is config