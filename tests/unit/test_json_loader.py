import json
from pathlib import Path

import pandas as pd
import pytest

from lite_cog_agent.loading import (
    DatasetFileNotFoundError,
    DatasetFileTooLargeError,
    DatasetLoadRequest,
    DatasetLoadingConfig,
    DatasetParseError,
    DatasetValidationError,
    JSONLoader,
    LoadedDataset,
    UnsupportedFileFormatError,
)


@pytest.fixture
def json_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "sales.json"

    payload = [
        {"product": "A", "price": 100, "quantity": 2},
        {"product": "B", "price": 200, "quantity": 3},
        {"product": "C", "price": 300, "quantity": 4},
    ]

    file_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    return file_path


@pytest.fixture
def loader() -> JSONLoader:
    return JSONLoader()


class TestJSONLoader:

    def test_loads_json_successfully(
        self,
        loader: JSONLoader,
        json_file: Path,
    ) -> None:
        result = loader.load(
            DatasetLoadRequest(file_path=json_file)
        )

        assert isinstance(result, LoadedDataset)
        assert isinstance(result.data, pd.DataFrame)
        assert result.data.shape == (3, 3)

        assert list(result.data.columns) == [
            "product",
            "price",
            "quantity",
        ]

    def test_creates_dataset_reference(
        self,
        loader: JSONLoader,
        json_file: Path,
    ) -> None:
        result = loader.load(
            DatasetLoadRequest(
                file_path=json_file,
                name="sales_dataset",
            )
        )

        reference = result.reference

        assert reference.name == "sales_dataset"
        assert reference.file_format == "json"
        assert reference.row_count == 3
        assert reference.column_count == 3
        assert reference.columns == (
            "product",
            "price",
            "quantity",
        )

    def test_uses_file_stem_as_default_name(
        self,
        loader: JSONLoader,
        json_file: Path,
    ) -> None:
        result = loader.load(
            DatasetLoadRequest(file_path=json_file)
        )

        assert result.reference.name == "sales"

    def test_reads_wrapped_records(
        self,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "wrapped.json"

        payload = {
            "data": [
                {"name": "Alice", "score": 90},
                {"name": "Bob", "score": 85},
            ]
        }

        file_path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

        result = JSONLoader().load(
            DatasetLoadRequest(
                file_path=file_path,
                options={"records_path": "data"},
            )
        )

        assert result.data.shape == (2, 2)
        assert result.reference.metadata["records_path"] == "data"

    def test_reads_nested_records_path(
        self,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "nested.json"

        payload = {
            "result": {
                "items": [
                    {"name": "Alice"},
                    {"name": "Bob"},
                ]
            }
        }

        file_path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

        result = JSONLoader().load(
            DatasetLoadRequest(
                file_path=file_path,
                options={"records_path": "result.items"},
            )
        )

        assert result.data["name"].tolist() == [
            "Alice",
            "Bob",
        ]

    def test_supports_utf8_sig(
        self,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "bom.json"

        file_path.write_text(
            json.dumps([{"name": "Alice"}]),
            encoding="utf-8-sig",
        )

        result = JSONLoader().load(
            DatasetLoadRequest(file_path=file_path)
        )

        assert result.data["name"].iloc[0] == "Alice"

    def test_rejects_missing_file(
        self,
        loader: JSONLoader,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(DatasetFileNotFoundError):
            loader.load(
                DatasetLoadRequest(
                    file_path=tmp_path / "missing.json"
                )
            )

    def test_rejects_oversized_file(
        self,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "sales.json"
        file_path.write_text(
            '[{"value": 1}]',
            encoding="utf-8",
        )

        loader = JSONLoader(
            DatasetLoadingConfig(max_file_size_bytes=1)
        )

        with pytest.raises(DatasetFileTooLargeError):
            loader.load(
                DatasetLoadRequest(file_path=file_path)
            )

    def test_rejects_non_json_format(
        self,
        loader: JSONLoader,
        json_file: Path,
    ) -> None:
        with pytest.raises(UnsupportedFileFormatError):
            loader.load(
                DatasetLoadRequest(
                    file_path=json_file,
                    file_format="csv",
                )
            )

    def test_rejects_unsupported_options(
        self,
        loader: JSONLoader,
        json_file: Path,
    ) -> None:
        with pytest.raises(DatasetValidationError):
            loader.load(
                DatasetLoadRequest(
                    file_path=json_file,
                    options={"unknown_option": True},
                )
            )

    def test_rejects_invalid_records_path(
        self,
        loader: JSONLoader,
        json_file: Path,
    ) -> None:
        with pytest.raises(DatasetValidationError):
            loader.load(
                DatasetLoadRequest(
                    file_path=json_file,
                    options={"records_path": ""},
                )
            )

    def test_rejects_missing_records_path(
        self,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "wrapped.json"
        file_path.write_text(
            '{"data": []}',
            encoding="utf-8",
        )

        with pytest.raises(DatasetValidationError):
            JSONLoader().load(
                DatasetLoadRequest(
                    file_path=file_path,
                    options={"records_path": "missing"},
                )
            )

    def test_rejects_non_array_root(
        self,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "object.json"
        file_path.write_text(
            '{"name": "Alice"}',
            encoding="utf-8",
        )

        with pytest.raises(DatasetValidationError):
            JSONLoader().load(
                DatasetLoadRequest(file_path=file_path)
            )

    @pytest.mark.parametrize(
        "payload",
        [
            ["Alice", "Bob"],
            [1, 2, 3],
            [[1, 2], [3, 4]],
            [None],
        ],
    )
    def test_rejects_non_object_records(
        self,
        tmp_path: Path,
        payload: list,
    ) -> None:
        file_path = tmp_path / "invalid_records.json"
        file_path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

        with pytest.raises(DatasetValidationError):
            JSONLoader().load(
                DatasetLoadRequest(file_path=file_path)
            )

    def test_respects_max_rows(
        self,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "large.json"
        file_path.write_text(
            json.dumps(
                [
                    {"value": 1},
                    {"value": 2},
                    {"value": 3},
                ]
            ),
            encoding="utf-8",
        )

        loader = JSONLoader(
            DatasetLoadingConfig(max_rows=2)
        )

        with pytest.raises(DatasetValidationError):
            loader.load(
                DatasetLoadRequest(file_path=file_path)
            )

    def test_rejects_too_many_columns(
        self,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "wide.json"
        file_path.write_text(
            json.dumps(
                [
                    {
                        "a": 1,
                        "b": 2,
                        "c": 3,
                    }
                ]
            ),
            encoding="utf-8",
        )

        loader = JSONLoader(
            DatasetLoadingConfig(max_columns=2)
        )

        with pytest.raises(DatasetValidationError):
            loader.load(
                DatasetLoadRequest(file_path=file_path)
            )

    def test_rejects_empty_records(
        self,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "empty.json"
        file_path.write_text(
            "[]",
            encoding="utf-8",
        )

        with pytest.raises(DatasetValidationError):
            JSONLoader().load(
                DatasetLoadRequest(file_path=file_path)
            )

    def test_rejects_corrupted_json(
        self,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "corrupted.json"
        file_path.write_text(
            '{"name": "Alice"',
            encoding="utf-8",
        )

        with pytest.raises(DatasetParseError):
            JSONLoader().load(
                DatasetLoadRequest(file_path=file_path)
            )

    def test_rejects_invalid_utf8(
        self,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "invalid_encoding.json"
        file_path.write_bytes(b"\xff\xfe\xfa")

        with pytest.raises(DatasetParseError):
            JSONLoader().load(
                DatasetLoadRequest(file_path=file_path)
            )