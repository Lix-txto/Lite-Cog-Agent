"""Unit tests for dataset loading contracts."""

from pathlib import Path

import pandas as pd
import pytest

from lite_cog_agent.execution.dataset_store import DatasetReference
from lite_cog_agent.loading.contracts import (
    DatasetLoadRequest,
    LoadedDataset,
)


class TestDatasetLoadRequest:
    def test_creates_request_with_path_object(self) -> None:
        request = DatasetLoadRequest(
            file_path=Path("data/sales.csv"),
        )

        assert request.file_path == Path("data/sales.csv")
        assert request.name is None
        assert request.file_format is None
        assert request.options == {}

    def test_converts_string_path_to_path_object(self) -> None:
        request = DatasetLoadRequest(
            file_path="data/sales.csv",
        )

        assert request.file_path == Path("data/sales.csv")
        assert isinstance(request.file_path, Path)

    def test_normalizes_name(self) -> None:
        request = DatasetLoadRequest(
            file_path="data/sales.csv",
            name="  sales  ",
        )

        assert request.name == "sales"

    def test_normalizes_file_format(self) -> None:
        request = DatasetLoadRequest(
            file_path="data/sales.csv",
            file_format=".CSV",
        )

        assert request.file_format == "csv"

    def test_preserves_options(self) -> None:
        request = DatasetLoadRequest(
            file_path="data/sales.csv",
            options={
                "encoding": "utf-8",
                "delimiter": ",",
            },
        )

        assert request.options["encoding"] == "utf-8"
        assert request.options["delimiter"] == ","

    def test_options_are_read_only(self) -> None:
        request = DatasetLoadRequest(
            file_path="data/sales.csv",
            options={"encoding": "utf-8"},
        )

        with pytest.raises(TypeError):
            request.options["encoding"] = "gbk"  # type: ignore[index]

    def test_options_are_copied(self) -> None:
        original_options = {"encoding": "utf-8"}

        request = DatasetLoadRequest(
            file_path="data/sales.csv",
            options=original_options,
        )

        original_options["encoding"] = "gbk"

        assert request.options["encoding"] == "utf-8"

    def test_request_is_frozen(self) -> None:
        request = DatasetLoadRequest(
            file_path="data/sales.csv",
        )

        with pytest.raises(AttributeError):
            request.name = "updated"  # type: ignore[misc]

    def test_rejects_empty_string_path(self) -> None:
        with pytest.raises(ValueError, match="file_path cannot be empty"):
            DatasetLoadRequest(file_path="")

    def test_rejects_whitespace_path(self) -> None:
        with pytest.raises(ValueError, match="file_path cannot be empty"):
            DatasetLoadRequest(file_path="   ")

    def test_rejects_invalid_path_type(self) -> None:
        with pytest.raises(TypeError, match="file_path must be"):
            DatasetLoadRequest(file_path=123)  # type: ignore[arg-type]

    def test_rejects_non_string_name(self) -> None:
        with pytest.raises(TypeError, match="name must be"):
            DatasetLoadRequest(
                file_path="data/sales.csv",
                name=123,  # type: ignore[arg-type]
            )

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="name cannot be empty"):
            DatasetLoadRequest(
                file_path="data/sales.csv",
                name="   ",
            )

    def test_rejects_non_string_file_format(self) -> None:
        with pytest.raises(TypeError, match="file_format must be"):
            DatasetLoadRequest(
                file_path="data/sales.csv",
                file_format=123,  # type: ignore[arg-type]
            )

    def test_rejects_empty_file_format(self) -> None:
        with pytest.raises(ValueError, match="file_format cannot be empty"):
            DatasetLoadRequest(
                file_path="data/sales.csv",
                file_format="   ",
            )

    def test_rejects_non_mapping_options(self) -> None:
        with pytest.raises(TypeError, match="options must be a mapping"):
            DatasetLoadRequest(
                file_path="data/sales.csv",
                options=["utf-8"],  # type: ignore[arg-type]
            )

    def test_rejects_non_string_option_keys(self) -> None:
        with pytest.raises(TypeError, match="all options keys must be strings"):
            DatasetLoadRequest(
                file_path="data/sales.csv",
                options={1: "utf-8"},  # type: ignore[dict-item]
            )


class TestLoadedDataset:
    def test_creates_loaded_dataset(self) -> None:
        reference = DatasetReference(
            name="sales",
            source="data/sales.csv",
            file_format="csv",
            row_count=2,
            column_count=2,
            columns=("product", "price"),
        )

        data = pd.DataFrame(
            {
                "product": ["A", "B"],
                "price": [100, 200],
            }
        )

        loaded = LoadedDataset(
            reference=reference,
            data=data,
        )

        assert loaded.reference is reference
        assert loaded.data is data

    def test_preserves_dataframe_reference(self) -> None:
        data = pd.DataFrame({"value": [1, 2, 3]})

        loaded = LoadedDataset(
            reference=DatasetReference(name="test"),
            data=data,
        )

        assert loaded.data is data

    def test_rejects_invalid_reference(self) -> None:
        with pytest.raises(TypeError, match="reference must be"):
            LoadedDataset(
                reference="invalid",  # type: ignore[arg-type]
                data=pd.DataFrame(),
            )

    def test_rejects_non_dataframe_data(self) -> None:
        with pytest.raises(TypeError, match="data must be"):
            LoadedDataset(
                reference=DatasetReference(name="test"),
                data=[1, 2, 3],  # type: ignore[arg-type]
            )

    def test_loaded_dataset_is_frozen(self) -> None:
        loaded = LoadedDataset(
            reference=DatasetReference(name="test"),
            data=pd.DataFrame({"value": [1]}),
        )

        with pytest.raises(AttributeError):
            loaded.data = pd.DataFrame()  # type: ignore[misc]