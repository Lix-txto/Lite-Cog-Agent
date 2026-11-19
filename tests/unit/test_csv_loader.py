from pathlib import Path

import pandas as pd
import pytest

from lite_cog_agent.loading import (
    CSVLoader,
    DatasetFileNotFoundError,
    DatasetFileTooLargeError,
    DatasetLoadRequest,
    DatasetLoadingConfig,
    DatasetParseError,
    DatasetValidationError,
    LoadedDataset,
    UnsupportedFileFormatError,
)


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "sales.csv"

    file_path.write_text(
        "product,price,quantity\n"
        "A,100,2\n"
        "B,200,3\n"
        "C,300,4\n",
        encoding="utf-8",
    )

    return file_path


@pytest.fixture
def loader() -> CSVLoader:
    return CSVLoader()


class TestCSVLoader:

    def test_loads_csv_successfully(
        self,
        loader: CSVLoader,
        csv_file: Path,
    ) -> None:
        result = loader.load(
            DatasetLoadRequest(file_path=csv_file)
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
        loader: CSVLoader,
        csv_file: Path,
    ) -> None:
        result = loader.load(
            DatasetLoadRequest(
                file_path=csv_file,
                name="sales_dataset",
            )
        )

        reference = result.reference

        assert reference.name == "sales_dataset"
        assert reference.file_format == "csv"
        assert reference.row_count == 3
        assert reference.column_count == 3
        assert reference.columns == (
            "product",
            "price",
            "quantity",
        )

    def test_uses_file_stem_as_default_name(
        self,
        loader: CSVLoader,
        csv_file: Path,
    ) -> None:
        result = loader.load(
            DatasetLoadRequest(file_path=csv_file)
        )

        assert result.reference.name == "sales"

    def test_supports_custom_delimiter(
        self,
        loader: CSVLoader,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "sales.csv"

        file_path.write_text(
            "product;price\n"
            "A;100\n"
            "B;200\n",
            encoding="utf-8",
        )

        result = loader.load(
            DatasetLoadRequest(
                file_path=file_path,
                options={"delimiter": ";"},
            )
        )

        assert result.data.shape == (2, 2)

    def test_supports_utf8_sig_encoding(
        self,
        loader: CSVLoader,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "sales.csv"

        file_path.write_text(
            "product,price\nA,100\n",
            encoding="utf-8-sig",
        )

        result = loader.load(
            DatasetLoadRequest(
                file_path=file_path,
                options={"encoding": "utf-8-sig"},
            )
        )

        assert result.data.columns[0] == "product"

    def test_respects_max_rows(
        self,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "sales.csv"

        file_path.write_text(
            "product,price\n"
            "A,100\n"
            "B,200\n"
            "C,300\n",
            encoding="utf-8",
        )

        loader = CSVLoader(
            DatasetLoadingConfig(max_rows=2)
        )

        result = loader.load(
            DatasetLoadRequest(file_path=file_path)
        )

        assert len(result.data) == 2

    def test_respects_requested_nrows(
        self,
        loader: CSVLoader,
        csv_file: Path,
    ) -> None:
        result = loader.load(
            DatasetLoadRequest(
                file_path=csv_file,
                options={"nrows": 1},
            )
        )

        assert len(result.data) == 1

    def test_rejects_missing_file(
        self,
        loader: CSVLoader,
        tmp_path: Path,
    ) -> None:
        request = DatasetLoadRequest(
            file_path=tmp_path / "missing.csv"
        )

        with pytest.raises(DatasetFileNotFoundError):
            loader.load(request)

    def test_rejects_oversized_file(
        self,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "large.csv"
        file_path.write_text(
            "a,b\n1,2\n",
            encoding="utf-8",
        )

        loader = CSVLoader(
            DatasetLoadingConfig(
                max_file_size_bytes=1
            )
        )

        with pytest.raises(DatasetFileTooLargeError):
            loader.load(
                DatasetLoadRequest(file_path=file_path)
            )

    def test_rejects_non_csv_format(
        self,
        loader: CSVLoader,
        csv_file: Path,
    ) -> None:
        with pytest.raises(UnsupportedFileFormatError):
            loader.load(
                DatasetLoadRequest(
                    file_path=csv_file,
                    file_format="excel",
                )
            )

    def test_rejects_unsupported_options(
        self,
        loader: CSVLoader,
        csv_file: Path,
    ) -> None:
        with pytest.raises(DatasetValidationError):
            loader.load(
                DatasetLoadRequest(
                    file_path=csv_file,
                    options={"unknown_option": True},
                )
            )

    def test_rejects_invalid_nrows(
        self,
        loader: CSVLoader,
        csv_file: Path,
    ) -> None:
        with pytest.raises(DatasetValidationError):
            loader.load(
                DatasetLoadRequest(
                    file_path=csv_file,
                    options={"nrows": -1},
                )
            )

    def test_rejects_empty_csv(
        self,
        loader: CSVLoader,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "empty.csv"
        file_path.write_text(
            "product,price\n",
            encoding="utf-8",
        )

        with pytest.raises(DatasetValidationError):
            loader.load(
                DatasetLoadRequest(file_path=file_path)
            )

    def test_rejects_invalid_csv(
        self,
        loader: CSVLoader,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "invalid.csv"
        file_path.write_bytes(b"\xff\xfe\xfa")

        with pytest.raises(DatasetParseError):
            loader.load(
                DatasetLoadRequest(
                    file_path=file_path,
                    options={"encoding": "utf-8"},
                )
            )