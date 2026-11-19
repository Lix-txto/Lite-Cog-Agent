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
    ExcelLoader,
    LoadedDataset,
    UnsupportedFileFormatError,
)


@pytest.fixture
def excel_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "sales.xlsx"

    dataframe = pd.DataFrame(
        {
            "product": ["A", "B", "C"],
            "price": [100, 200, 300],
            "quantity": [2, 3, 4],
        }
    )

    dataframe.to_excel(
        file_path,
        index=False,
        engine="openpyxl",
    )

    return file_path


@pytest.fixture
def loader() -> ExcelLoader:
    return ExcelLoader()


class TestExcelLoader:

    def test_loads_excel_successfully(
        self,
        loader: ExcelLoader,
        excel_file: Path,
    ) -> None:
        result = loader.load(
            DatasetLoadRequest(
                file_path=excel_file,
            )
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
        loader: ExcelLoader,
        excel_file: Path,
    ) -> None:
        result = loader.load(
            DatasetLoadRequest(
                file_path=excel_file,
                name="sales_dataset",
            )
        )

        reference = result.reference

        assert reference.name == "sales_dataset"
        assert reference.file_format == "excel"
        assert reference.row_count == 3
        assert reference.column_count == 3
        assert reference.columns == (
            "product",
            "price",
            "quantity",
        )

    def test_uses_file_stem_as_default_name(
        self,
        loader: ExcelLoader,
        excel_file: Path,
    ) -> None:
        result = loader.load(
            DatasetLoadRequest(file_path=excel_file)
        )

        assert result.reference.name == "sales"

    def test_reads_named_worksheet(
        self,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "multi_sheet.xlsx"

        with pd.ExcelWriter(
            file_path,
            engine="openpyxl",
        ) as writer:
            pd.DataFrame(
                {"name": ["Alice"], "score": [90]}
            ).to_excel(
                writer,
                sheet_name="students",
                index=False,
            )

            pd.DataFrame(
                {"product": ["Book"], "price": [30]}
            ).to_excel(
                writer,
                sheet_name="products",
                index=False,
            )

        loader = ExcelLoader()

        result = loader.load(
            DatasetLoadRequest(
                file_path=file_path,
                options={"sheet_name": "products"},
            )
        )

        assert list(result.data.columns) == [
            "product",
            "price",
        ]

        assert result.reference.metadata["sheet_name"] == "products"

    def test_reads_worksheet_by_index(
        self,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "multi_sheet.xlsx"

        with pd.ExcelWriter(
            file_path,
            engine="openpyxl",
        ) as writer:
            pd.DataFrame(
                {"first": [1]}
            ).to_excel(
                writer,
                sheet_name="first_sheet",
                index=False,
            )

            pd.DataFrame(
                {"second": [2]}
            ).to_excel(
                writer,
                sheet_name="second_sheet",
                index=False,
            )

        result = ExcelLoader().load(
            DatasetLoadRequest(
                file_path=file_path,
                options={"sheet_name": 1},
            )
        )

        assert list(result.data.columns) == ["second"]

    def test_supports_custom_header_row(
        self,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "header.xlsx"

        dataframe = pd.DataFrame(
            [
                ["Sales Report", None],
                ["product", "price"],
                ["A", 100],
                ["B", 200],
            ]
        )

        dataframe.to_excel(
            file_path,
            index=False,
            header=False,
        )

        result = ExcelLoader().load(
            DatasetLoadRequest(
                file_path=file_path,
                options={"header": 1},
            )
        )

        assert list(result.data.columns) == [
            "product",
            "price",
        ]

        assert len(result.data) == 2

    def test_supports_header_none(
        self,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "no_header.xlsx"

        pd.DataFrame(
            [
                ["A", 100],
                ["B", 200],
            ]
        ).to_excel(
            file_path,
            index=False,
            header=False,
        )

        result = ExcelLoader().load(
            DatasetLoadRequest(
                file_path=file_path,
                options={"header": None},
            )
        )

        assert len(result.data) == 2
        assert list(result.data.columns) == [0, 1]

    def test_respects_max_rows(
        self,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "sales.xlsx"

        pd.DataFrame(
            {
                "product": ["A", "B", "C"],
                "price": [100, 200, 300],
            }
        ).to_excel(
            file_path,
            index=False,
        )

        loader = ExcelLoader(
            DatasetLoadingConfig(max_rows=2)
        )

        result = loader.load(
            DatasetLoadRequest(file_path=file_path)
        )

        assert len(result.data) == 2

    def test_respects_requested_nrows(
        self,
        loader: ExcelLoader,
        excel_file: Path,
    ) -> None:
        result = loader.load(
            DatasetLoadRequest(
                file_path=excel_file,
                options={"nrows": 1},
            )
        )

        assert len(result.data) == 1

    def test_rejects_missing_file(
        self,
        loader: ExcelLoader,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(DatasetFileNotFoundError):
            loader.load(
                DatasetLoadRequest(
                    file_path=tmp_path / "missing.xlsx"
                )
            )

    def test_rejects_oversized_file(
        self,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "sales.xlsx"

        pd.DataFrame({"value": [1]}).to_excel(
            file_path,
            index=False,
        )

        loader = ExcelLoader(
            DatasetLoadingConfig(
                max_file_size_bytes=1
            )
        )

        with pytest.raises(DatasetFileTooLargeError):
            loader.load(
                DatasetLoadRequest(file_path=file_path)
            )

    def test_rejects_non_excel_format(
        self,
        loader: ExcelLoader,
        excel_file: Path,
    ) -> None:
        with pytest.raises(UnsupportedFileFormatError):
            loader.load(
                DatasetLoadRequest(
                    file_path=excel_file,
                    file_format="csv",
                )
            )

    def test_rejects_unsupported_options(
        self,
        loader: ExcelLoader,
        excel_file: Path,
    ) -> None:
        with pytest.raises(DatasetValidationError):
            loader.load(
                DatasetLoadRequest(
                    file_path=excel_file,
                    options={"unknown_option": True},
                )
            )

    @pytest.mark.parametrize(
        "sheet_name",
        [
            "",
            "   ",
            -1,
            True,
            1.5,
            None,
        ],
    )
    def test_rejects_invalid_sheet_name(
        self,
        loader: ExcelLoader,
        excel_file: Path,
        sheet_name: object,
    ) -> None:
        with pytest.raises(DatasetValidationError):
            loader.load(
                DatasetLoadRequest(
                    file_path=excel_file,
                    options={"sheet_name": sheet_name},
                )
            )

    @pytest.mark.parametrize(
        "header",
        [
            -1,
            True,
            1.5,
            "0",
        ],
    )
    def test_rejects_invalid_header(
        self,
        loader: ExcelLoader,
        excel_file: Path,
        header: object,
    ) -> None:
        with pytest.raises(DatasetValidationError):
            loader.load(
                DatasetLoadRequest(
                    file_path=excel_file,
                    options={"header": header},
                )
            )

    @pytest.mark.parametrize(
        "nrows",
        [
            0,
            -1,
            True,
            1.5,
        ],
    )
    def test_rejects_invalid_nrows(
        self,
        loader: ExcelLoader,
        excel_file: Path,
        nrows: object,
    ) -> None:
        with pytest.raises(DatasetValidationError):
            loader.load(
                DatasetLoadRequest(
                    file_path=excel_file,
                    options={"nrows": nrows},
                )
            )

    def test_rejects_empty_worksheet(
        self,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "empty.xlsx"

        pd.DataFrame().to_excel(
            file_path,
            index=False,
        )

        with pytest.raises(DatasetValidationError):
            ExcelLoader().load(
                DatasetLoadRequest(file_path=file_path)
            )

    def test_rejects_corrupted_excel_file(
        self,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "corrupted.xlsx"
        file_path.write_bytes(b"this is not a valid xlsx file")

        with pytest.raises(DatasetParseError):
            ExcelLoader().load(
                DatasetLoadRequest(file_path=file_path)
            )

    def test_rejects_too_many_columns(
        self,
        tmp_path: Path,
    ) -> None:
        file_path = tmp_path / "wide.xlsx"

        pd.DataFrame(
            {
                "column_a": [1],
                "column_b": [2],
                "column_c": [3],
            }
        ).to_excel(
            file_path,
            index=False,
        )

        loader = ExcelLoader(
            DatasetLoadingConfig(max_columns=2)
        )

        with pytest.raises(DatasetValidationError):
            loader.load(
                DatasetLoadRequest(file_path=file_path)
            )