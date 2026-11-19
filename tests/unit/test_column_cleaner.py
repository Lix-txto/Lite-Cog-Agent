"""Unit tests for ColumnCleaner."""

import pandas as pd
import pytest

from lite_cog_agent.cleaning.columns import ColumnCleaner
from lite_cog_agent.cleaning.exceptions import (
    CleaningOperationError,
    CleaningValidationError,
)


@pytest.fixture
def cleaner():
    return ColumnCleaner()


@pytest.fixture
def dataframe():
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "city": ["NY", "LA", "SF"],
        }
    )


class TestColumnSelect:
    def test_selects_requested_columns(self, cleaner, dataframe):
        result = cleaner.select(
            dataframe,
            columns=["name", "age"],
        )

        assert list(result.columns) == ["name", "age"]
        assert len(result) == 3

    def test_preserves_requested_column_order(
        self, cleaner, dataframe
    ):
        result = cleaner.select(
            dataframe,
            columns=["city", "name"],
        )

        assert list(result.columns) == ["city", "name"]

    def test_preserves_original_index(self, cleaner):
        df = pd.DataFrame(
            {"a": [1, 2, 3]},
            index=[10, 20, 30],
        )

        result = cleaner.select(df, columns=["a"])

        assert result.index.tolist() == [10, 20, 30]

    def test_supports_single_column(self, cleaner, dataframe):
        result = cleaner.select(
            dataframe,
            columns=["name"],
        )

        assert list(result.columns) == ["name"]

    def test_selects_all_columns(self, cleaner, dataframe):
        result = cleaner.select(
            dataframe,
            columns=["name", "age", "city"],
        )

        pd.testing.assert_frame_equal(result, dataframe)
        assert result is not dataframe

    def test_selects_from_empty_dataframe(self, cleaner):
        df = pd.DataFrame(
            {
                "name": pd.Series(dtype=str),
                "age": pd.Series(dtype=int),
            }
        )

        result = cleaner.select(df, columns=["name"])

        assert result.empty
        assert list(result.columns) == ["name"]


class TestColumnDrop:
    def test_drops_requested_columns(self, cleaner, dataframe):
        result = cleaner.drop(
            dataframe,
            columns=["city"],
        )

        assert list(result.columns) == ["name", "age"]
        assert len(result) == 3

    def test_drops_multiple_columns(self, cleaner, dataframe):
        result = cleaner.drop(
            dataframe,
            columns=["name", "city"],
        )

        assert list(result.columns) == ["age"]

    def test_drops_all_columns(self, cleaner, dataframe):
        result = cleaner.drop(
            dataframe,
            columns=["name", "age", "city"],
        )

        assert result.shape == (3, 0)

    def test_preserves_original_index(self, cleaner):
        df = pd.DataFrame(
            {
                "a": [1, 2],
                "b": [3, 4],
            },
            index=[10, 20],
        )

        result = cleaner.drop(df, columns=["b"])

        assert result.index.tolist() == [10, 20]


class TestColumnRename:
    def test_renames_single_column(self, cleaner, dataframe):
        result = cleaner.rename(
            dataframe,
            mapping={"name": "full_name"},
        )

        assert list(result.columns) == [
            "full_name",
            "age",
            "city",
        ]

    def test_renames_multiple_columns(self, cleaner, dataframe):
        result = cleaner.rename(
            dataframe,
            mapping={
                "name": "full_name",
                "age": "years",
            },
        )

        assert list(result.columns) == [
            "full_name",
            "years",
            "city",
        ]

    def test_preserves_unmapped_columns(self, cleaner, dataframe):
        result = cleaner.rename(
            dataframe,
            mapping={"name": "full_name"},
        )

        assert result["age"].equals(dataframe["age"])
        assert result["city"].equals(dataframe["city"])

    def test_supports_column_name_swap(self, cleaner):
        df = pd.DataFrame(
            {
                "a": [1, 2],
                "b": [3, 4],
            }
        )

        result = cleaner.rename(
            df,
            mapping={
                "a": "b",
                "b": "a",
            },
        )

        assert list(result.columns) == ["b", "a"]
        assert result["b"].tolist() == [1, 2]
        assert result["a"].tolist() == [3, 4]

    def test_rejects_duplicate_final_column_names(
        self, cleaner, dataframe
    ):
        with pytest.raises(
            CleaningValidationError,
            match="duplicate column names",
        ):
            cleaner.rename(
                dataframe,
                mapping={"name": "age"},
            )

    def test_preserves_column_order(self, cleaner, dataframe):
        result = cleaner.rename(
            dataframe,
            mapping={"city": "location"},
        )

        assert list(result.columns) == [
            "name",
            "age",
            "location",
        ]


class TestColumnValidation:
    @pytest.mark.parametrize(
        "columns",
        [
            "name",
            b"name",
            [],
            ["unknown"],
            ["name", "name"],
            [""],
            [" "],
            [None],
            [123],
            123,
            None,
        ],
    )
    @pytest.mark.parametrize("method", ["select", "drop"])
    def test_rejects_invalid_columns(
        self,
        cleaner,
        dataframe,
        columns,
        method,
    ):
        with pytest.raises(CleaningValidationError):
            getattr(cleaner, method)(
                dataframe,
                columns=columns,
            )

    @pytest.mark.parametrize(
        "invalid_dataframe",
        [
            None,
            [],
            "not a dataframe",
            123,
        ],
    )
    @pytest.mark.parametrize("method", ["select", "drop", "rename"])
    def test_rejects_non_dataframe(
        self,
        cleaner,
        invalid_dataframe,
        method,
    ):
        with pytest.raises(CleaningValidationError):
            if method == "rename":
                cleaner.rename(
                    invalid_dataframe,
                    mapping={"a": "b"},
                )
            else:
                getattr(cleaner, method)(
                    invalid_dataframe,
                    columns=["a"],
                )

    @pytest.mark.parametrize(
        "df",
        [
            pd.DataFrame([[1, 2]], columns=["a", "a"]),
            pd.DataFrame([[1, 2]], columns=["a", ""]),
            pd.DataFrame([[1, 2]], columns=["a", None]),
            pd.DataFrame([[1, 2]], columns=["a", 123]),
        ],
    )
    @pytest.mark.parametrize("method", ["select", "drop", "rename"])
    def test_rejects_invalid_dataframe_columns(
        self,
        cleaner,
        df,
        method,
    ):
        with pytest.raises(CleaningValidationError):
            if method == "rename":
                cleaner.rename(df, mapping={"a": "b"})
            else:
                getattr(cleaner, method)(
                    df,
                    columns=["a"],
                )

    @pytest.mark.parametrize(
        "mapping",
        [
            None,
            [],
            "invalid",
            {},
            {None: "new_name"},
            {" ": "new_name"},
            {"name": ""},
            {"name": " "},
            {"unknown": "new_name"},
            {"name": "new_name", "unknown": "other"},
        ],
    )
    def test_rejects_invalid_mapping(
        self,
        cleaner,
        dataframe,
        mapping,
    ):
        with pytest.raises(CleaningValidationError):
            cleaner.rename(dataframe, mapping=mapping)


class TestColumnCleanerSafety:
    @pytest.mark.parametrize(
        "method,kwargs",
        [
            ("select", {"columns": ["name", "age"]}),
            ("drop", {"columns": ["city"]}),
            ("rename", {"mapping": {"name": "full_name"}}),
        ],
    )
    def test_does_not_modify_original_dataframe(
        self,
        cleaner,
        dataframe,
        method,
        kwargs,
    ):
        original = dataframe.copy(deep=True)

        getattr(cleaner, method)(dataframe, **kwargs)

        pd.testing.assert_frame_equal(dataframe, original)

    @pytest.mark.parametrize(
        "method,kwargs",
        [
            ("select", {"columns": ["name"]}),
            ("drop", {"columns": ["city"]}),
            ("rename", {"mapping": {"name": "full_name"}}),
        ],
    )
    def test_returns_independent_dataframe(
        self,
        cleaner,
        dataframe,
        method,
        kwargs,
    ):
        result = getattr(cleaner, method)(dataframe, **kwargs)

        assert result is not dataframe

        if not result.empty and len(result.columns) > 0:
            original_value = dataframe.iloc[0, 0]
            result.iloc[0, 0] = "changed"

            assert dataframe.iloc[0, 0] == original_value

    @pytest.mark.parametrize(
        "method,operation,kwargs",
        [
            (
                "select",
                "columns.select",
                {"columns": ["name"]},
            ),
            (
                "drop",
                "columns.drop",
                {"columns": ["city"]},
            ),
            (
                "rename",
                "columns.rename",
                {"mapping": {"name": "full_name"}},
            ),
        ],
    )
    def test_wraps_pandas_execution_errors(
        self,
        cleaner,
        dataframe,
        method,
        operation,
        kwargs,
        monkeypatch,
    ):
        def raise_runtime_error(self, *args, **kwargs):
            raise RuntimeError("Simulated pandas execution failure")

        monkeypatch.setattr(
            pd.DataFrame,
            "copy",
            raise_runtime_error,
        )

        with pytest.raises(CleaningOperationError) as exc_info:
            getattr(cleaner, method)(dataframe, **kwargs)

        assert exc_info.value.operation == operation
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert str(exc_info.value.__cause__) == (
            "Simulated pandas execution failure"
        )
