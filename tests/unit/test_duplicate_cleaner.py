"""Unit tests for DuplicateCleaner."""

import pandas as pd
import pytest

from lite_cog_agent.cleaning.duplicates import DuplicateCleaner
from lite_cog_agent.cleaning.exceptions import (
    CleaningOperationError,
    CleaningValidationError,
)


@pytest.fixture
def cleaner() -> DuplicateCleaner:
    return DuplicateCleaner()


@pytest.fixture
def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Alice", "Alice", "Bob"],
            "age": [25, 30, 25, 26, 30],
            "city": ["NY", "LA", "NY", "NY", "LA"],
        }
    )


class TestDuplicateDrop:
    def test_drops_duplicate_rows_keep_first(
        self,
        cleaner,
        dataframe,
    ):
        result = cleaner.drop_duplicates(dataframe)

        assert len(result) == 3
        assert result.index.tolist() == [0, 1, 3]

    def test_drops_duplicate_rows_keep_last(
        self,
        cleaner,
        dataframe,
    ):
        result = cleaner.drop_duplicates(
            dataframe,
            keep="last",
        )

        assert len(result) == 3
        assert result.index.tolist() == [2, 3, 4]

    def test_drops_all_duplicate_rows(
        self,
        cleaner,
        dataframe,
    ):
        result = cleaner.drop_duplicates(
            dataframe,
            keep=False,
        )

        assert len(result) == 1
        assert result.index.tolist() == [3]
        assert result.iloc[0]["name"] == "Alice"
        assert result.iloc[0]["age"] == 26

    def test_uses_selected_columns_for_duplicate_detection(
        self,
        cleaner,
        dataframe,
    ):
        result = cleaner.drop_duplicates(
            dataframe,
            subset=["name"],
        )

        assert len(result) == 2
        assert result["name"].tolist() == ["Alice", "Bob"]

    def test_supports_multiple_subset_columns(
        self,
        cleaner,
        dataframe,
    ):
        result = cleaner.drop_duplicates(
            dataframe,
            subset=["name", "age"],
        )

        assert len(result) == 3

    def test_preserves_index_by_default(
        self,
        cleaner,
        dataframe,
    ):
        result = cleaner.drop_duplicates(dataframe)

        assert result.index.tolist() == [0, 1, 3]

    def test_resets_index_when_requested(
        self,
        cleaner,
        dataframe,
    ):
        result = cleaner.drop_duplicates(
            dataframe,
            ignore_index=True,
        )

        assert result.index.tolist() == [0, 1, 2]

    def test_returns_all_rows_when_no_duplicates(
        self,
        cleaner,
    ):
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
            }
        )

        result = cleaner.drop_duplicates(df)

        assert len(result) == 3

    def test_supports_empty_dataframe(
        self,
        cleaner,
    ):
        df = pd.DataFrame(
            {
                "id": pd.Series(dtype="int64"),
                "name": pd.Series(dtype="object"),
            }
        )

        result = cleaner.drop_duplicates(df)

        assert result.empty
        assert result.columns.tolist() == ["id", "name"]


class TestDuplicateValidation:
    @pytest.mark.parametrize(
        "keep",
        ["invalid", "", None, 1, 0, True],
    )
    def test_rejects_invalid_keep(
        self,
        cleaner,
        dataframe,
        keep,
    ):
        with pytest.raises(CleaningValidationError):
            cleaner.drop_duplicates(
                dataframe,
                keep=keep,
            )

    @pytest.mark.parametrize(
        "subset",
        [
            "name",
            [],
            ["name", "name"],
            ["unknown"],
            [""],
            [None],
            123,
        ],
    )
    def test_rejects_invalid_subset(
        self,
        cleaner,
        dataframe,
        subset,
    ):
        with pytest.raises(CleaningValidationError):
            cleaner.drop_duplicates(
                dataframe,
                subset=subset,
            )

    @pytest.mark.parametrize(
        "ignore_index",
        [1, 0, "true", None],
    )
    def test_rejects_invalid_ignore_index(
        self,
        cleaner,
        dataframe,
        ignore_index,
    ):
        with pytest.raises(CleaningValidationError):
            cleaner.drop_duplicates(
                dataframe,
                ignore_index=ignore_index,
            )

    @pytest.mark.parametrize(
        "value",
        [
            None,
            [],
            {},
            "not a dataframe",
        ],
    )
    def test_rejects_non_dataframe(
        self,
        cleaner,
        value,
    ):
        with pytest.raises(CleaningValidationError):
            cleaner.drop_duplicates(value)

    def test_rejects_duplicate_column_names(self, cleaner):
        df = pd.DataFrame(
            [[1, 1]],
            columns=["name", "name"],
        )

        with pytest.raises(CleaningValidationError):
            cleaner.drop_duplicates(df)

    def test_rejects_empty_column_names(self, cleaner):
        df = pd.DataFrame(
            [[1]],
            columns=[""],
        )

        with pytest.raises(CleaningValidationError):
            cleaner.drop_duplicates(df)


class TestDuplicateCleanerSafety:
    def test_does_not_modify_original_dataframe(
        self,
        cleaner,
        dataframe,
    ):
        original = dataframe.copy(deep=True)

        cleaner.drop_duplicates(dataframe)

        pd.testing.assert_frame_equal(
            dataframe,
            original,
        )

    def test_returns_independent_dataframe(
        self,
        cleaner,
        dataframe,
    ):
        result = cleaner.drop_duplicates(dataframe)

        result.loc[result.index[0], "name"] = "Changed"

        assert dataframe.loc[0, "name"] == "Alice"

    def test_wraps_pandas_execution_errors(
        self,
        cleaner,
        dataframe,
        monkeypatch,
    ):
        def raise_runtime_error(self, **kwargs):
            raise RuntimeError("Simulated pandas execution failure")

        monkeypatch.setattr(
            pd.DataFrame,
            "drop_duplicates",
            raise_runtime_error,
        )

        with pytest.raises(CleaningOperationError) as exc_info:
            cleaner.drop_duplicates(dataframe)

        assert exc_info.value.operation == "duplicates.drop"
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert str(exc_info.value.__cause__) == (
            "Simulated pandas execution failure"
        )
