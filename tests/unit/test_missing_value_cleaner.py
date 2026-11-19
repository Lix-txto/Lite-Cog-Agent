"""Unit tests for MissingValueCleaner."""

import numpy as np
import pandas as pd
import pytest

from lite_cog_agent.cleaning.exceptions import (
    CleaningOperationError,
    CleaningValidationError,
)
from lite_cog_agent.cleaning.missing import MissingValueCleaner


@pytest.fixture
def cleaner():
    return MissingValueCleaner()


@pytest.fixture
def dataframe():
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", None, "David"],
            "age": [25, None, 30, 40],
            "city": ["NY", "LA", None, "SF"],
        }
    )


class TestMissingValueDrop:
    def test_drops_rows_with_any_missing_value(
        self, cleaner, dataframe
    ):
        result = cleaner.drop(dataframe)

        assert len(result) == 2
        assert result.index.tolist() == [0, 3]
        assert result.iloc[0]["name"] == "Alice"
        assert result.iloc[1]["name"] == "David"

    def test_drops_rows_only_when_all_selected_values_are_missing(
        self, cleaner, dataframe
    ):
        result = cleaner.drop(
            dataframe,
            how="all",
            subset=["name", "age"],
        )

        assert len(result) == 4

    def test_drops_rows_with_all_values_missing_in_subset(
        self, cleaner
    ):
        df = pd.DataFrame(
            {
                "a": [None, None, 1],
                "b": [None, 2, None],
                "c": [1, 2, 3],
            }
        )

        result = cleaner.drop(
            df,
            how="all",
            subset=["a", "b"],
        )

        assert len(result) == 2
        assert result.index.tolist() == [1, 2]

    def test_drop_with_subset_only_checks_selected_columns(
        self, cleaner
    ):
        df = pd.DataFrame(
            {
                "a": [1, None, 3],
                "b": [None, 2, 3],
            }
        )

        result = cleaner.drop(df, subset=["a"])

        assert result.index.tolist() == [0, 2]

    def test_drop_with_thresh(self, cleaner):
        df = pd.DataFrame(
            {
                "a": [1, None, None],
                "b": [2, 2, None],
                "c": [3, 3, 3],
            }
        )

        result = cleaner.drop(df, thresh=2)

        assert result.index.tolist() == [0, 1]

    def test_thresh_can_be_used_with_subset(self, cleaner):
        df = pd.DataFrame(
            {
                "a": [1, None, None],
                "b": [2, 2, None],
                "c": [None, None, 3],
            }
        )

        result = cleaner.drop(
            df,
            thresh=1,
            subset=["a", "b"],
        )

        assert result.index.tolist() == [0, 1]

    def test_thresh_zero_keeps_all_rows(self, cleaner, dataframe):
        result = cleaner.drop(dataframe, thresh=0)

        assert len(result) == len(dataframe)

    def test_drop_without_missing_values(self, cleaner):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        result = cleaner.drop(df)

        pd.testing.assert_frame_equal(result, df)
        assert result is not df

    def test_drop_empty_dataframe(self, cleaner):
        df = pd.DataFrame({"a": pd.Series(dtype=float)})

        result = cleaner.drop(df)

        assert result.empty
        assert list(result.columns) == ["a"]

    @pytest.mark.parametrize(
        "how",
        ["invalid", "", 123],
    )
    def test_rejects_invalid_how(self, cleaner, dataframe, how):
        with pytest.raises(CleaningValidationError):
            cleaner.drop(dataframe, how=how)

    @pytest.mark.parametrize(
        "thresh",
        [-1, 1.5, True, "2"],
    )
    def test_rejects_invalid_thresh(
        self, cleaner, dataframe, thresh
    ):
        with pytest.raises(CleaningValidationError):
            cleaner.drop(dataframe, thresh=thresh)

    def test_rejects_how_and_thresh_together(
        self, cleaner, dataframe
    ):
        with pytest.raises(CleaningValidationError):
            cleaner.drop(dataframe, how="any", thresh=2)

    @pytest.mark.parametrize(
        "subset",
        ["name", [], ["unknown"], ["name", "name"], [None]],
    )
    def test_rejects_invalid_subset(
        self, cleaner, dataframe, subset
    ):
        with pytest.raises(CleaningValidationError):
            cleaner.drop(dataframe, subset=subset)


class TestMissingValueFill:
    def test_fills_missing_values_with_scalar(
        self, cleaner, dataframe
    ):
        result = cleaner.fill(dataframe, value="unknown")

        assert not result.isna().any().any()
        assert result.loc[2, "name"] == "unknown"
        assert result.loc[1, "age"] == "unknown"

    def test_fills_missing_values_with_mapping(self, cleaner):
        df = pd.DataFrame(
            {
                "name": ["Alice", None],
                "age": [None, 30],
                "city": ["NY", None],
            }
        )

        result = cleaner.fill(
            df,
            value={
                "name": "Unknown",
                "age": 0,
                "city": "N/A",
            },
        )

        assert result.loc[1, "name"] == "Unknown"
        assert result.loc[0, "age"] == 0
        assert result.loc[1, "city"] == "N/A"

    def test_fills_only_selected_columns(self, cleaner):
        df = pd.DataFrame(
            {
                "a": [None, 1],
                "b": [None, 2],
            }
        )

        result = cleaner.fill(
            df,
            value=0,
            columns=["a"],
        )

        assert result["a"].isna().sum() == 0
        assert result["b"].isna().sum() == 1

    def test_mapping_can_fill_only_some_columns(self, cleaner):
        df = pd.DataFrame(
            {
                "a": [None, 1],
                "b": [None, 2],
            }
        )

        result = cleaner.fill(df, value={"a": 0})

        assert result.loc[0, "a"] == 0
        assert pd.isna(result.loc[0, "b"])

    def test_mapping_respects_selected_columns(self, cleaner):
        df = pd.DataFrame(
            {
                "a": [None, 1],
                "b": [None, 2],
            }
        )

        result = cleaner.fill(
            df,
            value={"a": 0},
            columns=["a"],
        )

        assert result.loc[0, "a"] == 0
        assert pd.isna(result.loc[0, "b"])

    def test_fill_without_missing_values(self, cleaner):
        df = pd.DataFrame({"a": [1, 2]})

        result = cleaner.fill(df, value=0)

        pd.testing.assert_frame_equal(result, df)
        assert result is not df

    def test_fill_empty_dataframe(self, cleaner):
        df = pd.DataFrame({"a": pd.Series(dtype=float)})

        result = cleaner.fill(df, value=0)

        assert result.empty
        assert list(result.columns) == ["a"]

    def test_fills_pandas_nullable_values(self, cleaner):
        df = pd.DataFrame(
            {
                "integer": pd.Series([1, pd.NA], dtype="Int64"),
                "boolean": pd.Series([True, pd.NA], dtype="boolean"),
            }
        )

        result = cleaner.fill(
            df,
            value={
                "integer": 0,
                "boolean": False,
            },
        )

        assert result["integer"].tolist() == [1, 0]
        assert result["boolean"].tolist() == [True, False]

    def test_fills_numpy_nan_and_none(self, cleaner):
        df = pd.DataFrame(
            {
                "a": [np.nan, 1.0],
                "b": [None, "value"],
            }
        )

        result = cleaner.fill(df, value=0)

        assert result["a"].tolist() == [0.0, 1.0]
        assert result["b"].tolist() == [0, "value"]

    @pytest.mark.parametrize(
        "value",
        [None, pd.NA, np.nan, [1, 2], object()],
    )
    def test_rejects_invalid_scalar_fill_value(
        self, cleaner, dataframe, value
    ):
        with pytest.raises(CleaningValidationError):
            cleaner.fill(dataframe, value=value)

    @pytest.mark.parametrize(
        "value",
        [{}, {"age": None}, {"age": np.nan}, {"age": [1, 2]}],
    )
    def test_rejects_invalid_fill_mapping(
        self, cleaner, dataframe, value
    ):
        with pytest.raises(CleaningValidationError):
            cleaner.fill(dataframe, value=value)

    def test_rejects_unknown_mapping_column(self, cleaner, dataframe):
        with pytest.raises(CleaningValidationError):
            cleaner.fill(dataframe, value={"unknown": 0})

    def test_rejects_mapping_column_outside_selection(
        self, cleaner, dataframe
    ):
        with pytest.raises(CleaningValidationError):
            cleaner.fill(
                dataframe,
                value={"age": 0},
                columns=["name"],
            )

    @pytest.mark.parametrize(
        "columns",
        ["age", [], ["unknown"], ["age", "age"], [None]],
    )
    def test_rejects_invalid_columns(
        self, cleaner, dataframe, columns
    ):
        with pytest.raises(CleaningValidationError):
            cleaner.fill(
                dataframe,
                value=0,
                columns=columns,
            )


class TestMissingValueCleanerSafety:
    def test_drop_does_not_modify_original_dataframe(
        self, cleaner, dataframe
    ):
        original = dataframe.copy(deep=True)

        cleaner.drop(dataframe)

        pd.testing.assert_frame_equal(dataframe, original)

    def test_fill_does_not_modify_original_dataframe(
        self, cleaner, dataframe
    ):
        original = dataframe.copy(deep=True)

        cleaner.fill(dataframe, value=0)

        pd.testing.assert_frame_equal(dataframe, original)

    @pytest.mark.parametrize("method", ["drop", "fill"])
    def test_rejects_non_dataframe(self, cleaner, method):
        with pytest.raises(CleaningValidationError):
            if method == "drop":
                cleaner.drop([1, 2, 3])
            else:
                cleaner.fill([1, 2, 3], value=0)

    @pytest.mark.parametrize(
        "dataframe",
        [
            pd.DataFrame([[1, 2]], columns=["a", "a"]),
            pd.DataFrame([[1, 2]], columns=["a", ""]),
            pd.DataFrame([[1, 2]], columns=["a", 1]),
        ],
    )
    def test_rejects_invalid_dataframe_columns(
        self, cleaner, dataframe
    ):
        with pytest.raises(CleaningValidationError):
            cleaner.drop(dataframe)
