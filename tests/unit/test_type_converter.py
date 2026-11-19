"""Unit tests for TypeConverter."""

import numpy as np
import pandas as pd
import pytest

from lite_cog_agent.cleaning.exceptions import (
    CleaningOperationError,
    CleaningValidationError,
)
from lite_cog_agent.cleaning.type_converter import TypeConverter


@pytest.fixture
def converter():
    return TypeConverter()


@pytest.fixture
def dataframe():
    return pd.DataFrame(
        {
            "age": ["18", "25", "30"],
            "price": ["12.5", "20.8", "30.0"],
            "name": [1, 2, 3],
            "active": ["true", "false", "yes"],
            "created_at": [
                "2026-01-01",
                "2026-02-01",
                "2026-03-01",
            ],
            "untouched": ["a", "b", "c"],
        }
    )


class TestTypeConverter:
    def test_converts_integer_column(self, converter):
        df = pd.DataFrame({"age": ["18", "25", "30"]})

        result = converter.convert(
            df,
            conversions={"age": "int"},
        )

        assert str(result["age"].dtype) == "Int64"
        assert result["age"].tolist() == [18, 25, 30]

    def test_integer_conversion_preserves_missing_values(
        self,
        converter,
    ):
        df = pd.DataFrame({"age": ["18", None, "30"]})

        result = converter.convert(
            df,
            conversions={"age": "int"},
        )

        assert str(result["age"].dtype) == "Int64"
        assert result["age"].iloc[0] == 18
        assert pd.isna(result["age"].iloc[1])
        assert result["age"].iloc[2] == 30

    def test_rejects_fractional_values_for_integer_conversion(
        self,
        converter,
    ):
        df = pd.DataFrame({"age": ["18", "20.5"]})

        with pytest.raises(CleaningOperationError):
            converter.convert(
                df,
                conversions={"age": "int"},
            )

    def test_coerces_fractional_values_for_integer_conversion(
        self,
        converter,
    ):
        df = pd.DataFrame({"age": ["18", "20.5", "30"]})

        result = converter.convert(
            df,
            conversions={"age": "int"},
            errors="coerce",
        )

        assert result["age"].tolist()[0] == 18
        assert pd.isna(result["age"].tolist()[1])
        assert result["age"].tolist()[2] == 30

    def test_converts_float_column(self, converter):
        df = pd.DataFrame({"price": ["12.5", "20.8", "30"]})

        result = converter.convert(
            df,
            conversions={"price": "float"},
        )

        assert result["price"].dtype == np.dtype("float64")
        assert result["price"].tolist() == [12.5, 20.8, 30.0]

    def test_float_conversion_coerces_invalid_values(self, converter):
        df = pd.DataFrame({"price": ["12.5", "invalid", "30"]})

        result = converter.convert(
            df,
            conversions={"price": "float"},
            errors="coerce",
        )

        assert result["price"].iloc[0] == 12.5
        assert np.isnan(result["price"].iloc[1])
        assert result["price"].iloc[2] == 30.0

    def test_converts_string_column(self, converter):
        df = pd.DataFrame({"name": [1, 2, None]})

        result = converter.convert(
            df,
            conversions={"name": "string"},
        )

        assert str(result["name"].dtype) == "string"
        assert result["name"].iloc[0] == "1.0"
        assert result["name"].iloc[1] == "2.0"
        assert pd.isna(result["name"].iloc[2])

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            (["true", "false"], [True, False]),
            (["TRUE", "False"], [True, False]),
            (["yes", "no"], [True, False]),
            (["Y", "N"], [True, False]),
            (["1", "0"], [True, False]),
            ([1, 0], [True, False]),
            ([True, False], [True, False]),
        ],
    )
    def test_converts_supported_boolean_values(
        self,
        converter,
        values,
        expected,
    ):
        df = pd.DataFrame({"active": values})

        result = converter.convert(
            df,
            conversions={"active": "boolean"},
        )

        assert str(result["active"].dtype) == "boolean"
        assert result["active"].tolist() == expected

    def test_boolean_conversion_preserves_missing_values(
        self,
        converter,
    ):
        df = pd.DataFrame(
            {"active": ["true", None, "false"]}
        )

        result = converter.convert(
            df,
            conversions={"active": "boolean"},
        )

        assert result["active"].iloc[0] == True
        assert pd.isna(result["active"].iloc[1])
        assert result["active"].iloc[2] == False

    def test_rejects_unknown_boolean_values(self, converter):
        df = pd.DataFrame({"active": ["true", "unknown"]})

        with pytest.raises(CleaningOperationError):
            converter.convert(
                df,
                conversions={"active": "boolean"},
            )

    def test_coerces_unknown_boolean_values(self, converter):
        df = pd.DataFrame({"active": ["true", "unknown", "false"]})

        result = converter.convert(
            df,
            conversions={"active": "boolean"},
            errors="coerce",
        )

        assert result["active"].iloc[0] == True
        assert pd.isna(result["active"].iloc[1])
        assert result["active"].iloc[2] == False

    def test_converts_datetime_column(self, converter):
        df = pd.DataFrame(
            {"created_at": ["2026-01-01", "2026-02-01"]}
        )

        result = converter.convert(
            df,
            conversions={"created_at": "datetime"},
        )

        assert pd.api.types.is_datetime64_any_dtype(
            result["created_at"]
        )
        assert result["created_at"].iloc[0] == pd.Timestamp(
            "2026-01-01"
        )

    def test_supports_custom_datetime_format(self, converter):
        df = pd.DataFrame(
            {"created_at": ["01/18/2026", "02/18/2026"]}
        )

        result = converter.convert(
            df,
            conversions={"created_at": "datetime"},
            datetime_format="%m/%d/%Y",
        )

        assert result["created_at"].iloc[0] == pd.Timestamp(
            "2026-01-18"
        )

    def test_datetime_coerces_invalid_values(self, converter):
        df = pd.DataFrame(
            {"created_at": ["2026-01-01", "invalid"]}
        )

        result = converter.convert(
            df,
            conversions={"created_at": "datetime"},
            errors="coerce",
        )

        assert result["created_at"].iloc[0] == pd.Timestamp(
            "2026-01-01"
        )
        assert pd.isna(result["created_at"].iloc[1])

    def test_converts_multiple_columns(self, converter, dataframe):
        result = converter.convert(
            dataframe,
            conversions={
                "age": "int",
                "price": "float",
                "name": "string",
                "active": "boolean",
                "created_at": "datetime",
            },
        )

        assert str(result["age"].dtype) == "Int64"
        assert result["price"].dtype == np.dtype("float64")
        assert str(result["name"].dtype) == "string"
        assert str(result["active"].dtype) == "boolean"
        assert pd.api.types.is_datetime64_any_dtype(
            result["created_at"]
        )
        assert result["untouched"].tolist() == ["a", "b", "c"]


class TestTypeConverterValidation:
    @pytest.mark.parametrize(
        "dataframe",
        [None, [], "invalid", 123],
    )
    def test_rejects_non_dataframe(self, converter, dataframe):
        with pytest.raises(CleaningValidationError):
            converter.convert(
                dataframe,
                conversions={"age": "int"},
            )

    @pytest.mark.parametrize(
        "conversions",
        [
            None,
            [],
            {},
            "invalid",
        ],
    )
    def test_rejects_invalid_conversions(
        self,
        converter,
        dataframe,
        conversions,
    ):
        with pytest.raises(CleaningValidationError):
            converter.convert(
                dataframe,
                conversions=conversions,
            )

    @pytest.mark.parametrize(
        "conversions",
        [
            {"unknown": "int"},
            {"age": "decimal"},
            {"age": ""},
            {"": "int"},
            {None: "int"},
            {"age": None},
        ],
    )
    def test_rejects_invalid_conversion_entries(
        self,
        converter,
        dataframe,
        conversions,
    ):
        with pytest.raises(CleaningValidationError):
            converter.convert(
                dataframe,
                conversions=conversions,
            )

    @pytest.mark.parametrize("errors", ["ignore", "", None, True])
    def test_rejects_invalid_errors_policy(
        self,
        converter,
        dataframe,
        errors,
    ):
        with pytest.raises(CleaningValidationError):
            converter.convert(
                dataframe,
                conversions={"age": "int"},
                errors=errors,
            )

    @pytest.mark.parametrize(
        "datetime_format",
        ["", 123, [], False],
    )
    def test_rejects_invalid_datetime_format(
        self,
        converter,
        dataframe,
        datetime_format,
    ):
        with pytest.raises(CleaningValidationError):
            converter.convert(
                dataframe,
                conversions={"age": "int"},
                datetime_format=datetime_format,
            )

    def test_rejects_duplicate_dataframe_columns(self, converter):
        df = pd.DataFrame(
            [[1, 2]],
            columns=["age", "age"],
        )

        with pytest.raises(CleaningValidationError):
            converter.convert(
                df,
                conversions={"age": "int"},
            )

    def test_rejects_empty_dataframe_column_names(self, converter):
        df = pd.DataFrame([[1]], columns=[" "])

        with pytest.raises(CleaningValidationError):
            converter.convert(
                df,
                conversions={" ": "int"},
            )


class TestTypeConverterSafety:
    def test_does_not_modify_original_dataframe(
        self,
        converter,
        dataframe,
    ):
        original = dataframe.copy(deep=True)

        converter.convert(
            dataframe,
            conversions={
                "age": "int",
                "price": "float",
            },
        )

        pd.testing.assert_frame_equal(dataframe, original)

    def test_returns_independent_dataframe(
        self,
        converter,
        dataframe,
    ):
        result = converter.convert(
            dataframe,
            conversions={"age": "int"},
        )

        assert result is not dataframe

        result.loc[0, "untouched"] = "changed"

        assert dataframe.loc[0, "untouched"] == "a"

    def test_wraps_conversion_errors(self, converter, monkeypatch):
        df = pd.DataFrame({"age": ["18", "25"]})

        def raise_runtime_error(*args, **kwargs):
            raise RuntimeError("Simulated conversion failure")

        monkeypatch.setattr(
            pd,
            "to_numeric",
            raise_runtime_error,
        )

        with pytest.raises(CleaningOperationError) as exc_info:
            converter.convert(
                df,
                conversions={"age": "int"},
            )

        assert exc_info.value.operation == "types.convert"
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert str(exc_info.value.__cause__) == (
            "Simulated conversion failure"
        )
