"""Unit tests for dataset loading exceptions."""

from lite_cog_agent.loading.exceptions import (
    DatasetFileNotFoundError,
    DatasetFileTooLargeError,
    DatasetLoadingError,
    DatasetParseError,
    DatasetValidationError,
    UnsupportedFileFormatError,
)


def test_all_loading_exceptions_inherit_from_base() -> None:
    exception_types = [
        UnsupportedFileFormatError,
        DatasetFileNotFoundError,
        DatasetFileTooLargeError,
        DatasetParseError,
        DatasetValidationError,
    ]

    for exception_type in exception_types:
        assert issubclass(exception_type, DatasetLoadingError)


def test_loading_exception_can_be_raised_and_caught() -> None:
    try:
        raise DatasetParseError("Unable to parse CSV file")
    except DatasetLoadingError as exc:
        assert str(exc) == "Unable to parse CSV file"


def test_specific_exception_type_is_preserved() -> None:
    try:
        raise UnsupportedFileFormatError("Unsupported format: parquet")
    except DatasetLoadingError as exc:
        assert isinstance(exc, UnsupportedFileFormatError)