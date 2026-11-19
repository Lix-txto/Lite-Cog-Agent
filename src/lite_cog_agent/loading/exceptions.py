"""Exceptions for dataset loading."""

from __future__ import annotations


class DatasetLoadingError(Exception):
    """Base exception for dataset loading failures."""


class UnsupportedFileFormatError(DatasetLoadingError):
    """Raised when a dataset file format is not supported."""


class DatasetFileNotFoundError(DatasetLoadingError):
    """Raised when the requested dataset file does not exist."""


class DatasetFileTooLargeError(DatasetLoadingError):
    """Raised when a dataset file exceeds the configured size limit."""


class DatasetParseError(DatasetLoadingError):
    """Raised when a dataset file cannot be parsed."""


class DatasetValidationError(DatasetLoadingError):
    """Raised when the loaded dataset fails validation."""