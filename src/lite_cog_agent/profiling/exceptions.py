"""Exceptions for dataset profiling."""


class ProfilingException(Exception):
    """Base exception for profiling operations."""


class ProfilingValidationError(ProfilingException):
    """Raised when profiling input or output contracts are invalid."""


class ColumnProfilingError(ProfilingException):
    """Raised when profiling a column fails."""


class DatasetProfilingError(ProfilingException):
    """Raised when profiling a dataset fails."""


class DataQualityAnalysisError(ProfilingException):
    """Raised when data quality analysis fails."""
