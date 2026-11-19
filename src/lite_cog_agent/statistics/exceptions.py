"""Exceptions for statistical analysis."""


class StatisticalAnalysisException(Exception):
    """Base exception for statistical analysis."""


class StatisticalValidationError(StatisticalAnalysisException):
    """Raised when statistical analysis inputs are invalid."""


class DescriptiveAnalysisError(StatisticalAnalysisException):
    """Raised when descriptive statistics analysis fails."""


class FrequencyAnalysisError(StatisticalAnalysisException):
    """Raised when frequency analysis fails."""


class CorrelationAnalysisError(StatisticalAnalysisException):
    """Raised when correlation analysis fails."""


class GroupedAnalysisError(StatisticalAnalysisException):
    """Raised when grouped analysis fails."""


class DistributionAnalysisError(StatisticalAnalysisException):
    """Raised when distribution analysis fails."""


class StatisticalExecutionError(StatisticalAnalysisException):
    """Raised when statistical analysis execution fails."""
