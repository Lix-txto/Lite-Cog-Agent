"""Exceptions for visualization."""


class VisualizationException(Exception):
    """Base exception for visualization."""


class VisualizationValidationError(VisualizationException):
    """Raised when visualization inputs or contracts are invalid."""


class VisualizationExecutionError(VisualizationException):
    """Raised when visualization execution fails."""
