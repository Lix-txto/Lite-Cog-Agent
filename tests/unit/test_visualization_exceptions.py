"""Tests for visualization exceptions."""

import pytest

from lite_cog_agent.visualization.exceptions import (
    VisualizationExecutionError,
    VisualizationException,
    VisualizationValidationError,
)


@pytest.mark.parametrize(
    "exception_type",
    [
        VisualizationValidationError,
        VisualizationExecutionError,
    ],
)
def test_all_exceptions_inherit_from_base(exception_type):
    assert issubclass(
        exception_type,
        VisualizationException,
    )


@pytest.mark.parametrize(
    "exception_type",
    [
        VisualizationException,
        VisualizationValidationError,
        VisualizationExecutionError,
    ],
)
def test_exception_can_be_raised_with_message(exception_type):
    message = "visualization failed"

    with pytest.raises(exception_type, match=message):
        raise exception_type(message)
