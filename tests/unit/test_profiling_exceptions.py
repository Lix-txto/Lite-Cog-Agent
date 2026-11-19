"""Unit tests for profiling exceptions."""

import pytest

from lite_cog_agent.profiling.exceptions import (
    ColumnProfilingError,
    DataQualityAnalysisError,
    DatasetProfilingError,
    ProfilingException,
    ProfilingValidationError,
)


@pytest.mark.parametrize(
    "exception_type",
    [
        ProfilingValidationError,
        ColumnProfilingError,
        DatasetProfilingError,
        DataQualityAnalysisError,
    ],
)
def test_all_profiling_exceptions_inherit_from_base(
    exception_type: type[Exception],
) -> None:
    assert issubclass(exception_type, ProfilingException)


def test_profiling_exception_can_be_raised_and_caught() -> None:
    with pytest.raises(ProfilingException):
        raise ProfilingValidationError("Invalid profiling contract")


def test_specific_exception_type_is_preserved() -> None:
    with pytest.raises(ColumnProfilingError):
        raise ColumnProfilingError("Column profiling failed")
