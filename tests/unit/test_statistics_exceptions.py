"""Tests for statistical analysis exceptions."""

import pytest

from lite_cog_agent.statistics.exceptions import (
    CorrelationAnalysisError,
    DescriptiveAnalysisError,
    DistributionAnalysisError,
    FrequencyAnalysisError,
    GroupedAnalysisError,
    StatisticalAnalysisException,
    StatisticalExecutionError,
    StatisticalValidationError,
)


@pytest.mark.parametrize(
    "exception_type",
    [
        StatisticalValidationError,
        DescriptiveAnalysisError,
        FrequencyAnalysisError,
        CorrelationAnalysisError,
        GroupedAnalysisError,
        DistributionAnalysisError,
        StatisticalExecutionError,
    ],
)
def test_all_exceptions_inherit_from_base(exception_type):
    assert issubclass(
        exception_type,
        StatisticalAnalysisException,
    )


@pytest.mark.parametrize(
    "exception_type",
    [
        StatisticalValidationError,
        DescriptiveAnalysisError,
        FrequencyAnalysisError,
        CorrelationAnalysisError,
        GroupedAnalysisError,
        DistributionAnalysisError,
        StatisticalExecutionError,
    ],
)
def test_exception_can_be_raised_with_message(exception_type):
    message = "statistical analysis failed"

    with pytest.raises(exception_type, match=message):
        raise exception_type(message)


def test_base_exception_can_be_raised():
    with pytest.raises(StatisticalAnalysisException):
        raise StatisticalAnalysisException("base error")


def test_validation_error_is_not_analyzer_specific_error():
    assert not issubclass(
        StatisticalValidationError,
        DescriptiveAnalysisError,
    )
    assert not issubclass(
        StatisticalValidationError,
        FrequencyAnalysisError,
    )
