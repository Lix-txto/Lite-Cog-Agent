import pytest

from lite_cog_agent.tools.execution_config import (
    ToolExecutionConfig,
)


def test_default_execution_config():
    config = ToolExecutionConfig()

    assert config.timeout_seconds == 30.0
    assert config.max_concurrency == 5


def test_custom_execution_config():
    config = ToolExecutionConfig(
        timeout_seconds=60.0,
        max_concurrency=10,
    )

    assert config.timeout_seconds == 60.0
    assert config.max_concurrency == 10


def test_timeout_accepts_integer():
    config = ToolExecutionConfig(
        timeout_seconds=15,
    )

    assert config.timeout_seconds == 15


def test_timeout_must_be_numeric():
    with pytest.raises(TypeError, match="timeout_seconds"):
        ToolExecutionConfig(timeout_seconds="30")


def test_timeout_rejects_boolean():
    with pytest.raises(TypeError, match="timeout_seconds"):
        ToolExecutionConfig(timeout_seconds=True)


@pytest.mark.parametrize(
    "timeout",
    [
        0,
        -1,
        0.0,
        -10.0,
    ],
)
def test_timeout_must_be_positive(timeout):
    with pytest.raises(ValueError, match="greater than 0"):
        ToolExecutionConfig(timeout_seconds=timeout)


@pytest.mark.parametrize(
    "timeout",
    [
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
def test_timeout_must_be_finite(timeout):
    with pytest.raises(ValueError, match="finite"):
        ToolExecutionConfig(timeout_seconds=timeout)


def test_max_concurrency_must_be_integer():
    with pytest.raises(TypeError, match="max_concurrency"):
        ToolExecutionConfig(max_concurrency=2.5)


def test_max_concurrency_rejects_boolean():
    with pytest.raises(TypeError, match="max_concurrency"):
        ToolExecutionConfig(max_concurrency=True)


@pytest.mark.parametrize(
    "max_concurrency",
    [
        0,
        -1,
    ],
)
def test_max_concurrency_must_be_positive(max_concurrency):
    with pytest.raises(ValueError, match="greater than 0"):
        ToolExecutionConfig(
            max_concurrency=max_concurrency
        )


def test_execution_config_is_immutable():
    config = ToolExecutionConfig()

    with pytest.raises(AttributeError):
        config.timeout_seconds = 60.0


def test_execution_config_rejects_unknown_attributes():
    config = ToolExecutionConfig()

    with pytest.raises(AttributeError):
        config.unknown_option = 100
