import pytest

from lite_cog_agent.core.tool_calling_config import (
    ToolCallingConfig,
)


def test_default_tool_calling_config() -> None:
    config = ToolCallingConfig()

    assert config.max_rounds == 10
    assert config.max_tool_calls_per_round == 10


def test_custom_tool_calling_config() -> None:
    config = ToolCallingConfig(
        max_rounds=3,
        max_tool_calls_per_round=5,
    )

    assert config.max_rounds == 3
    assert config.max_tool_calls_per_round == 5


@pytest.mark.parametrize(
    "value",
    [0, -1, -10],
)
def test_max_rounds_must_be_positive(value: int) -> None:
    with pytest.raises(
        ValueError,
        match="max_rounds must be greater than 0",
    ):
        ToolCallingConfig(max_rounds=value)


@pytest.mark.parametrize(
    "value",
    [0, -1, -10],
)
def test_max_tool_calls_per_round_must_be_positive(
    value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_tool_calls_per_round must be greater than 0",
    ):
        ToolCallingConfig(max_tool_calls_per_round=value)


@pytest.mark.parametrize(
    "value",
    [True, False, 1.5, "10", None],
)
def test_max_rounds_must_be_integer(value: object) -> None:
    with pytest.raises(
        TypeError,
        match="max_rounds must be an integer",
    ):
        ToolCallingConfig(max_rounds=value)


@pytest.mark.parametrize(
    "value",
    [True, False, 1.5, "10", None],
)
def test_max_tool_calls_per_round_must_be_integer(
    value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="max_tool_calls_per_round must be an integer",
    ):
        ToolCallingConfig(max_tool_calls_per_round=value)


def test_tool_calling_config_is_immutable() -> None:
    config = ToolCallingConfig()

    with pytest.raises(AttributeError):
        config.max_rounds = 5


def test_tool_calling_config_rejects_unknown_attributes() -> None:
    config = ToolCallingConfig()

    with pytest.raises(AttributeError):
        config.unknown_attribute = 123
