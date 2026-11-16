import pytest

from lite_cog_agent.config.settings import (
    get_env,
    get_int_env,
    get_positive_int_env,
    resolve_project_path,
)


def test_get_env_strips_whitespace(monkeypatch):
    monkeypatch.setenv("TEST_CONFIG_VALUE", "  hello  ")

    assert get_env("TEST_CONFIG_VALUE", "") == "hello"


def test_get_int_env(monkeypatch):
    monkeypatch.setenv("TEST_INT_VALUE", " 42 ")

    assert get_int_env("TEST_INT_VALUE", 10) == 42


def test_get_int_env_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("TEST_INT_VALUE", "invalid")

    with pytest.raises(ValueError):
        get_int_env("TEST_INT_VALUE", 10)


def test_get_positive_int_env_rejects_zero(monkeypatch):
    monkeypatch.setenv("TEST_INT_VALUE", "0")

    with pytest.raises(ValueError):
        get_positive_int_env("TEST_INT_VALUE", 10)


def test_resolve_project_path():
    result = resolve_project_path(__import__("pathlib").Path("./data"))

    assert result.is_absolute()
