
"""
Application configuration loaded from environment variables.
"""

import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_FILE)


def get_env(key: str, default: str) -> str:
    """Read an environment variable and strip whitespace."""

    return os.getenv(key, default).strip()


def get_int_env(key: str, default: int) -> int:
    """Read an integer environment variable."""

    value = os.getenv(key)

    if value is None:
        return default

    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {key} must be an integer."
        ) from exc


def get_positive_int_env(key: str, default: int) -> int:
    """Read a positive integer environment variable."""

    value = get_int_env(key, default)

    if value <= 0:
        raise ValueError(
            f"Environment variable {key} must be greater than 0."
        )

    return value


def resolve_project_path(path: Path) -> Path:
    """Resolve a path relative to the project root."""

    if path.is_absolute():
        return path

    return (PROJECT_ROOT / path).resolve()


# ---------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------

LLM_PROVIDER = get_env("LLM_PROVIDER", "deepseek").lower()

DEEPSEEK_API_KEY = get_env("DEEPSEEK_API_KEY", "")

DEEPSEEK_BASE_URL = get_env(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com",
)

DEEPSEEK_MODEL = get_env(
    "DEEPSEEK_MODEL",
    "deepseek-flash",
)


# ---------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------

AGENT_MAX_ITERATIONS = get_positive_int_env(
    "AGENT_MAX_ITERATIONS", 10
)

AGENT_MAX_TOOL_CALLS = get_positive_int_env(
    "AGENT_MAX_TOOL_CALLS", 15
)


# ---------------------------------------------------------
# Python execution configuration
# ---------------------------------------------------------

PYTHON_EXECUTION_TIMEOUT = get_positive_int_env(
    "PYTHON_EXECUTION_TIMEOUT", 30
)

PYTHON_MAX_OUTPUT_SIZE = get_positive_int_env(
    "PYTHON_MAX_OUTPUT_SIZE", 10000
)


# ---------------------------------------------------------
# Storage configuration
# ---------------------------------------------------------

DATA_DIR = resolve_project_path(
    Path(get_env("DATA_DIR", "./data"))
)

SESSION_DIR = resolve_project_path(
    Path(get_env("SESSION_DIR", "./data/sessions"))
)

TEMP_DIR = resolve_project_path(
    Path(get_env("TEMP_DIR", "./data/temp"))
)


# ---------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------

LOG_LEVEL = get_env("LOG_LEVEL", "INFO").upper()

LOG_DIR = resolve_project_path(
    Path(get_env("LOG_DIR", "./logs"))
)


# ---------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------

if LLM_PROVIDER != "deepseek":
    raise ValueError(
        f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. "
        "Currently only 'deepseek' is supported."
    )

if not DEEPSEEK_BASE_URL:
    raise ValueError("DEEPSEEK_BASE_URL cannot be empty.")

parsed_base_url = urlparse(DEEPSEEK_BASE_URL)

if parsed_base_url.scheme not in {"http", "https"}:
    raise ValueError(
        "DEEPSEEK_BASE_URL must use HTTP or HTTPS."
    )

if not parsed_base_url.netloc:
    raise ValueError(
        "DEEPSEEK_BASE_URL must contain a valid host."
    )

if not DEEPSEEK_MODEL:
    raise ValueError("DEEPSEEK_MODEL cannot be empty.")

if LOG_LEVEL not in {
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
}:
    raise ValueError(f"Invalid LOG_LEVEL: {LOG_LEVEL}")
