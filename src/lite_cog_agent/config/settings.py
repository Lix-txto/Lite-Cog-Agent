
import os
from pathlib import Path

from dotenv import load_dotenv


# ==========================================
# Project Root
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ENV_FILE = PROJECT_ROOT / ".env"

# Load environment variables from project root.
load_dotenv(dotenv_path=ENV_FILE)


# ==========================================
# Environment Helpers
# ==========================================

def get_env(key: str, default: str) -> str:
    """Read a string environment variable."""
    return os.getenv(key, default)


def get_int_env(key: str, default: int) -> int:
    """Read and validate an integer environment variable."""
    value = os.getenv(key)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {key} must be an integer."
        ) from exc


def get_positive_int_env(key: str, default: int) -> int:
    """Read an integer environment variable and require it to be positive."""
    value = get_int_env(key, default)

    if value <= 0:
        raise ValueError(
            f"Environment variable {key} must be greater than 0."
        )

    return value


# ==========================================
# LLM Configuration
# ==========================================

LLM_PROVIDER = get_env("LLM_PROVIDER", "deepseek")

DEEPSEEK_API_KEY = get_env("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = get_env(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com",
)
DEEPSEEK_MODEL = get_env("DEEPSEEK_MODEL", "deepseek-chat")


# ==========================================
# Agent Configuration
# ==========================================

AGENT_MAX_ITERATIONS = get_positive_int_env(
    "AGENT_MAX_ITERATIONS",
    10,
)

AGENT_MAX_TOOL_CALLS = get_positive_int_env(
    "AGENT_MAX_TOOL_CALLS",
    15,
)


# ==========================================
# Python Execution Configuration
# ==========================================

PYTHON_EXECUTION_TIMEOUT = get_positive_int_env(
    "PYTHON_EXECUTION_TIMEOUT",
    30,
)

PYTHON_MAX_OUTPUT_SIZE = get_positive_int_env(
    "PYTHON_MAX_OUTPUT_SIZE",
    10000,
)


# ==========================================
# Storage Configuration
# ==========================================

DATA_DIR = Path(
    get_env("DATA_DIR", "./data")
)

SESSION_DIR = Path(
    get_env("SESSION_DIR", "./data/sessions")
)

TEMP_DIR = Path(
    get_env("TEMP_DIR", "./data/temp")
)


# ==========================================
# Logging Configuration
# ==========================================

LOG_LEVEL = get_env("LOG_LEVEL", "INFO").upper()

LOG_DIR = Path(
    get_env("LOG_DIR", "./logs")
)


# ==========================================
# Settings Validation
# ==========================================

if LLM_PROVIDER != "deepseek":
    raise ValueError(
        f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. "
        "Currently only 'deepseek' is supported."
    )

if LOG_LEVEL not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
    raise ValueError(
        f"Invalid LOG_LEVEL: {LOG_LEVEL}"
    )


# Resolve relative paths against the project root.
def resolve_project_path(path: Path) -> Path:
    """Resolve a relative path against the project root."""
    if path.is_absolute():
        return path

    return (PROJECT_ROOT / path).resolve()


DATA_DIR = resolve_project_path(DATA_DIR)
SESSION_DIR = resolve_project_path(SESSION_DIR)
TEMP_DIR = resolve_project_path(TEMP_DIR)
LOG_DIR = resolve_project_path(LOG_DIR)
