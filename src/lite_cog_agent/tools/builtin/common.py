"""Common utilities for builtin tools."""

from __future__ import annotations

import math
from dataclasses import fields, is_dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID

import numpy as np
import pandas as pd


def to_json_compatible(value: Any) -> Any:
    """Convert supported Python and scientific objects into JSON values.

    Supported values:
    - JSON primitives
    - Dataclass instances
    - Enum instances
    - Mapping objects, including MappingProxyType
    - Lists and tuples
    - NumPy scalar values and arrays
    - Decimal values
    - Date/time values
    - UUID and Path

    Non-finite numeric values are converted to None.

    Unsupported objects raise TypeError instead of silently becoming
    strings. Binary data is intentionally rejected.
    """

    if value is None:
        return None

    if value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, Enum):
        return to_json_compatible(value.value)

    # NumPy scalar checks must precede Python primitive checks.
    # In particular, np.float64 may also satisfy isinstance(value, float).
    if isinstance(value, np.generic):
        return to_json_compatible(value.item())

    if isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, np.ndarray):
        return to_json_compatible(value.tolist())

    if isinstance(value, Decimal):
        if not value.is_finite():
            return None

        converted = float(value)
        return converted if math.isfinite(converted) else None

    if isinstance(value, (datetime, date, time)):
        return value.isoformat()

    if isinstance(value, (UUID, Path)):
        return str(value)

    if isinstance(value, bytes | bytearray | memoryview):
        raise TypeError(
            "Binary data cannot be serialized as a builtin tool result."
        )

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_json_compatible(
                getattr(value, field.name)
            )
            for field in fields(value)
        }

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}

        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "Mapping keys must be strings for JSON serialization."
                )

            result[key] = to_json_compatible(item)

        return result

    if isinstance(value, (list, tuple)):
        return [
            to_json_compatible(item)
            for item in value
        ]

    raise TypeError(
        "Unsupported builtin tool result type: "
        f"{type(value).__name__}."
    )
