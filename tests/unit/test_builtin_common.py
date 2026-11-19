"""Tests for builtin tool common utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from uuid import UUID

import numpy as np
import pandas as pd
import pytest

from lite_cog_agent.tools.builtin.common import (
    to_json_compatible,
)


class SampleStatus(str, Enum):
    SUCCESS = "success"


@dataclass(frozen=True, slots=True)
class SampleResult:
    name: str
    status: SampleStatus
    values: tuple[int | None, ...]


def test_converts_dataclass_and_enum() -> None:
    value = SampleResult(
        name="sales",
        status=SampleStatus.SUCCESS,
        values=(1, 2, None),
    )

    result = to_json_compatible(value)

    assert result == {
        "name": "sales",
        "status": "success",
        "values": [1, 2, None],
    }


def test_converts_mapping_proxy_recursively() -> None:
    value = MappingProxyType(
        {
            "count": 3,
            "items": (
                MappingProxyType(
                    {
                        "name": "Sales",
                        "percentage": 50.0,
                    }
                ),
            ),
        }
    )

    result = to_json_compatible(value)

    assert result == {
        "count": 3,
        "items": [
            {
                "name": "Sales",
                "percentage": 50.0,
            }
        ],
    }


def test_converts_numpy_scalars() -> None:
    value = {
        "count": np.int64(10),
        "mean": np.float64(2.5),
        "missing": np.float64(np.nan),
    }

    result = to_json_compatible(value)

    assert result == {
        "count": 10,
        "mean": 2.5,
        "missing": None,
    }

    assert type(result["count"]) is int
    assert type(result["mean"]) is float


def test_converts_numpy_array() -> None:
    value = np.array(
        [1.0, 2.0, np.nan],
        dtype=np.float64,
    )

    assert to_json_compatible(value) == [
        1.0,
        2.0,
        None,
    ]


def test_converts_non_finite_floats_to_none() -> None:
    result = to_json_compatible(
        {
            "nan": float("nan"),
            "positive": float("inf"),
            "negative": float("-inf"),
            "normal": 1.25,
        }
    )

    assert result == {
        "nan": None,
        "positive": None,
        "negative": None,
        "normal": 1.25,
    }


def test_converts_pandas_missing_values() -> None:
    assert to_json_compatible(pd.NA) is None
    assert to_json_compatible(pd.NaT) is None


def test_converts_decimal_values() -> None:
    result = to_json_compatible(
        {
            "normal": Decimal("12.5"),
            "infinity": Decimal("Infinity"),
            "overflow": Decimal("1e9999"),
        }
    )

    assert result == {
        "normal": 12.5,
        "infinity": None,
        "overflow": None,
    }


def test_converts_datetime_values() -> None:
    result = to_json_compatible(
        {
            "date": date(2026, 11, 20),
            "datetime": datetime(2026, 11, 20, 12, 30),
        }
    )

    assert result == {
        "date": "2026-11-20",
        "datetime": "2026-11-20T12:30:00",
    }


def test_converts_uuid_and_path() -> None:
    value = {
        "id": UUID("12345678-1234-5678-1234-567812345678"),
        "path": Path("data/sales.csv"),
    }

    result = to_json_compatible(value)

    assert result == {
        "id": "12345678-1234-5678-1234-567812345678",
        "path": "data/sales.csv",
    }


def test_preserves_json_primitives() -> None:
    value = {
        "none": None,
        "bool": True,
        "int": 10,
        "float": 2.5,
        "string": "sales",
    }

    assert to_json_compatible(value) == value


def test_result_can_be_encoded_as_strict_json() -> None:
    value = {
        "count": np.int64(5),
        "mean": np.float64(np.nan),
        "groups": (
            {"name": "Sales", "count": np.int64(3)},
        ),
    }

    result = to_json_compatible(value)

    encoded = json.dumps(
        result,
        ensure_ascii=False,
        allow_nan=False,
    )

    assert json.loads(encoded) == {
        "count": 5,
        "mean": None,
        "groups": [
            {"name": "Sales", "count": 3},
        ],
    }


@pytest.mark.parametrize(
    "value",
    [
        b"image bytes",
        bytearray(b"image bytes"),
        memoryview(b"image bytes"),
    ],
)
def test_rejects_binary_data(value: object) -> None:
    with pytest.raises(
        TypeError,
        match="Binary data cannot be serialized",
    ):
        to_json_compatible(value)


def test_rejects_mapping_with_non_string_keys() -> None:
    with pytest.raises(
        TypeError,
        match="Mapping keys must be strings",
    ):
        to_json_compatible({1: "invalid"})


def test_rejects_unsupported_objects() -> None:
    with pytest.raises(
        TypeError,
        match="Unsupported builtin tool result type",
    ):
        to_json_compatible(object())
