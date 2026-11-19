"""Contracts for dataset loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import pandas as pd

from lite_cog_agent.execution.dataset_store import DatasetReference


@dataclass(frozen=True, slots=True)
class DatasetLoadRequest:
    """Request describing a dataset loading operation."""

    file_path: Path | str
    name: str | None = None
    file_format: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize and validate request fields."""

        if isinstance(self.file_path, str):
            if not self.file_path.strip():
                raise ValueError("file_path cannot be empty")

        if not isinstance(self.file_path, (str, Path)):
            raise TypeError("file_path must be a str or Path")

        normalized_path = Path(self.file_path)
        object.__setattr__(self, "file_path", normalized_path)

        if self.name is not None:
            if not isinstance(self.name, str):
                raise TypeError("name must be a str or None")

            normalized_name = self.name.strip()

            if not normalized_name:
                raise ValueError("name cannot be empty")

            object.__setattr__(self, "name", normalized_name)

        if self.file_format is not None:
            if not isinstance(self.file_format, str):
                raise TypeError("file_format must be a str or None")

            normalized_format = self.file_format.strip().lower().lstrip(".")

            if not normalized_format:
                raise ValueError("file_format cannot be empty")

            object.__setattr__(self, "file_format", normalized_format)

        if not isinstance(self.options, Mapping):
            raise TypeError("options must be a mapping")

        options = dict(self.options)

        if any(not isinstance(key, str) for key in options):
            raise TypeError("all options keys must be strings")

        object.__setattr__(
            self,
            "options",
            MappingProxyType(options),
        )


@dataclass(frozen=True, slots=True)
class LoadedDataset:
    """A successfully parsed dataset and its metadata."""

    reference: DatasetReference
    data: pd.DataFrame

    def __post_init__(self) -> None:
        """Validate the loaded dataset."""

        if not isinstance(self.reference, DatasetReference):
            raise TypeError("reference must be a DatasetReference")

        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")