"""In-memory storage for datasets used by execution contexts."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4


def _validate_non_empty_string(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _immutable_metadata(
    metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")

    copied = dict(metadata)

    if not all(isinstance(key, str) for key in copied):
        raise TypeError("metadata keys must be strings")

    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class DatasetReference:
    """Lightweight metadata reference to a stored dataset."""

    dataset_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = "dataset"
    source: str = "unknown"
    file_format: str | None = None
    row_count: int | None = None
    column_count: int | None = None
    columns: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.dataset_id, "dataset_id")
        _validate_non_empty_string(self.name, "name")
        _validate_non_empty_string(self.source, "source")

        if self.file_format is not None:
            _validate_non_empty_string(self.file_format, "file_format")

        for field_name in ("row_count", "column_count"):
            value = getattr(self, field_name)

            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"{field_name} must be an integer or None")

                if value < 0:
                    raise ValueError(f"{field_name} must be non-negative")

        if not isinstance(self.columns, (tuple, list)):
            raise TypeError("columns must be a tuple or list")

        columns = tuple(self.columns)

        if not all(isinstance(column, str) for column in columns):
            raise TypeError("column names must be strings")

        object.__setattr__(self, "columns", columns)
        object.__setattr__(
            self,
            "metadata",
            _immutable_metadata(self.metadata),
        )


class DatasetStore:
    """Thread-safe in-memory dataset object store.

    Dataset objects are stored by reference. The store does not copy
    DataFrames or other potentially large objects.
    """

    def __init__(self) -> None:
        self._datasets: dict[str, Any] = {}
        self._references: dict[str, DatasetReference] = {}
        self._lock = RLock()

    def register(
        self,
        reference: DatasetReference,
        dataset: Any,
    ) -> None:
        """Register a dataset and its metadata reference."""
        if not isinstance(reference, DatasetReference):
            raise TypeError("reference must be a DatasetReference")

        if dataset is None:
            raise ValueError("dataset cannot be None")

        dataset_id = reference.dataset_id

        with self._lock:
            if dataset_id in self._datasets:
                raise ValueError(f"Dataset already exists: {dataset_id}")

            self._datasets[dataset_id] = dataset
            self._references[dataset_id] = reference

    def get(self, dataset_id: str) -> Any:
        """Retrieve the original dataset object."""
        _validate_non_empty_string(dataset_id, "dataset_id")

        with self._lock:
            if dataset_id not in self._datasets:
                raise KeyError(f"Dataset not found: {dataset_id}")

            return self._datasets[dataset_id]

    def get_reference(self, dataset_id: str) -> DatasetReference:
        """Retrieve dataset metadata without exposing the data object."""
        _validate_non_empty_string(dataset_id, "dataset_id")

        with self._lock:
            if dataset_id not in self._references:
                raise KeyError(f"Dataset not found: {dataset_id}")

            return self._references[dataset_id]

    def contains(self, dataset_id: str) -> bool:
        """Check whether a dataset exists."""
        with self._lock:
            return dataset_id in self._datasets

    def remove(self, dataset_id: str) -> Any:
        """Remove a dataset and return its original object."""
        _validate_non_empty_string(dataset_id, "dataset_id")

        with self._lock:
            if dataset_id not in self._datasets:
                raise KeyError(f"Dataset not found: {dataset_id}")

            dataset = self._datasets.pop(dataset_id)
            self._references.pop(dataset_id, None)
            return dataset

    def list_references(self) -> tuple[DatasetReference, ...]:
        """Return a snapshot of all registered dataset references."""
        with self._lock:
            return tuple(self._references.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._datasets)
