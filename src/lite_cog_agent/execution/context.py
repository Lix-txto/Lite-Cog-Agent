"""Execution context for sharing state across analysis operations."""

from __future__ import annotations

from threading import RLock
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

from lite_cog_agent.execution.dataset_store import (
    DatasetReference,
    DatasetStore,
)


class ExecutionContext:
    """Shared context for a sequence of data analysis operations.

    A context owns the dataset references registered through it.
    Multiple contexts may share a DatasetStore, but each context
    can only access datasets registered by itself.
    """

    def __init__(
        self,
        *,
        context_id: str | None = None,
        dataset_store: DatasetStore | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if context_id is None:
            context_id = str(uuid4())

        if not isinstance(context_id, str) or not context_id.strip():
            raise ValueError("context_id must be a non-empty string")

        if dataset_store is not None and not isinstance(
            dataset_store, DatasetStore
        ):
            raise TypeError("dataset_store must be a DatasetStore")

        if metadata is None:
            metadata = {}

        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        copied_metadata = dict(metadata)

        if not all(isinstance(key, str) for key in copied_metadata):
            raise TypeError("metadata keys must be strings")

        self.context_id = context_id
        self.dataset_store =  (
            dataset_store if dataset_store is not None else DatasetStore()
        )
        self._metadata = MappingProxyType(copied_metadata)
        self._dataset_ids: set[str] = set()
        self._lock = RLock()
        self._closed = False

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return read-only context metadata."""
        return self._metadata

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        """Return dataset IDs owned by this context."""
        with self._lock:
            return tuple(sorted(self._dataset_ids))

    @property
    def is_closed(self) -> bool:
        """Whether this context has been closed."""
        return self._closed

    def register_dataset(
        self,
        reference: DatasetReference,
        dataset: Any,
    ) -> None:
        """Register a dataset and make it accessible in this context."""
        self._ensure_open()

        if not isinstance(reference, DatasetReference):
            raise TypeError("reference must be a DatasetReference")

        with self._lock:
            self._ensure_open()

            self.dataset_store.register(reference, dataset)
            self._dataset_ids.add(reference.dataset_id)

    def get_dataset(self, dataset_id: str) -> Any:
        """Retrieve a dataset owned by this context."""
        self._ensure_open()

        with self._lock:
            if dataset_id not in self._dataset_ids:
                raise KeyError(
                    f"Dataset is not available in context: {dataset_id}"
                )

            return self.dataset_store.get(dataset_id)

    def get_dataset_reference(
        self,
        dataset_id: str,
    ) -> DatasetReference:
        """Retrieve metadata for a dataset owned by this context."""
        self._ensure_open()

        with self._lock:
            if dataset_id not in self._dataset_ids:
                raise KeyError(
                    f"Dataset is not available in context: {dataset_id}"
                )

            return self.dataset_store.get_reference(dataset_id)

    def remove_dataset(self, dataset_id: str) -> Any:
        """Remove a dataset owned by this context."""
        self._ensure_open()

        with self._lock:
            if dataset_id not in self._dataset_ids:
                raise KeyError(
                    f"Dataset is not available in context: {dataset_id}"
                )

            dataset = self.dataset_store.remove(dataset_id)
            self._dataset_ids.remove(dataset_id)
            return dataset

    def close(self) -> None:
        """Release all datasets owned by this context."""
        with self._lock:
            if self._closed:
                return

            for dataset_id in tuple(self._dataset_ids):
                self.dataset_store.remove(dataset_id)

            self._dataset_ids.clear()
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ExecutionContext is closed")

    def __enter__(self) -> ExecutionContext:
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
