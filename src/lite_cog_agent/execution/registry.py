"""Registry for managing executable data analysis operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from threading import RLock
from typing import Any

from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.execution.contracts import ExecutionRequest
from lite_cog_agent.execution.exceptions import (
    ExecutionException,
    OperationNotFoundError,
)


class BaseOperation(ABC):
    """Abstract contract for a data analysis operation.

    Every operation must provide a unique name, a description,
    and a synchronous execute method.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique operation name."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable operation description."""

    @abstractmethod
    def execute(
        self,
        request: ExecutionRequest,
        context: ExecutionContext,
    ) -> Any:
        """Execute the operation and return its output."""


class DuplicateOperationError(ExecutionException):
    """Raised when an operation name has already been registered."""

    def __init__(self, operation: str) -> None:
        super().__init__(
            f"Operation is already registered: {operation}",
            code="duplicate_operation",
            retryable=False,
        )
        self.operation = operation


class ExecutionRegistry:
    """Thread-safe registry for executable analysis operations.

    Operations are indexed by their unique names.
    Registration order is preserved by the underlying dictionary.
    """

    def __init__(self) -> None:
        self._operations: dict[str, BaseOperation] = {}
        self._lock = RLock()

    def register(self, operation: BaseOperation) -> None:
        """Register an operation.

        Raises:
            TypeError: If the object is not a BaseOperation.
            ValueError: If its name or description is invalid.
            DuplicateOperationError: If its name is already registered.
        """

        if not isinstance(operation, BaseOperation):
            raise TypeError(
                "Registered operation must be an instance of BaseOperation."
            )

        name = operation.name
        description = operation.description

        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "Operation name must be a non-empty string."
            )

        if not isinstance(description, str) or not description.strip():
            raise ValueError(
                "Operation description must be a non-empty string."
            )

        with self._lock:
            if name in self._operations:
                raise DuplicateOperationError(name)

            self._operations[name] = operation

    def get(self, name: str) -> BaseOperation:
        """Retrieve a registered operation by name.

        Raises:
            TypeError: If the name is not a string.
            OperationNotFoundError: If the operation is not registered.
        """

        if not isinstance(name, str):
            raise TypeError("Operation name must be a string.")

        with self._lock:
            try:
                return self._operations[name]
            except KeyError as exc:
                raise OperationNotFoundError(name) from exc

    def contains(self, name: object) -> bool:
        """Check whether an operation name is registered."""

        if not isinstance(name, str):
            return False

        with self._lock:
            return name in self._operations

    def list_operations(self) -> list[BaseOperation]:
        """Return a new list of registered operations."""

        with self._lock:
            return list(self._operations.values())

    def __len__(self) -> int:
        """Return the number of registered operations."""

        with self._lock:
            return len(self._operations)
