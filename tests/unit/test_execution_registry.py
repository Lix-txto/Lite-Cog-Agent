from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from abc import ABC

import pytest

from lite_cog_agent.execution import (
    ExecutionContext,
    ExecutionRequest,
    OperationNotFoundError,
)
from lite_cog_agent.execution.registry import (
    BaseOperation,
    DuplicateOperationError,
    ExecutionRegistry,
)


class DummyOperation(BaseOperation):
    """A minimal operation used for registry tests."""

    @property
    def name(self) -> str:
        return "test.dummy"

    @property
    def description(self) -> str:
        return "A dummy operation for testing."

    def execute(
        self,
        request: ExecutionRequest,
        context: ExecutionContext,
    ) -> object:
        return {"status": "ok"}


class InvalidNameOperation(DummyOperation):
    @property
    def name(self) -> str:
        return " "


class InvalidDescriptionOperation(DummyOperation):
    @property
    def description(self) -> str:
        return ""


class TestBaseOperation:
    def test_base_operation_is_abstract(self):
        assert issubclass(BaseOperation, ABC)

        with pytest.raises(TypeError):
            BaseOperation()

    def test_operation_implements_contract(self):
        operation = DummyOperation()

        assert operation.name == "test.dummy"
        assert operation.description == "A dummy operation for testing."


class TestExecutionRegistry:
    def test_create_empty_registry(self):
        registry = ExecutionRegistry()

        assert len(registry) == 0
        assert registry.list_operations() == []

    def test_register_operation(self):
        registry = ExecutionRegistry()
        operation = DummyOperation()

        registry.register(operation)

        assert len(registry) == 1
        assert registry.contains("test.dummy") is True

    def test_register_invalid_object(self):
        registry = ExecutionRegistry()

        with pytest.raises(TypeError):
            registry.register(object())

    def test_register_invalid_operation_name(self):
        registry = ExecutionRegistry()

        with pytest.raises(ValueError):
            registry.register(InvalidNameOperation())

    def test_register_invalid_operation_description(self):
        registry = ExecutionRegistry()

        with pytest.raises(ValueError):
            registry.register(InvalidDescriptionOperation())

    def test_register_duplicate_operation(self):
        registry = ExecutionRegistry()

        registry.register(DummyOperation())

        with pytest.raises(DuplicateOperationError):
            registry.register(DummyOperation())

    def test_get_registered_operation(self):
        registry = ExecutionRegistry()
        operation = DummyOperation()

        registry.register(operation)

        result = registry.get("test.dummy")

        assert result is operation

    def test_get_unregistered_operation(self):
        registry = ExecutionRegistry()

        with pytest.raises(OperationNotFoundError):
            registry.get("statistics.describe")

    def test_get_invalid_name_type(self):
        registry = ExecutionRegistry()

        with pytest.raises(TypeError):
            registry.get(123)

    def test_contains_registered_operation(self):
        registry = ExecutionRegistry()
        registry.register(DummyOperation())

        assert registry.contains("test.dummy") is True

    def test_contains_unregistered_operation(self):
        registry = ExecutionRegistry()

        assert registry.contains("test.dummy") is False

    def test_contains_invalid_name(self):
        registry = ExecutionRegistry()

        assert registry.contains(123) is False

    def test_list_operations(self):
        registry = ExecutionRegistry()
        first = DummyOperation()

        class SecondOperation(DummyOperation):
            @property
            def name(self) -> str:
                return "test.second"

        second = SecondOperation()

        registry.register(first)
        registry.register(second)

        operations = registry.list_operations()

        assert operations == [first, second]

    def test_list_operations_returns_independent_list(self):
        registry = ExecutionRegistry()
        operation = DummyOperation()

        registry.register(operation)

        operations = registry.list_operations()
        operations.clear()

        assert len(registry) == 1
        assert registry.get("test.dummy") is operation

    def test_registry_preserves_registration_order(self):
        registry = ExecutionRegistry()

        class FirstOperation(DummyOperation):
            @property
            def name(self) -> str:
                return "test.first"

        class SecondOperation(DummyOperation):
            @property
            def name(self) -> str:
                return "test.second"

        first = FirstOperation()
        second = SecondOperation()

        registry.register(first)
        registry.register(second)

        assert registry.list_operations() == [first, second]

    def test_len_updates_after_registration(self):
        registry = ExecutionRegistry()

        assert len(registry) == 0

        registry.register(DummyOperation())

        assert len(registry) == 1

def test_concurrent_registration_of_same_operation_name() -> None:
    """Only one thread should register an operation with the same name."""

    registry = ExecutionRegistry()
    barrier = Barrier(8)

    class ConcurrentOperation(BaseOperation):
        @property
        def name(self) -> str:
            return "concurrent_test_operation"

        @property
        def description(self) -> str:
            return "Operation used for concurrent registration testing."

        def execute(self, request, context):
            return None

    def register_operation() -> str:
        operation = ConcurrentOperation()

        # Make all workers attempt registration at approximately the same time.
        barrier.wait()

        try:
            registry.register(operation)
            return "registered"
        except DuplicateOperationError:
            return "duplicate"

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _: register_operation(),
                range(8),
            )
        )

    assert results.count("registered") == 1
    assert results.count("duplicate") == 7
    assert len(registry) == 1
