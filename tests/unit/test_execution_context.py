"""Tests for ExecutionContext."""

import pytest

from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.execution.dataset_store import (
    DatasetReference,
    DatasetStore,
)


class TestExecutionContext:
    def setup_method(self):
        self.context = ExecutionContext()
        self.reference = DatasetReference(
            dataset_id="ds-001",
            name="sales",
            source="upload",
        )
        self.dataset = [{"amount": 100}, {"amount": 200}]

    def test_create_context(self):
        context = ExecutionContext()

        assert context.context_id
        assert context.is_closed is False
        assert context.dataset_ids == ()

    def test_create_context_with_metadata(self):
        context = ExecutionContext(
            context_id="ctx-001",
            metadata={"user_id": "user-001"},
        )

        assert context.context_id == "ctx-001"
        assert context.metadata["user_id"] == "user-001"

    def test_metadata_is_read_only(self):
        context = ExecutionContext(metadata={"user_id": "user-001"})

        with pytest.raises(TypeError):
            context.metadata["user_id"] = "user-002"

    def test_register_and_get_dataset(self):
        self.context.register_dataset(self.reference, self.dataset)

        assert self.context.get_dataset("ds-001") is self.dataset
        assert self.context.dataset_ids == ("ds-001",)

    def test_get_dataset_reference(self):
        self.context.register_dataset(self.reference, self.dataset)

        reference = self.context.get_dataset_reference("ds-001")

        assert reference is self.reference

    def test_get_unregistered_dataset(self):
        with pytest.raises(KeyError, match="not available"):
            self.context.get_dataset("missing")

    def test_remove_dataset(self):
        self.context.register_dataset(self.reference, self.dataset)

        removed = self.context.remove_dataset("ds-001")

        assert removed is self.dataset
        assert self.context.dataset_ids == ()

    def test_remove_unregistered_dataset(self):
        with pytest.raises(KeyError):
            self.context.remove_dataset("missing")

    def test_close_context(self):
        self.context.register_dataset(self.reference, self.dataset)

        self.context.close()

        assert self.context.is_closed is True
        assert self.context.dataset_ids == ()
        assert len(self.context.dataset_store) == 0

    def test_close_context_is_idempotent(self):
        self.context.close()
        self.context.close()

        assert self.context.is_closed is True

    def test_closed_context_cannot_register_dataset(self):
        self.context.close()

        with pytest.raises(RuntimeError, match="closed"):
            self.context.register_dataset(self.reference, self.dataset)

    def test_closed_context_cannot_get_dataset(self):
        self.context.register_dataset(self.reference, self.dataset)
        self.context.close()

        with pytest.raises(RuntimeError, match="closed"):
            self.context.get_dataset("ds-001")

    def test_context_manager(self):
        with ExecutionContext() as context:
            context.register_dataset(self.reference, self.dataset)

            assert context.get_dataset("ds-001") is self.dataset

        assert context.is_closed is True

    def test_context_isolation(self):
        shared_store = DatasetStore()

        context_a = ExecutionContext(dataset_store=shared_store)
        context_b = ExecutionContext(dataset_store=shared_store)

        assert context_a.dataset_store is shared_store
        assert context_b.dataset_store is shared_store

        context_a.register_dataset(self.reference, self.dataset)

        assert shared_store.contains("ds-001") is True
        assert context_a.get_dataset("ds-001") is self.dataset

        with pytest.raises(KeyError, match="not available"):
            context_b.get_dataset("ds-001")

    def test_context_close_only_cleans_its_own_datasets(self):
        shared_store = DatasetStore()

        context_a = ExecutionContext(dataset_store=shared_store)
        context_b = ExecutionContext(dataset_store=shared_store)

        context_a.register_dataset(
            DatasetReference(dataset_id="ds-a"),
            ["a"],
        )
        context_b.register_dataset(
            DatasetReference(dataset_id="ds-b"),
            ["b"],
        )

        context_a.close()

        assert shared_store.contains("ds-a") is False
        assert shared_store.contains("ds-b") is True
        assert context_b.get_dataset("ds-b") == ["b"]

    def test_context_cannot_be_created_with_invalid_store(self):
        with pytest.raises(TypeError):
            ExecutionContext(dataset_store="invalid")

    def test_context_cannot_be_created_with_invalid_metadata(self):
        with pytest.raises(TypeError):
            ExecutionContext(metadata=["invalid"])

    def test_close_releases_all_owned_datasets_without_affecting_shared_context(self):
        shared_store = DatasetStore()

        context_a = ExecutionContext(dataset_store=shared_store)
        context_b = ExecutionContext(dataset_store=shared_store)

        reference_a1 = DatasetReference(dataset_id="ds-a-001")
        reference_a2 = DatasetReference(dataset_id="ds-a-002")
        reference_b = DatasetReference(dataset_id="ds-b-001")

        context_a.register_dataset(reference_a1, ["a1"])
        context_a.register_dataset(reference_a2, ["a2"])
        context_b.register_dataset(reference_b, ["b"])

        assert len(shared_store) == 3

        context_a.close()

        assert context_a.is_closed is True
        assert context_a.dataset_ids == ()

        assert shared_store.contains("ds-a-001") is False
        assert shared_store.contains("ds-a-002") is False
        assert shared_store.contains("ds-b-001") is True

        assert len(shared_store) == 1
        assert context_b.get_dataset("ds-b-001") == ["b"]

        with pytest.raises(KeyError):
            shared_store.get_reference("ds-a-001")

        with pytest.raises(KeyError):
            shared_store.get_reference("ds-a-002")
