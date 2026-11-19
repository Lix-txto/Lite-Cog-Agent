"""Tests for DatasetReference and DatasetStore."""

import pytest

from lite_cog_agent.execution.dataset_store import (
    DatasetReference,
    DatasetStore,
)


class TestDatasetReference:
    def test_create_reference_with_defaults(self):
        reference = DatasetReference()

        assert reference.dataset_id
        assert reference.name == "dataset"
        assert reference.source == "unknown"
        assert reference.columns == ()

    def test_create_reference_with_metadata(self):
        reference = DatasetReference(
            dataset_id="ds-001",
            name="sales",
            source="upload",
            file_format="csv",
            row_count=100,
            column_count=3,
            columns=("date", "product", "amount"),
            metadata={"owner": "test"},
        )

        assert reference.dataset_id == "ds-001"
        assert reference.row_count == 100
        assert reference.column_count == 3
        assert reference.columns == ("date", "product", "amount")
        assert reference.metadata["owner"] == "test"

    def test_invalid_dataset_id(self):
        with pytest.raises(ValueError):
            DatasetReference(dataset_id=" ")

    @pytest.mark.parametrize("value", [-1, True, 1.5])
    def test_invalid_row_count(self, value):
        with pytest.raises((ValueError, TypeError)):
            DatasetReference(row_count=value)

    @pytest.mark.parametrize("value", [-1, True, 1.5])
    def test_invalid_column_count(self, value):
        with pytest.raises((ValueError, TypeError)):
            DatasetReference(column_count=value)

    def test_columns_are_immutable(self):
        columns = ["name", "age"]
        reference = DatasetReference(columns=columns)

        columns.append("salary")

        assert reference.columns == ("name", "age")

    def test_metadata_is_read_only(self):
        reference = DatasetReference(metadata={"owner": "test"})

        with pytest.raises(TypeError):
            reference.metadata["owner"] = "changed"


class TestDatasetStore:
    def setup_method(self):
        self.store = DatasetStore()
        self.reference = DatasetReference(
            dataset_id="ds-001",
            name="sales",
        )
        self.dataset = [{"amount": 100}, {"amount": 200}]

    def test_register_and_get_dataset(self):
        self.store.register(self.reference, self.dataset)

        assert self.store.get("ds-001") is self.dataset
        assert len(self.store) == 1

    def test_get_reference(self):
        self.store.register(self.reference, self.dataset)

        reference = self.store.get_reference("ds-001")

        assert reference is self.reference

    def test_register_duplicate_dataset(self):
        self.store.register(self.reference, self.dataset)

        with pytest.raises(ValueError, match="already exists"):
            self.store.register(self.reference, [{"amount": 300}])

    def test_register_invalid_reference(self):
        with pytest.raises(TypeError):
            self.store.register("invalid", self.dataset)

    def test_register_none_dataset(self):
        with pytest.raises(ValueError):
            self.store.register(self.reference, None)

    def test_get_missing_dataset(self):
        with pytest.raises(KeyError, match="not found"):
            self.store.get("missing")

    def test_get_missing_reference(self):
        with pytest.raises(KeyError):
            self.store.get_reference("missing")

    def test_contains(self):
        assert self.store.contains("ds-001") is False

        self.store.register(self.reference, self.dataset)

        assert self.store.contains("ds-001") is True

    def test_remove_dataset(self):
        self.store.register(self.reference, self.dataset)

        removed = self.store.remove("ds-001")

        assert removed is self.dataset
        assert self.store.contains("ds-001") is False
        assert len(self.store) == 0

    def test_remove_missing_dataset(self):
        with pytest.raises(KeyError):
            self.store.remove("missing")

    def test_list_references(self):
        self.store.register(self.reference, self.dataset)

        references = self.store.list_references()

        assert references == (self.reference,)

    def test_store_does_not_copy_dataset(self):
        self.store.register(self.reference, self.dataset)

        retrieved = self.store.get("ds-001")
        retrieved.append({"amount": 300})

        assert len(self.store.get("ds-001")) == 3

    def test_remove_dataset_also_removes_reference(self):
        self.store.register(self.reference, self.dataset)

        self.store.remove("ds-001")

        with pytest.raises(KeyError):
            self.store.get("ds-001")

        with pytest.raises(KeyError):
            self.store.get_reference("ds-001")

        assert self.store.contains("ds-001") is False
        assert len(self.store) == 0
