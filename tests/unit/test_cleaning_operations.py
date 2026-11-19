"""Unit tests for cleaning execution operations."""

import pandas as pd
import pytest

from lite_cog_agent.cleaning.operations import (
    ColumnDropOperation,
    ColumnRenameOperation,
    ColumnSelectOperation,
    DuplicateDropOperation,
    MissingDropOperation,
    MissingFillOperation,
    TypeConvertOperation,
)
from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.execution.contracts import ExecutionRequest
from lite_cog_agent.execution.dataset_store import DatasetReference
from lite_cog_agent.cleaning.exceptions import CleaningValidationError
from lite_cog_agent.execution.registry import ExecutionRegistry


@pytest.fixture
def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Alice", None],
            "age": ["20", "30", "20", "40"],
            "score": [80.0, None, 80.0, 90.0],
        }
    )


@pytest.fixture
def context_and_dataset(
    dataframe: pd.DataFrame,
) -> tuple[ExecutionContext, str]:
    context = ExecutionContext()

    reference = DatasetReference(
        name="test_dataset",
        source="unit_test",
        file_format="csv",
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        columns=tuple(dataframe.columns),
    )

    context.register_dataset(reference, dataframe)

    return context, reference.dataset_id


def make_request(
    operation: str,
    dataset_id: str | None,
    parameters: dict | None = None,
) -> ExecutionRequest:
    return ExecutionRequest(
        operation=operation,
        dataset_id=dataset_id,
        parameters=parameters or {},
    )


class TestCleaningOperationMetadata:
    def test_operation_names_are_unique(self) -> None:
        operations = [
            MissingDropOperation(),
            MissingFillOperation(),
            DuplicateDropOperation(),
            ColumnSelectOperation(),
            ColumnDropOperation(),
            ColumnRenameOperation(),
            TypeConvertOperation(),
        ]

        names = [operation.name for operation in operations]

        assert len(names) == 7
        assert len(set(names)) == 7

    @pytest.mark.parametrize(
        ("operation", "expected_name"),
        [
            (MissingDropOperation(), "missing.drop"),
            (MissingFillOperation(), "missing.fill"),
            (DuplicateDropOperation(), "duplicates.drop"),
            (ColumnSelectOperation(), "columns.select"),
            (ColumnDropOperation(), "columns.drop"),
            (ColumnRenameOperation(), "columns.rename"),
            (TypeConvertOperation(), "types.convert"),
        ],
    )
    def test_operation_name(
        self,
        operation,
        expected_name: str,
    ) -> None:
        assert operation.name == expected_name
        assert isinstance(operation.description, str)
        assert operation.description.strip()


class TestMissingOperations:
    def test_missing_drop(
        self,
        context_and_dataset,
    ) -> None:
        context, dataset_id = context_and_dataset

        operation = MissingDropOperation()
        request = make_request(
            "missing.drop",
            dataset_id,
            {"subset": ["name"]},
        )

        result = operation.execute(request, context)

        assert result["name"].notna().all()
        assert len(result) == 3

    def test_missing_fill(
        self,
        context_and_dataset,
    ) -> None:
        context, dataset_id = context_and_dataset

        operation = MissingFillOperation()
        request = make_request(
            "missing.fill",
            dataset_id,
            {
                "value": "unknown",
                "columns": ["name"],
            },
        )

        result = operation.execute(request, context)

        assert result["name"].isna().sum() == 0
        assert result.loc[3, "name"] == "unknown"


class TestDuplicateOperation:
    def test_duplicate_drop(
        self,
        context_and_dataset,
    ) -> None:
        context, dataset_id = context_and_dataset

        operation = DuplicateDropOperation()
        request = make_request(
            "duplicates.drop",
            dataset_id,
            {"subset": ["name", "age"]},
        )

        result = operation.execute(request, context)

        assert len(result) == 3
        assert not result.duplicated(
            subset=["name", "age"]
        ).any()


class TestColumnOperations:
    def test_column_select(
        self,
        context_and_dataset,
    ) -> None:
        context, dataset_id = context_and_dataset

        operation = ColumnSelectOperation()
        request = make_request(
            "columns.select",
            dataset_id,
            {"columns": ["name", "age"]},
        )

        result = operation.execute(request, context)

        assert list(result.columns) == ["name", "age"]

    def test_column_drop(
        self,
        context_and_dataset,
    ) -> None:
        context, dataset_id = context_and_dataset

        operation = ColumnDropOperation()
        request = make_request(
            "columns.drop",
            dataset_id,
            {"columns": ["score"]},
        )

        result = operation.execute(request, context)

        assert list(result.columns) == ["name", "age"]

    def test_column_rename(
        self,
        context_and_dataset,
    ) -> None:
        context, dataset_id = context_and_dataset

        operation = ColumnRenameOperation()
        request = make_request(
            "columns.rename",
            dataset_id,
            {"mapping": {"name": "full_name"}},
        )

        result = operation.execute(request, context)

        assert "full_name" in result.columns
        assert "name" not in result.columns


class TestTypeConvertOperation:
    def test_type_convert(
        self,
        context_and_dataset,
    ) -> None:
        context, dataset_id = context_and_dataset

        operation = TypeConvertOperation()
        request = make_request(
            "types.convert",
            dataset_id,
            {
                "conversions": {
                    "age": "int",
                    "name": "string",
                },
            },
        )

        result = operation.execute(request, context)

        assert str(result["age"].dtype) == "Int64"
        assert str(result["name"].dtype) == "string"


class TestCleaningOperationValidation:
    def test_requires_dataset_id(self) -> None:
        operation = ColumnSelectOperation()
        context = ExecutionContext()

        request = make_request(
            "columns.select",
            None,
            {"columns": ["name"]},
        )

        with pytest.raises(
            CleaningValidationError,
            match="dataset_id is required",
        ):
            operation.execute(request, context)

    def test_rejects_missing_dataset(
        self,
    ) -> None:
        operation = ColumnSelectOperation()
        context = ExecutionContext()

        request = make_request(
            "columns.select",
            "missing-dataset",
            {"columns": ["name"]},
        )

        with pytest.raises(
            CleaningValidationError,
            match="dataset is not available",
        ):
            operation.execute(request, context)

    def test_rejects_unknown_parameters(
        self,
        context_and_dataset,
    ) -> None:
        context, dataset_id = context_and_dataset

        operation = ColumnSelectOperation()
        request = make_request(
            "columns.select",
            dataset_id,
            {
                "columns": ["name"],
                "unexpected": True,
            },
        )

        with pytest.raises(
            CleaningValidationError,
            match="unexpected parameters",
        ):
            operation.execute(request, context)

    def test_rejects_missing_required_parameters(
        self,
        context_and_dataset,
    ) -> None:
        context, dataset_id = context_and_dataset

        operation = ColumnSelectOperation()
        request = make_request(
            "columns.select",
            dataset_id,
        )

        with pytest.raises(
            CleaningValidationError,
            match="missing required parameters",
        ):
            operation.execute(request, context)


class TestCleaningOperationSafety:
    def test_operation_does_not_modify_source_dataset(
        self,
        dataframe: pd.DataFrame,
        context_and_dataset,
    ) -> None:
        context, dataset_id = context_and_dataset
        original = dataframe.copy(deep=True)

        operation = ColumnDropOperation()
        request = make_request(
            "columns.drop",
            dataset_id,
            {"columns": ["score"]},
        )

        operation.execute(request, context)

        stored = context.get_dataset(dataset_id)

        pd.testing.assert_frame_equal(stored, original)


class TestCleaningOperationRegistry:
    def test_registers_all_cleaning_operations(self) -> None:
        registry = ExecutionRegistry()

        operations = [
            MissingDropOperation(),
            MissingFillOperation(),
            DuplicateDropOperation(),
            ColumnSelectOperation(),
            ColumnDropOperation(),
            ColumnRenameOperation(),
            TypeConvertOperation(),
        ]

        for operation in operations:
            registry.register(operation)

        assert len(registry) == 7

        for operation in operations:
            assert registry.contains(operation.name)
            assert registry.get(operation.name) is operation