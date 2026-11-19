from typing import Any

from lite_cog_agent.execution import (
    BaseOperation,
    DatasetReference,
    DatasetStore,
    ExecutionContext,
    ExecutionRegistry,
    ExecutionRequest,
    ExecutionStatus,
    PythonExecutor,
)

class AnalysisOperation(BaseOperation):
    """A representative registered analysis operation."""

    @property
    def name(self) -> str:
        return "calculate_summary"

    @property
    def description(self) -> str:
        return "Calculate a simple summary from request parameters."

    def execute(
        self,
        request: ExecutionRequest,
        context: ExecutionContext,
    ) -> Any:
        values = request.parameters["values"]

        return {
            "count": len(values),
            "total": sum(values),
            "context_id": context.context_id,
        }

class DatasetAnalysisOperation(BaseOperation):
    """Analyze a dataset registered in the execution context."""

    @property
    def name(self) -> str:
        return "analyze_dataset"

    @property
    def description(self) -> str:
        return "Calculate summary statistics from a registered dataset."

    def execute(
        self,
        request: ExecutionRequest,
        context: ExecutionContext,
    ) -> Any:
        if request.dataset_id is None:
            raise ValueError("dataset_id is required")

        dataset = context.get_dataset(request.dataset_id)
        reference = context.get_dataset_reference(request.dataset_id)

        amounts = [row["amount"] for row in dataset]

        total = sum(amounts)
        count = len(amounts)

        return {
            "dataset_id": reference.dataset_id,
            "dataset_name": reference.name,
            "row_count": reference.row_count,
            "column_count": reference.column_count,
            "columns": reference.columns,
            "count": count,
            "total": total,
            "average": total / count if count else 0,
        }

def test_executor_registry_operation_integration() -> None:
    registry = ExecutionRegistry()
    context = ExecutionContext()

    operation = AnalysisOperation()
    registry.register(operation)

    executor = PythonExecutor(registry)

    request = ExecutionRequest(
        operation="calculate_summary",
        parameters={"values": [10, 20, 30, 40]},
        execution_id="integration-execution-001",
    )

    result = executor.execute(request, context)

    assert result.status is ExecutionStatus.SUCCESS
    assert result.execution_id == "integration-execution-001"
    assert result.operation == "calculate_summary"
    assert result.output == {
        "count": 4,
        "total": 100,
        "context_id": context.context_id,
    }
    assert result.error is None
    assert result.duration_ms >= 0

    context.close()

def test_executor_operation_reads_dataset_from_context() -> None:
    context = ExecutionContext()

    dataset = [
        {"product": "A", "amount": 100},
        {"product": "B", "amount": 200},
        {"product": "C", "amount": 300},
    ]

    reference = DatasetReference(
        dataset_id="integration-dataset-001",
        name="sales",
        source="integration_test",
        file_format="memory",
        row_count=3,
        column_count=2,
        columns=("product", "amount"),
        metadata={"domain": "sales"},
    )

    context.register_dataset(reference, dataset)

    registry = ExecutionRegistry()
    registry.register(DatasetAnalysisOperation())

    executor = PythonExecutor(registry)

    request = ExecutionRequest(
        operation="analyze_dataset",
        dataset_id="integration-dataset-001",
        execution_id="integration-dataset-execution-001",
    )

    result = executor.execute(request, context)

    assert result.status is ExecutionStatus.SUCCESS
    assert result.execution_id == "integration-dataset-execution-001"
    assert result.operation == "analyze_dataset"
    assert result.error is None

    assert result.output == {
        "dataset_id": "integration-dataset-001",
        "dataset_name": "sales",
        "row_count": 3,
        "column_count": 2,
        "columns": ("product", "amount"),
        "count": 3,
        "total": 600,
        "average": 200,
    }

    assert result.duration_ms >= 0

    context.close()

    assert context.is_closed is True
    assert context.dataset_ids == ()
    assert len(context.dataset_store) == 0

def test_executor_returns_failure_for_missing_dataset() -> None:
    with ExecutionContext() as context:
        registry = ExecutionRegistry()
        registry.register(DatasetAnalysisOperation())

        executor = PythonExecutor(registry)

        request = ExecutionRequest(
            operation="analyze_dataset",
            dataset_id="missing-dataset-001",
            execution_id="integration-missing-dataset-001",
        )

        result = executor.execute(request, context)

        assert result.status is ExecutionStatus.FAILED
        assert result.is_success is False

        assert result.execution_id == "integration-missing-dataset-001"
        assert result.operation == "analyze_dataset"

        assert result.output is None
        assert result.error is not None

        assert result.error.code == "operation_execution_error"
        assert result.error.retryable is False
        assert result.error.details["exception_type"] == "KeyError"
        assert "not available" in result.error.message

        assert result.duration_ms >= 0

def test_executor_respects_context_dataset_isolation() -> None:
    shared_store = DatasetStore()

    with (
        ExecutionContext(dataset_store=shared_store) as context_a,
        ExecutionContext(dataset_store=shared_store) as context_b,
    ):
        dataset = [
            {"product": "A", "amount": 100},
            {"product": "B", "amount": 200},
        ]

        reference = DatasetReference(
            dataset_id="isolated-dataset-001",
            name="private_sales",
            source="integration_test",
            row_count=2,
            column_count=2,
            columns=("product", "amount"),
        )

        context_a.register_dataset(reference, dataset)

        # 两个 Context 确实共享同一个 DatasetStore。
        assert context_a.dataset_store is shared_store
        assert context_b.dataset_store is shared_store

        # 数据集在底层 Store 中确实存在。
        assert shared_store.contains("isolated-dataset-001") is True

        # Context A 可以访问。
        assert (
            context_a.get_dataset("isolated-dataset-001")
            is dataset
        )

        registry = ExecutionRegistry()
        registry.register(DatasetAnalysisOperation())

        executor = PythonExecutor(registry)

        request = ExecutionRequest(
            operation="analyze_dataset",
            dataset_id="isolated-dataset-001",
            execution_id="integration-isolation-001",
        )

        # 尝试通过 Context B 执行分析。
        result = executor.execute(request, context_b)

        assert result.status is ExecutionStatus.FAILED
        assert result.execution_id == "integration-isolation-001"
        assert result.operation == "analyze_dataset"

        assert result.output is None
        assert result.error is not None

        assert result.error.code == "operation_execution_error"
        assert result.error.details["exception_type"] == "KeyError"
        assert "not available" in result.error.message

        # Context A 的数据仍然存在。
        assert shared_store.contains("isolated-dataset-001") is True
        assert (
            context_a.get_dataset("isolated-dataset-001")
            is dataset
        )

def test_executor_returns_failure_when_dataset_id_is_missing() -> None:
    with ExecutionContext() as context:
        registry = ExecutionRegistry()
        registry.register(DatasetAnalysisOperation())

        executor = PythonExecutor(registry)

        request = ExecutionRequest(
            operation="analyze_dataset",
            execution_id="integration-missing-dataset-id-001",
        )

        result = executor.execute(request, context)

        assert result.status is ExecutionStatus.FAILED
        assert result.execution_id == (
            "integration-missing-dataset-id-001"
        )
        assert result.operation == "analyze_dataset"

        assert result.output is None
        assert result.error is not None

        assert result.error.code == "operation_execution_error"
        assert result.error.details["exception_type"] == "ValueError"
        assert result.error.message == "dataset_id is required"

        assert result.duration_ms >= 0

def test_executor_cannot_access_dataset_after_context_is_closed() -> None:
    context = ExecutionContext()

    dataset = [
        {"product": "A", "amount": 100},
        {"product": "B", "amount": 200},
    ]

    reference = DatasetReference(
        dataset_id="lifecycle-dataset-001",
        name="lifecycle_sales",
        source="integration_test",
        row_count=2,
        column_count=2,
        columns=("product", "amount"),
    )

    context.register_dataset(reference, dataset)

    registry = ExecutionRegistry()
    registry.register(DatasetAnalysisOperation())
    executor = PythonExecutor(registry)

    request = ExecutionRequest(
        operation="analyze_dataset",
        dataset_id="lifecycle-dataset-001",
        execution_id="lifecycle-before-close-001",
    )

    # Context 正常时，执行应该成功。
    first_result = executor.execute(request, context)

    assert first_result.status is ExecutionStatus.SUCCESS
    assert first_result.output is not None
    assert first_result.output["total"] == 300

    # 关闭 Context，验证数据集生命周期清理。
    context.close()

    assert context.dataset_store.contains(
        "lifecycle-dataset-001"
    ) is False

    # 关闭后的 Context 不应继续提供数据集访问。
    second_result = executor.execute(request, context)

    assert second_result.status is ExecutionStatus.FAILED
    assert second_result.execution_id == (
        "lifecycle-before-close-001"
    )
    assert second_result.operation == "analyze_dataset"
    assert second_result.output is None
    assert second_result.error is not None

def test_executor_supports_multiple_independent_executions() -> None:
    registry = ExecutionRegistry()
    registry.register(DatasetAnalysisOperation())

    executor = PythonExecutor(registry)

    with (
        ExecutionContext() as context_a,
        ExecutionContext() as context_b,
    ):
        dataset_a = [
            {"product": "A", "amount": 100},
            {"product": "B", "amount": 200},
        ]

        reference_a = DatasetReference(
            dataset_id="multi-execution-dataset-a",
            name="sales_a",
            source="integration_test",
            row_count=2,
            column_count=2,
            columns=("product", "amount"),
        )

        dataset_b = [
            {"product": "C", "amount": 300},
            {"product": "D", "amount": 400},
        ]

        reference_b = DatasetReference(
            dataset_id="multi-execution-dataset-b",
            name="sales_b",
            source="integration_test",
            row_count=2,
            column_count=2,
            columns=("product", "amount"),
        )

        context_a.register_dataset(reference_a, dataset_a)
        context_b.register_dataset(reference_b, dataset_b)

        request_a = ExecutionRequest(
            operation="analyze_dataset",
            dataset_id="multi-execution-dataset-a",
            execution_id="multi-execution-001",
        )

        request_b = ExecutionRequest(
            operation="analyze_dataset",
            dataset_id="multi-execution-dataset-b",
            execution_id="multi-execution-002",
        )

        # 同一个 Executor 连续执行两个请求。
        result_a = executor.execute(request_a, context_a)
        result_b = executor.execute(request_b, context_b)

        assert result_a.status is ExecutionStatus.SUCCESS
        assert result_b.status is ExecutionStatus.SUCCESS

        assert result_a.execution_id == "multi-execution-001"
        assert result_b.execution_id == "multi-execution-002"

        assert result_a.output is not None
        assert result_b.output is not None

        assert result_a.output["dataset_id"] == (
            "multi-execution-dataset-a"
        )
        assert result_b.output["dataset_id"] == (
            "multi-execution-dataset-b"
        )

        assert result_a.output["total"] == 300
        assert result_b.output["total"] == 700

        assert result_a.output["average"] == 150
        assert result_b.output["average"] == 350

        assert result_a.execution_id != result_b.execution_id