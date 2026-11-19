"""Builtin tools for dataset analysis operations."""

from __future__ import annotations

import asyncio
from typing import Any

from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.profiling.exceptions import ProfilingException
from lite_cog_agent.profiling.service import ProfilingService
from lite_cog_agent.statistics.exceptions import StatisticalAnalysisException, StatisticalExecutionError
from lite_cog_agent.statistics.service import StatisticalAnalysisService
from lite_cog_agent.tools.base import BaseTool
from lite_cog_agent.tools.builtin.common import to_json_compatible
from lite_cog_agent.tools.schemas import ToolResult


def _validate_arguments(
    arguments: dict[str, Any],
) -> str | None:
    """Validate the common tool argument container."""
    if not isinstance(arguments, dict):
        return "Tool arguments must be an object."

    return None


def _get_dataset_id(
    arguments: dict[str, Any],
) -> str:
    """Read and validate a dataset ID."""
    dataset_id = arguments.get("dataset_id")

    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be a non-empty string.")

    return dataset_id


def _success_result(
    *,
    tool_name: str,
    dataset_id: str,
    result: Any,
) -> ToolResult:
    """Convert a service result into a JSON-compatible ToolResult."""
    return ToolResult.ok(
        output=to_json_compatible(result),
        metadata={
            "tool": tool_name,
            "dataset_id": dataset_id,
        },
    )


def _failure_result(
    *,
    tool_name: str,
    dataset_id: str | None,
    error: Exception,
) -> ToolResult:
    """Convert an exception into a safe ToolResult failure."""
    if (
        isinstance(error, StatisticalExecutionError)
        and isinstance(error.__cause__, KeyError)
    ):
        return ToolResult.fail(
            "Dataset is not available in the current execution context."
        )
    if isinstance(error, (ProfilingException, StatisticalAnalysisException)):
        message = str(error)
    elif isinstance(error, ValueError):
        message = str(error)
    elif isinstance(error, KeyError | RuntimeError):
        message = "Dataset is not available in the current execution context."
    else:
        message = (
            "Analysis execution failed: "
            f"{type(error).__name__}."
        )

    return ToolResult.fail(
        error=message,
        metadata={
            "tool": tool_name,
            "dataset_id": dataset_id,
            "error_type": type(error).__name__,
        },
    )


class _BaseAnalysisTool(BaseTool):
    """Shared context and execution behavior for analysis tools."""

    def __init__(self, context: ExecutionContext) -> None:
        if not isinstance(context, ExecutionContext):
            raise TypeError("context must be an ExecutionContext")

        if context.is_closed:
            raise RuntimeError("ExecutionContext is closed")

        self._context = context

    def _validate_arguments(
        self,
        arguments: dict[str, Any],
    ) -> ToolResult | None:
        error = _validate_arguments(arguments)

        if error is not None:
            return ToolResult.fail(
                error,
                metadata={"tool": self.name},
            )

        if self._context.is_closed:
            return ToolResult.fail(
                "ExecutionContext is closed.",
                metadata={"tool": self.name},
            )

        return None


class DatasetProfileTool(_BaseAnalysisTool):
    """Generate a dataset profile and data quality report."""

    def __init__(
        self,
        context: ExecutionContext,
        profiling_service: ProfilingService,
    ) -> None:
        super().__init__(context)

        if not isinstance(profiling_service, ProfilingService):
            raise TypeError(
                "profiling_service must be a ProfilingService"
            )

        self._profiling_service = profiling_service

    @property
    def name(self) -> str:
        return "analysis.profile"

    @property
    def description(self) -> str:
        return (
            "Generate a dataset profile containing column statistics, "
            "missing values, duplicate rows, and data quality metrics."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "minLength": 1,
                    "description": "ID of a dataset already loaded in this session.",
                },
            },
            "required": ["dataset_id"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolResult:
        invalid = self._validate_arguments(arguments)
        if invalid is not None:
            return invalid

        dataset_id: str | None = None

        try:
            dataset_id = _get_dataset_id(arguments)

            dataframe = self._context.get_dataset(dataset_id)
            reference = self._context.get_dataset_reference(dataset_id)

            result = await asyncio.to_thread(
                self._profiling_service.profile,
                dataframe,
                reference,
            )

            return _success_result(
                tool_name=self.name,
                dataset_id=dataset_id,
                result=result,
            )

        except Exception as exc:
            return _failure_result(
                tool_name=self.name,
                dataset_id=dataset_id,
                error=exc,
            )


class DescriptiveAnalysisTool(_BaseAnalysisTool):
    """Calculate descriptive statistics for numeric columns."""

    def __init__(
        self,
        context: ExecutionContext,
        statistics_service: StatisticalAnalysisService,
    ) -> None:
        super().__init__(context)

        if not isinstance(
            statistics_service,
            StatisticalAnalysisService,
        ):
            raise TypeError(
                "statistics_service must be a StatisticalAnalysisService"
            )

        self._statistics_service = statistics_service

    @property
    def name(self) -> str:
        return "analysis.descriptive"

    @property
    def description(self) -> str:
        return (
            "Calculate descriptive statistics for numeric dataset columns, "
            "including count, mean, median, standard deviation, quartiles, "
            "minimum, maximum, variance, and range."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "minLength": 1},
                "columns": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "minLength": 1},
                    "uniqueItems": True,
                    "description": (
                        "Optional list of numeric columns. "
                        "Omit or pass null to analyze all eligible columns."
                    ),
                },
            },
            "required": ["dataset_id"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolResult:
        invalid = self._validate_arguments(arguments)
        if invalid is not None:
            return invalid

        dataset_id: str | None = None

        try:
            dataset_id = _get_dataset_id(arguments)
            columns = arguments.get("columns")

            if columns is not None and not isinstance(columns, list):
                raise ValueError("columns must be a list or null.")

            result = await asyncio.to_thread(
                self._statistics_service.analyze_descriptive,
                self._context,
                dataset_id,
                columns=columns,
            )

            return _success_result(
                tool_name=self.name,
                dataset_id=dataset_id,
                result=result,
            )

        except Exception as exc:
            return _failure_result(
                tool_name=self.name,
                dataset_id=dataset_id,
                error=exc,
            )


class FrequencyAnalysisTool(_BaseAnalysisTool):
    """Calculate category frequencies for a dataset column."""

    def __init__(
        self,
        context: ExecutionContext,
        statistics_service: StatisticalAnalysisService,
    ) -> None:
        super().__init__(context)

        if not isinstance(
            statistics_service,
            StatisticalAnalysisService,
        ):
            raise TypeError(
                "statistics_service must be a StatisticalAnalysisService"
            )

        self._statistics_service = statistics_service

    @property
    def name(self) -> str:
        return "analysis.frequency"

    @property
    def description(self) -> str:
        return (
            "Calculate category frequencies for a dataset column, "
            "including counts, percentages, and cumulative percentages."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "minLength": 1},
                "column": {"type": "string", "minLength": 1},
                "top_n": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "description": (
                        "Maximum number of categories to return. "
                        "Omit or pass null to return all categories."
                    ),
                },
                "dropna": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to exclude missing values.",
                },
            },
            "required": ["dataset_id", "column"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolResult:
        invalid = self._validate_arguments(arguments)
        if invalid is not None:
            return invalid

        dataset_id: str | None = None

        try:
            dataset_id = _get_dataset_id(arguments)
            column = arguments.get("column")
            top_n = arguments.get("top_n")
            dropna = arguments.get("dropna", True)

            if not isinstance(column, str) or not column.strip():
                raise ValueError("column must be a non-empty string.")

            if top_n is not None and (
                isinstance(top_n, bool)
                or not isinstance(top_n, int)
                or top_n <= 0
            ):
                raise ValueError("top_n must be a positive integer or null.")

            if not isinstance(dropna, bool):
                raise ValueError("dropna must be a boolean.")

            result = await asyncio.to_thread(
                self._statistics_service.analyze_frequency,
                self._context,
                dataset_id,
                column=column,
                top_n=top_n,
                dropna=dropna,
            )

            return _success_result(
                tool_name=self.name,
                dataset_id=dataset_id,
                result=result,
            )

        except Exception as exc:
            return _failure_result(
                tool_name=self.name,
                dataset_id=dataset_id,
                error=exc,
            )


class CorrelationAnalysisTool(_BaseAnalysisTool):
    """Calculate pairwise correlations between numeric columns."""

    def __init__(
        self,
        context: ExecutionContext,
        statistics_service: StatisticalAnalysisService,
    ) -> None:
        super().__init__(context)

        if not isinstance(
            statistics_service,
            StatisticalAnalysisService,
        ):
            raise TypeError(
                "statistics_service must be a StatisticalAnalysisService"
            )

        self._statistics_service = statistics_service

    @property
    def name(self) -> str:
        return "analysis.correlation"

    @property
    def description(self) -> str:
        return (
            "Calculate a Pearson or Spearman correlation matrix for "
            "numeric columns, including pairwise observation counts."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "minLength": 1},
                "columns": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "minLength": 1},
                    "uniqueItems": True,
                    "description": (
                        "Optional numeric columns to include. "
                        "Omit or pass null to select all eligible columns."
                    ),
                },
                "method": {
                    "type": "string",
                    "enum": ["pearson", "spearman"],
                    "default": "pearson",
                },
                "min_periods": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 2,
                },
            },
            "required": ["dataset_id"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolResult:
        invalid = self._validate_arguments(arguments)
        if invalid is not None:
            return invalid

        dataset_id: str | None = None

        try:
            dataset_id = _get_dataset_id(arguments)
            columns = arguments.get("columns")
            method = arguments.get("method", "pearson")
            min_periods = arguments.get("min_periods", 2)

            if columns is not None and not isinstance(columns, list):
                raise ValueError("columns must be a list or null.")

            if method not in ("pearson", "spearman"):
                raise ValueError(
                    "method must be 'pearson' or 'spearman'."
                )

            if (
                isinstance(min_periods, bool)
                or not isinstance(min_periods, int)
                or min_periods < 1
            ):
                raise ValueError("min_periods must be a positive integer.")

            result = await asyncio.to_thread(
                self._statistics_service.analyze_correlation,
                self._context,
                dataset_id,
                columns=columns,
                method=method,
                min_periods=min_periods,
            )

            return _success_result(
                tool_name=self.name,
                dataset_id=dataset_id,
                result=result,
            )

        except Exception as exc:
            return _failure_result(
                tool_name=self.name,
                dataset_id=dataset_id,
                error=exc,
            )


class GroupedAnalysisTool(_BaseAnalysisTool):
    """Perform grouped aggregations on a dataset."""

    SUPPORTED_AGGREGATIONS = (
        "count",
        "sum",
        "mean",
        "median",
        "min",
        "max",
        "std",
    )

    def __init__(
        self,
        context: ExecutionContext,
        statistics_service: StatisticalAnalysisService,
    ) -> None:
        super().__init__(context)

        if not isinstance(
            statistics_service,
            StatisticalAnalysisService,
        ):
            raise TypeError(
                "statistics_service must be a StatisticalAnalysisService"
            )

        self._statistics_service = statistics_service

    @property
    def name(self) -> str:
        return "analysis.grouped"

    @property
    def description(self) -> str:
        return (
            "Group dataset rows by one or more columns and calculate "
            "configured aggregations such as count, sum, mean, median, "
            "minimum, maximum, and standard deviation."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "minLength": 1},
                "group_by": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                    "uniqueItems": True,
                },
                "aggregations": {
                    "type": "object",
                    "minProperties": 1,
                    "description": (
                        "Mapping of column names to aggregation operations."
                    ),
                    "additionalProperties": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "string",
                            "enum": list(self.SUPPORTED_AGGREGATIONS),
                        },
                        "uniqueItems": True,
                    },
                },
                "dropna": {
                    "type": "boolean",
                    "default": True,
                },
                "sort": {
                    "type": "boolean",
                    "default": True,
                },
            },
            "required": ["dataset_id", "group_by", "aggregations"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolResult:
        invalid = self._validate_arguments(arguments)
        if invalid is not None:
            return invalid

        dataset_id: str | None = None

        try:
            dataset_id = _get_dataset_id(arguments)
            group_by = arguments.get("group_by")
            aggregations = arguments.get("aggregations")
            dropna = arguments.get("dropna", True)
            sort = arguments.get("sort", True)

            if (
                not isinstance(group_by, list)
                or not group_by
                or any(
                    not isinstance(column, str) or not column.strip()
                    for column in group_by
                )
            ):
                raise ValueError(
                    "group_by must be a non-empty list of column names."
                )

            if len(group_by) != len(set(group_by)):
                raise ValueError("group_by cannot contain duplicates.")

            if not isinstance(aggregations, dict) or not aggregations:
                raise ValueError("aggregations must be a non-empty object.")

            for column, operations in aggregations.items():
                if not isinstance(column, str) or not column.strip():
                    raise ValueError(
                        "aggregation column names must be non-empty strings."
                    )

                if (
                    not isinstance(operations, list)
                    or not operations
                    or any(
                        operation not in self.SUPPORTED_AGGREGATIONS
                        for operation in operations
                    )
                ):
                    raise ValueError(
                        f"Invalid aggregation operations for column '{column}'."
                    )

                if len(operations) != len(set(operations)):
                    raise ValueError(
                        f"Duplicate aggregation operations for column '{column}'."
                    )

            if not isinstance(dropna, bool):
                raise ValueError("dropna must be a boolean.")

            if not isinstance(sort, bool):
                raise ValueError("sort must be a boolean.")

            result = await asyncio.to_thread(
                self._statistics_service.analyze_grouped,
                self._context,
                dataset_id,
                group_by=group_by,
                aggregations=aggregations,
                dropna=dropna,
                sort=sort,
            )

            return _success_result(
                tool_name=self.name,
                dataset_id=dataset_id,
                result=result,
            )

        except Exception as exc:
            return _failure_result(
                tool_name=self.name,
                dataset_id=dataset_id,
                error=exc,
            )


class DistributionAnalysisTool(_BaseAnalysisTool):
    """Analyze a numeric column distribution and histogram."""

    def __init__(
        self,
        context: ExecutionContext,
        statistics_service: StatisticalAnalysisService,
    ) -> None:
        super().__init__(context)

        if not isinstance(
            statistics_service,
            StatisticalAnalysisService,
        ):
            raise TypeError(
                "statistics_service must be a StatisticalAnalysisService"
            )

        self._statistics_service = statistics_service

    @property
    def name(self) -> str:
        return "analysis.distribution"

    @property
    def description(self) -> str:
        return (
            "Analyze a numeric column distribution, including mean, "
            "median, standard deviation, skewness, kurtosis, quantiles, "
            "and histogram bins."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string", "minLength": 1},
                "column": {"type": "string", "minLength": 1},
                "bins": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 10,
                },
                "quantiles": {
                    "type": "array",
                    "items": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "uniqueItems": True,
                    "default": [0.25, 0.5, 0.75],
                    "description": (
                        "Quantile probabilities between 0 and 1."
                    ),
                },
            },
            "required": ["dataset_id", "column"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolResult:
        invalid = self._validate_arguments(arguments)
        if invalid is not None:
            return invalid

        dataset_id: str | None = None

        try:
            dataset_id = _get_dataset_id(arguments)
            column = arguments.get("column")
            bins = arguments.get("bins", 10)
            quantiles = arguments.get(
                "quantiles",
                [0.25, 0.5, 0.75],
            )

            if not isinstance(column, str) or not column.strip():
                raise ValueError("column must be a non-empty string.")

            if (
                isinstance(bins, bool)
                or not isinstance(bins, int)
                or bins <= 0
            ):
                raise ValueError("bins must be a positive integer.")

            if not isinstance(quantiles, list):
                raise ValueError("quantiles must be a list.")

            if len(quantiles) != len(set(quantiles)):
                raise ValueError("quantiles cannot contain duplicates.")

            for probability in quantiles:
                if (
                    isinstance(probability, bool)
                    or not isinstance(probability, int | float)
                    or not 0 <= probability <= 1
                ):
                    raise ValueError(
                        "quantiles must contain numbers between 0 and 1."
                    )

            result = await asyncio.to_thread(
                self._statistics_service.analyze_distribution,
                self._context,
                dataset_id,
                column=column,
                bins=bins,
                quantiles=tuple(quantiles),
            )

            return _success_result(
                tool_name=self.name,
                dataset_id=dataset_id,
                result=result,
            )

        except Exception as exc:
            return _failure_result(
                tool_name=self.name,
                dataset_id=dataset_id,
                error=exc,
            )