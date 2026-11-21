"""Tests for builtin tool composition factory."""

from __future__ import annotations

from pathlib import Path

import pytest

from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.tools.base import BaseTool
from lite_cog_agent.tools.builtin import (
    BuiltinToolServices,
    create_builtin_tool_services,
    create_builtin_tools,
)

EXPECTED_TOOL_NAMES = (
    "dataset.load",
    "dataset.clean",
    "analysis.profile",
    "analysis.descriptive",
    "analysis.frequency",
    "analysis.correlation",
    "analysis.grouped",
    "analysis.derived_metric",
    "analysis.distribution",
    "visualization.bar",
    "visualization.grouped_bar",
    "visualization.line",
    "visualization.histogram",
    "visualization.scatter",
    "visualization.pie",
    "visualization.box",
)


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext()


@pytest.fixture
def services(
    context: ExecutionContext,
    tmp_path: Path,
) -> BuiltinToolServices:
    return create_builtin_tool_services(
        context,
        artifact_root=tmp_path / "artifacts",
    )


def test_factory_creates_all_builtin_tools(
    services: BuiltinToolServices,
    tmp_path: Path,
) -> None:
    tools = create_builtin_tools(
        services,
        dataset_root=tmp_path / "datasets",
    )

    assert isinstance(tools, tuple)
    assert len(tools) == 16

    assert all(
        isinstance(tool, BaseTool)
        for tool in tools
    )

    assert tuple(tool.name for tool in tools) == EXPECTED_TOOL_NAMES


def test_factory_creates_tools_with_unique_names(
    services: BuiltinToolServices,
    tmp_path: Path,
) -> None:
    tools = create_builtin_tools(
        services,
        dataset_root=tmp_path,
    )

    names = [tool.name for tool in tools]

    assert len(names) == len(set(names))


def test_factory_reuses_shared_execution_context(
    services: BuiltinToolServices,
    tmp_path: Path,
) -> None:
    tools = create_builtin_tools(
        services,
        dataset_root=tmp_path,
    )

    for tool in tools:
        assert getattr(tool, "_context", None) is services.context


def test_factory_reuses_shared_services(
    services: BuiltinToolServices,
    tmp_path: Path,
) -> None:
    tools = create_builtin_tools(
        services,
        dataset_root=tmp_path,
    )

    by_name = {
        tool.name: tool
        for tool in tools
    }

    assert (
        by_name["dataset.load"]._loading_service
        is services.loading
    )

    assert (
        by_name["dataset.clean"]._cleaning_service
        is services.cleaning
    )

    assert (
        by_name["analysis.profile"]._profiling_service
        is services.profiling
    )

    assert (
        by_name["analysis.descriptive"]._statistics_service
        is services.statistics
    )
    assert (
        by_name["analysis.derived_metric"]._statistics_service
        is services.statistics
    )

    assert (
        by_name["visualization.bar"]._visualization_service
        is services.visualization
    )

    assert (
        by_name["visualization.bar"]._file_manager
        is services.file_manager
    )


def test_factory_does_not_close_execution_context(
    services: BuiltinToolServices,
    tmp_path: Path,
) -> None:
    create_builtin_tools(
        services,
        dataset_root=tmp_path,
    )

    assert services.context.is_closed is False


def test_factory_rejects_invalid_services(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        TypeError,
        match="BuiltinToolServices",
    ):
        create_builtin_tools(
            object(),  # type: ignore[arg-type]
            dataset_root=tmp_path,
        )


@pytest.mark.parametrize(
    "dataset_root",
    [
        None,
        123,
        object(),
    ],
)
def test_factory_rejects_invalid_dataset_root(
    services: BuiltinToolServices,
    dataset_root: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="dataset_root",
    ):
        create_builtin_tools(
            services,
            dataset_root=dataset_root,  # type: ignore[arg-type]
        )


def test_factory_rejects_closed_execution_context(
    context: ExecutionContext,
    tmp_path: Path,
) -> None:
    services = create_builtin_tool_services(context)

    context.close()

    with pytest.raises(
        RuntimeError,
        match="closed ExecutionContext",
    ):
        create_builtin_tools(
            services,
            dataset_root=tmp_path,
        )
