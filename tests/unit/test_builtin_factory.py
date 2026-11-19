"""Tests for builtin tool service factory."""

from __future__ import annotations

import pytest

from pathlib import Path

from lite_cog_agent.config.settings import DATA_DIR
from lite_cog_agent.storage.file_manager import FileManager
from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.loading import DatasetLoadingConfig
from lite_cog_agent.loading.service import DatasetLoadingService
from lite_cog_agent.profiling.service import ProfilingService
from lite_cog_agent.statistics.service import StatisticalAnalysisService
from lite_cog_agent.tools.builtin.factory import (
    BuiltinToolServices,
    create_builtin_tool_services,
)
from lite_cog_agent.visualization.service import VisualizationService


def test_factory_creates_all_builtin_services() -> None:
    context = ExecutionContext()

    services = create_builtin_tool_services(context)

    assert isinstance(services, BuiltinToolServices)
    assert services.context is context

    assert isinstance(
        services.loading,
        DatasetLoadingService,
    )
    assert isinstance(
        services.profiling,
        ProfilingService,
    )
    assert isinstance(
        services.statistics,
        StatisticalAnalysisService,
    )
    assert isinstance(
        services.visualization,
        VisualizationService,
    )

    assert isinstance(
        services.file_manager,
        FileManager,
    )

    assert services.cleaning.context is context


def test_visualization_reuses_shared_statistical_service() -> None:
    context = ExecutionContext()

    services = create_builtin_tool_services(context)

    assert (
        services.visualization._statistical_service
        is services.statistics
    )


def test_cleaning_reuses_shared_profiling_service() -> None:
    context = ExecutionContext()

    services = create_builtin_tool_services(context)

    assert (
        services.cleaning.profiling_service
        is services.profiling
    )


def test_different_contexts_receive_different_cleaning_services() -> None:
    first_context = ExecutionContext()
    second_context = ExecutionContext()

    first = create_builtin_tool_services(first_context)
    second = create_builtin_tool_services(second_context)

    assert first.cleaning is not second.cleaning
    assert first.cleaning.context is first_context
    assert second.cleaning.context is second_context

    assert first.context is not second.context


def test_factory_does_not_close_context() -> None:
    context = ExecutionContext()

    create_builtin_tool_services(context)

    assert context.is_closed is False


def test_factory_rejects_invalid_context() -> None:
    with pytest.raises(
        TypeError,
        match="context must be an ExecutionContext",
    ):
        create_builtin_tool_services(object())  # type: ignore[arg-type]


def test_factory_rejects_closed_context() -> None:
    context = ExecutionContext()
    context.close()

    with pytest.raises(
        RuntimeError,
        match="closed ExecutionContext",
    ):
        create_builtin_tool_services(context)


def test_factory_accepts_custom_loading_config() -> None:
    context = ExecutionContext()
    config = DatasetLoadingConfig(
        max_file_size_bytes=1024 * 1024,
        max_rows=500,
        max_columns=20,
    )

    services = create_builtin_tool_services(
        context,
        loading_config=config,
    )

    assert services.loading.config is config


def test_factory_rejects_invalid_loading_config() -> None:
    context = ExecutionContext()

    with pytest.raises(
        TypeError,
        match="loading_config must be a DatasetLoadingConfig",
    ):
        create_builtin_tool_services(
            context,
            loading_config=object(),  # type: ignore[arg-type]
        )


def test_services_container_is_frozen() -> None:
    context = ExecutionContext()
    services = create_builtin_tool_services(context)

    with pytest.raises(AttributeError):
        services.context = ExecutionContext()  # type: ignore[misc]


def test_services_container_rejects_cleaning_context_mismatch() -> None:
    first_context = ExecutionContext()
    second_context = ExecutionContext()

    services = create_builtin_tool_services(first_context)
    other_cleaning = create_builtin_tool_services(
        second_context
    ).cleaning

    with pytest.raises(
        ValueError,
        match="bundle execution context",
    ):
        BuiltinToolServices(
            context=first_context,
            loading=services.loading,
            profiling=services.profiling,
            statistics=services.statistics,
            visualization=services.visualization,
            cleaning=other_cleaning,
            file_manager=services.file_manager,
        )


def test_factory_creates_file_manager_in_default_artifact_directory() -> None:
    context = ExecutionContext()

    services = create_builtin_tool_services(context)

    expected_root = (
        DATA_DIR / "artifacts" / "visualizations"
    ).resolve()

    assert isinstance(services.file_manager, FileManager)
    assert services.file_manager.root == expected_root
    assert services.file_manager.root.is_dir()


def test_factory_accepts_custom_artifact_root(
    tmp_path: Path,
) -> None:
    context = ExecutionContext()
    artifact_root = tmp_path / "custom-artifacts"

    services = create_builtin_tool_services(
        context,
        artifact_root=artifact_root,
    )

    assert services.file_manager.root == artifact_root.resolve()
    assert services.file_manager.root.is_dir()


def test_different_factory_calls_receive_different_file_managers(
    tmp_path: Path,
) -> None:
    first_context = ExecutionContext()
    second_context = ExecutionContext()

    first = create_builtin_tool_services(
        first_context,
        artifact_root=tmp_path / "first",
    )
    second = create_builtin_tool_services(
        second_context,
        artifact_root=tmp_path / "second",
    )

    assert first.file_manager is not second.file_manager
    assert first.file_manager.root != second.file_manager.root


def test_factory_rejects_invalid_artifact_root() -> None:
    context = ExecutionContext()

    with pytest.raises(
        TypeError,
        match="artifact_root must be a Path, str, or None",
    ):
        create_builtin_tool_services(
            context,
            artifact_root=123,  # type: ignore[arg-type]
        )


def test_services_container_rejects_invalid_file_manager() -> None:
    context = ExecutionContext()
    services = create_builtin_tool_services(context)

    with pytest.raises(
        TypeError,
        match="file_manager must be a FileManager",
    ):
        BuiltinToolServices(
            context=services.context,
            loading=services.loading,
            profiling=services.profiling,
            statistics=services.statistics,
            visualization=services.visualization,
            cleaning=services.cleaning,
            file_manager=object(),  # type: ignore[arg-type]
        )
