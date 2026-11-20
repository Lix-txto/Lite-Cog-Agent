"""Dependency factory for builtin tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lite_cog_agent.cleaning.service import CleaningService
from lite_cog_agent.config.settings import DATA_DIR
from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.loading.config import DatasetLoadingConfig
from lite_cog_agent.loading.service import DatasetLoadingService
from lite_cog_agent.profiling.service import ProfilingService
from lite_cog_agent.statistics.service import StatisticalAnalysisService
from lite_cog_agent.storage.file_manager import FileManager
from lite_cog_agent.visualization.service import VisualizationService
from lite_cog_agent.tools.base import BaseTool
from lite_cog_agent.tools.builtin.analysis_tools import (
    CorrelationAnalysisTool,
    DatasetProfileTool,
    DescriptiveAnalysisTool,
    DistributionAnalysisTool,
    FrequencyAnalysisTool,
    GroupedAnalysisTool,
)
from lite_cog_agent.tools.builtin.cleaning_tools import (
    DatasetCleaningTool,
)
from lite_cog_agent.tools.builtin.dataset_tools import (
    DatasetLoadTool,
)
from lite_cog_agent.tools.builtin.visualization_tools import (
    create_visualization_tools,
)


@dataclass(frozen=True, slots=True)
class BuiltinToolServices:
    """Shared service dependencies for one execution context."""

    context: ExecutionContext
    loading: DatasetLoadingService
    profiling: ProfilingService
    statistics: StatisticalAnalysisService
    visualization: VisualizationService
    cleaning: CleaningService
    file_manager: FileManager

    def __post_init__(self) -> None:
        if not isinstance(self.context, ExecutionContext):
            raise TypeError("context must be an ExecutionContext")

        if not isinstance(self.loading, DatasetLoadingService):
            raise TypeError(
                "loading must be a DatasetLoadingService"
            )

        if not isinstance(self.profiling, ProfilingService):
            raise TypeError(
                "profiling must be a ProfilingService"
            )

        if not isinstance(
            self.statistics,
            StatisticalAnalysisService,
        ):
            raise TypeError(
                "statistics must be a StatisticalAnalysisService"
            )

        if not isinstance(
            self.visualization,
            VisualizationService,
        ):
            raise TypeError(
                "visualization must be a VisualizationService"
            )

        if not isinstance(self.cleaning, CleaningService):
            raise TypeError(
                "cleaning must be a CleaningService"
            )

        if not isinstance(self.file_manager, FileManager):
            raise TypeError(
                "file_manager must be a FileManager"
            )

        if self.cleaning.context is not self.context:
            raise ValueError(
                "cleaning service must use the bundle execution context"
            )


def create_builtin_tool_services(
    context: ExecutionContext,
    *,
    loading_config: DatasetLoadingConfig | None = None,
    artifact_root: Path | str | None = None,
) -> BuiltinToolServices:
    """Create builtin service dependencies for one execution context.

    The caller owns the ExecutionContext lifecycle.

    Loading, profiling, statistics, visualization, and file storage
    services are shared by tools within this dependency bundle.
    CleaningService is constructed specifically for this context and
    must not be reused by another execution context.

    Args:
        context: Execution context owned by the caller.
        loading_config: Optional dataset loading configuration.
        artifact_root: Optional artifact storage directory. Defaults
            to DATA_DIR / "artifacts" / "visualizations".
    """

    if not isinstance(context, ExecutionContext):
        raise TypeError("context must be an ExecutionContext")

    if context.is_closed:
        raise RuntimeError(
            "Cannot create builtin services for a closed ExecutionContext"
        )

    if (
        loading_config is not None
        and not isinstance(loading_config, DatasetLoadingConfig)
    ):
        raise TypeError(
            "loading_config must be a DatasetLoadingConfig or None"
        )

    if (
        artifact_root is not None
        and not isinstance(artifact_root, (Path, str))
    ):
        raise TypeError(
            "artifact_root must be a Path, str, or None"
        )

    loading_service = DatasetLoadingService(
        config=loading_config,
    )

    profiling_service = ProfilingService()

    statistical_service = StatisticalAnalysisService()

    visualization_service = VisualizationService(
        statistical_service=statistical_service,
    )

    cleaning_service = CleaningService(
        context=context,
        profiling_service=profiling_service,
    )

    resolved_artifact_root = (
        artifact_root
        if artifact_root is not None
        else DATA_DIR / "artifacts" / "visualizations"
    )

    file_manager = FileManager(resolved_artifact_root)

    return BuiltinToolServices(
        context=context,
        loading=loading_service,
        profiling=profiling_service,
        statistics=statistical_service,
        visualization=visualization_service,
        cleaning=cleaning_service,
        file_manager=file_manager,
    )



def create_builtin_tools(
    services: BuiltinToolServices,
    *,
    dataset_root: Path | str,
) -> tuple[BaseTool, ...]:
    """Create all builtin tools using one shared service bundle.

    All tools share the ExecutionContext and the service instances
    provided by BuiltinToolServices.

    Args:
        services: Shared builtin service dependencies.
        dataset_root: Allowed root directory for dataset loading.

    Returns:
        An immutable tuple containing all builtin tools.

    Raises:
        TypeError: If services or dataset_root has an invalid type.
        RuntimeError: If the execution context has been closed.
    """

    if not isinstance(services, BuiltinToolServices):
        raise TypeError(
            "services must be a BuiltinToolServices instance"
        )

    if not isinstance(dataset_root, (Path, str)):
        raise TypeError(
            "dataset_root must be a Path or string"
        )

    context = services.context

    if context.is_closed:
        raise RuntimeError(
            "Cannot create builtin tools for a closed ExecutionContext"
        )

    tools: tuple[BaseTool, ...] = (
        DatasetLoadTool(
            context=context,
            loading_service=services.loading,
            dataset_root=dataset_root,
        ),
        DatasetCleaningTool(
            context=context,
            cleaning_service=services.cleaning,
        ),
        DatasetProfileTool(
            context=context,
            profiling_service=services.profiling,
        ),
        DescriptiveAnalysisTool(
            context=context,
            statistics_service=services.statistics,
        ),
        FrequencyAnalysisTool(
            context=context,
            statistics_service=services.statistics,
        ),
        CorrelationAnalysisTool(
            context=context,
            statistics_service=services.statistics,
        ),
        GroupedAnalysisTool(
            context=context,
            statistics_service=services.statistics,
        ),
        DistributionAnalysisTool(
            context=context,
            statistics_service=services.statistics,
        ),
        *create_visualization_tools(
            context=context,
            visualization_service=services.visualization,
            file_manager=services.file_manager,
        ),
    )

    return tools
