"""Application service for dataset profiling."""

from __future__ import annotations

import pandas as pd

from lite_cog_agent.execution.dataset_store import DatasetReference
from lite_cog_agent.profiling.contracts import ProfilingReport
from lite_cog_agent.profiling.dataset_profiler import DatasetProfiler
from lite_cog_agent.profiling.quality import DataQualityAnalyzer
from lite_cog_agent.profiling.exceptions import ProfilingValidationError


class ProfilingService:
    """Coordinate dataset profiling and data quality analysis."""

    def __init__(
        self,
        dataset_profiler: DatasetProfiler | None = None,
        quality_analyzer: DataQualityAnalyzer | None = None,
    ) -> None:
        """Initialize the service with optional profiling components."""
        self.dataset_profiler = (
            dataset_profiler
            if dataset_profiler is not None
            else DatasetProfiler()
        )
        self.quality_analyzer = (
            quality_analyzer
            if quality_analyzer is not None
            else DataQualityAnalyzer()
        )

        if not callable(
            getattr(self.dataset_profiler, "profile", None)
        ):
            raise ProfilingValidationError(
                "dataset_profiler must provide a callable profile method"
            )

        if not callable(
            getattr(self.quality_analyzer, "analyze", None)
        ):
            raise ProfilingValidationError(
                "quality_analyzer must provide a callable analyze method"
            )

    def profile(
        self,
        dataframe: pd.DataFrame,
        reference: DatasetReference,
    ) -> ProfilingReport:
        """Generate a complete profiling report.

        Args:
            dataframe: Dataset to analyze.
            reference: Metadata reference identifying the dataset.

        Returns:
            A ProfilingReport containing dataset and quality metrics.

        Raises:
            ProfilingValidationError: If the input is invalid.
            DatasetProfilingError: If dataset profiling fails.
            DataQualityAnalysisError: If quality analysis fails.
        """
        dataset_profile = self.dataset_profiler.profile(
            dataframe,
            reference,
        )

        quality_report = self.quality_analyzer.analyze(dataframe)

        return ProfilingReport(
            dataset_profile=dataset_profile,
            quality_report=quality_report,
        )
