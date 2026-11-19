"""Statistical analysis module."""

from lite_cog_agent.statistics.contracts import (
    CorrelationAnalysisResult,
    DescriptiveStatistics,
    DescriptiveStatisticsResult,
    DistributionAnalysisResult,
    FrequencyAnalysisResult,
    FrequencyItem,
    GroupedAnalysisResult,
    HistogramBin,
)
from lite_cog_agent.statistics.correlation import CorrelationAnalyzer
from lite_cog_agent.statistics.descriptive import DescriptiveAnalyzer
from lite_cog_agent.statistics.distribution import (
    DistributionAnalysisData,
    DistributionAnalyzer,
)
from lite_cog_agent.statistics.exceptions import (
    CorrelationAnalysisError,
    DescriptiveAnalysisError,
    DistributionAnalysisError,
    FrequencyAnalysisError,
    GroupedAnalysisError,
    StatisticalAnalysisException,
    StatisticalExecutionError,
    StatisticalValidationError,
)
from lite_cog_agent.statistics.frequency import FrequencyAnalyzer
from lite_cog_agent.statistics.grouped import GroupedAnalyzer
from lite_cog_agent.statistics.service import StatisticalAnalysisService

__all__ = [
    "CorrelationAnalysisError",
    "CorrelationAnalysisResult",
    "CorrelationAnalyzer",
    "DescriptiveAnalysisError",
    "DescriptiveAnalyzer",
    "DescriptiveStatistics",
    "DescriptiveStatisticsResult",
    "DistributionAnalysisData",
    "DistributionAnalysisError",
    "DistributionAnalysisResult",
    "DistributionAnalyzer",
    "FrequencyAnalysisError",
    "FrequencyAnalysisResult",
    "FrequencyAnalyzer",
    "FrequencyItem",
    "GroupedAnalysisError",
    "GroupedAnalysisResult",
    "GroupedAnalyzer",
    "HistogramBin",
    "StatisticalAnalysisException",
    "StatisticalAnalysisService",
    "StatisticalExecutionError",
    "StatisticalValidationError",
]
