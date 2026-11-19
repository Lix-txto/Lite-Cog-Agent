"""Unit tests for ProfilingService."""

from unittest.mock import Mock, call

import pandas as pd
import pytest

from lite_cog_agent.execution.dataset_store import DatasetReference
from lite_cog_agent.profiling.contracts import (
    DataQualityReport,
    DatasetProfile,
    ProfilingReport,
)
from lite_cog_agent.profiling.dataset_profiler import DatasetProfiler
from lite_cog_agent.profiling.exceptions import (
    DataQualityAnalysisError,
    DatasetProfilingError,
    ProfilingValidationError,
)
from lite_cog_agent.profiling.quality import DataQualityAnalyzer
from lite_cog_agent.profiling.service import ProfilingService


@pytest.fixture
def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", None],
            "age": [20, None, 30],
        }
    )


@pytest.fixture
def reference() -> DatasetReference:
    return DatasetReference(
        dataset_id="dataset-1",
        name="test-dataset",
        source="test.csv",
        file_format="csv",
        row_count=3,
        column_count=2,
        columns=("name", "age"),
    )


def test_profile_returns_complete_report(
    dataframe: pd.DataFrame,
    reference: DatasetReference,
) -> None:
    service = ProfilingService()

    result = service.profile(dataframe, reference)

    assert isinstance(result, ProfilingReport)
    assert isinstance(result.dataset_profile, DatasetProfile)
    assert isinstance(result.quality_report, DataQualityReport)

    assert result.dataset_profile.dataset_id == reference.dataset_id
    assert result.dataset_profile.dataset_name == reference.name
    assert result.dataset_profile.row_count == 3
    assert result.dataset_profile.column_count == 2

    assert result.quality_report.total_cells == 6
    assert result.quality_report.missing_cells == 2
    assert result.quality_report.completeness == pytest.approx(
        4 / 6 * 100
    )


def test_profile_uses_default_components(
    dataframe: pd.DataFrame,
    reference: DatasetReference,
) -> None:
    service = ProfilingService()

    assert isinstance(service.dataset_profiler, DatasetProfiler)
    assert isinstance(service.quality_analyzer, DataQualityAnalyzer)

    result = service.profile(dataframe, reference)

    assert isinstance(result, ProfilingReport)


def test_profile_supports_custom_dataset_profiler(
    dataframe: pd.DataFrame,
    reference: DatasetReference,
) -> None:
    dataset_profiler = Mock(spec=DatasetProfiler)
    quality_analyzer = Mock(spec=DataQualityAnalyzer)

    expected_dataset_profile = DatasetProfiler().profile(
        dataframe,
        reference,
    )
    expected_quality_report = DataQualityAnalyzer().analyze(dataframe)

    dataset_profiler.profile.return_value = expected_dataset_profile
    quality_analyzer.analyze.return_value = expected_quality_report

    service = ProfilingService(
        dataset_profiler=dataset_profiler,
        quality_analyzer=quality_analyzer,
    )

    result = service.profile(dataframe, reference)

    assert result.dataset_profile is expected_dataset_profile
    dataset_profiler.profile.assert_called_once_with(
        dataframe,
        reference,
    )


def test_profile_supports_custom_quality_analyzer(
    dataframe: pd.DataFrame,
    reference: DatasetReference,
) -> None:
    dataset_profiler = Mock(spec=DatasetProfiler)
    quality_analyzer = Mock(spec=DataQualityAnalyzer)

    expected_dataset_profile = DatasetProfiler().profile(
        dataframe,
        reference,
    )
    expected_quality_report = DataQualityAnalyzer().analyze(dataframe)

    dataset_profiler.profile.return_value = expected_dataset_profile
    quality_analyzer.analyze.return_value = expected_quality_report

    service = ProfilingService(
        dataset_profiler=dataset_profiler,
        quality_analyzer=quality_analyzer,
    )

    result = service.profile(dataframe, reference)

    assert result.quality_report is expected_quality_report
    quality_analyzer.analyze.assert_called_once_with(dataframe)


def test_profile_passes_same_dataframe_to_both_components(
    dataframe: pd.DataFrame,
    reference: DatasetReference,
) -> None:
    dataset_profiler = Mock(spec=DatasetProfiler)
    quality_analyzer = Mock(spec=DataQualityAnalyzer)

    dataset_profiler.profile.return_value = (
        DatasetProfiler().profile(dataframe, reference)
    )
    quality_analyzer.analyze.return_value = (
        DataQualityAnalyzer().analyze(dataframe)
    )

    service = ProfilingService(
        dataset_profiler=dataset_profiler,
        quality_analyzer=quality_analyzer,
    )

    service.profile(dataframe, reference)

    dataset_profiler.profile.assert_called_once()
    quality_analyzer.analyze.assert_called_once()

    assert (
        dataset_profiler.profile.call_args
        == call(dataframe, reference)
    )
    assert quality_analyzer.analyze.call_args == call(dataframe)

    assert (
        dataset_profiler.profile.call_args.args[0]
        is quality_analyzer.analyze.call_args.args[0]
    )


def test_profile_propagates_dataset_profiling_error(
    dataframe: pd.DataFrame,
    reference: DatasetReference,
) -> None:
    dataset_profiler = Mock(spec=DatasetProfiler)
    quality_analyzer = Mock(spec=DataQualityAnalyzer)

    expected_error = DatasetProfilingError("dataset profiling failed")
    dataset_profiler.profile.side_effect = expected_error

    service = ProfilingService(
        dataset_profiler=dataset_profiler,
        quality_analyzer=quality_analyzer,
    )

    with pytest.raises(DatasetProfilingError) as exc_info:
        service.profile(dataframe, reference)

    assert exc_info.value is expected_error
    quality_analyzer.analyze.assert_not_called()


def test_profile_propagates_quality_analysis_error(
    dataframe: pd.DataFrame,
    reference: DatasetReference,
) -> None:
    dataset_profiler = Mock(spec=DatasetProfiler)
    quality_analyzer = Mock(spec=DataQualityAnalyzer)

    expected_error = DataQualityAnalysisError("quality analysis failed")
    dataset_profiler.profile.return_value = (
        DatasetProfiler().profile(dataframe, reference)
    )
    quality_analyzer.analyze.side_effect = expected_error

    service = ProfilingService(
        dataset_profiler=dataset_profiler,
        quality_analyzer=quality_analyzer,
    )

    with pytest.raises(DataQualityAnalysisError) as exc_info:
        service.profile(dataframe, reference)

    assert exc_info.value is expected_error
    dataset_profiler.profile.assert_called_once()


def test_profile_propagates_validation_error(
    dataframe: pd.DataFrame,
    reference: DatasetReference,
) -> None:
    dataset_profiler = Mock(spec=DatasetProfiler)
    quality_analyzer = Mock(spec=DataQualityAnalyzer)

    expected_error = ProfilingValidationError("invalid dataset")
    dataset_profiler.profile.side_effect = expected_error

    service = ProfilingService(
        dataset_profiler=dataset_profiler,
        quality_analyzer=quality_analyzer,
    )

    with pytest.raises(ProfilingValidationError) as exc_info:
        service.profile(dataframe, reference)

    assert exc_info.value is expected_error


def test_constructor_rejects_invalid_dataset_profiler() -> None:
    with pytest.raises(
        ProfilingValidationError,
        match="dataset_profiler must provide",
    ):
        ProfilingService(dataset_profiler=object())  # type: ignore[arg-type]


def test_constructor_rejects_invalid_quality_analyzer() -> None:
    with pytest.raises(
        ProfilingValidationError,
        match="quality_analyzer must provide",
    ):
        ProfilingService(quality_analyzer=object())  # type: ignore[arg-type]


def test_profile_does_not_modify_original_dataframe(
    dataframe: pd.DataFrame,
    reference: DatasetReference,
) -> None:
    original = dataframe.copy(deep=True)
    service = ProfilingService()

    service.profile(dataframe, reference)

    pd.testing.assert_frame_equal(dataframe, original)
