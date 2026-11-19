"""Robustness integration tests for the dataset profiling pipeline."""

import json
from pathlib import Path

import pandas as pd
import pytest

from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.loading import DatasetLoadRequest, DatasetLoadingService
from lite_cog_agent.profiling import ProfilingService
from lite_cog_agent.profiling.exceptions import (
    DatasetProfilingError,
)


@pytest.fixture
def loading_service() -> DatasetLoadingService:
    return DatasetLoadingService()


@pytest.fixture
def profiling_service() -> ProfilingService:
    return ProfilingService()


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext()


def create_json_dataset(
    tmp_path: Path,
    records: list[dict],
    filename: str = "robustness.json",
) -> Path:
    """Create a JSON records file for integration testing."""
    file_path = tmp_path / filename

    file_path.write_text(
        json.dumps(records, ensure_ascii=False),
        encoding="utf-8",
    )

    return file_path


class TestMixedValuesIntegration:
    def test_profiles_json_dataset_with_mixed_special_values(
        self,
        tmp_path: Path,
        loading_service: DatasetLoadingService,
        profiling_service: ProfilingService,
        context: ExecutionContext,
    ) -> None:
        """Profile JSON data containing booleans, nulls and special strings."""
        records = [
            {
                "active": True,
                "score": 1,
                "label": "None",
            },
            {
                "active": False,
                "score": None,
                "label": "NaN",
            },
            {
                "active": True,
                "score": 3,
                "label": "",
            },
        ]

        file_path = create_json_dataset(tmp_path, records)

        reference = loading_service.load_into_context(
            DatasetLoadRequest(file_path=file_path),
            context,
        )

        dataframe = context.get_dataset(reference.dataset_id)

        report = profiling_service.profile(
            dataframe,
            reference,
        )

        dataset_profile = report.dataset_profile
        quality_report = report.quality_report

        assert dataset_profile.row_count == 3
        assert dataset_profile.column_count == 3

        assert quality_report.total_cells == 9
        assert quality_report.missing_cells == 1
        assert quality_report.completeness == pytest.approx(8 / 9 * 100)
        assert quality_report.missing_rate == pytest.approx(1 / 9 * 100)

        profiles = {
            column.name: column
            for column in dataset_profile.columns
        }

        assert profiles["active"].inferred_type == "boolean"
        assert profiles["active"].null_count == 0

        assert profiles["score"].null_count == 1
        assert profiles["score"].non_null_count == 2

        assert profiles["label"].null_count == 0
        assert profiles["label"].unique_count == 3

        assert quality_report.column_completeness == {
            "active": 100.0,
            "score": pytest.approx(200 / 3),
            "label": 100.0,
        }

    def test_profiles_json_dataset_with_special_strings(
        self,
        tmp_path: Path,
        loading_service: DatasetLoadingService,
        profiling_service: ProfilingService,
        context: ExecutionContext,
    ) -> None:
        """String representations of missing values must remain ordinary strings."""
        records = [
            {"value": "None"},
            {"value": "NaN"},
            {"value": "null"},
            {"value": "NULL"},
            {"value": "nan"},
        ]

        file_path = create_json_dataset(
            tmp_path,
            records,
            filename="special_strings.json",
        )

        reference = loading_service.load_into_context(
            DatasetLoadRequest(file_path=file_path),
            context,
        )

        dataframe = context.get_dataset(reference.dataset_id)

        report = profiling_service.profile(dataframe, reference)

        column_profile = report.dataset_profile.columns[0]

        assert column_profile.total_count == 5
        assert column_profile.non_null_count == 5
        assert column_profile.null_count == 0
        assert column_profile.unique_count == 5

        assert report.quality_report.missing_cells == 0
        assert report.quality_report.completeness == 100.0


class TestComplexObjectIntegration:
    @pytest.mark.parametrize(
        "complex_values",
        [
            [[1, 2], [3, 4]],
            [{"name": "Alice"}, {"name": "Bob"}],
        ],
        ids=["list-values", "dict-values"],
    )
    def test_profiling_rejects_complex_json_objects(
        self,
        tmp_path: Path,
        loading_service: DatasetLoadingService,
        profiling_service: ProfilingService,
        context: ExecutionContext,
        complex_values: list,
    ) -> None:
        """Complex object values must fail explicitly during profiling."""
        records = [
            {"id": 1, "payload": complex_values[0]},
            {"id": 2, "payload": complex_values[1]},
        ]

        file_path = create_json_dataset(
            tmp_path,
            records,
            filename="complex_objects.json",
        )

        reference = loading_service.load_into_context(
            DatasetLoadRequest(file_path=file_path),
            context,
        )

        dataframe = context.get_dataset(reference.dataset_id)

        with pytest.raises(DatasetProfilingError):
            profiling_service.profile(dataframe, reference)

        # Profiling failure must not silently remove the loaded dataset.
        assert context.get_dataset(reference.dataset_id) is dataframe

        assert context.dataset_store.contains(reference.dataset_id)


class TestProfilingFailureRecovery:
    def test_context_can_be_closed_after_profiling_failure(
        self,
        tmp_path: Path,
        loading_service: DatasetLoadingService,
        profiling_service: ProfilingService,
        context: ExecutionContext,
    ) -> None:
        """A profiling failure must not prevent normal context cleanup."""
        records = [
            {"id": 1, "payload": [1, 2]},
            {"id": 2, "payload": [3, 4]},
        ]

        file_path = create_json_dataset(
            tmp_path,
            records,
            filename="cleanup_after_failure.json",
        )

        reference = loading_service.load_into_context(
            DatasetLoadRequest(file_path=file_path),
            context,
        )

        dataframe = context.get_dataset(reference.dataset_id)

        with pytest.raises(DatasetProfilingError):
            profiling_service.profile(dataframe, reference)

        context.close()

        assert context.is_closed is True
        assert context.dataset_ids == ()
        assert context.dataset_store.contains(reference.dataset_id) is False
