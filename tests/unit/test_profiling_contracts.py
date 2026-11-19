"""Unit tests for profiling contracts."""

from types import MappingProxyType

import pytest

from lite_cog_agent.profiling import (
    ColumnProfile,
    DataQualityReport,
    DatasetProfile,
    ProfilingReport,
    ProfilingValidationError,
)


@pytest.fixture
def column_profile() -> ColumnProfile:
    return ColumnProfile(
        name="price",
        dtype="float64",
        inferred_type="numeric",
        total_count=5,
        non_null_count=4,
        null_count=1,
        null_percentage=20.0,
        unique_count=3,
        unique_percentage=60.0,
        statistics={"mean": 175.0, "min": 100.0, "max": 300.0},
        sample_values=(100.0, 200.0, 300.0),
    )


@pytest.fixture
def dataset_profile(
    column_profile: ColumnProfile,
) -> DatasetProfile:
    return DatasetProfile(
        dataset_id="dataset-001",
        dataset_name="sales",
        row_count=5,
        column_count=1,
        memory_usage_bytes=1024,
        columns=(column_profile,),
        duplicate_row_count=1,
        duplicate_percentage=20.0,
    )


@pytest.fixture
def quality_report() -> DataQualityReport:
    return DataQualityReport(
        total_cells=5,
        missing_cells=1,
        completeness=80.0,
        missing_rate=20.0,
        duplicate_row_count=1,
        duplicate_rate=20.0,
        column_completeness={"price": 80.0},
    )


class TestColumnProfile:
    def test_creates_column_profile(
        self,
        column_profile: ColumnProfile,
    ) -> None:
        assert column_profile.name == "price"
        assert column_profile.total_count == 5
        assert column_profile.null_count == 1
        assert column_profile.null_percentage == 20.0

    def test_statistics_are_read_only(self) -> None:
        profile = ColumnProfile(
            name="price",
            dtype="float64",
            inferred_type="numeric",
            total_count=1,
            non_null_count=1,
            null_count=0,
            null_percentage=0.0,
            unique_count=1,
            unique_percentage=100.0,
            statistics={"mean": 100.0},
            sample_values=(100.0,),
        )

        assert isinstance(profile.statistics, MappingProxyType)

        with pytest.raises(TypeError):
            profile.statistics["mean"] = 200.0  # type: ignore[index]

    def test_statistics_are_copied(self) -> None:
        statistics = {"mean": 100.0}
        profile = ColumnProfile(
            name="price",
            dtype="float64",
            inferred_type="numeric",
            total_count=1,
            non_null_count=1,
            null_count=0,
            null_percentage=0.0,
            unique_count=1,
            unique_percentage=100.0,
            statistics=statistics,
            sample_values=(100.0,),
        )

        statistics["mean"] = 200.0

        assert profile.statistics["mean"] == 100.0

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ProfilingValidationError, match="name"):
            ColumnProfile(
                name=" ",
                dtype="float64",
                inferred_type="numeric",
                total_count=1,
                non_null_count=1,
                null_count=0,
                null_percentage=0.0,
                unique_count=1,
                unique_percentage=100.0,
                statistics={},
                sample_values=(),
            )

    @pytest.mark.parametrize("value", [-1, 1.5, True])
    def test_rejects_invalid_total_count(self, value: object) -> None:
        with pytest.raises(ProfilingValidationError):
            ColumnProfile(
                name="price",
                dtype="float64",
                inferred_type="numeric",
                total_count=value,  # type: ignore[arg-type]
                non_null_count=1,
                null_count=0,
                null_percentage=0.0,
                unique_count=1,
                unique_percentage=100.0,
                statistics={},
                sample_values=(),
            )

    def test_rejects_inconsistent_counts(self) -> None:
        with pytest.raises(ProfilingValidationError, match="total_count"):
            ColumnProfile(
                name="price",
                dtype="float64",
                inferred_type="numeric",
                total_count=5,
                non_null_count=3,
                null_count=1,
                null_percentage=20.0,
                unique_count=1,
                unique_percentage=20.0,
                statistics={},
                sample_values=(),
            )

    def test_rejects_unique_count_exceeding_non_null_count(self) -> None:
        with pytest.raises(ProfilingValidationError, match="unique_count"):
            ColumnProfile(
                name="price",
                dtype="float64",
                inferred_type="numeric",
                total_count=2,
                non_null_count=1,
                null_count=1,
                null_percentage=50.0,
                unique_count=2,
                unique_percentage=100.0,
                statistics={},
                sample_values=(),
            )

    def test_rejects_incorrect_null_percentage(self) -> None:
        with pytest.raises(ProfilingValidationError, match="null_percentage"):
            ColumnProfile(
                name="price",
                dtype="float64",
                inferred_type="numeric",
                total_count=5,
                non_null_count=4,
                null_count=1,
                null_percentage=10.0,
                unique_count=1,
                unique_percentage=20.0,
                statistics={},
                sample_values=(),
            )

    def test_rejects_incorrect_unique_percentage(self) -> None:
        with pytest.raises(
            ProfilingValidationError,
            match="unique_percentage",
        ):
            ColumnProfile(
                name="price",
                dtype="float64",
                inferred_type="numeric",
                total_count=5,
                non_null_count=4,
                null_count=1,
                null_percentage=20.0,
                unique_count=2,
                unique_percentage=20.0,
                statistics={},
                sample_values=(),
            )

    def test_rejects_non_mapping_statistics(self) -> None:
        with pytest.raises(ProfilingValidationError, match="statistics"):
            ColumnProfile(
                name="price",
                dtype="float64",
                inferred_type="numeric",
                total_count=1,
                non_null_count=1,
                null_count=0,
                null_percentage=0.0,
                unique_count=1,
                unique_percentage=100.0,
                statistics=[],  # type: ignore[arg-type]
                sample_values=(),
            )

    def test_rejects_non_string_statistics_key(self) -> None:
        with pytest.raises(ProfilingValidationError, match="keys"):
            ColumnProfile(
                name="price",
                dtype="float64",
                inferred_type="numeric",
                total_count=1,
                non_null_count=1,
                null_count=0,
                null_percentage=0.0,
                unique_count=1,
                unique_percentage=100.0,
                statistics={1: 100.0},  # type: ignore[dict-item]
                sample_values=(),
            )

    def test_rejects_non_tuple_sample_values(self) -> None:
        with pytest.raises(ProfilingValidationError, match="sample_values"):
            ColumnProfile(
                name="price",
                dtype="float64",
                inferred_type="numeric",
                total_count=1,
                non_null_count=1,
                null_count=0,
                null_percentage=0.0,
                unique_count=1,
                unique_percentage=100.0,
                statistics={},
                sample_values=[100.0],  # type: ignore[arg-type]
            )

    def test_rejects_nan_percentage(self) -> None:
        with pytest.raises(ProfilingValidationError):
            ColumnProfile(
                name="price",
                dtype="float64",
                inferred_type="numeric",
                total_count=1,
                non_null_count=1,
                null_count=0,
                null_percentage=float("nan"),
                unique_count=1,
                unique_percentage=100.0,
                statistics={},
                sample_values=(),
            )


class TestDatasetProfile:
    def test_creates_dataset_profile(
        self,
        dataset_profile: DatasetProfile,
    ) -> None:
        assert dataset_profile.dataset_id == "dataset-001"
        assert dataset_profile.row_count == 5
        assert dataset_profile.column_count == 1

    def test_rejects_column_count_mismatch(
        self,
        column_profile: ColumnProfile,
    ) -> None:
        with pytest.raises(ProfilingValidationError, match="column_count"):
            DatasetProfile(
                dataset_id="dataset-001",
                dataset_name="sales",
                row_count=5,
                column_count=2,
                memory_usage_bytes=1024,
                columns=(column_profile,),
                duplicate_row_count=0,
                duplicate_percentage=0.0,
            )

    def test_rejects_invalid_column_type(self) -> None:
        with pytest.raises(ProfilingValidationError, match="ColumnProfile"):
            DatasetProfile(
                dataset_id="dataset-001",
                dataset_name="sales",
                row_count=5,
                column_count=1,
                memory_usage_bytes=1024,
                columns=("invalid",),  # type: ignore[arg-type]
                duplicate_row_count=0,
                duplicate_percentage=0.0,
            )

    def test_rejects_duplicate_count_exceeding_row_count(self) -> None:
        with pytest.raises(
            ProfilingValidationError,
            match="duplicate_row_count",
        ):
            DatasetProfile(
                dataset_id="dataset-001",
                dataset_name="sales",
                row_count=2,
                column_count=0,
                memory_usage_bytes=0,
                columns=(),
                duplicate_row_count=3,
                duplicate_percentage=0.0,
            )

    def test_rejects_incorrect_duplicate_percentage(self) -> None:
        with pytest.raises(
            ProfilingValidationError,
            match="duplicate_percentage",
        ):
            DatasetProfile(
                dataset_id="dataset-001",
                dataset_name="sales",
                row_count=5,
                column_count=0,
                memory_usage_bytes=0,
                columns=(),
                duplicate_row_count=1,
                duplicate_percentage=10.0,
            )


class TestDataQualityReport:
    def test_creates_quality_report(
        self,
        quality_report: DataQualityReport,
    ) -> None:
        assert quality_report.completeness == 80.0
        assert quality_report.missing_rate == 20.0
        assert quality_report.column_completeness["price"] == 80.0

    def test_column_completeness_is_read_only(
        self,
        quality_report: DataQualityReport,
    ) -> None:
        with pytest.raises(TypeError):
            quality_report.column_completeness["price"] = 0.0  # type: ignore[index]

    def test_column_completeness_is_copied(self) -> None:
        values = {"price": 80.0}
        report = DataQualityReport(
            total_cells=5,
            missing_cells=1,
            completeness=80.0,
            missing_rate=20.0,
            duplicate_row_count=0,
            duplicate_rate=0.0,
            column_completeness=values,
        )

        values["price"] = 0.0

        assert report.column_completeness["price"] == 80.0

    def test_rejects_missing_cells_exceeding_total(self) -> None:
        with pytest.raises(ProfilingValidationError, match="missing_cells"):
            DataQualityReport(
                total_cells=5,
                missing_cells=6,
                completeness=0.0,
                missing_rate=100.0,
                duplicate_row_count=0,
                duplicate_rate=0.0,
                column_completeness={},
            )

    def test_rejects_incorrect_completeness(self) -> None:
        with pytest.raises(ProfilingValidationError, match="completeness"):
            DataQualityReport(
                total_cells=5,
                missing_cells=1,
                completeness=90.0,
                missing_rate=20.0,
                duplicate_row_count=0,
                duplicate_rate=0.0,
                column_completeness={},
            )

    def test_rejects_incorrect_missing_rate(self) -> None:
        with pytest.raises(ProfilingValidationError, match="missing_rate"):
            DataQualityReport(
                total_cells=5,
                missing_cells=1,
                completeness=80.0,
                missing_rate=10.0,
                duplicate_row_count=0,
                duplicate_rate=0.0,
                column_completeness={},
            )

    def test_rejects_invalid_column_completeness(self) -> None:
        with pytest.raises(ProfilingValidationError):
            DataQualityReport(
                total_cells=5,
                missing_cells=1,
                completeness=80.0,
                missing_rate=20.0,
                duplicate_row_count=0,
                duplicate_rate=0.0,
                column_completeness={"price": 120.0},
            )

    def test_supports_empty_dataset_quality(self) -> None:
        report = DataQualityReport(
            total_cells=0,
            missing_cells=0,
            completeness=0.0,
            missing_rate=0.0,
            duplicate_row_count=0,
            duplicate_rate=0.0,
            column_completeness={},
        )

        assert report.total_cells == 0
        assert report.completeness == 0.0
        assert report.missing_rate == 0.0


class TestProfilingReport:
    def test_creates_profiling_report(
        self,
        dataset_profile: DatasetProfile,
        quality_report: DataQualityReport,
    ) -> None:
        report = ProfilingReport(
            dataset_profile=dataset_profile,
            quality_report=quality_report,
        )

        assert report.dataset_profile is dataset_profile
        assert report.quality_report is quality_report

    def test_rejects_invalid_dataset_profile(
        self,
        quality_report: DataQualityReport,
    ) -> None:
        with pytest.raises(ProfilingValidationError, match="dataset_profile"):
            ProfilingReport(
                dataset_profile="invalid",  # type: ignore[arg-type]
                quality_report=quality_report,
            )

    def test_rejects_invalid_quality_report(
        self,
        dataset_profile: DatasetProfile,
    ) -> None:
        with pytest.raises(ProfilingValidationError, match="quality_report"):
            ProfilingReport(
                dataset_profile=dataset_profile,
                quality_report="invalid",  # type: ignore[arg-type]
            )
