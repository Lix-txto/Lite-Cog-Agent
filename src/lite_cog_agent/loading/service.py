"""Dataset loading service."""

from lite_cog_agent.execution.context import ExecutionContext
from lite_cog_agent.execution.dataset_store import DatasetReference
from lite_cog_agent.loading.config import DatasetLoadingConfig
from lite_cog_agent.loading.contracts import DatasetLoadRequest
from lite_cog_agent.loading.detector import detect_file_format
from lite_cog_agent.loading.exceptions import UnsupportedFileFormatError
from lite_cog_agent.loading.loaders import (
    CSVLoader,
    ExcelLoader,
    JSONLoader,
)


class DatasetLoadingService:
    """Coordinate dataset loading and execution context registration."""

    def __init__(
        self,
        config: DatasetLoadingConfig | None = None,
    ) -> None:
        if config is not None and not isinstance(
            config,
            DatasetLoadingConfig,
        ):
            raise TypeError("config must be a DatasetLoadingConfig or None")

        self.config = (
            config if config is not None else DatasetLoadingConfig()
        )

        self._loaders = {
            "csv": CSVLoader(self.config),
            "excel": ExcelLoader(self.config),
            "json": JSONLoader(self.config),
        }

    def load_into_context(
        self,
        request: DatasetLoadRequest,
        context: ExecutionContext,
    ) -> DatasetReference:
        """Load a dataset and register it in the given execution context.

        An explicit file format takes precedence over extension detection.
        If no format is specified, the format is inferred from the file
        extension.
        """

        if not isinstance(request, DatasetLoadRequest):
            raise TypeError("request must be a DatasetLoadRequest")

        if not isinstance(context, ExecutionContext):
            raise TypeError("context must be an ExecutionContext")

        file_format = (
            request.file_format
            if request.file_format is not None
            else detect_file_format(request.file_path)
        )

        loader = self._loaders.get(file_format)

        if loader is None:
            raise UnsupportedFileFormatError(
                f"Unsupported file format: {file_format}"
            )

        resolved_request = DatasetLoadRequest(
            file_path=request.file_path,
            name=request.name,
            file_format=file_format,
            options=request.options,
        )

        loaded_dataset = loader.load(resolved_request)

        context.register_dataset(
            loaded_dataset.reference,
            loaded_dataset.data,
        )

        return loaded_dataset.reference