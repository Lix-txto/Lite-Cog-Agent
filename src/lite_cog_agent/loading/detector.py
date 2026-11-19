"""Dataset file format detection."""

from pathlib import Path


SUPPORTED_FORMATS = {
    ".csv": "csv",
    ".xlsx": "excel",
    ".json": "json",
}


def detect_file_format(file_path: Path) -> str:
    """Detect dataset format from file extension."""

    if not isinstance(file_path, Path):
        raise TypeError("file_path must be a Path")

    suffix = file_path.suffix.lower()

    file_format = SUPPORTED_FORMATS.get(suffix)

    if file_format is None:
        from lite_cog_agent.loading.exceptions import (
            UnsupportedFileFormatError,
        )

        raise UnsupportedFileFormatError(
            f"Unsupported file format: {suffix or '<no extension>'}"
        )

    return file_format