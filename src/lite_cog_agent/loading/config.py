"""Configuration for dataset loading."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetLoadingConfig:
    """Resource limits and defaults for dataset loading."""

    max_file_size_bytes: int = 50 * 1024 * 1024
    max_rows: int = 100_000
    max_columns: int = 500

    default_encoding: str = "utf-8"
    default_delimiter: str = ","

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_file_size_bytes, bool)
            or not isinstance(self.max_file_size_bytes, int)
            or self.max_file_size_bytes <= 0
        ):
            raise ValueError("max_file_size_bytes must be a positive integer")

        if (
            isinstance(self.max_rows, bool)
            or not isinstance(self.max_rows, int)
            or self.max_rows <= 0
        ):
            raise ValueError("max_rows must be a positive integer")

        if (
            isinstance(self.max_columns, bool)
            or not isinstance(self.max_columns, int)
            or self.max_columns <= 0
        ):
            raise ValueError("max_columns must be a positive integer")

        if not isinstance(self.default_encoding, str):
            raise TypeError("default_encoding must be a string")

        if not self.default_encoding.strip():
            raise ValueError("default_encoding cannot be empty")

        if not isinstance(self.default_delimiter, str):
            raise TypeError("default_delimiter must be a string")

        if len(self.default_delimiter) != 1:
            raise ValueError("default_delimiter must contain exactly one character")