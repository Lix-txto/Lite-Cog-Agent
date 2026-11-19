"""File storage infrastructure for binary artifacts."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


class FileManagerError(Exception):
    """Base exception for file storage operations."""


class FileValidationError(FileManagerError):
    """Raised when file input or configuration is invalid."""


class FileStorageError(FileManagerError):
    """Raised when the filesystem operation fails."""


@dataclass(frozen=True, slots=True)
class StoredFile:
    """Metadata describing a successfully stored file."""

    artifact_id: str
    filename: str
    path: Path
    media_type: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, str) or not self.artifact_id:
            raise FileValidationError(
                "artifact_id must be a non-empty string"
            )

        if not isinstance(self.filename, str) or not self.filename:
            raise FileValidationError(
                "filename must be a non-empty string"
            )

        if not isinstance(self.path, Path):
            raise FileValidationError("path must be a Path instance")

        if not isinstance(self.media_type, str) or not self.media_type:
            raise FileValidationError(
                "media_type must be a non-empty string"
            )

        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
        ):
            raise FileValidationError(
                "size_bytes must be a positive integer"
            )


class FileManager:
    """Save binary artifacts under a controlled storage directory."""

    _SUPPORTED_TYPES = {
        ".png": "image/png",
        ".svg": "image/svg+xml",
    }

    def __init__(
        self,
        root: Path | str,
        *,
        max_file_size_bytes: int = 20 * 1024 * 1024,
    ) -> None:
        if not isinstance(root, (Path, str)):
            raise FileValidationError(
                "root must be a Path or string"
            )

        if (
            isinstance(max_file_size_bytes, bool)
            or not isinstance(max_file_size_bytes, int)
            or max_file_size_bytes <= 0
        ):
            raise FileValidationError(
                "max_file_size_bytes must be a positive integer"
            )

        self._root = Path(root).expanduser().resolve()
        self._max_file_size_bytes = max_file_size_bytes

        try:
            self._root.mkdir(parents=True, exist_ok=True)

            if not self._root.is_dir():
                raise FileStorageError(
                    "Storage root must be a directory"
                )
        except FileStorageError:
            raise
        except OSError as exc:
            raise FileStorageError(
                "Failed to initialize file storage"
            ) from exc

    @property
    def root(self) -> Path:
        """Return the resolved storage root."""

        return self._root

    @property
    def max_file_size_bytes(self) -> int:
        """Return the configured maximum file size."""

        return self._max_file_size_bytes

    def save_bytes(
        self,
        content: bytes,
        *,
        extension: str,
        media_type: str,
    ) -> StoredFile:
        """Atomically save binary content and return its metadata."""

        if not isinstance(content, bytes):
            raise FileValidationError("content must be bytes")

        if not content:
            raise FileValidationError("content cannot be empty")

        if len(content) > self._max_file_size_bytes:
            raise FileValidationError(
                "File exceeds the configured size limit"
            )

        if not isinstance(extension, str):
            raise FileValidationError(
                "extension must be a string"
            )

        normalized_extension = extension.lower()

        if normalized_extension not in self._SUPPORTED_TYPES:
            raise FileValidationError(
                "Unsupported file extension"
            )

        expected_media_type = self._SUPPORTED_TYPES[
            normalized_extension
        ]

        if media_type != expected_media_type:
            raise FileValidationError(
                "File extension and media type do not match"
            )

        artifact_id = str(uuid4())
        filename = f"{artifact_id}{normalized_extension}"
        destination = self._root / filename

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self._root,
                prefix=".artifact-",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            # Publish atomically without replacing an existing file.
            # The temporary file and destination are on the same filesystem.
            os.link(temporary_path, destination)
            temporary_path.unlink()
            temporary_path = None

            return StoredFile(
                artifact_id=artifact_id,
                filename=filename,
                path=destination,
                media_type=expected_media_type,
                size_bytes=len(content),
            )

        except FileManagerError:
            raise
        except OSError as exc:
            raise FileStorageError(
                "Failed to save artifact"
            ) from exc
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
