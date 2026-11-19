"""Unit tests for FileManager."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from lite_cog_agent.storage.file_manager import (
    FileManager,
    FileManagerError,
    FileStorageError,
    FileValidationError,
    StoredFile,
)


@pytest.fixture
def storage_root(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


@pytest.fixture
def file_manager(storage_root: Path) -> FileManager:
    return FileManager(storage_root)


# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------


def test_initialization_creates_storage_directory(
    storage_root: Path,
) -> None:
    assert not storage_root.exists()

    manager = FileManager(storage_root)

    assert storage_root.is_dir()
    assert manager.root == storage_root.resolve()


def test_initialization_accepts_string_root(
    storage_root: Path,
) -> None:
    manager = FileManager(str(storage_root))

    assert manager.root == storage_root.resolve()


@pytest.mark.parametrize("invalid_root", [None, 123, object()])
def test_initialization_rejects_invalid_root(
    invalid_root: object,
) -> None:
    with pytest.raises(FileValidationError):
        FileManager(invalid_root)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_size",
    [0, -1, True, 1.5, "1024", None],
)
def test_initialization_rejects_invalid_max_file_size(
    tmp_path: Path,
    invalid_size: object,
) -> None:
    with pytest.raises(FileValidationError):
        FileManager(
            tmp_path / "artifacts",
            max_file_size_bytes=invalid_size,  # type: ignore[arg-type]
        )


def test_initialization_rejects_existing_file_as_root(
    tmp_path: Path,
) -> None:
    root_file = tmp_path / "not-a-directory"
    root_file.write_text("existing file")

    with pytest.raises(FileStorageError):
        FileManager(root_file)


def test_initialization_wraps_filesystem_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_mkdir(*args: object, **kwargs: object) -> None:
        raise PermissionError("permission denied")

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)

    with pytest.raises(FileStorageError):
        FileManager(tmp_path / "artifacts")


def test_max_file_size_property(
    file_manager: FileManager,
) -> None:
    assert file_manager.max_file_size_bytes == 20 * 1024 * 1024


# ---------------------------------------------------------
# StoredFile contract
# ---------------------------------------------------------


def test_stored_file_is_immutable(
    tmp_path: Path,
) -> None:
    stored = StoredFile(
        artifact_id="artifact-1",
        filename="artifact-1.png",
        path=tmp_path / "artifact-1.png",
        media_type="image/png",
        size_bytes=128,
    )

    with pytest.raises(FrozenInstanceError):
        stored.filename = "changed.png"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_id", ""),
        ("filename", ""),
        ("path", "artifact.png"),
        ("media_type", ""),
        ("size_bytes", 0),
        ("size_bytes", -1),
        ("size_bytes", True),
    ],
)
def test_stored_file_rejects_invalid_fields(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    values = {
        "artifact_id": "artifact-1",
        "filename": "artifact-1.png",
        "path": tmp_path / "artifact-1.png",
        "media_type": "image/png",
        "size_bytes": 128,
    }
    values[field] = value

    with pytest.raises(FileValidationError):
        StoredFile(**values)  # type: ignore[arg-type]


# ---------------------------------------------------------
# Successful file storage
# ---------------------------------------------------------


@pytest.mark.parametrize(
    ("extension", "media_type", "content"),
    [
        (".png", "image/png", b"\x89PNG\r\n\x1a\nbinary-data"),
        (".svg", "image/svg+xml", b"<svg></svg>"),
    ],
)
def test_save_bytes_success(
    file_manager: FileManager,
    extension: str,
    media_type: str,
    content: bytes,
) -> None:
    stored = file_manager.save_bytes(
        content,
        extension=extension,
        media_type=media_type,
    )

    assert isinstance(stored, StoredFile)
    assert stored.path.is_file()
    assert stored.path.parent == file_manager.root
    assert stored.path.read_bytes() == content
    assert stored.size_bytes == len(content)
    assert stored.media_type == media_type
    assert stored.filename == f"{stored.artifact_id}{extension}"
    assert stored.path.name == stored.filename


def test_save_bytes_normalizes_extension(
    file_manager: FileManager,
) -> None:
    stored = file_manager.save_bytes(
        b"svg-content",
        extension=".SVG",
        media_type="image/svg+xml",
    )

    assert stored.filename.endswith(".svg")


def test_save_bytes_generates_unique_artifact_ids(
    file_manager: FileManager,
) -> None:
    first = file_manager.save_bytes(
        b"first",
        extension=".png",
        media_type="image/png",
    )
    second = file_manager.save_bytes(
        b"second",
        extension=".png",
        media_type="image/png",
    )

    assert first.artifact_id != second.artifact_id
    assert first.filename != second.filename
    assert first.path != second.path
    assert first.path.read_bytes() == b"first"
    assert second.path.read_bytes() == b"second"


# ---------------------------------------------------------
# Input validation
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "invalid_content",
    [None, "", bytearray(b"data"), memoryview(b"data"), 123],
)
def test_save_bytes_rejects_invalid_content(
    file_manager: FileManager,
    invalid_content: object,
) -> None:
    with pytest.raises(FileValidationError):
        file_manager.save_bytes(
            invalid_content,  # type: ignore[arg-type]
            extension=".png",
            media_type="image/png",
        )


def test_save_bytes_rejects_oversized_content(
    tmp_path: Path,
) -> None:
    manager = FileManager(
        tmp_path / "artifacts",
        max_file_size_bytes=4,
    )

    with pytest.raises(FileValidationError):
        manager.save_bytes(
            b"12345",
            extension=".png",
            media_type="image/png",
        )


@pytest.mark.parametrize(
    "extension",
    ["png", "jpg", ".jpg", ".pdf", ".exe", "../png", ""],
)
def test_save_bytes_rejects_unsupported_extension(
    file_manager: FileManager,
    extension: str,
) -> None:
    with pytest.raises(FileValidationError):
        file_manager.save_bytes(
            b"content",
            extension=extension,
            media_type="image/png",
        )


@pytest.mark.parametrize(
    ("extension", "media_type"),
    [
        (".png", "image/svg+xml"),
        (".svg", "image/png"),
        (".png", "text/plain"),
        (".svg", ""),
    ],
)
def test_save_bytes_rejects_mismatched_media_type(
    file_manager: FileManager,
    extension: str,
    media_type: str,
) -> None:
    with pytest.raises(FileValidationError):
        file_manager.save_bytes(
            b"content",
            extension=extension,
            media_type=media_type,
        )


# ---------------------------------------------------------
# Concurrent writes
# ---------------------------------------------------------


def test_concurrent_saves_generate_unique_files(
    file_manager: FileManager,
) -> None:
    file_count = 32

    def save(index: int) -> StoredFile:
        return file_manager.save_bytes(
            f"content-{index}".encode(),
            extension=".png",
            media_type="image/png",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        stored_files = list(
            executor.map(save, range(file_count))
        )

    artifact_ids = {
        stored.artifact_id for stored in stored_files
    }
    filenames = {
        stored.filename for stored in stored_files
    }

    assert len(artifact_ids) == file_count
    assert len(filenames) == file_count

    for stored in stored_files:
        assert stored.path.is_file()


# ---------------------------------------------------------
# Failure recovery
# ---------------------------------------------------------


def test_save_failure_cleans_temporary_file(
    file_manager: FileManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_link(
        source: str | Path,
        destination: str | Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        raise OSError("simulated link failure")

    monkeypatch.setattr(
        "lite_cog_agent.storage.file_manager.os.link",
        fail_link,
    )

    with pytest.raises(FileStorageError):
        file_manager.save_bytes(
            b"content",
            extension=".png",
            media_type="image/png",
        )

    assert list(file_manager.root.iterdir()) == []


def test_save_failure_preserves_original_exception(
    file_manager: FileManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_error = OSError("simulated filesystem error")

    def fail_link(
        source: str | Path,
        destination: str | Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        raise original_error

    monkeypatch.setattr(
        "lite_cog_agent.storage.file_manager.os.link",
        fail_link,
    )

    with pytest.raises(FileStorageError) as exc_info:
        file_manager.save_bytes(
            b"content",
            extension=".png",
            media_type="image/png",
        )

    assert exc_info.value.__cause__ is original_error


def test_file_manager_errors_share_common_base() -> None:
    assert issubclass(FileValidationError, FileManagerError)
    assert issubclass(FileStorageError, FileManagerError)
