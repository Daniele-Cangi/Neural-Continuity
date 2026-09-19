"""Deterministic lossless storage for bounded ONNX Runtime profiles."""

from __future__ import annotations

import gzip
import hashlib
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from neural_continuity.evidence import sha256_file
from neural_continuity.m1_diagnostics.cuda_null_paths import has_linked_ancestor

_CHUNK_BYTES = 1024 * 1024
_MAX_RAW_PROFILE_BYTES = 256 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
_SHA256_HEX_LENGTH = 64


class ProfileArchiveBlocked(ValueError):
    """A profile could not be archived or restored exactly."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def compress_profile(source: Path, archive: Path) -> dict[str, Any]:
    """Create a byte-deterministic gzip member without modifying the source."""
    source = Path(source).absolute()
    archive = Path(archive).absolute()
    if not source.is_file() or has_linked_ancestor(source):
        raise ProfileArchiveBlocked("profile source missing or linked")
    if archive.exists():
        raise ProfileArchiveBlocked("profile archive target exists")
    raw_size = source.stat().st_size
    if raw_size > _MAX_RAW_PROFILE_BYTES:
        raise ProfileArchiveBlocked("raw profile exceeds archive limit")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if has_linked_ancestor(archive.parent):
        raise ProfileArchiveBlocked("profile archive target is linked")
    raw_digest = hashlib.sha256()
    with (
        source.open("rb") as source_stream,
        archive.open("xb") as archive_stream,
        gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=9,
            fileobj=archive_stream,
            mtime=0,
        ) as compressed_stream,
    ):
        while chunk := source_stream.read(_CHUNK_BYTES):
            raw_digest.update(chunk)
            compressed_stream.write(chunk)
    archive_size = archive.stat().st_size
    if archive_size > _MAX_ARCHIVE_BYTES:
        archive.unlink(missing_ok=True)
        raise ProfileArchiveBlocked("compressed profile exceeds archive limit")
    return {
        "compression": "gzip",
        "compression_level": 9,
        "gzip_mtime": 0,
        "gzip_filename": "",
        "raw_size_bytes": raw_size,
        "raw_sha256": raw_digest.hexdigest(),
        "archive_size_bytes": archive_size,
        "archive_sha256": sha256_file(archive),
    }


def _validate_descriptor(descriptor: Mapping[str, object], archive: Path) -> tuple[int, str]:
    expected_keys = {
        "compression",
        "compression_level",
        "gzip_mtime",
        "gzip_filename",
        "raw_size_bytes",
        "raw_sha256",
        "archive_size_bytes",
        "archive_sha256",
    }
    if set(descriptor) != expected_keys:
        raise ProfileArchiveBlocked("profile archive descriptor schema differs")
    raw_size = descriptor.get("raw_size_bytes")
    archive_size = descriptor.get("archive_size_bytes")
    raw_sha256 = descriptor.get("raw_sha256")
    archive_sha256 = descriptor.get("archive_sha256")
    if (
        descriptor.get("compression") != "gzip"
        or descriptor.get("compression_level") != 9
        or descriptor.get("gzip_mtime") != 0
        or descriptor.get("gzip_filename") != ""
        or not isinstance(raw_size, int)
        or isinstance(raw_size, bool)
        or not 0 <= raw_size <= _MAX_RAW_PROFILE_BYTES
        or not isinstance(archive_size, int)
        or isinstance(archive_size, bool)
        or not 0 <= archive_size <= _MAX_ARCHIVE_BYTES
        or not _is_sha256(raw_sha256)
        or not _is_sha256(archive_sha256)
    ):
        raise ProfileArchiveBlocked("profile archive descriptor differs")
    if (
        not archive.is_file()
        or has_linked_ancestor(archive)
        or archive.stat().st_size != archive_size
        or sha256_file(archive) != archive_sha256
    ):
        raise ProfileArchiveBlocked("profile archive integrity differs")
    return raw_size, str(raw_sha256)


@contextmanager
def materialize_profile(
    archive: Path,
    descriptor: Mapping[str, object],
    scratch_root: Path,
) -> Iterator[Path]:
    """Yield one verified raw profile and remove it immediately afterwards."""
    archive = Path(archive).absolute()
    scratch_root = Path(scratch_root).absolute()
    raw_size, raw_sha256 = _validate_descriptor(descriptor, archive)
    if not scratch_root.is_dir() or has_linked_ancestor(scratch_root):
        raise ProfileArchiveBlocked("profile replay scratch root missing or linked")
    with tempfile.TemporaryDirectory(prefix="profile-replay-", dir=scratch_root) as temporary:
        raw_path = Path(temporary) / "profile.json"
        digest = hashlib.sha256()
        written = 0
        try:
            with gzip.open(archive, "rb") as source_stream, raw_path.open("xb") as raw_stream:
                while chunk := source_stream.read(_CHUNK_BYTES):
                    written += len(chunk)
                    if written > raw_size or written > _MAX_RAW_PROFILE_BYTES:
                        raise ProfileArchiveBlocked("decompressed profile exceeds declared size")
                    digest.update(chunk)
                    raw_stream.write(chunk)
        except (OSError, EOFError) as exc:
            raise ProfileArchiveBlocked("profile archive cannot be decompressed") from exc
        if written != raw_size or digest.hexdigest() != raw_sha256:
            raise ProfileArchiveBlocked("decompressed profile integrity differs")
        yield raw_path
