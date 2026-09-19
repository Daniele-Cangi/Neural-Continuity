from __future__ import annotations

from pathlib import Path

import pytest

from neural_continuity.m1_diagnostics.cuda_null_full_profile_archive import (
    ProfileArchiveBlocked,
    compress_profile,
    materialize_profile,
)


def test_profile_archive_is_deterministic_and_lossless(tmp_path: Path) -> None:
    source = tmp_path / "profile.json"
    payload = b'[{"cat":"Node","name":"kernel","args":{"provider":"CUDA"}}]' * 4096
    source.write_bytes(payload)
    first = tmp_path / "first" / "profile.json.gz"
    second = tmp_path / "second" / "profile.json.gz"

    first_descriptor = compress_profile(source, first)
    second_descriptor = compress_profile(source, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_descriptor == second_descriptor
    with materialize_profile(first, first_descriptor, tmp_path) as restored:
        assert restored.read_bytes() == payload
    assert not list(tmp_path.glob("profile-replay-*"))


def test_profile_archive_tamper_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "profile.json"
    source.write_bytes(b"[]")
    archive = tmp_path / "profile.json.gz"
    descriptor = compress_profile(source, archive)
    archive.write_bytes(archive.read_bytes() + b"tamper")

    with (
        pytest.raises(ProfileArchiveBlocked, match="archive integrity"),
        materialize_profile(archive, descriptor, tmp_path),
    ):
        pass


def test_profile_archive_declared_size_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "profile.json"
    source.write_bytes(b"[]")
    archive = tmp_path / "profile.json.gz"
    descriptor = compress_profile(source, archive)
    descriptor["raw_size_bytes"] = 1

    with (
        pytest.raises(ProfileArchiveBlocked, match="exceeds declared size"),
        materialize_profile(archive, descriptor, tmp_path),
    ):
        pass
