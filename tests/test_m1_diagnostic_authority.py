from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from neural_continuity.m1_diagnostics import authority
from neural_continuity.m1_diagnostics.authority import (
    DiagnosticPreflightError,
    FrozenAuthorityPaths,
    verify_frozen_authority_set,
)
from neural_continuity.m1_diagnostics.preflight import run_static_preflight


def _paths(tmp_path: Path) -> FrozenAuthorityPaths:
    values = {}
    for role in authority.FROZEN_AUTHORITY_SHA256:
        path = tmp_path / role
        path.write_bytes(role.encode("ascii"))
        values[role] = path
    return FrozenAuthorityPaths(**values)


def test_complete_authority_set_is_required_before_graph_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    expected = {
        role: authority._sha256_file(paths.path_for(role))
        for role in authority.FROZEN_AUTHORITY_SHA256
    }
    monkeypatch.setattr(authority, "FROZEN_AUTHORITY_SHA256", expected)

    verified = verify_frozen_authority_set(paths)

    verified.assert_complete()
    assert len(verified.authorities) == 8
    assert verified.to_dict()["all_authorities_verified"] is True


def test_hash_mismatch_blocks_complete_authority_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    expected = {
        role: authority._sha256_file(paths.path_for(role))
        for role in authority.FROZEN_AUTHORITY_SHA256
    }
    expected["calibration_manifest"] = "0" * 64
    monkeypatch.setattr(authority, "FROZEN_AUTHORITY_SHA256", expected)

    with pytest.raises(DiagnosticPreflightError) as captured:
        verify_frozen_authority_set(paths)

    assert captured.value.status == "BLOCKED"
    assert captured.value.code == "FROZEN_AUTHORITY_HASH_MISMATCH"


def test_authority_failure_precedes_onnx_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "onnx" or name.startswith("onnx."):
            raise AssertionError("ONNX imported before complete authority verification")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(DiagnosticPreflightError) as captured:
        run_static_preflight(paths, tmp_path / "output")

    assert captured.value.code == "FROZEN_AUTHORITY_HASH_MISMATCH"
