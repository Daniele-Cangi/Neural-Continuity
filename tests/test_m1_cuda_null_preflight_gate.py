"""The preflight gate cannot silently substitute live authority or execute."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from neural_continuity.evidence import canonical_json_bytes
from neural_continuity.m1_diagnostics import cuda_null_preflight_gate as gate
from neural_continuity.m1_diagnostics.cuda_null_authority import CudaNullAuthorityBlocked


def _arguments(tmp_path: Path) -> dict[str, Any]:
    return {
        "static_bundle": tmp_path / "replay-bundle.json",
        "external_static_manifest_sha256": "a" * 64,
        "config_path": tmp_path / "config.yaml",
        "external_config_sha256": "b" * 64,
        "dataset_root": tmp_path / "dataset",
        "transition_a_bundle": tmp_path / "transition-a" / "replay-bundle.json",
        "teacher_snapshot_root": tmp_path / "snapshot",
        "cpu_extension_bundle": tmp_path / "cpu" / "replay-bundle.json",
        "historical_cuda_bundle": tmp_path / "cuda" / "replay-bundle.json",
    }


def _mock_verified_inputs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    authority = {"config_sha256": "b" * 64, "execution_authorized": False}
    authority_sha256 = hashlib.sha256(canonical_json_bytes(authority) + b"\n").hexdigest()
    monkeypatch.setattr(
        gate,
        "replay_static_package",
        lambda *_: {
            "integrity_replay_status": "PASS",
            "status_match": True,
            "decision_match": True,
            "execution_authorized": False,
            "manifest_sha256": "a" * 64,
            "authority_sha256": authority_sha256,
        },
    )
    monkeypatch.setattr(gate, "build_static_authority", lambda **_: authority)
    return authority


def test_matching_live_authority_remains_non_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_verified_inputs(monkeypatch)
    result = gate.verify_live_static_authority(**_arguments(tmp_path))
    assert result["status"] == "FROZEN_STATIC_INPUTS_REVERIFIED"
    assert result["runtime_verified"] is False
    assert result["onnx_graph_loaded"] is False
    assert result["session_created"] is False
    assert result["execution_authorized"] is False


def test_changed_live_authority_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _mock_verified_inputs(monkeypatch)
    authority["extra_identity"] = "changed"
    with pytest.raises(CudaNullAuthorityBlocked, match="live authority differs"):
        gate.verify_live_static_authority(**_arguments(tmp_path))


def test_missing_replay_match_fails_before_live_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_verified_inputs(monkeypatch)
    monkeypatch.setattr(
        gate,
        "replay_static_package",
        lambda *_: {"integrity_replay_status": "PASS", "status_match": True},
    )

    def forbidden_live_verification(**_: Any) -> dict[str, Any]:
        raise AssertionError("live verification must not run after incomplete replay")

    monkeypatch.setattr(gate, "build_static_authority", forbidden_live_verification)
    with pytest.raises(CudaNullAuthorityBlocked, match="static replay"):
        gate.verify_live_static_authority(**_arguments(tmp_path))
