from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from neural_continuity.m1_diagnostics import runtime_provenance_evidence
from neural_continuity.m1_diagnostics.runtime_provenance_authority import RuntimeProvenanceError


def _authority(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        stage0_bundle=tmp_path / "stage0" / "replay-bundle.json",
        stage0_manifest_sha256="a" * 64,
        archive_manifest_sha256="b" * 64,
        package_authority_sha256={"dataset": "c" * 64},
    )


def _inventory() -> dict[str, object]:
    return {
        "kind": "m1-runtime-provenance-inventory",
        "model_execution_used": False,
        "onnx_graph_loaded": False,
        "activation_read": False,
    }


def _audit() -> dict[str, object]:
    return {
        "status": "BLOCKED",
        "attribution": {
            "status": "INCONCLUSIVE",
            "classification": "CROSS_EPOCH_DRIFT_WITH_INCOMPLETE_RUNTIME_AUTHORITY",
        },
    }


def _patch(monkeypatch: pytest.MonkeyPatch, authority: SimpleNamespace) -> None:
    monkeypatch.setattr(
        runtime_provenance_evidence, "verify_runtime_provenance_authority", lambda *_args: authority
    )
    monkeypatch.setattr(runtime_provenance_evidence, "capture_runtime_inventory", _inventory)
    monkeypatch.setattr(
        runtime_provenance_evidence, "build_runtime_drift_audit", lambda *_args: _audit()
    )


def test_runtime_provenance_capture_and_replay_are_model_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority(tmp_path)
    _patch(monkeypatch, authority)
    output = tmp_path / "output"
    captured = runtime_provenance_evidence.capture_runtime_provenance_package(
        authority.stage0_bundle, authority.stage0_manifest_sha256, output
    )
    replayed = runtime_provenance_evidence.replay_runtime_provenance_package(
        output / "replay-bundle.json", str(captured["artifact_manifest_sha256"])
    )
    assert captured["model_execution_used"] is False
    assert replayed["replay_verified"] is True
    assert replayed["onnx_graph_loaded"] is False
    assert replayed["activation_read"] is False
    assert replayed["stage_1_execution_started"] is False


def test_runtime_provenance_replay_rejects_tampered_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority(tmp_path)
    _patch(monkeypatch, authority)
    output = tmp_path / "output"
    captured = runtime_provenance_evidence.capture_runtime_provenance_package(
        authority.stage0_bundle, authority.stage0_manifest_sha256, output
    )
    (output / "runtime-inventory.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeProvenanceError) as error:
        runtime_provenance_evidence.replay_runtime_provenance_package(
            output / "replay-bundle.json", str(captured["artifact_manifest_sha256"])
        )
    assert error.value.code == "RUNTIME_PROVENANCE_ARTIFACT_INTEGRITY_FAILED"
