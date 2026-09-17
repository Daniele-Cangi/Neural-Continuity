"""Fail-closed checks for the non-executing CUDA-null preflight gate."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from neural_continuity.evidence import canonical_json_bytes
from neural_continuity.m1_diagnostics import cuda_null_preflight_readiness as readiness


def _prepared_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, object]:
    spec = tmp_path / "preflight.yaml"
    protocol = tmp_path / "protocol.md"
    config = tmp_path / "config.yaml"
    for path, content in (
        (spec, b"frozen preflight"),
        (protocol, b"frozen protocol"),
        (config, b"frozen configuration"),
    ):
        path.write_bytes(content)
    monkeypatch.setattr(readiness, "PREFLIGHT_SPEC_PATH", spec)
    monkeypatch.setattr(readiness, "PROTOCOL_PATH", protocol)
    monkeypatch.setattr(readiness, "CONFIG_PATH", config)
    monkeypatch.setattr(
        readiness, "PREFLIGHT_SPEC_SHA256", hashlib.sha256(spec.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        readiness, "PROTOCOL_SHA256", hashlib.sha256(protocol.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(readiness, "CONFIG_SHA256", hashlib.sha256(config.read_bytes()).hexdigest())
    static = {
        "status": "FROZEN_STATIC_INPUTS_REVERIFIED",
        "static_manifest_sha256": readiness.STATIC_MANIFEST_SHA256,
        "static_authority_sha256": readiness.STATIC_AUTHORITY_SHA256,
        "config_sha256": readiness.CONFIG_SHA256,
        "runtime_verified": False,
        "onnx_graph_loaded": False,
        "session_created": False,
        "execution_authorized": False,
    }
    runtime = {
        "status": "RUNTIME_IDENTITY_VERIFIED_EXECUTION_BLOCKED",
        "config_sha256": readiness.CONFIG_SHA256,
        "onnx_graph_loaded": False,
        "session_created": False,
        "execution_authorized": False,
    }
    monkeypatch.setattr(readiness, "verify_live_static_authority", lambda **_: static)
    monkeypatch.setattr(readiness, "verify_runtime_identity", lambda *_: runtime)
    return {
        "preflight_spec": spec,
        "external_preflight_spec_sha256": readiness.PREFLIGHT_SPEC_SHA256,
        "static_bundle": tmp_path / "static-bundle.json",
        "external_static_manifest_sha256": readiness.STATIC_MANIFEST_SHA256,
        "config_path": config,
        "external_config_sha256": readiness.CONFIG_SHA256,
        "dataset_root": tmp_path / "dataset",
        "transition_a_bundle": tmp_path / "transition-a.json",
        "teacher_snapshot_root": tmp_path / "teacher",
        "cpu_extension_bundle": tmp_path / "cpu-extension.json",
        "historical_cuda_bundle": tmp_path / "historical-cuda.json",
    }


def test_readiness_is_deterministic_and_never_authorizes_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _prepared_inputs(monkeypatch, tmp_path)
    first = readiness.verify_preflight_readiness(**inputs)
    second = readiness.verify_preflight_readiness(**inputs)
    assert first == second
    assert first["status"] == "PREFLIGHT_PREREQUISITES_VERIFIED_NOT_AUTHORIZED"
    assert first["technical_preflight_permission"] == "NOT_GRANTED"
    assert first["execution_authorized"] is False
    assert first["onnx_graph_loaded"] is False
    assert first["session_created"] is False
    payload = {key: value for key, value in first.items() if key != "record_sha256"}
    assert (
        first["record_sha256"] == hashlib.sha256(canonical_json_bytes(payload) + b"\n").hexdigest()
    )


def test_changed_spec_blocks_before_static_or_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _prepared_inputs(monkeypatch, tmp_path)
    Path(inputs["preflight_spec"]).write_bytes(b"changed")
    monkeypatch.setattr(
        readiness,
        "verify_live_static_authority",
        lambda **_: pytest.fail("static authority ran after spec mismatch"),
    )
    with pytest.raises(readiness.CudaNullPreflightBlocked, match="specification hash mismatch"):
        readiness.verify_preflight_readiness(**inputs)


def test_external_spec_hash_and_canonical_path_are_required(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _prepared_inputs(monkeypatch, tmp_path)
    inputs["external_preflight_spec_sha256"] = "0" * 64
    with pytest.raises(readiness.CudaNullPreflightBlocked, match="external preflight"):
        readiness.verify_preflight_readiness(**inputs)
    inputs["external_preflight_spec_sha256"] = readiness.PREFLIGHT_SPEC_SHA256
    shadow = tmp_path / "shadow.yaml"
    shadow.write_bytes(Path(inputs["preflight_spec"]).read_bytes())
    inputs["preflight_spec"] = shadow
    with pytest.raises(readiness.CudaNullPreflightBlocked, match="not canonical"):
        readiness.verify_preflight_readiness(**inputs)


def test_protocol_drift_blocks_before_static_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _prepared_inputs(monkeypatch, tmp_path)
    readiness.PROTOCOL_PATH.write_bytes(b"changed protocol")
    monkeypatch.setattr(
        readiness,
        "verify_live_static_authority",
        lambda **_: pytest.fail("static authority ran after protocol mismatch"),
    )
    with pytest.raises(readiness.CudaNullPreflightBlocked, match="protocol hash mismatch"):
        readiness.verify_preflight_readiness(**inputs)


def test_inconsistent_static_authority_blocks_before_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _prepared_inputs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        readiness,
        "verify_live_static_authority",
        lambda **_: {"status": "FROZEN_STATIC_INPUTS_REVERIFIED"},
    )
    monkeypatch.setattr(
        readiness,
        "verify_runtime_identity",
        lambda *_: pytest.fail("runtime ran after static mismatch"),
    )
    with pytest.raises(readiness.CudaNullPreflightBlocked, match="static authority is incomplete"):
        readiness.verify_preflight_readiness(**inputs)


def test_runtime_must_report_no_session_and_no_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _prepared_inputs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        readiness,
        "verify_runtime_identity",
        lambda *_: {
            "status": "RUNTIME_IDENTITY_VERIFIED_EXECUTION_BLOCKED",
            "config_sha256": readiness.CONFIG_SHA256,
            "onnx_graph_loaded": False,
            "session_created": True,
            "execution_authorized": False,
        },
    )
    with pytest.raises(readiness.CudaNullPreflightBlocked, match="runtime identity is incomplete"):
        readiness.verify_preflight_readiness(**inputs)
