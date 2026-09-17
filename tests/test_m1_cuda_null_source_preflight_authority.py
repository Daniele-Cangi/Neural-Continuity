"""Versioned source-only preflight authority stays fail-closed."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from neural_continuity.m1_diagnostics import cuda_null_source_preflight_authority as authority


def _inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, object]:
    parent = tmp_path / "parent.yaml"
    parent.write_bytes(b"frozen parent")
    monkeypatch.setattr(authority, "PREFLIGHT_SPEC_PATH", parent)
    monkeypatch.setattr(
        authority, "PREFLIGHT_SPEC_SHA256", hashlib.sha256(parent.read_bytes()).hexdigest()
    )
    proposed = tmp_path / "authorization.yaml"
    expected = {
        **authority.EXPECTED_AUTHORIZATION_SPEC,
        "parent_spec_sha256": authority.PREFLIGHT_SPEC_SHA256,
    }
    proposed.write_text(yaml.safe_dump(expected, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(authority, "AUTHORIZATION_SPEC_PATH", proposed)
    monkeypatch.setattr(authority, "EXPECTED_AUTHORIZATION_SPEC", expected)
    readiness = {
        "status": "PREFLIGHT_PREREQUISITES_VERIFIED_NOT_AUTHORIZED",
        "record_sha256": authority.READINESS_RECORD_SHA256,
        "runtime_identity_sha256": authority.RUNTIME_IDENTITY_SHA256,
        "preflight_spec_sha256": authority.PREFLIGHT_SPEC_SHA256,
        "protocol_sha256": authority.PROTOCOL_SHA256,
        "config_sha256": authority.CONFIG_SHA256,
        "static_manifest_sha256": authority.STATIC_MANIFEST_SHA256,
        "static_authority_sha256": authority.STATIC_AUTHORITY_SHA256,
        "technical_preflight_permission": "NOT_GRANTED",
        "onnx_graph_loaded": False,
        "session_created": False,
        "execution_authorized": False,
    }
    monkeypatch.setattr(authority, "verify_preflight_readiness", lambda **_: readiness)
    return {
        "authorization_spec": proposed,
        "external_reviewed_spec_sha256": hashlib.sha256(proposed.read_bytes()).hexdigest(),
        "preflight_spec": parent,
        "external_preflight_spec_sha256": authority.PREFLIGHT_SPEC_SHA256,
        "static_bundle": tmp_path / "static-bundle.json",
        "external_static_manifest_sha256": authority.STATIC_MANIFEST_SHA256,
        "config_path": tmp_path / "config.yaml",
        "external_config_sha256": authority.CONFIG_SHA256,
        "dataset_root": tmp_path / "dataset",
        "transition_a_bundle": tmp_path / "transition-a.json",
        "teacher_snapshot_root": tmp_path / "teacher",
        "cpu_extension_bundle": tmp_path / "cpu.json",
        "historical_cuda_bundle": tmp_path / "historical-cuda.json",
    }


def test_checked_in_authorization_scope_is_exact() -> None:
    proposed = yaml.safe_load(authority.AUTHORIZATION_SPEC_PATH.read_text(encoding="utf-8"))
    assert proposed == authority.EXPECTED_AUTHORIZATION_SPEC


def test_authority_requires_fresh_readiness_and_never_loads_a_graph(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _inputs(monkeypatch, tmp_path)
    record = authority.verify_source_preflight_authority(**inputs)
    assert record["status"] == "SOURCE_ONLY_PREFLIGHT_AUTHORITY_VERIFIED"
    assert record["technical_preflight_permission"] == "GRANTED_AFTER_REVIEW"
    assert record["onnx_graph_loaded"] is False
    assert record["session_created"] is False
    assert record["int8_allowed"] is False


def test_changed_reviewed_spec_fails_before_readiness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _inputs(monkeypatch, tmp_path)
    Path(inputs["authorization_spec"]).write_text("status: tampered\n", encoding="utf-8")
    monkeypatch.setattr(
        authority,
        "verify_preflight_readiness",
        lambda **_: pytest.fail("readiness ran after specification mismatch"),
    )
    with pytest.raises(authority.CudaNullSourcePreflightBlocked, match="hash mismatch"):
        authority.verify_source_preflight_authority(**inputs)


def test_scope_change_fails_even_with_matching_external_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _inputs(monkeypatch, tmp_path)
    proposed = Path(inputs["authorization_spec"])
    altered = yaml.safe_load(proposed.read_text(encoding="utf-8"))
    altered["scope"]["int8_allowed"] = True
    proposed.write_text(yaml.safe_dump(altered, sort_keys=True), encoding="utf-8")
    inputs["external_reviewed_spec_sha256"] = hashlib.sha256(proposed.read_bytes()).hexdigest()
    with pytest.raises(authority.CudaNullSourcePreflightBlocked, match="frozen source-only scope"):
        authority.verify_source_preflight_authority(**inputs)


def test_changed_readiness_blocks_authorization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _inputs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        authority,
        "verify_preflight_readiness",
        lambda **_: {"status": "PREFLIGHT_PREREQUISITES_VERIFIED_NOT_AUTHORIZED"},
    )
    with pytest.raises(authority.CudaNullSourcePreflightBlocked, match="readiness identity"):
        authority.verify_source_preflight_authority(**inputs)
