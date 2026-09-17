"""Versioned source-only CUDA preflight authority, never an inference runner.

An independently reviewed digest must be supplied outside the v2 specification.
The caller must invoke this gate afresh immediately before loading any ONNX graph.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

from neural_continuity.evidence import canonical_json_bytes
from neural_continuity.m1_diagnostics.cuda_null_preflight_readiness import (
    CONFIG_SHA256,
    PREFLIGHT_SPEC_PATH,
    PREFLIGHT_SPEC_SHA256,
    PROTOCOL_SHA256,
    STATIC_AUTHORITY_SHA256,
    STATIC_MANIFEST_SHA256,
    CudaNullPreflightBlocked,
    _pinned_file,
    verify_preflight_readiness,
)

AUTHORIZATION_SPEC_PATH = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "m1-cuda-null-source-preflight-authorization-v2.yaml"
)
AUTHORIZATION_SPEC_SHA256 = "5986cc2b0ed5ffeeab83d695bdc56840dae8caafd951af2217be6919c534c17a"
READINESS_RECORD_SHA256 = "78c3ccc8dcc5cc62103c743cf69dde1b71f738b2a13793febf2918758ba990a3"
RUNTIME_IDENTITY_SHA256 = "7df5acab2d9982362e83db0e34fcc926d22ff75efe832451e3073942b116e669"

EXPECTED_AUTHORIZATION_SPEC: dict[str, Any] = {
    "spec_id": "m1-cuda-null-source-preflight-authorization-v2",
    "version": 2,
    "status": "AUTHORIZED_AFTER_INDEPENDENT_REVIEW",
    "parent_spec_sha256": PREFLIGHT_SPEC_SHA256,
    "readiness_record_sha256": READINESS_RECORD_SHA256,
    "runtime_identity_sha256": RUNTIME_IDENTITY_SHA256,
    "scope": {
        "source_only": True,
        "dataset_role": "measurement_null",
        "document_count": 64,
        "query_count": 64,
        "ordered_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "batch_sizes": [1, 16, 64],
        "qualifying_m1_evidence": False,
        "sentinel_allowed": False,
        "full_corpus_allowed": False,
        "int8_allowed": False,
        "holdout_allowed": False,
        "tolerance_change_allowed": False,
    },
    "authorization": {
        "technical_preflight_permission": "GRANTED_AFTER_REVIEW",
        "independent_review_required": True,
        "external_reviewed_spec_sha256_required": True,
    },
}


class CudaNullSourcePreflightBlocked(ValueError):
    """A frozen source-only preflight prerequisite did not verify."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise CudaNullSourcePreflightBlocked(reason)


def verify_source_preflight_authority(
    *,
    authorization_spec: Path,
    external_reviewed_spec_sha256: str,
    preflight_spec: Path,
    external_preflight_spec_sha256: str,
    static_bundle: Path,
    external_static_manifest_sha256: str,
    config_path: Path,
    external_config_sha256: str,
    dataset_root: Path,
    transition_a_bundle: Path,
    teacher_snapshot_root: Path,
    cpu_extension_bundle: Path,
    historical_cuda_bundle: Path,
    runtime_inventory_out: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify the reviewed v2 scope and repeat every live prerequisite check.

    The external digest is an operator-supplied review attestation. This
    function checks its identity, not whether a human review actually occurred.
    """
    _require(
        isinstance(external_reviewed_spec_sha256, str)
        and len(external_reviewed_spec_sha256) == 64
        and all(character in "0123456789abcdef" for character in external_reviewed_spec_sha256),
        "external reviewed specification SHA-256 is invalid",
    )
    try:
        _pinned_file(
            authorization_spec,
            AUTHORIZATION_SPEC_PATH,
            external_reviewed_spec_sha256,
            "reviewed source-only preflight specification",
        )
        _pinned_file(
            PREFLIGHT_SPEC_PATH,
            PREFLIGHT_SPEC_PATH,
            PREFLIGHT_SPEC_SHA256,
            "parent preflight specification",
        )
    except CudaNullPreflightBlocked as exc:
        raise CudaNullSourcePreflightBlocked(str(exc)) from exc
    _require(
        external_reviewed_spec_sha256 == AUTHORIZATION_SPEC_SHA256,
        "reviewed specification SHA-256 differs from the frozen authorization",
    )
    try:
        proposed = yaml.safe_load(authorization_spec.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CudaNullSourcePreflightBlocked("reviewed specification is unreadable") from exc
    _require(
        proposed == EXPECTED_AUTHORIZATION_SPEC,
        "reviewed specification differs from the frozen source-only scope",
    )
    try:
        readiness = verify_preflight_readiness(
            preflight_spec=preflight_spec,
            external_preflight_spec_sha256=external_preflight_spec_sha256,
            static_bundle=static_bundle,
            external_static_manifest_sha256=external_static_manifest_sha256,
            config_path=config_path,
            external_config_sha256=external_config_sha256,
            dataset_root=dataset_root,
            transition_a_bundle=transition_a_bundle,
            teacher_snapshot_root=teacher_snapshot_root,
            cpu_extension_bundle=cpu_extension_bundle,
            historical_cuda_bundle=historical_cuda_bundle,
            runtime_inventory_out=runtime_inventory_out,
        )
    except CudaNullPreflightBlocked as exc:
        raise CudaNullSourcePreflightBlocked("fresh preflight readiness did not verify") from exc
    _require(
        readiness.get("status") == "PREFLIGHT_PREREQUISITES_VERIFIED_NOT_AUTHORIZED"
        and readiness.get("record_sha256") == READINESS_RECORD_SHA256
        and readiness.get("runtime_identity_sha256") == RUNTIME_IDENTITY_SHA256
        and readiness.get("preflight_spec_sha256") == PREFLIGHT_SPEC_SHA256
        and readiness.get("protocol_sha256") == PROTOCOL_SHA256
        and readiness.get("config_sha256") == CONFIG_SHA256
        and readiness.get("static_manifest_sha256") == STATIC_MANIFEST_SHA256
        and readiness.get("static_authority_sha256") == STATIC_AUTHORITY_SHA256
        and readiness.get("technical_preflight_permission") == "NOT_GRANTED"
        and readiness.get("onnx_graph_loaded") is False
        and readiness.get("session_created") is False
        and readiness.get("execution_authorized") is False,
        "fresh readiness identity or non-execution boundary mismatch",
    )
    record: dict[str, Any] = {
        "status": "SOURCE_ONLY_PREFLIGHT_AUTHORITY_VERIFIED",
        "authorization_spec_sha256": external_reviewed_spec_sha256,
        "parent_spec_sha256": PREFLIGHT_SPEC_SHA256,
        "readiness_record_sha256": READINESS_RECORD_SHA256,
        "runtime_identity_sha256": RUNTIME_IDENTITY_SHA256,
        "source_only": True,
        "document_count": 64,
        "query_count": 64,
        "qualifying_m1_evidence": False,
        "sentinel_allowed": False,
        "full_corpus_allowed": False,
        "int8_allowed": False,
        "holdout_allowed": False,
        "onnx_graph_loaded": False,
        "session_created": False,
        "technical_preflight_permission": "GRANTED_AFTER_REVIEW",
    }
    return {
        **record,
        "record_sha256": hashlib.sha256(canonical_json_bytes(record) + b"\n").hexdigest(),
    }
