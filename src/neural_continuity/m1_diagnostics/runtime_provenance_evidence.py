from __future__ import annotations

import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.runtime_provenance_authority import (
    RuntimeProvenanceAuthority,
    RuntimeProvenanceError,
    verify_runtime_provenance_authority,
)
from neural_continuity.m1_diagnostics.runtime_provenance_environment import (
    capture_runtime_inventory,
)
from neural_continuity.m1_diagnostics.runtime_provenance_metrics import build_runtime_drift_audit
from neural_continuity.m1_teacher_evidence import _load_json, _safe_artifact_path

ARTIFACTS = (
    "drift-audit.json",
    "provenance-plan.json",
    "replay-bundle.json",
    "runtime-inventory.json",
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _plan(authority: RuntimeProvenanceAuthority) -> dict[str, Any]:
    return {
        "kind": "m1-runtime-provenance-plan",
        "version": "1.1.0",
        "stage0_bundle": str(authority.stage0_bundle),
        "stage0_manifest_sha256": authority.stage0_manifest_sha256,
        "archive_manifest_sha256": authority.archive_manifest_sha256,
        "package_authority_sha256": dict(authority.package_authority_sha256),
        "analysis_scope": [
            "current_runtime_inventory",
            "intra_package_batch_variation",
            "cross_epoch_repeat_controls",
            "historical_runtime_coverage",
        ],
        "model_execution_authorized": False,
        "onnx_graph_loading_authorized": False,
        "activation_read_authorized": False,
        "stage_1_execution_authorized": False,
        "threshold_change_authorized": False,
        "frozen_evidence_mutation_authorized": False,
        "status": "AUTHORIZED_MODEL_FREE_ONLY",
    }


def _replay_bundle(authority: RuntimeProvenanceAuthority) -> dict[str, Any]:
    return {
        "replay_format_version": "1.0.0",
        "stage0_bundle": str(authority.stage0_bundle),
        "stage0_manifest_sha256": authority.stage0_manifest_sha256,
        "plan_path": "provenance-plan.json",
        "inventory_path": "runtime-inventory.json",
        "audit_path": "drift-audit.json",
        "expected_status": "BLOCKED",
        "expected_attribution_status": "INCONCLUSIVE",
        "replay_requires_model_execution": False,
    }


def _write_manifest(root: Path, status: str) -> str:
    artifacts = [
        {
            "path": relative,
            "sha256": sha256_file(root / relative),
            "size_bytes": (root / relative).stat().st_size,
        }
        for relative in ARTIFACTS
    ]
    manifest = {
        "kind": "m1-runtime-provenance-artifact-manifest",
        "version": "1.1.0",
        "status": status,
        "tamper_evident": True,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "model_execution_used": False,
        "onnx_graph_loaded": False,
        "activation_read": False,
        "stage_1_execution_started": False,
        "replay_requires_model_execution": False,
    }
    path = root / "artifact-manifest.json"
    _write_json(path, manifest)
    return sha256_file(path)


def _verify_manifest(root: Path, expected_sha256: str) -> None:
    path = root / "artifact-manifest.json"
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise RuntimeProvenanceError(
            "RUNTIME_PROVENANCE_MANIFEST_HASH_MISMATCH", "runtime provenance manifest hash mismatch"
        )
    manifest = _load_json(path, "RUNTIME_PROVENANCE_MANIFEST_INVALID")
    if manifest.get("kind") != "m1-runtime-provenance-artifact-manifest":
        raise RuntimeProvenanceError(
            "RUNTIME_PROVENANCE_MANIFEST_INVALID", "runtime provenance manifest kind is invalid"
        )
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        raise RuntimeProvenanceError(
            "RUNTIME_PROVENANCE_MANIFEST_INVALID", "runtime provenance artifacts are missing"
        )
    observed: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise RuntimeProvenanceError(
                "RUNTIME_PROVENANCE_MANIFEST_INVALID",
                "runtime provenance artifact entry is invalid",
            )
        relative = entry.get("path")
        if not isinstance(relative, str) or relative in observed:
            raise RuntimeProvenanceError(
                "RUNTIME_PROVENANCE_MANIFEST_INVALID",
                "runtime provenance artifact path is invalid or duplicated",
            )
        observed.add(relative)
        artifact = _safe_artifact_path(root, relative)
        if (
            not artifact.is_file()
            or entry.get("sha256") != sha256_file(artifact)
            or entry.get("size_bytes") != artifact.stat().st_size
        ):
            raise RuntimeProvenanceError(
                "RUNTIME_PROVENANCE_ARTIFACT_INTEGRITY_FAILED",
                f"runtime provenance artifact integrity failed: {relative}",
            )
    if observed != set(ARTIFACTS):
        raise RuntimeProvenanceError(
            "RUNTIME_PROVENANCE_ARTIFACT_SET_MISMATCH",
            "runtime provenance artifact set is incomplete or unexpected",
        )


def capture_runtime_provenance_package(
    stage0_bundle: str | Path, stage0_manifest_sha256: str, output_directory: str | Path
) -> dict[str, Any]:
    authority = verify_runtime_provenance_authority(stage0_bundle, stage0_manifest_sha256)
    output = Path(output_directory).resolve()
    if output.exists():
        raise RuntimeProvenanceError(
            "RUNTIME_PROVENANCE_OUTPUT_EXISTS", f"output directory already exists: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.building-{uuid.uuid4().hex}")
    temporary.mkdir()
    try:
        plan = _plan(authority)
        inventory = capture_runtime_inventory()
        audit = build_runtime_drift_audit(authority)
        replay = _replay_bundle(authority)
        _write_json(temporary / "provenance-plan.json", plan)
        _write_json(temporary / "runtime-inventory.json", inventory)
        _write_json(temporary / "drift-audit.json", audit)
        _write_json(temporary / "replay-bundle.json", replay)
        manifest_hash = _write_manifest(temporary, str(audit["status"]))
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "status": audit["status"],
        "attribution_status": audit["attribution"]["status"],
        "classification": audit["attribution"]["classification"],
        "stage_1_execution_started": False,
        "model_execution_used": False,
        "output_directory": str(output),
        "artifact_manifest_sha256": manifest_hash,
    }


def replay_runtime_provenance_package(
    bundle_path: str | Path, expected_manifest_sha256: str
) -> dict[str, Any]:
    bundle = Path(bundle_path).resolve()
    root = bundle.parent
    _verify_manifest(root, expected_manifest_sha256)
    replay = _load_json(bundle, "RUNTIME_PROVENANCE_REPLAY_BUNDLE_INVALID")
    if replay.get("replay_requires_model_execution") is not False:
        raise RuntimeProvenanceError(
            "RUNTIME_PROVENANCE_REPLAY_BUNDLE_INVALID",
            "runtime provenance replay must be model free",
        )
    authority = verify_runtime_provenance_authority(
        str(replay.get("stage0_bundle", "")), str(replay.get("stage0_manifest_sha256", ""))
    )
    stored_plan = _load_json(
        _safe_artifact_path(root, str(replay.get("plan_path", ""))),
        "RUNTIME_PROVENANCE_PLAN_INVALID",
    )
    stored_inventory = _load_json(
        _safe_artifact_path(root, str(replay.get("inventory_path", ""))),
        "RUNTIME_PROVENANCE_INVENTORY_INVALID",
    )
    stored_audit = _load_json(
        _safe_artifact_path(root, str(replay.get("audit_path", ""))),
        "RUNTIME_PROVENANCE_AUDIT_INVALID",
    )
    plan_match = stored_plan == _plan(authority)
    audit_match = stored_audit == build_runtime_drift_audit(authority)
    status_match = stored_audit.get("status") == replay.get("expected_status")
    attribution = stored_audit.get("attribution")
    if not isinstance(attribution, Mapping):
        raise RuntimeProvenanceError(
            "RUNTIME_PROVENANCE_REPLAY_MISMATCH",
            "runtime provenance attribution is missing",
        )
    attribution_match = attribution.get("status") == replay.get("expected_attribution_status")
    inventory_integrity = (
        stored_inventory.get("kind") == "m1-runtime-provenance-inventory"
        and stored_inventory.get("model_execution_used") is False
        and stored_inventory.get("onnx_graph_loaded") is False
        and stored_inventory.get("activation_read") is False
    )
    if not all([plan_match, audit_match, status_match, attribution_match, inventory_integrity]):
        raise RuntimeProvenanceError(
            "RUNTIME_PROVENANCE_REPLAY_MISMATCH",
            "runtime provenance replay differs from captured evidence",
        )
    return {
        "status": stored_audit["status"],
        "attribution_status": attribution["status"],
        "classification": attribution["classification"],
        "replay_verified": True,
        "plan_match": plan_match,
        "audit_match": audit_match,
        "status_match": status_match,
        "attribution_match": attribution_match,
        "runtime_inventory_integrity": inventory_integrity,
        "model_execution_used": False,
        "onnx_graph_loaded": False,
        "activation_read": False,
        "stage_1_execution_started": False,
    }
