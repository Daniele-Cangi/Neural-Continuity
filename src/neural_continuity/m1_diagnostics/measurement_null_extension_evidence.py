from __future__ import annotations

import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.measurement_null_extension_authority import (
    MeasurementNullPlanAuthority,
    MeasurementNullPlanError,
    verify_measurement_null_plan_authority,
)
from neural_continuity.m1_diagnostics.measurement_null_extension_plan import (
    build_measurement_null_extension_plan,
)
from neural_continuity.m1_teacher_evidence import _load_json, _safe_artifact_path

ARTIFACTS = ("measurement-null-extension-plan.json", "replay-bundle.json")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _replay_bundle(authority: MeasurementNullPlanAuthority) -> dict[str, Any]:
    return {
        "replay_format_version": "1.0.0",
        "runtime_provenance_bundle": str(authority.provenance_bundle),
        "runtime_provenance_manifest_sha256": authority.provenance_manifest_sha256,
        "plan_path": "measurement-null-extension-plan.json",
        "expected_status": "PREREGISTERED_NOT_EXECUTED",
        "replay_requires_model_execution": False,
    }


def _write_manifest(root: Path) -> str:
    artifacts = [
        {
            "path": relative,
            "sha256": sha256_file(root / relative),
            "size_bytes": (root / relative).stat().st_size,
        }
        for relative in ARTIFACTS
    ]
    manifest = {
        "kind": "m1-measurement-null-extension-plan-manifest",
        "version": "1.0.0",
        "status": "PREREGISTERED_NOT_EXECUTED",
        "tamper_evident": True,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "model_execution_used": False,
        "execution_started": False,
        "candidate_or_holdout_result_used": False,
        "replay_requires_model_execution": False,
    }
    path = root / "artifact-manifest.json"
    _write_json(path, manifest)
    return sha256_file(path)


def _verify_manifest(root: Path, expected_sha256: str) -> None:
    path = root / "artifact-manifest.json"
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise MeasurementNullPlanError(
            "MEASUREMENT_NULL_PLAN_MANIFEST_HASH_MISMATCH",
            "measurement-null plan manifest hash mismatch",
        )
    manifest = _load_json(path, "MEASUREMENT_NULL_PLAN_MANIFEST_INVALID")
    entries = manifest.get("artifacts")
    if manifest.get("kind") != "m1-measurement-null-extension-plan-manifest" or not isinstance(
        entries, list
    ):
        raise MeasurementNullPlanError(
            "MEASUREMENT_NULL_PLAN_MANIFEST_INVALID",
            "measurement-null plan manifest is invalid",
        )
    observed: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise MeasurementNullPlanError(
                "MEASUREMENT_NULL_PLAN_MANIFEST_INVALID", "artifact entry is invalid"
            )
        relative = entry.get("path")
        if not isinstance(relative, str) or relative in observed:
            raise MeasurementNullPlanError(
                "MEASUREMENT_NULL_PLAN_MANIFEST_INVALID", "artifact path is invalid or duplicated"
            )
        observed.add(relative)
        artifact = _safe_artifact_path(root, relative)
        if (
            not artifact.is_file()
            or entry.get("sha256") != sha256_file(artifact)
            or entry.get("size_bytes") != artifact.stat().st_size
        ):
            raise MeasurementNullPlanError(
                "MEASUREMENT_NULL_PLAN_ARTIFACT_INTEGRITY_FAILED",
                f"measurement-null plan artifact integrity failed: {relative}",
            )
    if observed != set(ARTIFACTS):
        raise MeasurementNullPlanError(
            "MEASUREMENT_NULL_PLAN_ARTIFACT_SET_MISMATCH",
            "measurement-null plan artifact set is incomplete or unexpected",
        )


def capture_measurement_null_extension_plan(
    provenance_bundle: str | Path,
    provenance_manifest_sha256: str,
    output_directory: str | Path,
) -> dict[str, Any]:
    authority = verify_measurement_null_plan_authority(
        provenance_bundle, provenance_manifest_sha256
    )
    output = Path(output_directory).resolve()
    if output.exists():
        raise MeasurementNullPlanError(
            "MEASUREMENT_NULL_PLAN_OUTPUT_EXISTS", f"output directory already exists: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.building-{uuid.uuid4().hex}")
    temporary.mkdir()
    try:
        plan = build_measurement_null_extension_plan(authority)
        replay = _replay_bundle(authority)
        _write_json(temporary / "measurement-null-extension-plan.json", plan)
        _write_json(temporary / "replay-bundle.json", replay)
        manifest_hash = _write_manifest(temporary)
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "status": plan["status"],
        "execution_started": False,
        "model_execution_used": False,
        "process_epoch_count": plan["frozen_design"]["process_epoch_count"],
        "total_planned_passes": plan["frozen_design"]["total_planned_passes"],
        "output_directory": str(output),
        "artifact_manifest_sha256": manifest_hash,
    }


def replay_measurement_null_extension_plan(
    bundle_path: str | Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    bundle = Path(bundle_path).resolve()
    root = bundle.parent
    _verify_manifest(root, expected_manifest_sha256)
    replay = _load_json(bundle, "MEASUREMENT_NULL_PLAN_REPLAY_BUNDLE_INVALID")
    if replay.get("replay_requires_model_execution") is not False:
        raise MeasurementNullPlanError(
            "MEASUREMENT_NULL_PLAN_REPLAY_BUNDLE_INVALID", "plan replay must be model free"
        )
    authority = verify_measurement_null_plan_authority(
        str(replay.get("runtime_provenance_bundle", "")),
        str(replay.get("runtime_provenance_manifest_sha256", "")),
    )
    stored = _load_json(
        _safe_artifact_path(root, str(replay.get("plan_path", ""))),
        "MEASUREMENT_NULL_PLAN_INVALID",
    )
    recomputed = build_measurement_null_extension_plan(authority)
    plan_match = stored == recomputed
    status_match = stored.get("status") == replay.get("expected_status")
    invariants_match = (
        stored.get("execution_started") is False
        and stored.get("model_execution_used_for_preregistration") is False
        and stored.get("candidate_or_holdout_result_selected_design") is False
    )
    if not all([plan_match, status_match, invariants_match]):
        raise MeasurementNullPlanError(
            "MEASUREMENT_NULL_PLAN_REPLAY_MISMATCH",
            "measurement-null extension plan replay differs",
        )
    return {
        "status": stored["status"],
        "replay_verified": True,
        "plan_match": plan_match,
        "status_match": status_match,
        "invariants_match": invariants_match,
        "execution_started": False,
        "model_execution_used": False,
        "stage_1_execution_started": False,
    }
