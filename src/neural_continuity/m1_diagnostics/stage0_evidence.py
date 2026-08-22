from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.stage0_authority import (
    Stage0Authority,
    Stage0ControlError,
    verify_stage0_authority,
)
from neural_continuity.m1_diagnostics.stage0_execution import (
    capture_stage0_observations,
)
from neural_continuity.m1_diagnostics.stage0_metrics import (
    build_stage0_control_report,
)

PLAN_NAME = "stage0-plan.json"
REPORT_NAME = "control-report.json"
REPLAY_NAME = "replay-bundle.json"
MANIFEST_NAME = "artifact-manifest.json"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage0ControlError(
            "STAGE0_ARTIFACT_INVALID",
            f"cannot load {path.name}: {exc}",
        ) from exc
    if not isinstance(payload, Mapping):
        raise Stage0ControlError(
            "STAGE0_ARTIFACT_INVALID",
            f"{path.name} must contain an object",
        )
    return payload


def _safe(root: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise Stage0ControlError("STAGE0_ARTIFACT_INVALID", "artifact path is missing")
    relative = Path(relative_path)
    if relative.is_absolute():
        raise Stage0ControlError(
            "STAGE0_ARTIFACT_INVALID",
            "artifact path must be relative",
        )
    candidate = (root / relative).resolve()
    if root != candidate and root not in candidate.parents:
        raise Stage0ControlError(
            "STAGE0_ARTIFACT_INVALID",
            "artifact path escapes the package",
        )
    if not candidate.is_file():
        raise Stage0ControlError(
            "STAGE0_ARTIFACT_MISSING",
            f"artifact is missing: {relative_path}",
        )
    return candidate


def _plan(authority: Stage0Authority) -> dict[str, Any]:
    return {
        "kind": "m1-diagnostic-stage0-control-plan",
        "version": "1.0.0",
        "status": "AUTHORIZED_FOR_STAGE0_ONLY",
        "causal_plan_bundle": str(authority.causal_plan_bundle),
        "causal_plan_manifest_sha256": authority.causal_plan_manifest_sha256,
        "archive_manifest_path": str(authority.archive_manifest_path),
        "archive_manifest_sha256": authority.archive_manifest_sha256,
        "runtime_authority_root": str(authority.runtime_root),
        "runtime_authority_manifest_sha256": authority.runtime_manifest_sha256,
        "package_authority_sha256": dict(authority.package_authority_sha256),
        "controls": [
            "FROZEN_INT8_EXACT_REPLAY",
            "VERIFIED_ONNX_FP32_REFERENCE",
        ],
        "batch_sizes": [1, 16, 64],
        "execution_provider": "CPUExecutionProvider",
        "detection_limit_family": "repeated_inference",
        "candidate_or_holdout_result_selected_threshold": False,
        "operational_tolerances_changed": False,
        "frozen_int8_candidate_mutated": False,
        "derived_diagnostic_candidate_created": False,
        "stage_1_intervention_executed": False,
        "frozen_transition_b_v1_scientific_decision": "FAIL",
    }


def _artifact_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(
        (path for path in root.rglob("*") if path.is_file() and path.name != MANIFEST_NAME),
        key=lambda path: path.relative_to(root).as_posix().encode("utf-8"),
    ):
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def _write_manifest(root: Path, status: str) -> str:
    artifacts = _artifact_records(root)
    manifest = {
        "kind": "m1-diagnostic-stage0-control-manifest",
        "status": status,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "tamper_evident": True,
        "model_execution_used_for_capture": True,
        "replay_requires_model_execution": False,
        "stage_1_intervention_executed": False,
    }
    _write_json(root / MANIFEST_NAME, manifest)
    return sha256_file(root / MANIFEST_NAME)


def _verify_manifest(root: Path, expected_sha256: str) -> None:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected_sha256:
        raise Stage0ControlError(
            "STAGE0_MANIFEST_HASH_MISMATCH",
            "Stage 0 manifest hash does not match",
        )
    manifest = _load_json(manifest_path)
    if (
        manifest.get("kind") != "m1-diagnostic-stage0-control-manifest"
        or manifest.get("tamper_evident") is not True
        or manifest.get("model_execution_used_for_capture") is not True
        or manifest.get("replay_requires_model_execution") is not False
        or manifest.get("stage_1_intervention_executed") is not False
    ):
        raise Stage0ControlError(
            "STAGE0_MANIFEST_INVALID",
            "Stage 0 manifest declaration is invalid",
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or manifest.get("artifact_count") != len(artifacts):
        raise Stage0ControlError(
            "STAGE0_MANIFEST_INVALID",
            "Stage 0 artifact declaration is incomplete",
        )
    declared: set[str] = set()
    for record in artifacts:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise Stage0ControlError(
                "STAGE0_MANIFEST_INVALID",
                "Stage 0 artifact record is invalid",
            )
        relative = str(record["path"])
        artifact = _safe(root, relative)
        if (
            relative in declared
            or sha256_file(artifact) != record.get("sha256")
            or artifact.stat().st_size != record.get("size_bytes")
        ):
            raise Stage0ControlError(
                "STAGE0_ARTIFACT_HASH_MISMATCH",
                f"Stage 0 artifact differs: {relative}",
            )
        declared.add(relative)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
    }
    if declared != actual:
        raise Stage0ControlError(
            "STAGE0_MANIFEST_INVALID",
            "Stage 0 artifact set contains undeclared or missing files",
        )


def create_stage0_control_package(
    causal_plan_bundle: str | Path,
    causal_plan_manifest_sha256: str,
    archive_manifest_path: str | Path,
    archive_manifest_sha256: str,
    runtime_manifest_path: str | Path,
    runtime_manifest_sha256: str,
    output_directory: str | Path,
) -> dict[str, Any]:
    authority = verify_stage0_authority(
        causal_plan_bundle,
        causal_plan_manifest_sha256,
        archive_manifest_path,
        archive_manifest_sha256,
        runtime_manifest_path,
        runtime_manifest_sha256,
    )
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise Stage0ControlError(
            "STAGE0_OUTPUT_EXISTS",
            "Stage 0 output directory already exists",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    build_root = Path(
        tempfile.mkdtemp(prefix=".stage0-control-build-", dir=destination.parent)
    ).resolve()
    try:
        fp32_capture, int8_capture = capture_stage0_observations(
            authority,
            build_root,
        )
        report = build_stage0_control_report(
            authority.baseline_fp32_bundle,
            build_root / "fp32-control" / "replay-bundle.json",
            authority.baseline_int8_bundle,
            build_root / "int8-control" / "replay-bundle.json",
            authority.onnx_null_report,
        )
        plan = _plan(authority)
        _write_json(build_root / PLAN_NAME, plan)
        _write_json(build_root / REPORT_NAME, report)
        replay = {
            "replay_format_version": "1.0.0",
            "stage0_plan_path": PLAN_NAME,
            "control_report_path": REPORT_NAME,
            "fresh_fp32_bundle_path": "fp32-control/replay-bundle.json",
            "fresh_int8_bundle_path": "int8-control/replay-bundle.json",
            "causal_plan_bundle": str(authority.causal_plan_bundle),
            "causal_plan_manifest_sha256": authority.causal_plan_manifest_sha256,
            "archive_manifest_path": str(authority.archive_manifest_path),
            "archive_manifest_sha256": authority.archive_manifest_sha256,
            "runtime_manifest_path": str(authority.runtime_root / "authority-manifest.json"),
            "runtime_manifest_sha256": authority.runtime_manifest_sha256,
            "expected_status": report["status"],
            "replay_requires_model_execution": False,
        }
        _write_json(build_root / REPLAY_NAME, replay)
        manifest_sha256 = _write_manifest(build_root, str(report["status"]))
        build_root.rename(destination)
    except Exception:
        if build_root.exists() and build_root.parent == destination.parent:
            shutil.rmtree(build_root)
        raise
    return {
        "status": report["status"],
        "stage_1_gate_status": report["stage_1_gate_status"],
        "output_directory": str(destination),
        "artifact_manifest_sha256": manifest_sha256,
        "fp32_control_outcome": report["controls"]["verified_onnx_fp32_reference"]["outcome"],
        "int8_control_outcome": report["controls"]["frozen_int8_exact_replay"]["outcome"],
        "fp32_capture": fp32_capture["status"],
        "int8_capture": int8_capture["status"],
        "stage_1_execution_started": False,
    }


def replay_stage0_control_package(
    bundle_path: str | Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    bundle = Path(bundle_path).resolve()
    root = bundle.parent
    if bundle != (root / REPLAY_NAME).resolve():
        raise Stage0ControlError(
            "STAGE0_REPLAY_INVALID",
            "replay bundle is not the manifest-declared replay bundle",
        )
    _verify_manifest(root, expected_manifest_sha256)
    replay = _load_json(bundle)
    required = {
        "stage0_plan_path": PLAN_NAME,
        "control_report_path": REPORT_NAME,
        "fresh_fp32_bundle_path": "fp32-control/replay-bundle.json",
        "fresh_int8_bundle_path": "int8-control/replay-bundle.json",
        "replay_requires_model_execution": False,
    }
    if any(replay.get(key) != value for key, value in required.items()):
        raise Stage0ControlError(
            "STAGE0_REPLAY_INVALID",
            "Stage 0 replay declaration is invalid or not model-free",
        )
    authority = verify_stage0_authority(
        str(replay.get("causal_plan_bundle", "")),
        str(replay.get("causal_plan_manifest_sha256", "")),
        str(replay.get("archive_manifest_path", "")),
        str(replay.get("archive_manifest_sha256", "")),
        str(replay.get("runtime_manifest_path", "")),
        str(replay.get("runtime_manifest_sha256", "")),
    )
    fresh_fp32 = _safe(root, replay["fresh_fp32_bundle_path"])
    fresh_int8 = _safe(root, replay["fresh_int8_bundle_path"])
    report = build_stage0_control_report(
        authority.baseline_fp32_bundle,
        fresh_fp32,
        authority.baseline_int8_bundle,
        fresh_int8,
        authority.onnx_null_report,
    )
    plan = _plan(authority)
    stored_plan = _load_json(_safe(root, replay["stage0_plan_path"]))
    stored_report = _load_json(_safe(root, replay["control_report_path"]))
    plan_match = canonical_json_bytes(plan) == canonical_json_bytes(stored_plan)
    report_match = canonical_json_bytes(report) == canonical_json_bytes(stored_report)
    status_match = report["status"] == replay.get("expected_status")
    if not plan_match or not report_match or not status_match:
        raise Stage0ControlError(
            "STAGE0_REPLAY_MISMATCH",
            "recomputed Stage 0 controls do not match",
        )
    return {
        "status": report["status"],
        "replay_verified": True,
        "plan_match": plan_match,
        "control_report_match": report_match,
        "status_match": status_match,
        "fp32_control_outcome": report["controls"]["verified_onnx_fp32_reference"]["outcome"],
        "int8_control_outcome": report["controls"]["frozen_int8_exact_replay"]["outcome"],
        "stage_1_gate_status": report["stage_1_gate_status"],
        "model_execution_used": False,
        "stage_1_execution_started": False,
    }
