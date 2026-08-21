"""Verification of the frozen M1 diagnostic static-preflight package."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from neural_continuity.m1_diagnostics.authority import (
    DiagnosticPreflightError,
    FrozenAuthorityPaths,
    VerifiedAuthoritySet,
    verify_frozen_authority_set,
)

STATIC_PREFLIGHT_MANIFEST_SHA256 = (
    "efbd9f60588d4b7b080b41f48de3691860634778b0a4f255c3fc54e6d690e507"
)
STATIC_PROBE_PLAN_SHA256 = "f432ed99a8d0747b2d763863397fd9e8e95569419937eaa12a716f6ff7e626e7"
STATIC_PROBE_COUNT = 283

_REQUIRED_ARTIFACTS = frozenset(
    {
        "diagnostic-authority.json",
        "probe-plan.json",
        "quantization-parameter-audit.json",
        "source-graph-inventory.json",
        "static-preflight-report.json",
        "target-graph-inventory.json",
    }
)


@dataclass(frozen=True)
class VerifiedStaticPreflight:
    package_directory: Path
    manifest_sha256: str
    probe_plan_sha256: str
    authorities: VerifiedAuthoritySet
    probe_plan: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "package_directory": str(self.package_directory),
            "manifest_sha256": self.manifest_sha256,
            "probe_plan_sha256": self.probe_plan_sha256,
            "probe_count": STATIC_PROBE_COUNT,
            "static_preflight_status": "STATIC_PREFLIGHT_COMPLETE",
            "authorities": self.authorities.to_dict(),
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="STATIC_ARTIFACT_INVALID",
            message="Static diagnostic artifact is not valid UTF-8 JSON",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    if not isinstance(payload, dict):
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="STATIC_ARTIFACT_INVALID",
            message="Static diagnostic artifact must contain a JSON object",
            details={"path": str(path)},
        )
    return cast(dict[str, object], payload)


def _require_string(mapping: dict[str, object], key: str, *, artifact: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="STATIC_ARTIFACT_FIELD_INVALID",
            message=f"Static artifact field must be a non-empty string: {key}",
            details={"artifact": artifact, "field": key},
        )
    return value


def _safe_artifact_path(root: Path, relative_name: str) -> Path:
    relative = Path(relative_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="STATIC_ARTIFACT_PATH_INVALID",
            message="Static artifact path must remain inside its package",
            details={"path": relative_name},
        )
    resolved = (root / relative).resolve()
    if resolved.parent != root:
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="STATIC_ARTIFACT_PATH_INVALID",
            message="Static artifact path escaped its package",
            details={"path": relative_name},
        )
    return resolved


def _verify_declared_artifacts(root: Path, manifest: dict[str, object]) -> None:
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="STATIC_MANIFEST_INVALID",
            message="Static manifest artifacts must be a list",
        )
    declared: set[str] = set()
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise DiagnosticPreflightError(
                status="BLOCKED",
                code="STATIC_MANIFEST_INVALID",
                message="Static manifest artifact entry must be an object",
            )
        entry = cast(dict[str, object], raw_entry)
        name = _require_string(entry, "path", artifact="artifact-manifest.json")
        expected_sha256 = _require_string(entry, "sha256", artifact="artifact-manifest.json")
        expected_size = entry.get("size_bytes")
        if name in declared or not isinstance(expected_size, int) or expected_size < 0:
            raise DiagnosticPreflightError(
                status="BLOCKED",
                code="STATIC_MANIFEST_INVALID",
                message="Static manifest contains a duplicate path or invalid size",
                details={"path": name},
            )
        path = _safe_artifact_path(root, name)
        if not path.is_file():
            raise DiagnosticPreflightError(
                status="BLOCKED",
                code="STATIC_ARTIFACT_MISSING",
                message="Declared static artifact is missing",
                details={"path": str(path)},
            )
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256 or path.stat().st_size != expected_size:
            raise DiagnosticPreflightError(
                status="BLOCKED",
                code="STATIC_ARTIFACT_INTEGRITY_MISMATCH",
                message="Declared static artifact failed integrity verification",
                details={"path": str(path)},
            )
        declared.add(name)
    if declared != _REQUIRED_ARTIFACTS:
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="STATIC_ARTIFACT_SET_MISMATCH",
            message="Static package artifact set differs from the frozen declaration",
            details={
                "missing": sorted(_REQUIRED_ARTIFACTS - declared),
                "unexpected": sorted(declared - _REQUIRED_ARTIFACTS),
            },
        )


def _verify_authorities(root: Path, authority_document: dict[str, object]) -> VerifiedAuthoritySet:
    entries = authority_document.get("authorities")
    if authority_document.get("all_authorities_verified") is not True or not isinstance(
        entries, list
    ):
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="STATIC_AUTHORITY_RECORD_INVALID",
            message="Static authority record is incomplete",
        )
    paths_by_role: dict[str, Path] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise DiagnosticPreflightError(
                status="BLOCKED",
                code="STATIC_AUTHORITY_RECORD_INVALID",
                message="Static authority entry must be an object",
            )
        entry = cast(dict[str, object], raw_entry)
        role = _require_string(entry, "role", artifact="diagnostic-authority.json")
        path = _require_string(entry, "path", artifact="diagnostic-authority.json")
        if role in paths_by_role:
            raise DiagnosticPreflightError(
                status="BLOCKED",
                code="STATIC_AUTHORITY_RECORD_INVALID",
                message="Static authority role is duplicated",
                details={"role": role},
            )
        paths_by_role[role] = Path(path)
    required_roles = {
        "onnx_fp32_source",
        "onnx_int8_candidate",
        "calibration_manifest",
        "paired_fp32_evidence",
        "int8_target_evidence",
        "transition_b_decision",
        "transition_a_contract",
        "transition_b_v1_contract",
    }
    if set(paths_by_role) != required_roles:
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="STATIC_AUTHORITY_RECORD_INVALID",
            message="Static authority roles differ from the frozen set",
        )
    del root
    return verify_frozen_authority_set(
        FrozenAuthorityPaths(
            onnx_fp32_source=paths_by_role["onnx_fp32_source"],
            onnx_int8_candidate=paths_by_role["onnx_int8_candidate"],
            calibration_manifest=paths_by_role["calibration_manifest"],
            paired_fp32_evidence=paths_by_role["paired_fp32_evidence"],
            int8_target_evidence=paths_by_role["int8_target_evidence"],
            transition_b_decision=paths_by_role["transition_b_decision"],
            transition_a_contract=paths_by_role["transition_a_contract"],
            transition_b_v1_contract=paths_by_role["transition_b_v1_contract"],
        )
    )


def _probe_plan_hash(plan: dict[str, object]) -> str:
    payload = dict(plan)
    payload.pop("probe_count", None)
    payload.pop("probe_plan_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )
    return hashlib.sha256(encoded).hexdigest()


def verify_static_preflight_package(
    package_directory: Path,
) -> VerifiedStaticPreflight:
    """Verify the complete static package before importing or loading ONNX."""

    root = package_directory.resolve()
    manifest_path = root / "artifact-manifest.json"
    if not manifest_path.is_file():
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="STATIC_MANIFEST_MISSING",
            message="Frozen static-preflight manifest is missing",
            details={"path": str(manifest_path)},
        )
    manifest_sha256 = _sha256_file(manifest_path)
    if manifest_sha256 != STATIC_PREFLIGHT_MANIFEST_SHA256:
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="STATIC_MANIFEST_HASH_MISMATCH",
            message="Static-preflight manifest does not match the frozen authority",
            details={
                "expected_sha256": STATIC_PREFLIGHT_MANIFEST_SHA256,
                "actual_sha256": manifest_sha256,
            },
        )
    manifest = _load_mapping(manifest_path)
    if manifest.get("kind") != "m1_transition_b_v2_static_preflight_manifest":
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="STATIC_MANIFEST_INVALID",
            message="Static-preflight manifest kind is invalid",
        )
    _verify_declared_artifacts(root, manifest)

    report = _load_mapping(root / "static-preflight-report.json")
    if (
        report.get("status") != "STATIC_PREFLIGHT_COMPLETE"
        or report.get("quantization_audit_integrity_status") != "PASS"
        or report.get("model_execution_used") is not False
        or report.get("onnx_runtime_session_created") is not False
        or report.get("activations_read") is not False
    ):
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="STATIC_PREFLIGHT_NOT_QUALIFIED",
            message="Static-preflight report is not qualified for instrumentation",
        )

    plan = _load_mapping(root / "probe-plan.json")
    plan_sha256 = _probe_plan_hash(plan)
    probes = plan.get("probes")
    if (
        plan.get("probe_plan_sha256") != STATIC_PROBE_PLAN_SHA256
        or plan_sha256 != STATIC_PROBE_PLAN_SHA256
        or plan.get("probe_count") != STATIC_PROBE_COUNT
        or not isinstance(probes, list)
        or len(probes) != STATIC_PROBE_COUNT
    ):
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="STATIC_PROBE_PLAN_MISMATCH",
            message="Probe plan differs from the frozen static authority",
        )
    probe_ids: set[str] = set()
    for raw_probe in probes:
        if not isinstance(raw_probe, dict):
            raise DiagnosticPreflightError(
                status="BLOCKED",
                code="STATIC_PROBE_PLAN_INVALID",
                message="Probe plan entry must be an object",
            )
        probe = cast(dict[str, object], raw_probe)
        probe_id = _require_string(probe, "probe_id", artifact="probe-plan.json")
        _require_string(probe, "source_tensor", artifact="probe-plan.json")
        _require_string(probe, "target_tensor", artifact="probe-plan.json")
        if probe_id in probe_ids:
            raise DiagnosticPreflightError(
                status="BLOCKED",
                code="STATIC_PROBE_PLAN_INVALID",
                message="Probe plan contains a duplicate probe identity",
                details={"probe_id": probe_id},
            )
        probe_ids.add(probe_id)

    authorities = _verify_authorities(root, _load_mapping(root / "diagnostic-authority.json"))
    return VerifiedStaticPreflight(
        package_directory=root,
        manifest_sha256=manifest_sha256,
        probe_plan_sha256=plan_sha256,
        authorities=authorities,
        probe_plan=plan,
    )
