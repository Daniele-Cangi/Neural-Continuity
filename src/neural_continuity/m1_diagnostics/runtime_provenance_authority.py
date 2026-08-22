from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural_continuity.m1_diagnostics.stage0_authority import Stage0ControlError, _verify_archive
from neural_continuity.m1_diagnostics.stage0_evidence import replay_stage0_control_package
from neural_continuity.m1_teacher_evidence import _load_json, _safe_artifact_path


class RuntimeProvenanceError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.status = "BLOCKED"


@dataclass(frozen=True)
class RuntimeProvenanceAuthority:
    stage0_bundle: Path
    stage0_manifest_sha256: str
    stage0_root: Path
    stage0_report: Mapping[str, Any]
    archive_manifest_path: Path
    archive_manifest_sha256: str
    baseline_fp32_bundle: Path
    baseline_int8_bundle: Path
    fresh_fp32_bundle: Path
    fresh_int8_bundle: Path
    null_report: Mapping[str, Any]
    top_k: int
    package_authority_sha256: Mapping[str, str]


def _validate_stage0_replay(result: Mapping[str, Any]) -> None:
    if result.get("replay_verified") is not True:
        raise RuntimeProvenanceError(
            "RUNTIME_PROVENANCE_STAGE0_REPLAY_FAILED", "Stage 0 package is not replay verified"
        )
    if result.get("status") != "BLOCKED":
        raise RuntimeProvenanceError(
            "RUNTIME_PROVENANCE_STAGE0_STATUS_INVALID", "Stage 0 status must remain BLOCKED"
        )
    if result.get("stage_1_execution_started") is not False:
        raise RuntimeProvenanceError(
            "RUNTIME_PROVENANCE_STAGE1_ALREADY_STARTED", "Stage 1 execution must not have started"
        )
    if result.get("model_execution_used") is not False:
        raise RuntimeProvenanceError(
            "RUNTIME_PROVENANCE_REPLAY_USED_MODEL", "Stage 0 authority replay must be model free"
        )


def _require_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeProvenanceError(
            "RUNTIME_PROVENANCE_STAGE0_BUNDLE_INVALID", f"missing Stage 0 replay field: {key}"
        )
    return value


def verify_runtime_provenance_authority(
    stage0_bundle: str | Path, stage0_manifest_sha256: str
) -> RuntimeProvenanceAuthority:
    bundle = Path(stage0_bundle).resolve()
    try:
        replay = replay_stage0_control_package(bundle, stage0_manifest_sha256)
        _validate_stage0_replay(replay)
        payload = _load_json(bundle, "RUNTIME_PROVENANCE_STAGE0_BUNDLE_INVALID")
        root = bundle.parent.resolve()
        if payload.get("replay_requires_model_execution") is not False:
            raise RuntimeProvenanceError(
                "RUNTIME_PROVENANCE_STAGE0_BUNDLE_INVALID",
                "Stage 0 replay must not require model execution",
            )
        fresh_fp32 = _safe_artifact_path(root, _require_string(payload, "fresh_fp32_bundle_path"))
        fresh_int8 = _safe_artifact_path(root, _require_string(payload, "fresh_int8_bundle_path"))
        report_path = _safe_artifact_path(root, _require_string(payload, "control_report_path"))
        report = _load_json(report_path, "RUNTIME_PROVENANCE_STAGE0_REPORT_INVALID")
        if (
            report.get("status") != "BLOCKED"
            or report.get("stage_1_execution_started") is not False
        ):
            raise RuntimeProvenanceError(
                "RUNTIME_PROVENANCE_STAGE0_REPORT_INVALID", "Stage 0 report is not fail-closed"
            )
        if report.get("frozen_transition_b_v1_scientific_decision") != "FAIL":
            raise RuntimeProvenanceError(
                "RUNTIME_PROVENANCE_FROZEN_DECISION_MISMATCH",
                "frozen Transition B decision is not FAIL",
            )
        archive_path = Path(_require_string(payload, "archive_manifest_path")).resolve()
        archive_hash = _require_string(payload, "archive_manifest_sha256")
        packages = _verify_archive(archive_path, archive_hash)
        null_report = _load_json(
            packages["onnx_null"][0] / "comparison-report.json",
            "RUNTIME_PROVENANCE_NULL_REPORT_INVALID",
        )
        top_k = null_report.get("top_k")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise RuntimeProvenanceError(
                "RUNTIME_PROVENANCE_NULL_REPORT_INVALID", "measurement-null top_k is invalid"
            )
    except RuntimeProvenanceError:
        raise
    except Stage0ControlError as exc:
        raise RuntimeProvenanceError(exc.code, str(exc)) from exc
    except Exception as exc:
        raise RuntimeProvenanceError(
            "RUNTIME_PROVENANCE_AUTHORITY_INVALID",
            f"cannot verify runtime provenance authority: {exc}",
        ) from exc
    return RuntimeProvenanceAuthority(
        stage0_bundle=bundle,
        stage0_manifest_sha256=stage0_manifest_sha256,
        stage0_root=root,
        stage0_report=report,
        archive_manifest_path=archive_path,
        archive_manifest_sha256=archive_hash,
        baseline_fp32_bundle=packages["fp32_observation"][0] / "replay-bundle.json",
        baseline_int8_bundle=packages["int8_observation"][0] / "replay-bundle.json",
        fresh_fp32_bundle=fresh_fp32,
        fresh_int8_bundle=fresh_int8,
        null_report=null_report,
        top_k=top_k,
        package_authority_sha256={
            role: authority_hash for role, (_, authority_hash) in packages.items()
        },
    )
