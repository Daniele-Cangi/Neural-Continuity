from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural_continuity.m1_diagnostics.runtime_provenance_evidence import (
    replay_runtime_provenance_package,
)
from neural_continuity.m1_teacher_evidence import _load_json, _safe_artifact_path


class MeasurementNullPlanError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.status = "BLOCKED"


@dataclass(frozen=True)
class MeasurementNullPlanAuthority:
    provenance_bundle: Path
    provenance_manifest_sha256: str
    provenance_root: Path
    provenance_audit: Mapping[str, Any]
    runtime_inventory: Mapping[str, Any]


def _validate_provenance_replay(result: Mapping[str, Any]) -> None:
    required = {
        "replay_verified": True,
        "status": "BLOCKED",
        "attribution_status": "INCONCLUSIVE",
        "classification": "FROZEN_BATCH_ENVELOPE_DOES_NOT_COVER_CANONICAL_BASELINE",
        "model_execution_used": False,
        "onnx_graph_loaded": False,
        "activation_read": False,
        "stage_1_execution_started": False,
    }
    for key, expected in required.items():
        if result.get(key) != expected:
            raise MeasurementNullPlanError(
                "MEASUREMENT_NULL_PLAN_PROVENANCE_INVALID",
                f"runtime provenance replay field differs: {key}",
            )


def verify_measurement_null_plan_authority(
    provenance_bundle: str | Path,
    provenance_manifest_sha256: str,
) -> MeasurementNullPlanAuthority:
    bundle = Path(provenance_bundle).resolve()
    try:
        result = replay_runtime_provenance_package(bundle, provenance_manifest_sha256)
        _validate_provenance_replay(result)
        payload = _load_json(bundle, "MEASUREMENT_NULL_PLAN_PROVENANCE_BUNDLE_INVALID")
        root = bundle.parent.resolve()
        if payload.get("replay_requires_model_execution") is not False:
            raise MeasurementNullPlanError(
                "MEASUREMENT_NULL_PLAN_PROVENANCE_BUNDLE_INVALID",
                "runtime provenance replay must remain model free",
            )
        audit_path = payload.get("audit_path")
        inventory_path = payload.get("inventory_path")
        if not isinstance(audit_path, str) or not isinstance(inventory_path, str):
            raise MeasurementNullPlanError(
                "MEASUREMENT_NULL_PLAN_PROVENANCE_BUNDLE_INVALID",
                "runtime provenance artifact paths are missing",
            )
        audit = _load_json(
            _safe_artifact_path(root, audit_path),
            "MEASUREMENT_NULL_PLAN_PROVENANCE_AUDIT_INVALID",
        )
        inventory = _load_json(
            _safe_artifact_path(root, inventory_path),
            "MEASUREMENT_NULL_PLAN_RUNTIME_INVENTORY_INVALID",
        )
        attribution = audit.get("attribution")
        if not isinstance(attribution, Mapping) or attribution.get("classification") != (
            "FROZEN_BATCH_ENVELOPE_DOES_NOT_COVER_CANONICAL_BASELINE"
        ):
            raise MeasurementNullPlanError(
                "MEASUREMENT_NULL_PLAN_PROVENANCE_AUDIT_INVALID",
                "runtime provenance classification is not the frozen-envelope coverage failure",
            )
        if audit.get("stage_1_execution_started") is not False:
            raise MeasurementNullPlanError(
                "MEASUREMENT_NULL_PLAN_STAGE1_ALREADY_STARTED",
                "Stage 1 execution must not have started",
            )
        if any(
            audit.get(field) is not False
            for field in ("model_execution_used", "onnx_graph_loaded", "activation_read")
        ):
            raise MeasurementNullPlanError(
                "MEASUREMENT_NULL_PLAN_PROVENANCE_AUDIT_INVALID",
                "runtime provenance audit is not model free",
            )
    except MeasurementNullPlanError:
        raise
    except Exception as exc:
        raise MeasurementNullPlanError(
            "MEASUREMENT_NULL_PLAN_AUTHORITY_INVALID",
            f"cannot verify measurement-null plan authority: {exc}",
        ) from exc
    return MeasurementNullPlanAuthority(
        provenance_bundle=bundle,
        provenance_manifest_sha256=provenance_manifest_sha256,
        provenance_root=root,
        provenance_audit=audit,
        runtime_inventory=inventory,
    )
