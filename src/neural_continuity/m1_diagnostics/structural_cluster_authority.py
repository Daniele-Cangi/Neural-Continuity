from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural_continuity.evidence import sha256_file

EXPECTED_ANALYSIS_ARTIFACTS = frozenset(
    {
        "analysis-plan.json",
        "probe-diagnostics.json",
        "diagnostic-report.json",
        "replay-bundle.json",
    }
)
EXPECTED_PROBE_COUNT = 283
EXPECTED_INTEGER_PROBE_COUNT = 248
EXPECTED_QUERY_COUNT = 364
EXPECTED_BATCH_COUNT = 23


class StructuralClusterError(RuntimeError):
    def __init__(self, code: str, message: str, status: str = "BLOCKED") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "status": self.status}


@dataclass(frozen=True)
class VerifiedStructuralClusterInput:
    root: Path
    bundle_path: Path
    manifest_sha256: str
    analysis_plan: Mapping[str, Any]
    probe_diagnostics: Mapping[str, Any]
    diagnostic_report: Mapping[str, Any]
    artifact_sha256: Mapping[str, str]


def load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_ARTIFACT_INVALID",
            f"cannot load {path.name}: {exc}",
        ) from exc
    if not isinstance(payload, Mapping):
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_ARTIFACT_INVALID",
            f"{path.name} must contain an object",
        )
    return payload


def safe_artifact(root: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_ARTIFACT_INVALID",
            "artifact path is missing",
        )
    relative = Path(relative_path)
    if relative.is_absolute():
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_ARTIFACT_INVALID",
            "artifact path must be relative",
        )
    candidate = (root / relative).resolve()
    if root != candidate and root not in candidate.parents:
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_ARTIFACT_INVALID",
            "artifact path escapes the package",
        )
    if not candidate.is_file():
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_ARTIFACT_MISSING",
            f"declared artifact is missing: {relative_path}",
        )
    return candidate


def verify_manifest(
    root: Path,
    expected_sha256: str,
    *,
    expected_kind: str,
    expected_artifacts: frozenset[str],
) -> tuple[Mapping[str, Any], dict[str, str]]:
    manifest_path = root / "artifact-manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected_sha256:
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_MANIFEST_HASH_MISMATCH",
            "artifact manifest hash does not match",
        )
    manifest = load_json(manifest_path)
    required_header = {
        "kind": expected_kind,
        "status": "COMPLETE",
        "artifact_count": len(expected_artifacts),
        "tamper_evident": True,
        "model_execution_used": False,
        "replay_requires_model_execution": False,
    }
    if any(manifest.get(key) != value for key, value in required_header.items()):
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_MANIFEST_INVALID",
            "artifact manifest authority fields are invalid",
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(expected_artifacts):
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_MANIFEST_INVALID",
            "artifact manifest declaration is incomplete",
        )
    hashes: dict[str, str] = {}
    for record in artifacts:
        if not isinstance(record, Mapping):
            raise StructuralClusterError(
                "STRUCTURAL_CLUSTER_MANIFEST_INVALID",
                "artifact record is invalid",
            )
        relative_path = record.get("path")
        artifact = safe_artifact(root, relative_path)
        if not isinstance(relative_path, str) or relative_path in hashes:
            raise StructuralClusterError(
                "STRUCTURAL_CLUSTER_MANIFEST_INVALID",
                "artifact identity is invalid or duplicated",
            )
        expected_hash = record.get("sha256")
        if not isinstance(expected_hash, str) or sha256_file(artifact) != expected_hash:
            raise StructuralClusterError(
                "STRUCTURAL_CLUSTER_ARTIFACT_HASH_MISMATCH",
                f"artifact hash does not match: {relative_path}",
            )
        if artifact.stat().st_size != record.get("size_bytes"):
            raise StructuralClusterError(
                "STRUCTURAL_CLUSTER_ARTIFACT_SIZE_MISMATCH",
                f"artifact size does not match: {relative_path}",
            )
        hashes[relative_path] = expected_hash
    if frozenset(hashes) != expected_artifacts:
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_MANIFEST_INVALID",
            "artifact set differs from the frozen declaration",
        )
    return manifest, hashes


def _require_source_semantics(
    plan: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    required_plan = {
        "kind": "m1-diagnostic-activation-analysis-plan",
        "status": "READY",
        "query_count": EXPECTED_QUERY_COUNT,
        "probe_count": EXPECTED_PROBE_COUNT,
        "integer_probe_count": EXPECTED_INTEGER_PROBE_COUNT,
        "numerical_threshold_selected": False,
        "scientific_decision_recomputed": False,
        "frozen_transition_b_v1_scientific_decision": "FAIL",
        "model_execution_used": False,
        "onnx_graph_loaded": False,
    }
    required_report = {
        "kind": "m1-diagnostic-activation-report",
        "status": "COMPLETE",
        "diagnostic_status": "DESCRIPTIVE_ONLY",
        "batch_count": EXPECTED_BATCH_COUNT,
        "query_count": EXPECTED_QUERY_COUNT,
        "probe_count": EXPECTED_PROBE_COUNT,
        "integer_probe_count": EXPECTED_INTEGER_PROBE_COUNT,
        "scientific_decision_recomputed": False,
        "frozen_transition_b_v1_scientific_decision": "FAIL",
        "model_execution_used": False,
        "causal_claim_made": False,
    }
    if any(plan.get(key) != value for key, value in required_plan.items()):
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_SOURCE_SCOPE_MISMATCH",
            "analysis plan is outside the frozen diagnostic scope",
        )
    if any(report.get(key) != value for key, value in required_report.items()):
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_SOURCE_SCOPE_MISMATCH",
            "diagnostic report is outside the frozen diagnostic scope",
        )
    if (
        diagnostics.get("kind") != "m1-diagnostic-activation-probe-diagnostics"
        or diagnostics.get("status") != "COMPLETE"
        or diagnostics.get("probe_count") != EXPECTED_PROBE_COUNT
    ):
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_SOURCE_INVALID",
            "probe diagnostics declaration is invalid",
        )
    probes = diagnostics.get("probes")
    if not isinstance(probes, list) or len(probes) != EXPECTED_PROBE_COUNT:
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_SOURCE_INVALID",
            "probe diagnostics are incomplete",
        )
    source_manifest = plan.get("source_activation_manifest_sha256")
    if (
        not isinstance(source_manifest, str)
        or report.get("source_activation_manifest_sha256") != source_manifest
    ):
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_SOURCE_IDENTITY_MISMATCH",
            "activation authority identity differs across analysis artifacts",
        )


def verify_structural_cluster_input(
    bundle_path: str | Path,
    expected_manifest_sha256: str,
) -> VerifiedStructuralClusterInput:
    bundle = Path(bundle_path).resolve()
    if not bundle.is_file():
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_SOURCE_MISSING",
            "analysis replay bundle is missing",
        )
    root = bundle.parent
    if bundle != (root / "replay-bundle.json").resolve():
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_SOURCE_REPLAY_INVALID",
            "analysis replay bundle is not the manifest-declared replay bundle",
        )
    _, artifact_hashes = verify_manifest(
        root,
        expected_manifest_sha256,
        expected_kind="m1-diagnostic-activation-analysis-manifest",
        expected_artifacts=EXPECTED_ANALYSIS_ARTIFACTS,
    )
    replay_bundle = load_json(bundle)
    required_bundle = {
        "analysis_plan_path": "analysis-plan.json",
        "probe_diagnostics_path": "probe-diagnostics.json",
        "diagnostic_report_path": "diagnostic-report.json",
        "expected_status": "COMPLETE",
        "replay_requires_model_execution": False,
    }
    if any(replay_bundle.get(key) != value for key, value in required_bundle.items()):
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_SOURCE_REPLAY_INVALID",
            "analysis replay bundle is not model-free or has unexpected paths",
        )
    plan = load_json(safe_artifact(root, replay_bundle["analysis_plan_path"]))
    diagnostics = load_json(safe_artifact(root, replay_bundle["probe_diagnostics_path"]))
    report = load_json(safe_artifact(root, replay_bundle["diagnostic_report_path"]))
    _require_source_semantics(plan, diagnostics, report)
    return VerifiedStructuralClusterInput(
        root=root,
        bundle_path=bundle,
        manifest_sha256=expected_manifest_sha256,
        analysis_plan=plan,
        probe_diagnostics=diagnostics,
        diagnostic_report=report,
        artifact_sha256=artifact_hashes,
    )
