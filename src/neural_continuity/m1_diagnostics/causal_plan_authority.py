from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural_continuity.m1_diagnostics.structural_cluster_authority import (
    StructuralClusterError,
    load_json,
    safe_artifact,
    verify_manifest,
)
from neural_continuity.m1_diagnostics.structural_cluster_evidence import (
    CLUSTER_ARTIFACT_SET,
)

EXPECTED_PROBE_COUNT = 283
EXPECTED_CLUSTER_COUNT = 64
EXPECTED_FINITE_CLUSTER_COUNT = 58
EXPECTED_NONFINITE_CLUSTER_COUNT = 6
EXPECTED_EQUAL_PROBE_COUNT = 52
EXPECTED_FINITE_PROBE_COUNT = 219
EXPECTED_NONFINITE_PROBE_COUNT = 12


class CausalPlanError(RuntimeError):
    def __init__(self, code: str, message: str, status: str = "BLOCKED") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "status": self.status}


@dataclass(frozen=True)
class VerifiedCausalPlanInput:
    root: Path
    bundle_path: Path
    manifest_sha256: str
    cluster_plan: Mapping[str, Any]
    membership: Mapping[str, Any]
    cluster_report: Mapping[str, Any]
    artifact_sha256: Mapping[str, str]


def _load(path: Path) -> Mapping[str, Any]:
    try:
        return load_json(path)
    except StructuralClusterError as exc:
        raise CausalPlanError(
            "CAUSAL_PLAN_SOURCE_ARTIFACT_INVALID",
            exc.message,
        ) from exc


def _artifact(root: Path, relative_path: Any) -> Path:
    try:
        return safe_artifact(root, relative_path)
    except StructuralClusterError as exc:
        raise CausalPlanError(
            "CAUSAL_PLAN_SOURCE_ARTIFACT_INVALID",
            exc.message,
        ) from exc


def _verify_source_manifest(
    root: Path,
    expected_sha256: str,
) -> dict[str, str]:
    try:
        _, hashes = verify_manifest(
            root,
            expected_sha256,
            expected_kind="m1-diagnostic-structural-cluster-manifest",
            expected_artifacts=CLUSTER_ARTIFACT_SET,
        )
    except StructuralClusterError as exc:
        raise CausalPlanError(
            "CAUSAL_PLAN_SOURCE_VERIFICATION_FAILED",
            exc.message,
        ) from exc
    return hashes


def _require_source_semantics(
    plan: Mapping[str, Any],
    membership: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    required_plan = {
        "kind": "m1-diagnostic-structural-cluster-plan",
        "status": "READY",
        "probe_count": EXPECTED_PROBE_COUNT,
        "integer_probe_count": 248,
        "bitwise_equal_is_separator_and_recovery": True,
        "finite_and_nonfinite_clusters_independently_reported": True,
        "nonfinite_ranked_with_finite": False,
        "numerical_threshold_selected": False,
        "candidate_specific_exception_used": False,
        "scientific_decision_recomputed": False,
        "frozen_transition_b_v1_scientific_decision": "FAIL",
        "model_execution_used": False,
        "onnx_graph_loaded": False,
        "activation_artifact_loaded": False,
    }
    required_report = {
        "kind": "m1-diagnostic-structural-cluster-report",
        "status": "COMPLETE",
        "diagnostic_status": "DESCRIPTIVE_ONLY",
        "probe_count": EXPECTED_PROBE_COUNT,
        "cluster_count": EXPECTED_CLUSTER_COUNT,
        "numerical_threshold_selected": False,
        "scientific_decision_recomputed": False,
        "frozen_transition_b_v1_scientific_decision": "FAIL",
        "model_execution_used": False,
        "onnx_graph_loaded": False,
        "activation_artifact_loaded": False,
        "causal_claim_made": False,
    }
    if any(plan.get(key) != value for key, value in required_plan.items()):
        raise CausalPlanError(
            "CAUSAL_PLAN_SOURCE_SCOPE_MISMATCH",
            "cluster plan is outside the frozen diagnostic scope",
        )
    if any(report.get(key) != value for key, value in required_report.items()):
        raise CausalPlanError(
            "CAUSAL_PLAN_SOURCE_SCOPE_MISMATCH",
            "cluster report is outside the frozen diagnostic scope",
        )
    if (
        membership.get("kind") != "m1-diagnostic-probe-cluster-membership"
        or membership.get("status") != "COMPLETE"
        or membership.get("probe_count") != EXPECTED_PROBE_COUNT
        or membership.get("cluster_count") != EXPECTED_CLUSTER_COUNT
    ):
        raise CausalPlanError(
            "CAUSAL_PLAN_SOURCE_INVALID",
            "probe-cluster membership declaration is invalid",
        )
    summary = report.get("summary")
    expected_summary = {
        "bitwise_equal_probe_count": EXPECTED_EQUAL_PROBE_COUNT,
        "finite_drift_probe_count": EXPECTED_FINITE_PROBE_COUNT,
        "nonfinite_probe_count": EXPECTED_NONFINITE_PROBE_COUNT,
        "finite_cluster_count": EXPECTED_FINITE_CLUSTER_COUNT,
        "nonfinite_cluster_count": EXPECTED_NONFINITE_CLUSTER_COUNT,
        "nonfinite_ranked_with_finite": False,
    }
    if not isinstance(summary, Mapping) or any(
        summary.get(key) != value for key, value in expected_summary.items()
    ):
        raise CausalPlanError(
            "CAUSAL_PLAN_SOURCE_IDENTITY_MISMATCH",
            "cluster summary differs from the frozen evidence",
        )
    clusters = report.get("clusters")
    probes = membership.get("probes")
    if (
        not isinstance(clusters, list)
        or len(clusters) != EXPECTED_CLUSTER_COUNT
        or not isinstance(probes, list)
        or len(probes) != EXPECTED_PROBE_COUNT
    ):
        raise CausalPlanError(
            "CAUSAL_PLAN_SOURCE_INVALID",
            "cluster records or memberships are incomplete",
        )
    cluster_ids = {
        item.get("cluster_id")
        for item in clusters
        if isinstance(item, Mapping) and isinstance(item.get("cluster_id"), str)
    }
    if len(cluster_ids) != EXPECTED_CLUSTER_COUNT:
        raise CausalPlanError(
            "CAUSAL_PLAN_SOURCE_INVALID",
            "cluster identities are incomplete or duplicated",
        )
    for probe in probes:
        if not isinstance(probe, Mapping):
            raise CausalPlanError(
                "CAUSAL_PLAN_SOURCE_INVALID",
                "probe membership record is invalid",
            )
        classification = probe.get("classification")
        cluster_id = probe.get("cluster_id")
        if classification == "BITWISE_EQUAL":
            valid = cluster_id is None
        else:
            valid = (
                classification
                in {
                    "FINITE_BITWISE_DRIFT",
                    "NONFINITE_OBSERVED",
                }
                and cluster_id in cluster_ids
            )
        if not valid:
            raise CausalPlanError(
                "CAUSAL_PLAN_SOURCE_INVALID",
                "probe membership is inconsistent with cluster identities",
            )
    source_analysis_manifest = plan.get("source_analysis_manifest_sha256")
    if (
        not isinstance(source_analysis_manifest, str)
        or report.get("source_analysis_manifest_sha256") != source_analysis_manifest
    ):
        raise CausalPlanError(
            "CAUSAL_PLAN_SOURCE_IDENTITY_MISMATCH",
            "analysis authority identity differs across cluster artifacts",
        )


def verify_causal_plan_input(
    bundle_path: str | Path,
    expected_manifest_sha256: str,
) -> VerifiedCausalPlanInput:
    bundle = Path(bundle_path).resolve()
    if not bundle.is_file():
        raise CausalPlanError(
            "CAUSAL_PLAN_SOURCE_MISSING",
            "structural cluster replay bundle is missing",
        )
    root = bundle.parent
    if bundle != (root / "replay-bundle.json").resolve():
        raise CausalPlanError(
            "CAUSAL_PLAN_SOURCE_REPLAY_INVALID",
            "cluster replay bundle is not the manifest-declared replay bundle",
        )
    artifact_hashes = _verify_source_manifest(root, expected_manifest_sha256)
    replay = _load(bundle)
    required_replay = {
        "cluster_plan_path": "cluster-plan.json",
        "probe_cluster_membership_path": "probe-cluster-membership.json",
        "structural_cluster_report_path": "structural-cluster-report.json",
        "expected_status": "COMPLETE",
        "replay_requires_model_execution": False,
    }
    if any(replay.get(key) != value for key, value in required_replay.items()):
        raise CausalPlanError(
            "CAUSAL_PLAN_SOURCE_REPLAY_INVALID",
            "cluster replay bundle is invalid or not model-free",
        )
    plan = _load(_artifact(root, replay["cluster_plan_path"]))
    membership = _load(_artifact(root, replay["probe_cluster_membership_path"]))
    report = _load(_artifact(root, replay["structural_cluster_report_path"]))
    _require_source_semantics(plan, membership, report)
    return VerifiedCausalPlanInput(
        root=root,
        bundle_path=bundle,
        manifest_sha256=expected_manifest_sha256,
        cluster_plan=plan,
        membership=membership,
        cluster_report=report,
        artifact_sha256=artifact_hashes,
    )
