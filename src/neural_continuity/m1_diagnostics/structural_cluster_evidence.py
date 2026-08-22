from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.structural_cluster_analysis import (
    analyze_structural_clusters,
)
from neural_continuity.m1_diagnostics.structural_cluster_authority import (
    StructuralClusterError,
    VerifiedStructuralClusterInput,
    load_json,
    safe_artifact,
    verify_manifest,
    verify_structural_cluster_input,
)

CLUSTER_ARTIFACTS = (
    "cluster-plan.json",
    "probe-cluster-membership.json",
    "structural-cluster-report.json",
    "replay-bundle.json",
)
CLUSTER_ARTIFACT_SET = frozenset(CLUSTER_ARTIFACTS)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _cluster_plan(authority: VerifiedStructuralClusterInput) -> dict[str, Any]:
    return {
        "kind": "m1-diagnostic-structural-cluster-plan",
        "version": "1.0.0",
        "status": "READY",
        "source_analysis_bundle": str(authority.bundle_path),
        "source_analysis_manifest_sha256": authority.manifest_sha256,
        "source_analysis_plan_sha256": authority.artifact_sha256["analysis-plan.json"],
        "source_probe_diagnostics_sha256": authority.artifact_sha256["probe-diagnostics.json"],
        "source_diagnostic_report_sha256": authority.artifact_sha256["diagnostic-report.json"],
        "probe_count": authority.probe_diagnostics["probe_count"],
        "integer_probe_count": authority.analysis_plan["integer_probe_count"],
        "cluster_adjacency_rule": "contiguous_probe_order",
        "cluster_compatibility_rule": (
            "same_classification_and_target_tensor_basis_and_shared_structural_family"
        ),
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


def _documents(
    authority: VerifiedStructuralClusterInput,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    analysis = analyze_structural_clusters(authority.probe_diagnostics)
    plan = _cluster_plan(authority)
    membership = {
        "kind": "m1-diagnostic-probe-cluster-membership",
        "status": "COMPLETE",
        "probe_count": analysis["probe_count"],
        "cluster_count": analysis["cluster_count"],
        "probes": analysis["membership"],
    }
    report = {
        "kind": "m1-diagnostic-structural-cluster-report",
        "status": "COMPLETE",
        "diagnostic_status": "DESCRIPTIVE_ONLY",
        "source_analysis_manifest_sha256": authority.manifest_sha256,
        "probe_count": analysis["probe_count"],
        "cluster_count": analysis["cluster_count"],
        "summary": analysis["summary"],
        "clusters": analysis["clusters"],
        "numerical_threshold_selected": False,
        "scientific_decision_recomputed": False,
        "frozen_transition_b_v1_scientific_decision": "FAIL",
        "model_execution_used": False,
        "onnx_graph_loaded": False,
        "activation_artifact_loaded": False,
        "causal_claim_made": False,
    }
    return plan, membership, report


def _write_manifest(build_root: Path) -> str:
    artifacts = [
        {
            "path": name,
            "sha256": sha256_file(build_root / name),
            "size_bytes": (build_root / name).stat().st_size,
        }
        for name in CLUSTER_ARTIFACTS
    ]
    manifest = {
        "kind": "m1-diagnostic-structural-cluster-manifest",
        "status": "COMPLETE",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "tamper_evident": True,
        "model_execution_used": False,
        "replay_requires_model_execution": False,
    }
    _write_json(build_root / "artifact-manifest.json", manifest)
    return sha256_file(build_root / "artifact-manifest.json")


def create_structural_cluster_package(
    source_bundle: str | Path,
    source_manifest_sha256: str,
    output_directory: str | Path,
) -> dict[str, Any]:
    authority = verify_structural_cluster_input(source_bundle, source_manifest_sha256)
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_OUTPUT_EXISTS",
            "structural cluster output directory already exists",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    build_root = Path(
        tempfile.mkdtemp(
            prefix=".structural-cluster-build-",
            dir=destination.parent,
        )
    ).resolve()
    try:
        plan, membership, report = _documents(authority)
        _write_json(build_root / "cluster-plan.json", plan)
        _write_json(build_root / "probe-cluster-membership.json", membership)
        _write_json(build_root / "structural-cluster-report.json", report)
        replay_bundle = {
            "replay_format_version": "1.0.0",
            "cluster_plan_path": "cluster-plan.json",
            "probe_cluster_membership_path": "probe-cluster-membership.json",
            "structural_cluster_report_path": "structural-cluster-report.json",
            "source_analysis_bundle": str(authority.bundle_path),
            "source_analysis_manifest_sha256": authority.manifest_sha256,
            "expected_status": "COMPLETE",
            "replay_requires_model_execution": False,
        }
        _write_json(build_root / "replay-bundle.json", replay_bundle)
        manifest_sha256 = _write_manifest(build_root)
        build_root.rename(destination)
    except Exception:
        if build_root.exists() and build_root.parent == destination.parent:
            shutil.rmtree(build_root)
        raise
    summary = report["summary"]
    return {
        "status": "COMPLETE",
        "diagnostic_status": "DESCRIPTIVE_ONLY",
        "output_directory": str(destination),
        "artifact_manifest_sha256": manifest_sha256,
        "probe_count": report["probe_count"],
        "cluster_count": report["cluster_count"],
        "finite_cluster_count": summary["finite_cluster_count"],
        "nonfinite_cluster_count": summary["nonfinite_cluster_count"],
        "scientific_decision_recomputed": False,
        "model_execution_used": False,
        "activation_artifact_loaded": False,
    }


def replay_structural_cluster_package(
    bundle_path: str | Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    bundle = Path(bundle_path).resolve()
    root = bundle.parent
    if bundle != (root / "replay-bundle.json").resolve():
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_REPLAY_INVALID",
            "replay bundle is not the manifest-declared replay bundle",
        )
    verify_manifest(
        root,
        expected_manifest_sha256,
        expected_kind="m1-diagnostic-structural-cluster-manifest",
        expected_artifacts=CLUSTER_ARTIFACT_SET,
    )
    replay_bundle = load_json(bundle)
    required_bundle = {
        "cluster_plan_path": "cluster-plan.json",
        "probe_cluster_membership_path": "probe-cluster-membership.json",
        "structural_cluster_report_path": "structural-cluster-report.json",
        "expected_status": "COMPLETE",
        "replay_requires_model_execution": False,
    }
    if any(replay_bundle.get(key) != value for key, value in required_bundle.items()):
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_REPLAY_INVALID",
            "structural cluster replay bundle is invalid or not model-free",
        )
    source_bundle = replay_bundle.get("source_analysis_bundle")
    source_manifest = replay_bundle.get("source_analysis_manifest_sha256")
    if not isinstance(source_bundle, str) or not isinstance(source_manifest, str):
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_REPLAY_INVALID",
            "source analysis authority is missing",
        )
    authority = verify_structural_cluster_input(source_bundle, source_manifest)
    plan, membership, report = _documents(authority)
    stored_plan = load_json(safe_artifact(root, replay_bundle["cluster_plan_path"]))
    stored_membership = load_json(
        safe_artifact(root, replay_bundle["probe_cluster_membership_path"])
    )
    stored_report = load_json(safe_artifact(root, replay_bundle["structural_cluster_report_path"]))
    plan_match = canonical_json_bytes(plan) == canonical_json_bytes(stored_plan)
    membership_match = canonical_json_bytes(membership) == canonical_json_bytes(stored_membership)
    report_match = canonical_json_bytes(report) == canonical_json_bytes(stored_report)
    if not plan_match or not membership_match or not report_match:
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_REPLAY_MISMATCH",
            "recomputed structural cluster evidence does not match",
        )
    summary = report["summary"]
    return {
        "status": "COMPLETE",
        "diagnostic_status": "DESCRIPTIVE_ONLY",
        "replay_verified": True,
        "plan_match": plan_match,
        "probe_cluster_membership_match": membership_match,
        "structural_cluster_report_match": report_match,
        "probe_count": report["probe_count"],
        "cluster_count": report["cluster_count"],
        "finite_cluster_count": summary["finite_cluster_count"],
        "nonfinite_cluster_count": summary["nonfinite_cluster_count"],
        "scientific_decision_recomputed": False,
        "model_execution_used": False,
        "activation_artifact_loaded": False,
    }
