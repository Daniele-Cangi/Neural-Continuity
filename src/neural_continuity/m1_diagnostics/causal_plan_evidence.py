from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.causal_plan_authority import (
    CausalPlanError,
    VerifiedCausalPlanInput,
    verify_causal_plan_input,
)
from neural_continuity.m1_diagnostics.causal_plan_design import (
    build_causal_design,
)
from neural_continuity.m1_diagnostics.structural_cluster_authority import (
    StructuralClusterError,
    load_json,
    safe_artifact,
    verify_manifest,
)

CAUSAL_PLAN_ARTIFACTS = (
    "causal-plan.json",
    "cluster-hypotheses.json",
    "intervention-matrix.json",
    "replay-bundle.json",
)
CAUSAL_PLAN_ARTIFACT_SET = frozenset(CAUSAL_PLAN_ARTIFACTS)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _plan(authority: VerifiedCausalPlanInput) -> dict[str, Any]:
    return {
        "kind": "m1-diagnostic-causal-plan",
        "version": "1.0.0",
        "status": "PRE_REGISTERED",
        "source_cluster_bundle": str(authority.bundle_path),
        "source_cluster_manifest_sha256": authority.manifest_sha256,
        "source_cluster_plan_sha256": authority.artifact_sha256["cluster-plan.json"],
        "source_cluster_membership_sha256": authority.artifact_sha256[
            "probe-cluster-membership.json"
        ],
        "source_cluster_report_sha256": authority.artifact_sha256["structural-cluster-report.json"],
        "probe_count": authority.cluster_report["probe_count"],
        "cluster_count": authority.cluster_report["cluster_count"],
        "frozen_transition_b_v1_scientific_decision": "FAIL",
        "hypothesis_decision_states": [
            "SUPPORTED",
            "NOT_SUPPORTED",
            "INCONCLUSIVE",
        ],
        "technical_states": ["BLOCKED", "EXECUTION_ERROR"],
        "measurement_detection_authority": (
            "verified_M0_detection_limit_to_be_frozen_before_execution"
        ),
        "operational_tolerance_authority": ("frozen_transition_B_v1_contract_unchanged"),
        "candidate_or_holdout_result_may_select_tolerance": False,
        "measurement_detection_and_operational_tolerance_separate": True,
        "all_clusters_included": True,
        "cluster_selection_cutoff_used": False,
        "numerical_threshold_selected": False,
        "candidate_specific_exception_used": False,
        "frozen_int8_candidate_mutated": False,
        "derived_diagnostic_candidate_created": False,
        "intervention_execution_authorized": False,
        "scientific_decision_recomputed": False,
        "causal_claim_made": False,
        "model_execution_used": False,
        "onnx_graph_loaded": False,
        "activation_artifact_loaded": False,
    }


def _documents(
    authority: VerifiedCausalPlanInput,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    design = build_causal_design(authority.cluster_report)
    plan = _plan(authority)
    hypotheses = {
        "kind": "m1-diagnostic-cluster-hypotheses",
        "status": "PRE_REGISTERED",
        "source_cluster_manifest_sha256": authority.manifest_sha256,
        "observed_structural_families": design["observed_structural_families"],
        "family_hypotheses": design["family_hypotheses"],
        "interaction_hypotheses": design["interaction_hypotheses"],
        "summary": design["summary"],
        "causal_claim_made": False,
    }
    matrix = {
        "kind": "m1-diagnostic-intervention-matrix",
        "status": "PRE_REGISTERED_NOT_AUTHORIZED",
        "source_cluster_manifest_sha256": authority.manifest_sha256,
        "controls": design["controls"],
        "stage_1_single_family_interventions": design["single_family_interventions"],
        "stage_2_pair_interventions": design["pair_interventions"],
        "execution_order": [
            "controls",
            "stage_1_single_family_interventions",
            "stage_2_pair_interventions_if_activation_condition_met",
        ],
        "shared_frozen_requirements": [
            "dataset_materialization_identity",
            "query_and_document_identities",
            "role_ordering_and_qrels",
            "tokenizer_and_preprocessing_identity",
            "normalization_semantics",
            "CPUExecutionProvider",
            "execution_batch_sizes_1_16_64",
            "artifact_integrity",
        ],
        "missing_authority_or_observation_status": "BLOCKED",
        "technical_execution_failure_status": "EXECUTION_ERROR",
        "scientific_fail_used_for_execution_error": False,
        "execution_authorized": False,
    }
    return plan, hypotheses, matrix


def _write_manifest(build_root: Path) -> str:
    artifacts = [
        {
            "path": name,
            "sha256": sha256_file(build_root / name),
            "size_bytes": (build_root / name).stat().st_size,
        }
        for name in CAUSAL_PLAN_ARTIFACTS
    ]
    manifest = {
        "kind": "m1-diagnostic-causal-plan-manifest",
        "status": "COMPLETE",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "tamper_evident": True,
        "model_execution_used": False,
        "replay_requires_model_execution": False,
    }
    _write_json(build_root / "artifact-manifest.json", manifest)
    return sha256_file(build_root / "artifact-manifest.json")


def _output_artifact(root: Path, relative_path: Any) -> Path:
    try:
        return safe_artifact(root, relative_path)
    except StructuralClusterError as exc:
        raise CausalPlanError(
            "CAUSAL_PLAN_REPLAY_ARTIFACT_INVALID",
            exc.message,
        ) from exc


def _verify_output_manifest(root: Path, expected_sha256: str) -> None:
    try:
        verify_manifest(
            root,
            expected_sha256,
            expected_kind="m1-diagnostic-causal-plan-manifest",
            expected_artifacts=CAUSAL_PLAN_ARTIFACT_SET,
        )
    except StructuralClusterError as exc:
        raise CausalPlanError(
            "CAUSAL_PLAN_REPLAY_VERIFICATION_FAILED",
            exc.message,
        ) from exc


def create_causal_plan_package(
    source_bundle: str | Path,
    source_manifest_sha256: str,
    output_directory: str | Path,
) -> dict[str, Any]:
    authority = verify_causal_plan_input(source_bundle, source_manifest_sha256)
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise CausalPlanError(
            "CAUSAL_PLAN_OUTPUT_EXISTS",
            "causal plan output directory already exists",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    build_root = Path(
        tempfile.mkdtemp(
            prefix=".causal-plan-build-",
            dir=destination.parent,
        )
    ).resolve()
    try:
        plan, hypotheses, matrix = _documents(authority)
        _write_json(build_root / "causal-plan.json", plan)
        _write_json(build_root / "cluster-hypotheses.json", hypotheses)
        _write_json(build_root / "intervention-matrix.json", matrix)
        replay = {
            "replay_format_version": "1.0.0",
            "causal_plan_path": "causal-plan.json",
            "cluster_hypotheses_path": "cluster-hypotheses.json",
            "intervention_matrix_path": "intervention-matrix.json",
            "source_cluster_bundle": str(authority.bundle_path),
            "source_cluster_manifest_sha256": authority.manifest_sha256,
            "expected_status": "PRE_REGISTERED",
            "replay_requires_model_execution": False,
        }
        _write_json(build_root / "replay-bundle.json", replay)
        manifest_sha256 = _write_manifest(build_root)
        build_root.rename(destination)
    except Exception:
        if build_root.exists() and build_root.parent == destination.parent:
            shutil.rmtree(build_root)
        raise
    summary = hypotheses["summary"]
    return {
        "status": "PRE_REGISTERED",
        "output_directory": str(destination),
        "artifact_manifest_sha256": manifest_sha256,
        "cluster_count": authority.cluster_report["cluster_count"],
        "family_hypothesis_count": summary["family_hypothesis_count"],
        "interaction_hypothesis_count": summary["interaction_hypothesis_count"],
        "intervention_execution_authorized": False,
        "causal_claim_made": False,
        "model_execution_used": False,
        "activation_artifact_loaded": False,
    }


def replay_causal_plan_package(
    bundle_path: str | Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    bundle = Path(bundle_path).resolve()
    root = bundle.parent
    if bundle != (root / "replay-bundle.json").resolve():
        raise CausalPlanError(
            "CAUSAL_PLAN_REPLAY_INVALID",
            "replay bundle is not the manifest-declared replay bundle",
        )
    _verify_output_manifest(root, expected_manifest_sha256)
    replay = load_json(bundle)
    required_replay = {
        "causal_plan_path": "causal-plan.json",
        "cluster_hypotheses_path": "cluster-hypotheses.json",
        "intervention_matrix_path": "intervention-matrix.json",
        "expected_status": "PRE_REGISTERED",
        "replay_requires_model_execution": False,
    }
    if any(replay.get(key) != value for key, value in required_replay.items()):
        raise CausalPlanError(
            "CAUSAL_PLAN_REPLAY_INVALID",
            "causal plan replay bundle is invalid or not model-free",
        )
    source_bundle = replay.get("source_cluster_bundle")
    source_manifest = replay.get("source_cluster_manifest_sha256")
    if not isinstance(source_bundle, str) or not isinstance(source_manifest, str):
        raise CausalPlanError(
            "CAUSAL_PLAN_REPLAY_INVALID",
            "source cluster authority is missing",
        )
    authority = verify_causal_plan_input(source_bundle, source_manifest)
    plan, hypotheses, matrix = _documents(authority)
    stored_plan = load_json(_output_artifact(root, replay["causal_plan_path"]))
    stored_hypotheses = load_json(_output_artifact(root, replay["cluster_hypotheses_path"]))
    stored_matrix = load_json(_output_artifact(root, replay["intervention_matrix_path"]))
    plan_match = canonical_json_bytes(plan) == canonical_json_bytes(stored_plan)
    hypotheses_match = canonical_json_bytes(hypotheses) == canonical_json_bytes(stored_hypotheses)
    matrix_match = canonical_json_bytes(matrix) == canonical_json_bytes(stored_matrix)
    if not plan_match or not hypotheses_match or not matrix_match:
        raise CausalPlanError(
            "CAUSAL_PLAN_REPLAY_MISMATCH",
            "recomputed causal plan evidence does not match",
        )
    summary = hypotheses["summary"]
    return {
        "status": "PRE_REGISTERED",
        "replay_verified": True,
        "plan_match": plan_match,
        "hypotheses_match": hypotheses_match,
        "intervention_matrix_match": matrix_match,
        "cluster_count": authority.cluster_report["cluster_count"],
        "family_hypothesis_count": summary["family_hypothesis_count"],
        "interaction_hypothesis_count": summary["interaction_hypothesis_count"],
        "intervention_execution_authorized": False,
        "causal_claim_made": False,
        "model_execution_used": False,
        "activation_artifact_loaded": False,
    }
