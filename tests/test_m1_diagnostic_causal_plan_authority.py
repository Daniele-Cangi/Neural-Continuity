from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.causal_plan_authority import (
    CausalPlanError,
    verify_causal_plan_input,
)
from neural_continuity.m1_diagnostics.structural_cluster_evidence import (
    CLUSTER_ARTIFACTS,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _source_package(
    root: Path,
    *,
    replay_requires_model_execution: bool = False,
) -> tuple[Path, str]:
    analysis_manifest = "c" * 64
    cluster_ids = [
        *[f"finite-{index:04d}" for index in range(1, 59)],
        *[f"nonfinite-{index:04d}" for index in range(1, 7)],
    ]
    clusters = [
        {
            "cluster_id": cluster_id,
            "cluster_type": (
                "FINITE_DRIFT" if cluster_id.startswith("finite-") else "NONFINITE_OBSERVED"
            ),
            "start_probe_order": index,
            "target_tensor_basis": "direct_compute_output",
            "structural_families": ["QUANTIZED_COMPUTE"],
            "causal_interpretation": "NOT_ESTABLISHED",
        }
        for index, cluster_id in enumerate(cluster_ids, start=1)
    ]
    classifications = [
        *(["BITWISE_EQUAL"] * 52),
        *(["FINITE_BITWISE_DRIFT"] * 219),
        *(["NONFINITE_OBSERVED"] * 12),
    ]
    memberships = []
    finite_index = 0
    nonfinite_index = 0
    for order, classification in enumerate(classifications, start=1):
        if classification == "BITWISE_EQUAL":
            cluster_id = None
        elif classification == "FINITE_BITWISE_DRIFT":
            cluster_id = f"finite-{finite_index % 58 + 1:04d}"
            finite_index += 1
        else:
            cluster_id = f"nonfinite-{nonfinite_index % 6 + 1:04d}"
            nonfinite_index += 1
        memberships.append(
            {
                "probe_id": f"probe-{order:04d}",
                "probe_order": order,
                "classification": classification,
                "cluster_id": cluster_id,
            }
        )
    plan = {
        "kind": "m1-diagnostic-structural-cluster-plan",
        "status": "READY",
        "source_analysis_manifest_sha256": analysis_manifest,
        "probe_count": 283,
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
    membership = {
        "kind": "m1-diagnostic-probe-cluster-membership",
        "status": "COMPLETE",
        "probe_count": 283,
        "cluster_count": 64,
        "probes": memberships,
    }
    report = {
        "kind": "m1-diagnostic-structural-cluster-report",
        "status": "COMPLETE",
        "diagnostic_status": "DESCRIPTIVE_ONLY",
        "source_analysis_manifest_sha256": analysis_manifest,
        "probe_count": 283,
        "cluster_count": 64,
        "summary": {
            "bitwise_equal_probe_count": 52,
            "finite_drift_probe_count": 219,
            "nonfinite_probe_count": 12,
            "finite_cluster_count": 58,
            "nonfinite_cluster_count": 6,
            "nonfinite_ranked_with_finite": False,
        },
        "clusters": clusters,
        "numerical_threshold_selected": False,
        "scientific_decision_recomputed": False,
        "frozen_transition_b_v1_scientific_decision": "FAIL",
        "model_execution_used": False,
        "onnx_graph_loaded": False,
        "activation_artifact_loaded": False,
        "causal_claim_made": False,
    }
    replay = {
        "cluster_plan_path": "cluster-plan.json",
        "probe_cluster_membership_path": "probe-cluster-membership.json",
        "structural_cluster_report_path": "structural-cluster-report.json",
        "expected_status": "COMPLETE",
        "replay_requires_model_execution": replay_requires_model_execution,
    }
    documents = {
        "cluster-plan.json": plan,
        "probe-cluster-membership.json": membership,
        "structural-cluster-report.json": report,
        "replay-bundle.json": replay,
    }
    for name, payload in documents.items():
        _write_json(root / name, payload)
    artifacts = [
        {
            "path": name,
            "sha256": sha256_file(root / name),
            "size_bytes": (root / name).stat().st_size,
        }
        for name in CLUSTER_ARTIFACTS
    ]
    manifest = {
        "kind": "m1-diagnostic-structural-cluster-manifest",
        "status": "COMPLETE",
        "artifact_count": 4,
        "artifacts": artifacts,
        "tamper_evident": True,
        "model_execution_used": False,
        "replay_requires_model_execution": False,
    }
    _write_json(root / "artifact-manifest.json", manifest)
    return root / "replay-bundle.json", sha256_file(root / "artifact-manifest.json")


def test_causal_plan_authority_accepts_frozen_cluster_package(
    tmp_path: Path,
) -> None:
    bundle, manifest_sha256 = _source_package(tmp_path)

    authority = verify_causal_plan_input(bundle, manifest_sha256)

    assert authority.cluster_report["cluster_count"] == 64
    assert authority.membership["probe_count"] == 283


def test_causal_plan_authority_rejects_model_requiring_source(
    tmp_path: Path,
) -> None:
    bundle, manifest_sha256 = _source_package(
        tmp_path,
        replay_requires_model_execution=True,
    )

    with pytest.raises(CausalPlanError) as error:
        verify_causal_plan_input(bundle, manifest_sha256)
    assert error.value.code == "CAUSAL_PLAN_SOURCE_REPLAY_INVALID"


def test_causal_plan_authority_rejects_undeclared_bundle(
    tmp_path: Path,
) -> None:
    bundle, manifest_sha256 = _source_package(tmp_path)
    alternate = tmp_path / "alternate-replay-bundle.json"
    alternate.write_bytes(bundle.read_bytes())

    with pytest.raises(CausalPlanError) as error:
        verify_causal_plan_input(alternate, manifest_sha256)
    assert error.value.code == "CAUSAL_PLAN_SOURCE_REPLAY_INVALID"
