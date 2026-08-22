from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.structural_cluster_authority import (
    EXPECTED_ANALYSIS_ARTIFACTS,
    StructuralClusterError,
    verify_structural_cluster_input,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _source_package(
    root: Path,
    *,
    replay_requires_model_execution: bool = False,
) -> tuple[Path, str]:
    activation_manifest = "b" * 64
    plan = {
        "kind": "m1-diagnostic-activation-analysis-plan",
        "status": "READY",
        "query_count": 364,
        "probe_count": 283,
        "integer_probe_count": 248,
        "source_activation_manifest_sha256": activation_manifest,
        "numerical_threshold_selected": False,
        "scientific_decision_recomputed": False,
        "frozen_transition_b_v1_scientific_decision": "FAIL",
        "model_execution_used": False,
        "onnx_graph_loaded": False,
    }
    probes = [
        {
            "probe_id": f"probe-{order:04d}",
            "probe_order": order,
            "target_tensor_basis": "direct_compute_output",
            "structural_families": ["DIRECT_COMPUTE"],
            "floating": {
                "classification": "BITWISE_EQUAL",
                "differing_value_count": 0,
                "bitwise_difference_rate": 0.0,
                "relative_l2_error": 0.0,
            },
            "integer_dtype_extremes": None,
        }
        for order in range(1, 284)
    ]
    diagnostics = {
        "kind": "m1-diagnostic-activation-probe-diagnostics",
        "status": "COMPLETE",
        "probe_count": 283,
        "probes": probes,
    }
    report = {
        "kind": "m1-diagnostic-activation-report",
        "status": "COMPLETE",
        "diagnostic_status": "DESCRIPTIVE_ONLY",
        "source_activation_manifest_sha256": activation_manifest,
        "batch_count": 23,
        "query_count": 364,
        "probe_count": 283,
        "integer_probe_count": 248,
        "scientific_decision_recomputed": False,
        "frozen_transition_b_v1_scientific_decision": "FAIL",
        "model_execution_used": False,
        "causal_claim_made": False,
    }
    replay = {
        "analysis_plan_path": "analysis-plan.json",
        "probe_diagnostics_path": "probe-diagnostics.json",
        "diagnostic_report_path": "diagnostic-report.json",
        "expected_status": "COMPLETE",
        "replay_requires_model_execution": replay_requires_model_execution,
    }
    documents = {
        "analysis-plan.json": plan,
        "probe-diagnostics.json": diagnostics,
        "diagnostic-report.json": report,
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
        for name in sorted(EXPECTED_ANALYSIS_ARTIFACTS)
    ]
    manifest = {
        "kind": "m1-diagnostic-activation-analysis-manifest",
        "status": "COMPLETE",
        "artifact_count": 4,
        "artifacts": artifacts,
        "tamper_evident": True,
        "model_execution_used": False,
        "replay_requires_model_execution": False,
    }
    _write_json(root / "artifact-manifest.json", manifest)
    return root / "replay-bundle.json", sha256_file(root / "artifact-manifest.json")


def test_structural_cluster_authority_accepts_complete_model_free_analysis(
    tmp_path: Path,
) -> None:
    bundle, manifest_sha256 = _source_package(tmp_path)

    authority = verify_structural_cluster_input(bundle, manifest_sha256)

    assert authority.manifest_sha256 == manifest_sha256
    assert authority.probe_diagnostics["probe_count"] == 283
    assert set(authority.artifact_sha256) == set(EXPECTED_ANALYSIS_ARTIFACTS)


def test_structural_cluster_authority_rejects_model_requiring_replay(
    tmp_path: Path,
) -> None:
    bundle, manifest_sha256 = _source_package(
        tmp_path,
        replay_requires_model_execution=True,
    )

    with pytest.raises(StructuralClusterError) as error:
        verify_structural_cluster_input(bundle, manifest_sha256)
    assert error.value.code == "STRUCTURAL_CLUSTER_SOURCE_REPLAY_INVALID"


def test_structural_cluster_authority_rejects_undeclared_replay_bundle(
    tmp_path: Path,
) -> None:
    bundle, manifest_sha256 = _source_package(tmp_path)
    alternate = tmp_path / "alternate-replay-bundle.json"
    alternate.write_bytes(bundle.read_bytes())

    with pytest.raises(StructuralClusterError) as error:
        verify_structural_cluster_input(alternate, manifest_sha256)
    assert error.value.code == "STRUCTURAL_CLUSTER_SOURCE_REPLAY_INVALID"


def test_structural_cluster_authority_rejects_tampered_analysis_artifact(
    tmp_path: Path,
) -> None:
    bundle, manifest_sha256 = _source_package(tmp_path)
    diagnostics = tmp_path / "probe-diagnostics.json"
    diagnostics.write_text(
        diagnostics.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )

    with pytest.raises(StructuralClusterError) as error:
        verify_structural_cluster_input(bundle, manifest_sha256)
    assert error.value.code == "STRUCTURAL_CLUSTER_ARTIFACT_HASH_MISMATCH"
