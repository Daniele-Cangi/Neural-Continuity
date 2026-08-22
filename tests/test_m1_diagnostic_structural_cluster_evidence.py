from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics import (
    structural_cluster_evidence,
)
from neural_continuity.m1_diagnostics.structural_cluster_authority import (
    StructuralClusterError,
    VerifiedStructuralClusterInput,
)
from neural_continuity.m1_diagnostics.structural_cluster_evidence import (
    create_structural_cluster_package,
    replay_structural_cluster_package,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _probe(order: int, classification: str) -> dict[str, Any]:
    differing = 0 if classification == "BITWISE_EQUAL" else 1
    return {
        "probe_id": f"probe-{order:04d}",
        "probe_order": order,
        "target_tensor_basis": "direct_compute_output",
        "structural_families": ["QUANTIZED_COMPUTE"],
        "floating": {
            "classification": classification,
            "differing_value_count": differing,
            "bitwise_difference_rate": float(differing),
            "relative_l2_error": 0.0 if not differing else 0.25,
        },
        "integer_dtype_extremes": {
            "classification": "NO_DTYPE_EXTREME_VALUES",
            "dtype_extreme_rate": 0.0,
        },
    }


def _authority(root: Path) -> VerifiedStructuralClusterInput:
    plan = {
        "kind": "m1-diagnostic-activation-analysis-plan",
        "status": "READY",
        "probe_count": 3,
        "integer_probe_count": 3,
    }
    diagnostics = {
        "kind": "m1-diagnostic-activation-probe-diagnostics",
        "status": "COMPLETE",
        "probe_count": 3,
        "probes": [
            _probe(1, "BITWISE_EQUAL"),
            _probe(2, "FINITE_BITWISE_DRIFT"),
            _probe(3, "BITWISE_EQUAL"),
        ],
    }
    report = {
        "kind": "m1-diagnostic-activation-report",
        "status": "COMPLETE",
        "diagnostic_status": "DESCRIPTIVE_ONLY",
    }
    documents = {
        "analysis-plan.json": plan,
        "probe-diagnostics.json": diagnostics,
        "diagnostic-report.json": report,
    }
    for name, payload in documents.items():
        _write_json(root / name, payload)
    bundle = root / "replay-bundle.json"
    _write_json(bundle, {"replay_requires_model_execution": False})
    hashes = {name: sha256_file(root / name) for name in documents}
    hashes["replay-bundle.json"] = sha256_file(bundle)
    return VerifiedStructuralClusterInput(
        root=root,
        bundle_path=bundle,
        manifest_sha256="a" * 64,
        analysis_plan=plan,
        probe_diagnostics=diagnostics,
        diagnostic_report=report,
        artifact_sha256=hashes,
    )


def test_structural_cluster_package_replays_without_model_or_activations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    authority = _authority(source)
    monkeypatch.setattr(
        structural_cluster_evidence,
        "verify_structural_cluster_input",
        lambda *_args, **_kwargs: authority,
    )
    output = tmp_path / "clusters"
    result = create_structural_cluster_package(
        authority.bundle_path,
        authority.manifest_sha256,
        output,
    )
    replay = replay_structural_cluster_package(
        output / "replay-bundle.json",
        result["artifact_manifest_sha256"],
    )

    assert replay["replay_verified"] is True
    assert replay["plan_match"] is True
    assert replay["probe_cluster_membership_match"] is True
    assert replay["structural_cluster_report_match"] is True
    assert replay["model_execution_used"] is False
    assert replay["activation_artifact_loaded"] is False


def test_structural_cluster_replay_rejects_tampered_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    authority = _authority(source)
    monkeypatch.setattr(
        structural_cluster_evidence,
        "verify_structural_cluster_input",
        lambda *_args, **_kwargs: authority,
    )
    output = tmp_path / "clusters"
    result = create_structural_cluster_package(
        authority.bundle_path,
        authority.manifest_sha256,
        output,
    )
    report = output / "structural-cluster-report.json"
    report.write_text(report.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(StructuralClusterError) as error:
        replay_structural_cluster_package(
            output / "replay-bundle.json",
            result["artifact_manifest_sha256"],
        )
    assert error.value.code == "STRUCTURAL_CLUSTER_ARTIFACT_HASH_MISMATCH"


def test_structural_cluster_cli_scope_is_declared_model_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    authority = _authority(source)
    monkeypatch.setattr(
        structural_cluster_evidence,
        "verify_structural_cluster_input",
        lambda *_args, **_kwargs: authority,
    )
    output = tmp_path / "clusters"
    create_structural_cluster_package(
        authority.bundle_path,
        authority.manifest_sha256,
        output,
    )
    plan = json.loads((output / "cluster-plan.json").read_text(encoding="utf-8"))

    assert plan["numerical_threshold_selected"] is False
    assert plan["candidate_specific_exception_used"] is False
    assert plan["onnx_graph_loaded"] is False
    assert plan["activation_artifact_loaded"] is False
