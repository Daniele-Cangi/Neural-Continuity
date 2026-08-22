from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics import causal_plan_evidence
from neural_continuity.m1_diagnostics.causal_plan_authority import (
    CausalPlanError,
    VerifiedCausalPlanInput,
)
from neural_continuity.m1_diagnostics.causal_plan_evidence import (
    create_causal_plan_package,
    replay_causal_plan_package,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _authority(root: Path) -> VerifiedCausalPlanInput:
    plan = {
        "kind": "m1-diagnostic-structural-cluster-plan",
        "probe_count": 3,
        "integer_probe_count": 2,
    }
    membership = {
        "kind": "m1-diagnostic-probe-cluster-membership",
        "probe_count": 3,
    }
    report = {
        "kind": "m1-diagnostic-structural-cluster-report",
        "probe_count": 3,
        "cluster_count": 2,
        "clusters": [
            {
                "cluster_id": "finite-0001",
                "cluster_type": "FINITE_DRIFT",
                "start_probe_order": 1,
                "target_tensor_basis": "direct_compute_output",
                "structural_families": ["A", "B"],
                "causal_interpretation": "NOT_ESTABLISHED",
            },
            {
                "cluster_id": "nonfinite-0001",
                "cluster_type": "NONFINITE_OBSERVED",
                "start_probe_order": 2,
                "target_tensor_basis": "direct_compute_output",
                "structural_families": ["B"],
                "causal_interpretation": "NOT_ESTABLISHED",
            },
        ],
    }
    documents = {
        "cluster-plan.json": plan,
        "probe-cluster-membership.json": membership,
        "structural-cluster-report.json": report,
    }
    for name, payload in documents.items():
        _write_json(root / name, payload)
    bundle = root / "replay-bundle.json"
    _write_json(bundle, {"replay_requires_model_execution": False})
    hashes = {name: sha256_file(root / name) for name in documents}
    hashes["replay-bundle.json"] = sha256_file(bundle)
    return VerifiedCausalPlanInput(
        root=root,
        bundle_path=bundle,
        manifest_sha256="a" * 64,
        cluster_plan=plan,
        membership=membership,
        cluster_report=report,
        artifact_sha256=hashes,
    )


def test_causal_plan_package_replays_without_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    authority = _authority(source)
    monkeypatch.setattr(
        causal_plan_evidence,
        "verify_causal_plan_input",
        lambda *_args, **_kwargs: authority,
    )
    output = tmp_path / "causal-plan"
    result = create_causal_plan_package(
        authority.bundle_path,
        authority.manifest_sha256,
        output,
    )
    replay = replay_causal_plan_package(
        output / "replay-bundle.json",
        result["artifact_manifest_sha256"],
    )

    assert replay["replay_verified"] is True
    assert replay["plan_match"] is True
    assert replay["hypotheses_match"] is True
    assert replay["intervention_matrix_match"] is True
    assert replay["intervention_execution_authorized"] is False
    assert replay["model_execution_used"] is False
    assert replay["activation_artifact_loaded"] is False


def test_causal_plan_replay_rejects_tampered_hypotheses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    authority = _authority(source)
    monkeypatch.setattr(
        causal_plan_evidence,
        "verify_causal_plan_input",
        lambda *_args, **_kwargs: authority,
    )
    output = tmp_path / "causal-plan"
    result = create_causal_plan_package(
        authority.bundle_path,
        authority.manifest_sha256,
        output,
    )
    hypotheses = output / "cluster-hypotheses.json"
    hypotheses.write_text(
        hypotheses.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )

    with pytest.raises(CausalPlanError) as error:
        replay_causal_plan_package(
            output / "replay-bundle.json",
            result["artifact_manifest_sha256"],
        )
    assert error.value.code == "CAUSAL_PLAN_REPLAY_VERIFICATION_FAILED"


def test_causal_plan_declares_no_threshold_or_candidate_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    authority = _authority(source)
    monkeypatch.setattr(
        causal_plan_evidence,
        "verify_causal_plan_input",
        lambda *_args, **_kwargs: authority,
    )
    output = tmp_path / "causal-plan"
    create_causal_plan_package(
        authority.bundle_path,
        authority.manifest_sha256,
        output,
    )
    plan = json.loads((output / "causal-plan.json").read_text(encoding="utf-8"))

    assert plan["numerical_threshold_selected"] is False
    assert plan["frozen_int8_candidate_mutated"] is False
    assert plan["derived_diagnostic_candidate_created"] is False
    assert plan["intervention_execution_authorized"] is False
    assert plan["causal_claim_made"] is False
