from __future__ import annotations

from typing import Any

import pytest

from neural_continuity.m1_diagnostics.causal_plan_authority import (
    CausalPlanError,
)
from neural_continuity.m1_diagnostics.causal_plan_design import (
    build_causal_design,
)


def _cluster(
    cluster_id: str,
    onset: int,
    cluster_type: str,
    families: list[str],
) -> dict[str, Any]:
    return {
        "cluster_id": cluster_id,
        "cluster_type": cluster_type,
        "start_probe_order": onset,
        "target_tensor_basis": "post_quantize_dequantize_output",
        "structural_families": families,
        "causal_interpretation": "NOT_ESTABLISHED",
    }


def _report() -> dict[str, Any]:
    return {
        "clusters": [
            _cluster("finite-0001", 1, "FINITE_DRIFT", ["A", "B"]),
            _cluster("finite-0002", 8, "FINITE_DRIFT", ["B"]),
            _cluster(
                "nonfinite-0001",
                15,
                "NONFINITE_OBSERVED",
                ["B", "C"],
            ),
        ]
    }


def test_causal_design_includes_every_observed_family_without_cutoff() -> None:
    design = build_causal_design(_report())

    assert design["observed_structural_families"] == ["A", "B", "C"]
    assert design["summary"]["family_hypothesis_count"] == 3
    assert design["summary"]["all_clusters_included"] is True
    assert design["summary"]["cluster_selection_cutoff_used"] is False


def test_causal_design_separates_finite_and_nonfinite_exposure() -> None:
    design = build_causal_design(_report())
    hypotheses = {item["structural_family"]: item for item in design["family_hypotheses"]}

    assert hypotheses["B"]["finite_cluster_ids"] == [
        "finite-0001",
        "finite-0002",
    ]
    assert hypotheses["B"]["nonfinite_cluster_ids"] == ["nonfinite-0001"]
    assert design["summary"]["nonfinite_ranked_with_finite"] is False


def test_causal_design_preregisters_only_observed_pairwise_interactions() -> None:
    design = build_causal_design(_report())

    pairs = [tuple(item["structural_families"]) for item in design["interaction_hypotheses"]]
    assert pairs == [("A", "B"), ("B", "C")]
    assert all(
        item["execution_status"] == "DEFERRED_NOT_AUTHORIZED"
        for item in design["pair_interventions"]
    )


def test_causal_design_keeps_controls_and_execution_states_separate() -> None:
    design = build_causal_design(_report())

    assert [item["control_type"] for item in design["controls"]] == [
        "FROZEN_INT8_EXACT_REPLAY",
        "VERIFIED_ONNX_FP32_REFERENCE",
    ]
    assert all(
        item["execution_status"] == "NOT_AUTHORIZED"
        for item in design["single_family_interventions"]
    )


def test_causal_design_rejects_preexisting_causal_claim() -> None:
    report = _report()
    report["clusters"][0]["causal_interpretation"] = "ESTABLISHED"

    with pytest.raises(CausalPlanError) as error:
        build_causal_design(report)
    assert error.value.code == "CAUSAL_PLAN_CLUSTER_INVALID"
