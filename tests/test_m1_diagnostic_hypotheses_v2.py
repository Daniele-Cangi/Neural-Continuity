from __future__ import annotations

from neural_continuity.m1_diagnostics.graph_predecessors import (
    ProbePredecessor,
    localize_dominant_onset,
)
from neural_continuity.m1_diagnostics.hypothesis_report import evaluate_hypotheses_v2


def test_hypotheses_apply_family_ties_and_saturation_intersection() -> None:
    localization = localize_dominant_onset(
        probe_ids=("p1", "p2"),
        predecessor_edges=(
            ProbePredecessor("p1", "input-0", synthetic_predecessor_id="graph-input:p1:input-0"),
            ProbePredecessor("p2", "input-0", "p1"),
        ),
        probe_scores_hex={
            "p1": "0x1.0000000000000p-2",
            "p2": "0x1.0000000000000p-1",
        },
    )
    report = evaluate_hypotheses_v2(
        localization=localization,
        boundary_families={
            "p1": ("ATTENTION_OR_MATMUL",),
            "p2": ("ATTENTION_OR_MATMUL", "OUTPUT_PATH"),
        },
        saturation_boundary_ids=("p1", "p2"),
        saturation={
            "p1": {
                "measurable": True,
                "aggregate_count": 1,
                "aggregate_total": 4,
                "aggregate_fraction": "0x1.0000000000000p-2",
            },
            "p2": {
                "measurable": True,
                "aggregate_count": 2,
                "aggregate_total": 4,
                "aggregate_fraction": "0x1.0000000000000p-1",
            },
        },
    )

    assert report["hypotheses"]["H1_NORMALIZATION_SENSITIVITY"]["status"] == "NOT_SUPPORTED"
    assert report["hypotheses"]["H2_ATTENTION_OR_MATMUL_SENSITIVITY"]["status"] == "SUPPORTED"
    assert report["hypotheses"]["H3_CALIBRATION_RANGE_MISMATCH"]["status"] == "SUPPORTED"
    assert report["hypotheses"]["H4_OUTPUT_PATH_QUANTIZATION"]["status"] == "UNRESOLVED"
    assert report["transition_decision_emitted"] is False


def test_zero_growth_keeps_every_hypothesis_unresolved() -> None:
    localization = localize_dominant_onset(
        probe_ids=("p1",),
        predecessor_edges=(
            ProbePredecessor("p1", "input-0", synthetic_predecessor_id="graph-input:p1:input-0"),
        ),
        probe_scores_hex={"p1": "0x0.0p+0"},
    )
    report = evaluate_hypotheses_v2(
        localization=localization,
        boundary_families={"p1": ()},
        saturation_boundary_ids=("p1",),
        saturation={
            "p1": {
                "measurable": True,
                "aggregate_count": 0,
                "aggregate_total": 4,
                "aggregate_fraction": "0x0.0p+0",
            }
        },
    )

    assert {entry["status"] for entry in report["hypotheses"].values()} == {"UNRESOLVED"}
