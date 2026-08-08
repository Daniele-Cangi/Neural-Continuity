from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from neural_continuity.m1_onnx_transition import _decision_for_comparison
from neural_continuity.m1_teacher_evidence import _fail


def decide_transition_b(
    comparison_report: Mapping[str, Any],
    transition_b_contract: Mapping[str, Any],
    transition_a_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if comparison_report.get("comparison_state") != "CAPTURED_NOT_DECIDED":
        raise _fail("TRANSITION_B_COMPARISON_INVALID", "comparison is not complete and undecided")
    runs = comparison_report.get("runs")
    if not isinstance(runs, list) or len(runs) != 3:
        raise _fail("MISSING_DECLARED_SOURCE_OBSERVATION", "three paired runs are required")
    required_batches = transition_b_contract["preconditions"]["target_capture"][
        "required_batch_sizes"
    ]
    if [run.get("batch_size") for run in runs] != required_batches:
        raise _fail(
            "MISSING_DECLARED_SOURCE_OBSERVATION",
            "paired run batches differ from contract",
        )
    run_decisions = [_decision_for_comparison(run, transition_a_contract) for run in runs]
    statuses = [decision["status"] for decision in run_decisions]
    status = "FAIL" if "FAIL" in statuses else "PASS"
    return {
        "transition_id": "B",
        "measurement_integrity_state": "VALID",
        "transition_b_status": status,
        "run_decisions": run_decisions,
        "scientific_claim": "evidence-bounded continuity decision; not universal equivalence",
    }
