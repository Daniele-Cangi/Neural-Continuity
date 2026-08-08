from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_continuity.m1_b.decision_policy import decide_transition_b
from neural_continuity.m1_teacher_evidence import ROLE_ORDER, TeacherEvidenceError


def _contract(name: str) -> dict[str, object]:
    return json.loads(Path(f"contracts/{name}.json").read_text(encoding="utf-8"))


def _comparison(batch_sizes: list[int]) -> dict[str, object]:
    roles = {
        role: {
            "query_count": 1,
            "ranking_change_count": 0,
            "ranking_change_fraction": 0.0,
            "metric_decrease": {"recall_at_k": 0.0, "mrr_at_k": 0.0, "ndcg_at_k": 0.0},
        }
        for role in ROLE_ORDER
    }
    return {
        "comparison_state": "CAPTURED_NOT_DECIDED",
        "runs": [
            {
                "run_id": f"batch-size-{batch_size:04d}",
                "batch_size": batch_size,
                "functional": {
                    "document_max_abs_delta": 0.1,
                    "query_max_abs_delta": 0.1,
                    "document_min_cosine_similarity": 0.9,
                    "query_min_cosine_similarity": 0.9,
                },
                "roles": roles,
            }
            for batch_size in batch_sizes
        ],
    }


def test_transition_b_decision_is_scientific_fail_for_frozen_tolerance_violation() -> None:
    decision = decide_transition_b(
        _comparison([1, 16, 64]),
        _contract("m1-transition-b-v1"),
        _contract("m1-transition-a-v1"),
    )

    assert decision["transition_b_status"] == "FAIL"
    assert all(run["status"] == "FAIL" for run in decision["run_decisions"])


def test_transition_b_decision_blocks_wrong_execution_batches() -> None:
    with pytest.raises(TeacherEvidenceError) as error:
        decide_transition_b(
            _comparison([1, 8, 64]),
            _contract("m1-transition-b-v1"),
            _contract("m1-transition-a-v1"),
        )

    assert error.value.status == "BLOCKED"
    assert error.value.code == "MISSING_DECLARED_SOURCE_OBSERVATION"
