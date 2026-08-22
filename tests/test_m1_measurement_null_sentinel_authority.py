from __future__ import annotations

import copy

import pytest

from neural_continuity.m1_diagnostics.measurement_null_sentinel_authority import (
    EPOCH_LAYOUT,
    PROCESS_EPOCH_COUNT,
    SENTINEL_DOCUMENT_COUNT,
    SENTINEL_SELECTION_DOMAIN,
    _select_sentinel_ids,
    _validate_extension_plan,
)
from neural_continuity.m1_teacher_evidence import TeacherEvidenceError


def _valid_extension_plan() -> dict[str, object]:
    return {
        "kind": "m1-measurement-null-extension-plan",
        "status": "PREREGISTERED_NOT_EXECUTED",
        "execution_started": False,
        "model_execution_used_for_preregistration": False,
        "candidate_or_holdout_result_selected_design": False,
        "scope": {
            "query_role": "measurement_null",
            "execution_provider": "CPUExecutionProvider",
            "candidate_or_int8_execution_allowed": False,
            "holdout_query_access_allowed": False,
            "stage_1_execution_allowed": False,
            "existing_evidence_mutation_allowed": False,
            "operational_tolerance_change_allowed": False,
        },
        "frozen_design": {
            "process_epoch_count": PROCESS_EPOCH_COUNT,
            "passes_per_epoch": 4,
            "passes_per_phase": 480,
            "total_planned_passes": 960,
            "batch_sizes": [1, 16, 64],
            "repeat_batch_size": 16,
            "early_stopping_allowed": False,
            "adaptive_sample_size_allowed": False,
            "independent_process_required_per_epoch": True,
        },
        "epoch_layout": [
            {"run": run_id, "batch_size": batch_size} for run_id, batch_size in EPOCH_LAYOUT
        ],
        "phases": [
            {
                "phase_id": "tensor_sentinel_preflight",
                "process_epoch_count": PROCESS_EPOCH_COUNT,
                "qualifying_detection_evidence": False,
                "documents": {
                    "count": SENTINEL_DOCUMENT_COUNT,
                    "selection": "first IDs after domain-separated SHA-256 ordering",
                    "selection_domain": SENTINEL_SELECTION_DOMAIN,
                    "text_or_qrel_dependent_selection": False,
                },
            },
            {
                "phase_id": "full_corpus_qualification",
                "process_epoch_count": PROCESS_EPOCH_COUNT,
                "qualifying_detection_evidence": True,
            },
        ],
    }


def test_extension_plan_accepts_only_frozen_sentinel_scope() -> None:
    _validate_extension_plan(_valid_extension_plan())


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("scope", "execution_provider", "CUDAExecutionProvider"),
        ("scope", "holdout_query_access_allowed", True),
        ("frozen_design", "process_epoch_count", 119),
        ("frozen_design", "early_stopping_allowed", True),
    ],
)
def test_extension_plan_fails_closed_for_scope_drift(
    section: str,
    field: str,
    value: object,
) -> None:
    plan = copy.deepcopy(_valid_extension_plan())
    nested = plan[section]
    assert isinstance(nested, dict)
    nested[field] = value

    with pytest.raises(TeacherEvidenceError) as error:
        _validate_extension_plan(plan)

    assert error.value.status == "BLOCKED"


def test_sentinel_selection_depends_only_on_document_identity() -> None:
    document_ids = tuple(f"doc-{index:04d}" for index in range(300))

    first = _select_sentinel_ids(document_ids)
    second = _select_sentinel_ids(tuple(reversed(document_ids)))

    assert first == second
    assert len(first) == SENTINEL_DOCUMENT_COUNT
    assert len(set(first)) == SENTINEL_DOCUMENT_COUNT
