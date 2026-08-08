from __future__ import annotations

import json
from pathlib import Path

CONTRACT_PATH = Path("contracts/m1-transition-a-v1.json")


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_transition_a_contract_is_the_single_declared_authority():
    contract = _contract()

    assert contract["contract_id"] == "m1-transition-a-v1"
    assert contract["contract_authority"] == "contracts/m1-transition-a-v1.json"
    assert contract["transition"] == {
        "id": "A",
        "source": "PyTorch FP32 real teacher",
        "target": "ONNX FP32",
        "purpose": "evidence-bounded continuity decision, not universal equivalence certification",
    }
    assert contract["decision_states"] == {
        "scientific": ["PASS", "FAIL", "INCONCLUSIVE"],
        "technical": ["BLOCKED", "EXECUTION_ERROR"],
        "scientific_fail_rule": (
            "FAIL is permitted only when complete, valid, replayable source and target evidence "
            "demonstrates an operational-tolerance violation."
        ),
    }


def test_transition_a_contract_binds_the_replayed_source_null_evidence():
    contract = _contract()

    measurement_null = contract["preconditions"]["measurement_null"]
    assert measurement_null["evidence_manifest_sha256"] == (
        "f3250f96577c1594b356c89252e3482e914f8b891d1bedbe84040206743a2f3d"
    )
    assert measurement_null["status"] == "CAPTURED_NOT_DECIDED"
    assert measurement_null["replay_required"] is True
    assert contract["detection_limits"]["batch_size_variation"]["document_max_abs_delta"] == (
        1.043081283569336e-07
    )
    assert contract["detection_limits"]["batch_size_variation"]["ranking_change_count"] == 0
    assert contract["detection_limits"]["repeated_inference"]["comparison_count"] == 2


def test_transition_a_tolerances_are_pre_candidate_and_transition_b_is_gated():
    contract = _contract()

    selection = contract["operational_tolerances"]["selection_basis"]
    assert selection["method"] == "pre-candidate governance declaration"
    assert selection["not_derived_directly_from"] == "measurement null envelope"
    assert "final_holdout results" in selection["prohibited_inputs"]
    assert (
        contract["operational_tolerances"]["topology_frozen_critical"]["ranking_change_count_lte"]
        == 0
    )
    assert contract["preconditions"]["transition_b"]["may_begin_only_when_transition_a"] == "PASS"
    assert contract["replay_policy"]["missing_declared_control_or_observation_fails_closed"] is True
