from __future__ import annotations

import json
from pathlib import Path

CONTRACT_PATH = Path("contracts/m1-transition-b-v1.json")


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_transition_b_contract_requires_verified_transition_a_and_onnx_null():
    contract = _contract()

    assert contract["contract_id"] == "m1-transition-b-v1"
    assert contract["contract_authority"] == "contracts/m1-transition-b-v1.json"
    assert contract["transition"]["id"] == "B"
    assert contract["preconditions"]["transition_a"]["required_status"] == "PASS"
    assert contract["preconditions"]["onnx_fp32_measurement_null"] == {
        "required_before_candidate": True,
        "required_source": "verified Transition A ONNX FP32 artifact",
        "required_families": ["repeated_inference", "batch_size_variation"],
        "required_batch_sizes": [1, 16, 64],
        "minimum_repeated_inference_count": 3,
        "evidence_status": "NOT_YET_CAPTURED",
    }


def test_transition_b_contract_freezes_static_calibration_and_prevents_leakage():
    contract = _contract()

    calibration = contract["calibration"]
    assert calibration["mode"] == "static"
    assert calibration["quantization_format"] == "QDQ"
    assert calibration["activation_type"] == "QUInt8"
    assert calibration["weight_type"] == "QInt8"
    assert calibration["data_role"] == "quantization_calibration"
    assert calibration["max_query_count"] == 162
    assert calibration["leakage_behavior"] == "BLOCKED"
    assert set(calibration["prohibited_data_roles"]) == {
        "measurement_null",
        "contract_development",
        "validation",
        "frozen_critical",
        "final_holdout",
    }


def test_transition_b_contract_declares_non_monolithic_responsibilities():
    contract = _contract()

    boundaries = contract["implementation_boundaries"]
    assert set(boundaries) == {
        "calibration_data",
        "quantization",
        "onnx_observation",
        "comparison",
        "decision",
        "replay",
        "orchestration",
    }
    assert contract["decision_states"]["scientific"] == ["PASS", "FAIL", "INCONCLUSIVE"]
    assert contract["decision_states"]["technical"] == ["BLOCKED", "EXECUTION_ERROR"]
