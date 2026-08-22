from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from neural_continuity.evidence import canonical_json_bytes
from neural_continuity.m1_diagnostics import stage0_evidence
from neural_continuity.m1_diagnostics.stage0_authority import (
    Stage0Authority,
    Stage0ControlError,
)
from neural_continuity.m1_diagnostics.stage0_evidence import (
    create_stage0_control_package,
    replay_stage0_control_package,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _authority(root: Path) -> Stage0Authority:
    runtime = root / "runtime"
    runtime.mkdir()
    (runtime / "authority-manifest.json").write_text("{}")
    placeholder = root / "placeholder.json"
    placeholder.write_text("{}")
    return Stage0Authority(
        causal_plan_bundle=placeholder,
        causal_plan_manifest_sha256="a" * 64,
        archive_manifest_path=placeholder,
        archive_manifest_sha256="b" * 64,
        runtime_root=runtime,
        runtime_manifest_sha256="c" * 64,
        contract_path=placeholder,
        fp32_config_path=placeholder,
        int8_config_path=placeholder,
        dataset_root=root,
        transition_a_bundle=placeholder,
        onnx_null_bundle=placeholder,
        candidate_root=root,
        baseline_int8_bundle=placeholder,
        baseline_fp32_bundle=placeholder,
        transition_b_bundle=placeholder,
        onnx_null_report={},
        package_authority_sha256={"dataset": "d" * 64},
    )


def _capture(
    _authority: Stage0Authority, build_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    for directory, prefix in (
        (build_root / "fp32-control", "source"),
        (build_root / "int8-control", "target"),
    ):
        directory.mkdir()
        _write_json(
            directory / "replay-bundle.json",
            {"replay_requires_model_execution": False},
        )
        _write_json(
            directory / "evidence-manifest.json",
            {"package_kind": prefix, "artifacts": []},
        )
    return {"status": "PASS"}, {"status": "PASS"}


def _report() -> dict[str, Any]:
    return {
        "kind": "m1-diagnostic-stage0-control-report",
        "status": "PASS",
        "controls": {
            "verified_onnx_fp32_reference": {"outcome": "PASS"},
            "frozen_int8_exact_replay": {"outcome": "PASS"},
        },
        "stage_1_gate_status": "PASS",
        "stage_1_execution_started": False,
        "frozen_transition_b_v1_scientific_decision": "FAIL",
        "scientific_decision_recomputed": False,
        "causal_claim_made": False,
    }


def test_stage0_package_replays_without_model_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority(tmp_path)
    monkeypatch.setattr(
        stage0_evidence,
        "verify_stage0_authority",
        lambda *_args, **_kwargs: authority,
    )
    monkeypatch.setattr(stage0_evidence, "capture_stage0_observations", _capture)
    monkeypatch.setattr(
        stage0_evidence,
        "build_stage0_control_report",
        lambda *_args, **_kwargs: _report(),
    )
    output = tmp_path / "output"
    result = create_stage0_control_package(
        "causal",
        "a" * 64,
        "archive",
        "b" * 64,
        "runtime",
        "c" * 64,
        output,
    )
    replay = replay_stage0_control_package(
        output / "replay-bundle.json",
        result["artifact_manifest_sha256"],
    )

    assert replay["replay_verified"] is True
    assert replay["plan_match"] is True
    assert replay["control_report_match"] is True
    assert replay["status_match"] is True
    assert replay["model_execution_used"] is False


def test_stage0_replay_rejects_tampered_control_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority(tmp_path)
    monkeypatch.setattr(
        stage0_evidence,
        "verify_stage0_authority",
        lambda *_args, **_kwargs: authority,
    )
    monkeypatch.setattr(stage0_evidence, "capture_stage0_observations", _capture)
    monkeypatch.setattr(
        stage0_evidence,
        "build_stage0_control_report",
        lambda *_args, **_kwargs: _report(),
    )
    output = tmp_path / "output"
    result = create_stage0_control_package(
        "causal",
        "a" * 64,
        "archive",
        "b" * 64,
        "runtime",
        "c" * 64,
        output,
    )
    report = output / "control-report.json"
    report.write_text(report.read_text(encoding="utf-8") + " ")

    with pytest.raises(Stage0ControlError) as error:
        replay_stage0_control_package(
            output / "replay-bundle.json",
            result["artifact_manifest_sha256"],
        )
    assert error.value.code == "STAGE0_ARTIFACT_HASH_MISMATCH"


def test_stage0_plan_does_not_authorize_stage1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority(tmp_path)
    monkeypatch.setattr(
        stage0_evidence,
        "verify_stage0_authority",
        lambda *_args, **_kwargs: authority,
    )
    monkeypatch.setattr(stage0_evidence, "capture_stage0_observations", _capture)
    monkeypatch.setattr(
        stage0_evidence,
        "build_stage0_control_report",
        lambda *_args, **_kwargs: _report(),
    )
    output = tmp_path / "output"
    create_stage0_control_package(
        "causal",
        "a" * 64,
        "archive",
        "b" * 64,
        "runtime",
        "c" * 64,
        output,
    )
    plan = json.loads((output / "stage0-plan.json").read_text())

    assert plan["stage_1_intervention_executed"] is False
    assert plan["frozen_int8_candidate_mutated"] is False
    assert plan["operational_tolerances_changed"] is False
