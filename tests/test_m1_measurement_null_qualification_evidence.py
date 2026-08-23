from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_continuity.m1_diagnostics import (
    measurement_null_qualification_evidence as module,
)
from neural_continuity.m1_diagnostics.measurement_null_qualification_authority import (
    QualificationPreflightAuthority,
    QualificationPreflightError,
)
from neural_continuity.m1_diagnostics.measurement_null_qualification_evidence import (
    capture_qualification_preflight_package,
    replay_qualification_preflight_package,
)


def _authority(tmp_path: Path) -> QualificationPreflightAuthority:
    return QualificationPreflightAuthority(
        config_path=tmp_path / "config.yaml",
        dataset_directory=tmp_path / "dataset",
        transition_a_bundle=tmp_path / "transition-a.json",
        extension_plan_bundle=tmp_path / "extension" / "replay-bundle.json",
        extension_plan_manifest_sha256="a" * 64,
        extension_plan_sha256="b" * 64,
        qualification_phase={
            "phase_id": "full_corpus_qualification",
            "qualifying_detection_evidence": True,
            "process_epoch_count": 120,
            "documents": "all frozen document IDs",
            "queries": "all and only measurement_null query IDs",
            "start_condition": "integrity verified",
        },
        qualification_phase_sha256="c" * 64,
        sentinel_run_directory=tmp_path / "sentinel",
        sentinel_root_manifest_sha256="d" * 64,
        sentinel_checkpoint_sha256="e" * 64,
        sentinel_run_plan_sha256="f" * 64,
        sentinel_authority_sha256="1" * 64,
        epoch_manifest_sha256s=tuple(f"{number:064x}" for number in range(1, 121)),
        authority_sha256="2" * 64,
    )


def test_package_capture_and_replay_remain_model_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority(tmp_path)
    monkeypatch.setattr(
        module,
        "verify_qualification_preflight_authority",
        lambda *args: authority,
    )
    output = tmp_path / "package"

    captured = capture_qualification_preflight_package(
        authority.config_path,
        authority.dataset_directory,
        authority.transition_a_bundle,
        authority.extension_plan_bundle,
        authority.extension_plan_manifest_sha256,
        authority.sentinel_run_directory,
        authority.sentinel_root_manifest_sha256,
        authority.sentinel_checkpoint_sha256,
        output,
    )
    replayed = replay_qualification_preflight_package(
        output / "replay-bundle.json",
        str(captured["artifact_manifest_sha256"]),
    )

    assert replayed["replay_verified"] is True
    assert replayed["full_corpus_execution_authorized"] is False
    assert replayed["model_execution_used"] is False
    assert replayed["onnx_graph_loaded"] is False
    assert replayed["numeric_observation_read"] is False


def test_package_replay_fails_closed_for_tampered_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority(tmp_path)
    monkeypatch.setattr(
        module,
        "verify_qualification_preflight_authority",
        lambda *args: authority,
    )
    output = tmp_path / "package"
    captured = capture_qualification_preflight_package(
        authority.config_path,
        authority.dataset_directory,
        authority.transition_a_bundle,
        authority.extension_plan_bundle,
        authority.extension_plan_manifest_sha256,
        authority.sentinel_run_directory,
        authority.sentinel_root_manifest_sha256,
        authority.sentinel_checkpoint_sha256,
        output,
    )
    path = output / "qualification-authority.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["gate"]["full_corpus_execution_authorized"] = True
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(QualificationPreflightError) as error:
        replay_qualification_preflight_package(
            output / "replay-bundle.json",
            str(captured["artifact_manifest_sha256"]),
        )

    assert error.value.status == "BLOCKED"
    assert error.value.code == "QUALIFICATION_PACKAGE_ARTIFACT_INTEGRITY_FAILED"
