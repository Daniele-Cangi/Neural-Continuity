from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics import (
    measurement_null_qualification_authority as module,
)
from neural_continuity.m1_diagnostics.measurement_null_qualification_authority import (
    QualificationPreflightError,
    verify_qualification_preflight_authority,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _entry(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _build_frozen_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    extension_root = tmp_path / "extension"
    extension_root.mkdir()
    phase = {
        "phase_id": "full_corpus_qualification",
        "qualifying_detection_evidence": True,
        "process_epoch_count": 120,
        "documents": "all frozen document IDs",
        "queries": "all and only measurement_null query IDs",
        "start_condition": (
            "phase 1 artifacts complete, integrity verified, and no execution error"
        ),
    }
    plan_path = extension_root / "measurement-null-extension-plan.json"
    _write_json(
        plan_path,
        {"phases": [{"phase_id": "tensor_sentinel_preflight"}, phase]},
    )
    replay_bundle = extension_root / "replay-bundle.json"
    _write_json(replay_bundle, {"plan_path": plan_path.name})
    plan_sha256 = sha256_file(plan_path)
    extension_manifest_sha256 = "a" * 64
    sentinel_authority = SimpleNamespace(
        authority_sha256="b" * 64,
        config_sha256="c" * 64,
        extension_plan_sha256=plan_sha256,
        selected_document_ids=("doc-1", "doc-2"),
        query_ids=("query-1",),
        qrels={"query-1": {"doc-1": 1}},
    )
    monkeypatch.setattr(
        module,
        "verify_sentinel_authority",
        lambda *args: sentinel_authority,
    )
    monkeypatch.setattr(
        module,
        "replay_measurement_null_extension_plan",
        lambda *args: {
            "status": "PREREGISTERED_NOT_EXECUTED",
            "replay_verified": True,
            "plan_match": True,
            "status_match": True,
            "invariants_match": True,
            "execution_started": False,
            "model_execution_used": False,
            "stage_1_execution_started": False,
        },
    )
    sentinel_root = tmp_path / "sentinel"
    sentinel_root.mkdir()
    run_plan = {
        "kind": "m1-measurement-null-sentinel-run-plan",
        "version": "1.0.0",
        "phase_id": "tensor_sentinel_preflight",
        "process_epoch_count": 120,
        "authority": {
            "authority_sha256": sentinel_authority.authority_sha256,
            "config_sha256": sentinel_authority.config_sha256,
            "extension_plan_bundle": str(replay_bundle.resolve()),
            "extension_plan_manifest_sha256": extension_manifest_sha256,
            "extension_plan_sha256": plan_sha256,
        },
        "document_ids": list(sentinel_authority.selected_document_ids),
        "query_ids": list(sentinel_authority.query_ids),
        "query_roles": ["measurement_null"],
        "qrels": sentinel_authority.qrels,
        "evidence_policy": {
            "operational_tolerance_selection_allowed": False,
            "qualifying_detection_evidence": False,
            "scientific_decision": "NOT_EVALUATED",
            "technical_preflight_only": True,
        },
        "execution_policy": {
            "adaptive_sample_size_allowed": False,
            "candidate_or_int8_execution_allowed": False,
            "early_stopping_allowed": False,
            "full_corpus_execution_allowed": False,
            "holdout_query_access_allowed": False,
            "one_epoch_per_process": True,
            "resume_requires_checkpoint_hash": True,
        },
    }
    run_plan_path = sentinel_root / "sentinel-run-plan.json"
    _write_json(run_plan_path, run_plan)
    root_manifest = {
        "kind": "m1-measurement-null-sentinel-root-manifest",
        "version": "1.0.0",
        "artifacts": [_entry(sentinel_root, run_plan_path)],
    }
    root_manifest_path = sentinel_root / "artifact-manifest.json"
    _write_json(root_manifest_path, root_manifest)
    root_sha256 = sha256_file(root_manifest_path)
    previous_sha256 = root_sha256
    for epoch_number in range(1, 121):
        epoch_root = sentinel_root / f"epoch-{epoch_number:04d}"
        epoch_root.mkdir()
        for name in (
            "epoch-plan.json",
            "epoch-summary.json",
            "runtime-inventory.json",
        ):
            _write_json(epoch_root / name, {"epoch_number": epoch_number})
        (epoch_root / "raw-observations.npz").write_bytes(b"opaque-observation-archive")
        manifest = {
            "kind": "m1-measurement-null-sentinel-epoch-manifest",
            "version": "1.0.0",
            "epoch_number": epoch_number,
            "previous_checkpoint_sha256": previous_sha256,
            "qualifying_detection_evidence": False,
            "full_corpus_execution": False,
            "integrity": {
                "hash_algorithm": "SHA-256",
                "missing_or_tampered_artifact_behavior": "BLOCKED",
            },
            "artifacts": [
                _entry(epoch_root, epoch_root / name) for name in sorted(module.EPOCH_ARTIFACTS)
            ],
        }
        manifest_path = epoch_root / "epoch-manifest.json"
        _write_json(manifest_path, manifest)
        previous_sha256 = sha256_file(manifest_path)
    return {
        "config": tmp_path / "config.yaml",
        "dataset": tmp_path / "dataset",
        "transition": tmp_path / "transition-a.json",
        "extension_bundle": replay_bundle,
        "extension_manifest_sha256": extension_manifest_sha256,
        "sentinel_root": sentinel_root,
        "root_sha256": root_sha256,
        "checkpoint_sha256": previous_sha256,
    }


def _verify(inputs: dict[str, object]) -> object:
    return verify_qualification_preflight_authority(
        inputs["config"],
        inputs["dataset"],
        inputs["transition"],
        inputs["extension_bundle"],
        inputs["extension_manifest_sha256"],
        inputs["sentinel_root"],
        inputs["root_sha256"],
        inputs["checkpoint_sha256"],
    )


def test_authority_verifies_complete_chain_without_deserializing_npz(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _build_frozen_inputs(tmp_path, monkeypatch)

    authority = _verify(inputs)

    assert authority.sentinel_checkpoint_sha256 == inputs["checkpoint_sha256"]
    assert len(authority.epoch_manifest_sha256s) == 120


def test_authority_fails_closed_for_tampered_observation_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _build_frozen_inputs(tmp_path, monkeypatch)
    sentinel = Path(str(inputs["sentinel_root"]))
    (sentinel / "epoch-0060" / "raw-observations.npz").write_bytes(b"tampered")

    with pytest.raises(QualificationPreflightError) as error:
        _verify(inputs)

    assert error.value.status == "BLOCKED"
    assert error.value.code == "QUALIFICATION_SENTINEL_EPOCH_ARTIFACT_INTEGRITY_FAILED"


def test_authority_fails_closed_for_missing_declared_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _build_frozen_inputs(tmp_path, monkeypatch)
    sentinel = Path(str(inputs["sentinel_root"]))
    for path in (sentinel / "epoch-0120").iterdir():
        path.unlink()
    (sentinel / "epoch-0120").rmdir()

    with pytest.raises(QualificationPreflightError) as error:
        _verify(inputs)

    assert error.value.status == "BLOCKED"
    assert error.value.code == "QUALIFICATION_SENTINEL_CHECKPOINT_SET_MISMATCH"
