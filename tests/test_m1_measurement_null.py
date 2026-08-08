from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_measurement_null import (
    BATCH_SIZE_VARIATION_FAMILY,
    REPEATED_INFERENCE_FAMILY,
    SourceRun,
    capture_measurement_null,
    replay_measurement_null,
    write_measurement_null_package,
)
from neural_continuity.m1_teacher_evidence import (
    MaterializedDataset,
    RoleData,
    TeacherEvidenceError,
    TeacherObservation,
)


def _dataset() -> MaterializedDataset:
    roles = {
        role: RoleData(
            role,
            [f"{role}-q"],
            [f"{role} query"],
            {f"{role}-q": ["d1"]},
        )
        for role in (
            "measurement_null",
            "quantization_calibration",
            "contract_development",
            "validation",
            "frozen_critical",
            "final_holdout",
        )
    }
    return MaterializedDataset(
        dataset_id="test-dataset",
        manifest_sha256="a" * 64,
        materialization_policy_sha256="b" * 64,
        partition_policy_sha256="c" * 64,
        document_ids=["d1", "d2"],
        document_texts=["one", "two"],
        roles=roles,
    )


def _observation() -> TeacherObservation:
    return TeacherObservation(
        document_ids=["d1", "d2"],
        document_embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        query_ids=["measurement_null-q"],
        query_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        query_roles=["measurement_null"],
        relevant_document_ids={"measurement_null-q": ["d1"]},
    )


def _runs() -> list[SourceRun]:
    observation = _observation()
    return [
        SourceRun("repeated-inference-001", REPEATED_INFERENCE_FAMILY, 2, observation),
        SourceRun("repeated-inference-002", REPEATED_INFERENCE_FAMILY, 2, observation),
        SourceRun("repeated-inference-003", REPEATED_INFERENCE_FAMILY, 2, observation),
        SourceRun("batch-size-0001", BATCH_SIZE_VARIATION_FAMILY, 1, observation),
        SourceRun("batch-size-0002", BATCH_SIZE_VARIATION_FAMILY, 2, observation),
    ]


def _write_package(tmp_path: Path) -> Path:
    output = tmp_path / "source-null"
    write_measurement_null_package(
        output_directory=output,
        dataset=_dataset(),
        runs=_runs(),
        teacher_manifest={"model_id": "synthetic-test-teacher"},
        evidence_scope={"classification": "test", "qualifying_m1_evidence": False},
        config_sha256="d" * 64,
        top_k=2,
        source_identity={
            "transition_a_evidence_manifest_sha256": "a" * 64,
            "onnx_fp32_artifact_sha256": "b" * 64,
            "execution_provider": "CPUExecutionProvider",
        },
        evidence_kind="onnx_fp32_measurement_null",
    )
    return output


def test_replay_reconstructs_source_only_measurement_null_without_model(tmp_path: Path):
    output = _write_package(tmp_path)

    replay = replay_measurement_null(output / "replay-bundle.json")

    assert replay == {
        "status": "PASS",
        "replay_verified": True,
        "model_execution_used": False,
        "dataset_id": "test-dataset",
        "source_run_count": 5,
        "query_count": 1,
        "document_count": 2,
        "measurement_null_status": "CAPTURED_NOT_DECIDED",
    }
    report = json.loads((output / "comparison-report.json").read_text(encoding="utf-8"))
    assert report["transition_a_decision"] == "NOT_APPLICABLE"
    assert set(report["empirical_envelopes"]) == {
        REPEATED_INFERENCE_FAMILY,
        BATCH_SIZE_VARIATION_FAMILY,
    }
    assert report["operational_tolerance"] == "NOT_SELECTED"
    evidence_manifest = json.loads((output / "evidence-manifest.json").read_text(encoding="utf-8"))
    replay_bundle = json.loads((output / "replay-bundle.json").read_text(encoding="utf-8"))
    assert evidence_manifest["evidence_kind"] == "onnx_fp32_measurement_null"
    assert replay_bundle["source_identity"]["execution_provider"] == "CPUExecutionProvider"


def test_replay_fails_closed_when_a_declared_source_run_is_missing(tmp_path: Path):
    output = _write_package(tmp_path)
    observations_path = output / "source-null-observations.npz"
    with np.load(observations_path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    for key in ("run_ids", "families", "batch_sizes", "document_embeddings", "query_embeddings"):
        values[key] = values[key][:-1]
    np.savez_compressed(observations_path, **values)
    manifest_path = output / "evidence-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        if artifact["path"] == "source-null-observations.npz":
            artifact["size_bytes"] = observations_path.stat().st_size
            artifact["sha256"] = sha256_file(observations_path)
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    with pytest.raises(TeacherEvidenceError) as error:
        replay_measurement_null(output / "replay-bundle.json")

    assert error.value.code == "MISSING_DECLARED_SOURCE_OBSERVATION"


def test_capture_uses_only_measurement_null_and_replays(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "test",
                "evidence_scope": {"classification": "test", "qualifying_m1_evidence": True},
                "dataset": {"dataset_id": "test-dataset"},
                "model": {
                    "model_id": "test-teacher",
                    "revision": "r1",
                    "device": "cpu",
                    "cache_only": True,
                    "output_dtype": "float32",
                },
                "evaluation": {"top_k": 2},
                "measurement_null": {
                    "repeated_inference": {"count": 3, "batch_size": 2},
                    "batch_size_variation": {"batch_sizes": [1, 2]},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "neural_continuity.m1_measurement_null.load_materialized_dataset", lambda _: _dataset()
    )
    monkeypatch.setattr(
        "neural_continuity.m1_measurement_null._load_teacher",
        lambda _: (object(), {"model_id": "test-teacher"}),
    )

    def encode(_: object, texts: list[str], __: int, ___: str) -> np.ndarray:
        if len(texts) == 2:
            return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        assert texts == ["measurement_null query"]
        return np.asarray([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr("neural_continuity.m1_measurement_null._encode", encode)
    output = tmp_path / "captured"

    result = capture_measurement_null(config_path, tmp_path / "materialized", output)

    assert result["source_run_count"] == 5
    assert replay_measurement_null(output / "replay-bundle.json")["status"] == "PASS"
