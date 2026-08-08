from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_onnx_transition import (
    PairedRun,
    replay_transition_a,
    write_transition_a_package,
)
from neural_continuity.m1_teacher_evidence import (
    MaterializedDataset,
    RoleData,
    TeacherEvidenceError,
    TeacherObservation,
)


def _dataset() -> MaterializedDataset:
    roles = {
        role: RoleData(role, [f"{role}-q"], [f"{role} query"], {f"{role}-q": ["d1"]})
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
    query_ids = [
        "measurement_null-q",
        "quantization_calibration-q",
        "contract_development-q",
        "validation-q",
        "frozen_critical-q",
        "final_holdout-q",
    ]
    return TeacherObservation(
        document_ids=["d1", "d2"],
        document_embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        query_ids=query_ids,
        query_embeddings=np.asarray([[1.0, 0.0]] * len(query_ids), dtype=np.float32),
        query_roles=[query_id.removesuffix("-q") for query_id in query_ids],
        relevant_document_ids={query_id: ["d1"] for query_id in query_ids},
    )


def _contract() -> dict[str, object]:
    return json.loads(Path("contracts/m1-transition-a-v1.json").read_text(encoding="utf-8"))


def _write_package(tmp_path: Path) -> Path:
    output = tmp_path / "transition"
    onnx_path = tmp_path / "teacher.onnx"
    onnx_path.write_bytes(b"synthetic-onnx")
    observation = _observation()
    write_transition_a_package(
        output_directory=output,
        dataset=_dataset(),
        runs=[
            PairedRun("batch-size-0001", 1, observation, observation),
            PairedRun("batch-size-0016", 16, observation, observation),
            PairedRun("batch-size-0064", 64, observation, observation),
        ],
        contract=_contract(),
        contract_sha256=sha256_file(Path("contracts/m1-transition-a-v1.json")),
        config_sha256="d" * 64,
        teacher_manifest={"model_id": "synthetic-test-teacher"},
        onnx_source_path=onnx_path,
        onnx_manifest={"opset_version": 18, "requested_execution_provider": "CPUExecutionProvider"},
        evidence_scope={"classification": "test", "qualifying_m1_evidence": False},
        top_k=2,
    )
    return output


def test_replay_reconstructs_transition_a_decision_without_models(tmp_path: Path):
    output = _write_package(tmp_path)

    replay = replay_transition_a(output / "replay-bundle.json")

    assert replay == {
        "status": "PASS",
        "replay_verified": True,
        "model_execution_used": False,
        "transition_a_status": "PASS",
        "source_target_pair_count": 3,
        "dataset_id": "test-dataset",
    }


def test_replay_fails_closed_when_a_declared_pair_is_missing(tmp_path: Path):
    output = _write_package(tmp_path)
    observations_path = output / "transition-observations.npz"
    with np.load(observations_path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    for key in (
        "run_ids",
        "batch_sizes",
        "source_document_embeddings",
        "source_query_embeddings",
        "target_document_embeddings",
        "target_query_embeddings",
    ):
        values[key] = values[key][:-1]
    np.savez_compressed(observations_path, **values)
    manifest_path = output / "evidence-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        if artifact["path"] == "transition-observations.npz":
            artifact["size_bytes"] = observations_path.stat().st_size
            artifact["sha256"] = sha256_file(observations_path)
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    with pytest.raises(TeacherEvidenceError) as error:
        replay_transition_a(output / "replay-bundle.json")

    assert error.value.code == "MISSING_DECLARED_SOURCE_OBSERVATION"
