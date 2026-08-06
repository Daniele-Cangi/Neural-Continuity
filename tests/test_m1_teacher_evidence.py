from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neural_continuity.m1_teacher_evidence import (
    MaterializedDataset,
    RoleData,
    TeacherEvidenceError,
    TeacherObservation,
    _rank_and_measure,
    replay_teacher_evidence,
    write_teacher_evidence_package,
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
    query_ids = [
        "measurement_null-q",
        "quantization_calibration-q",
        "contract_development-q",
        "validation-q",
        "frozen_critical-q",
        "final_holdout-q",
    ]
    query_roles = [
        "measurement_null",
        "quantization_calibration",
        "contract_development",
        "validation",
        "frozen_critical",
        "final_holdout",
    ]
    return TeacherObservation(
        document_ids=["d1", "d2"],
        document_embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        query_ids=query_ids,
        query_embeddings=np.asarray([[1.0, 0.0]] * len(query_ids), dtype=np.float32),
        query_roles=query_roles,
        relevant_document_ids={query_id: ["d1"] for query_id in query_ids},
    )


def test_rankings_break_ties_by_document_id():
    observation = TeacherObservation(
        document_ids=["d2", "d1"],
        document_embeddings=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        query_ids=["q1"],
        query_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        query_roles=["measurement_null"],
        relevant_document_ids={"q1": ["d1"]},
    )

    rankings, metrics = _rank_and_measure(observation, top_k=2)

    assert rankings[0]["ranked_document_ids"] == ["d1", "d2"]
    assert metrics["roles"]["measurement_null"]["metrics"]["mrr_at_k"] == 1.0


def test_replay_reconstructs_captured_teacher_evidence_without_model(tmp_path: Path):
    output = tmp_path / "evidence"
    result = write_teacher_evidence_package(
        output_directory=output,
        dataset=_dataset(),
        observation=_observation(),
        teacher_manifest={"model_id": "synthetic-test-teacher"},
        evidence_scope={"classification": "test", "qualifying_m1_evidence": False},
        config_sha256="d" * 64,
        top_k=2,
    )

    replay = replay_teacher_evidence(output / "replay-bundle.json")

    assert result["evidence_status"] == "CAPTURED_PENDING_REPLAY"
    assert replay == {
        "status": "PASS",
        "replay_verified": True,
        "model_execution_used": False,
        "dataset_id": "test-dataset",
        "query_count": 6,
        "document_count": 2,
    }


def test_replay_fails_closed_when_captured_artifact_changes(tmp_path: Path):
    output = tmp_path / "evidence"
    write_teacher_evidence_package(
        output_directory=output,
        dataset=_dataset(),
        observation=_observation(),
        teacher_manifest={"model_id": "synthetic-test-teacher"},
        evidence_scope={"classification": "test", "qualifying_m1_evidence": False},
        config_sha256="d" * 64,
        top_k=2,
    )
    rankings_path = output / "rankings.jsonl"
    rankings_path.write_bytes(
        rankings_path.read_bytes().replace(b"measurement_null", b"measurement-nulx", 1)
    )

    with pytest.raises(TeacherEvidenceError) as error:
        replay_teacher_evidence(output / "replay-bundle.json")

    assert error.value.code == "EVIDENCE_ARTIFACT_HASH_MISMATCH"
