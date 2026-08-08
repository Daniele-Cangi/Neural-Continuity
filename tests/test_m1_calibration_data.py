from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_continuity.m1_b import calibration_data
from neural_continuity.m1_teacher_evidence import (
    MaterializedDataset,
    RoleData,
    TeacherEvidenceError,
)


def _materialized_dataset() -> MaterializedDataset:
    role_names = (
        "measurement_null",
        "quantization_calibration",
        "contract_development",
        "validation",
        "frozen_critical",
        "final_holdout",
    )
    roles = {}
    for role_name in role_names:
        count = 162 if role_name == "quantization_calibration" else 1
        query_ids = [f"{role_name}-q-{index:03d}" for index in range(count)]
        roles[role_name] = RoleData(
            name=role_name,
            query_ids=query_ids,
            query_texts=[f"text for {query_id}" for query_id in query_ids],
            relevant_document_ids={query_id: ["document-001"] for query_id in query_ids},
        )
    return MaterializedDataset(
        dataset_id="nc-m1-canonical-fixture-v1",
        manifest_sha256="a" * 64,
        materialization_policy_sha256="b" * 64,
        partition_policy_sha256="c" * 64,
        document_ids=["document-001"],
        document_texts=["canonical document"],
        roles=roles,
    )


def test_build_and_verify_calibration_package(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        calibration_data,
        "load_materialized_dataset",
        lambda _: _materialized_dataset(),
    )
    output_directory = tmp_path / "calibration-package"

    result = calibration_data.build_calibration_package(
        Path("contracts/m1-transition-b-v1.json"),
        tmp_path / "dataset",
        output_directory,
    )

    assert result["status"] == "PASS"
    assert result["query_count"] == 162
    assert result["model_execution_used"] is False

    records = [
        json.loads(line)
        for line in (output_directory / "calibration-inputs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    query_ids = [record["query_id"] for record in records]
    assert query_ids == sorted(query_ids, key=lambda query_id: query_id.encode("utf-8"))
    assert records[0]["text"] == f"text for {query_ids[0]}"

    verification = calibration_data.verify_calibration_package(output_directory)
    assert verification == {
        "status": "PASS",
        "query_count": 162,
        "model_execution_used": False,
    }


def test_verify_calibration_package_fails_closed_for_tampered_inputs(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        calibration_data,
        "load_materialized_dataset",
        lambda _: _materialized_dataset(),
    )
    output_directory = tmp_path / "calibration-package"
    calibration_data.build_calibration_package(
        Path("contracts/m1-transition-b-v1.json"),
        tmp_path / "dataset",
        output_directory,
    )
    (output_directory / "calibration-inputs.jsonl").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(TeacherEvidenceError) as error:
        calibration_data.verify_calibration_package(output_directory)

    assert error.value.status == "BLOCKED"
    assert error.value.code == "EVIDENCE_ARTIFACT_SIZE_MISMATCH"


def test_verify_calibration_package_blocks_role_relabeling(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        calibration_data,
        "load_materialized_dataset",
        lambda _: _materialized_dataset(),
    )
    output_directory = tmp_path / "calibration-package"
    calibration_data.build_calibration_package(
        Path("contracts/m1-transition-b-v1.json"),
        tmp_path / "dataset",
        output_directory,
    )
    manifest_path = output_directory / "calibration-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["data_role"] = "validation"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(TeacherEvidenceError) as error:
        calibration_data.verify_calibration_package(output_directory)

    assert error.value.status == "BLOCKED"
    assert error.value.code == "CALIBRATION_LEAKAGE"
