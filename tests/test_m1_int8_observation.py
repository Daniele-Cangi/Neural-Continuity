from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_b.onnx_int8_observation import replay_int8_target_observation
from neural_continuity.m1_teacher_evidence import TeacherEvidenceError


def _artifact(path: Path) -> dict[str, object]:
    return {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _write_target_package(tmp_path: Path) -> Path:
    output = tmp_path / "target"
    output.mkdir()
    (output / "teacher-int8-qdq.onnx").write_bytes(b"synthetic-int8")
    np.savez_compressed(
        output / "target-observations.npz",
        run_ids=np.asarray(["batch-size-0001", "batch-size-0016", "batch-size-0064"]),
        batch_sizes=np.asarray([1, 16, 64]),
        document_embeddings=np.ones((3, 2, 2), dtype=np.float32),
        query_embeddings=np.ones((3, 3, 2), dtype=np.float32),
    )
    required_runs = [
        {"run_id": "batch-size-0001", "batch_size": 1},
        {"run_id": "batch-size-0016", "batch_size": 16},
        {"run_id": "batch-size-0064", "batch_size": 64},
    ]
    metadata = {"required_runs": required_runs}
    metadata_path = output / "observation-metadata.json"
    metadata_path.write_bytes(canonical_json_bytes(metadata) + b"\n")
    bundle = {
        "observation_path": "target-observations.npz",
        "metadata_path": "observation-metadata.json",
        "required_runs": required_runs,
    }
    bundle_path = output / "replay-bundle.json"
    bundle_path.write_bytes(canonical_json_bytes(bundle) + b"\n")
    manifest = {
        "package_kind": "m1_onnx_int8_target_observation",
        "artifacts": [
            _artifact(output / "teacher-int8-qdq.onnx"),
            _artifact(output / "target-observations.npz"),
            _artifact(metadata_path),
            _artifact(bundle_path),
        ],
    }
    (output / "evidence-manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return bundle_path


def test_replay_verifies_target_observation_without_model_execution(tmp_path: Path) -> None:
    replay = replay_int8_target_observation(_write_target_package(tmp_path))

    assert replay == {
        "status": "PASS",
        "replay_verified": True,
        "model_execution_used": False,
        "target_run_count": 3,
        "transition_b_decision": "NOT_EVALUATED",
    }


def test_replay_blocks_missing_declared_target_run(tmp_path: Path) -> None:
    bundle_path = _write_target_package(tmp_path)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["required_runs"] = bundle["required_runs"][:-1]
    bundle_path.write_bytes(canonical_json_bytes(bundle) + b"\n")
    manifest_path = bundle_path.parent / "evidence-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        if artifact["path"] == bundle_path.name:
            artifact.update(_artifact(bundle_path))
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    with pytest.raises(TeacherEvidenceError) as error:
        replay_int8_target_observation(bundle_path)

    assert error.value.code == "MISSING_DECLARED_TARGET_OBSERVATION"
