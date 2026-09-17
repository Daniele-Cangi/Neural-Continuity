"""Synthetic epoch packages test plumbing only, never CUDA qualification."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics import cuda_null_sentinel_epoch_package as package
from neural_continuity.m1_diagnostics.cuda_null_preflight_readiness import CONFIG_SHA256
from neural_continuity.m1_diagnostics.cuda_null_sentinel_epoch_format import (
    DOCUMENT_COUNT,
    EMBEDDING_DIMENSION,
    EPOCH_LAYOUT,
    QUERY_COUNT,
    CudaNullSentinelEpochBlocked,
)
from neural_continuity.m1_diagnostics.cuda_null_sentinel_readiness import (
    SOURCE_PREFLIGHT_MANIFEST_SHA256,
)
from neural_continuity.m1_diagnostics.cuda_null_source_preflight_inputs import (
    DATASET_MANIFEST_SHA256,
    SOURCE_ONNX_SHA256,
)


def _capture() -> (
    tuple[dict[str, object], dict[str, object], np.ndarray, np.ndarray, list[dict[str, object]]]
):
    plan: dict[str, object] = {
        "kind": "m1_cuda_null_sentinel_epoch_plan",
        "version": "1.0.0",
        "phase_id": "tensor_sentinel_preflight",
        "epoch_number": 1,
        "previous_checkpoint_sha256": "a" * 64,
        "config_sha256": CONFIG_SHA256,
        "source_preflight_manifest_sha256": SOURCE_PREFLIGHT_MANIFEST_SHA256,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "source_onnx_sha256": SOURCE_ONNX_SHA256,
        "document_ids": [f"d{index:04d}" for index in range(DOCUMENT_COUNT)],
        "query_ids": [f"q{index:03d}" for index in range(QUERY_COUNT)],
        "query_role": "measurement_null",
        "runs": [{"label": label, "batch_size": batch} for label, batch in EPOCH_LAYOUT],
        "qualifying_detection_evidence": False,
        "scientific_decision": "NOT_EVALUATED",
        "full_corpus_execution": False,
        "int8_execution": False,
    }
    runtime: dict[str, object] = {
        "process_instance_id": "synthetic-process-1",
        "runtime_identity_sha256": "b" * 64,
        "session_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "source_only": True,
        "onnx_graph_loaded": True,
        "session_created": True,
        "model_execution_used": True,
        "int8_executed": False,
        "holdout_accessed": False,
    }
    documents = np.zeros((len(EPOCH_LAYOUT), DOCUMENT_COUNT, EMBEDDING_DIMENSION), dtype="<f4")
    queries = np.zeros((len(EPOCH_LAYOUT), QUERY_COUNT, EMBEDDING_DIMENSION), dtype="<f4")
    documents[:, :, 0] = 1.0
    queries[:, :, 1] = 1.0
    profiles: list[dict[str, object]] = []
    for index, (label, batch) in enumerate(EPOCH_LAYOUT):
        for role, observation in (
            ("documents", documents[index]),
            ("measurement_null_queries", queries[index]),
        ):
            profiles.append(
                {
                    "run_label": label,
                    "role": role,
                    "batch_size": batch,
                    "item_count": observation.shape[0],
                    "observation_sha256": hashlib.sha256(observation.tobytes()).hexdigest(),
                    "provider_event_counts": {"CUDAExecutionProvider": 1},
                    "operator_event_counts": {"CUDAExecutionProvider": {"MatMul": 1}},
                    "cpu_fallback_operator_types": [],
                    "unclassified_cpu_events": 0,
                    "undeclared_providers": [],
                }
            )
    return plan, runtime, documents, queries, profiles


def _write(tmp_path: Path) -> tuple[Path, str]:
    plan, runtime, documents, queries, profiles = _capture()
    return package.write_cuda_sentinel_epoch(
        tmp_path / "epoch",
        plan=plan,
        runtime=runtime,
        document_embeddings=documents,
        query_embeddings=queries,
        profiles=profiles,
    )


def test_synthetic_epoch_replays_without_model_or_authority(tmp_path: Path) -> None:
    output, manifest_hash = _write(tmp_path)
    replay = package.replay_cuda_sentinel_epoch(output / "replay-bundle.json", manifest_hash)
    assert replay["replay_status"] == "PASS"
    assert replay["summary_match"] is True
    assert replay["authority_verified"] is False
    assert replay["checkpoint_chain_verified"] is False
    assert replay["qualifying_detection_evidence"] is False
    assert replay["model_loaded"] is False


def test_missing_declared_artifact_blocks(tmp_path: Path) -> None:
    output, manifest_hash = _write(tmp_path)
    (output / "provider-profile.jsonl").unlink()
    replay = package.replay_cuda_sentinel_epoch(output / "replay-bundle.json", manifest_hash)
    assert replay["replay_status"] == "BLOCKED"


def test_resealed_technical_summary_still_blocks(tmp_path: Path) -> None:
    output, _ = _write(tmp_path)
    summary_path = output / "technical-summary.json"
    summary = package._read_json(summary_path)
    summary["scientific_decision"] = "PASS"
    package._write_json(summary_path, summary)
    manifest_path = output / "artifact-manifest.json"
    manifest = package._read_json(manifest_path)
    for entry in manifest["artifacts"]:
        if entry["path"] == "technical-summary.json":
            entry["sha256"] = sha256_file(summary_path)
    package._write_json(manifest_path, manifest)
    replay = package.replay_cuda_sentinel_epoch(
        output / "replay-bundle.json", sha256_file(manifest_path)
    )
    assert replay["replay_status"] == "BLOCKED"
    assert replay["summary_match"] is False


def test_cpu_only_profile_cannot_be_packaged(tmp_path: Path) -> None:
    plan, runtime, documents, queries, profiles = _capture()
    profiles[0]["provider_event_counts"] = {"CPUExecutionProvider": 1}
    profiles[0]["operator_event_counts"] = {"CPUExecutionProvider": {"MatMul": 1}}
    profiles[0]["cpu_fallback_operator_types"] = ["MatMul"]
    with pytest.raises(CudaNullSentinelEpochBlocked, match="CUDA provider activity"):
        package.write_cuda_sentinel_epoch(
            tmp_path / "epoch",
            plan=plan,
            runtime=runtime,
            document_embeddings=documents,
            query_embeddings=queries,
            profiles=profiles,
        )


def test_wrong_observation_dtype_cannot_be_packaged(tmp_path: Path) -> None:
    plan, runtime, documents, queries, profiles = _capture()
    with pytest.raises(CudaNullSentinelEpochBlocked, match="float32"):
        package.write_cuda_sentinel_epoch(
            tmp_path / "epoch",
            plan=plan,
            runtime=runtime,
            document_embeddings=documents.astype(np.float64),
            query_embeddings=queries,
            profiles=profiles,
        )


def test_profile_observation_hash_mismatch_blocks(tmp_path: Path) -> None:
    plan, runtime, documents, queries, profiles = _capture()
    profiles[0]["observation_sha256"] = hashlib.sha256(canonical_json_bytes({})).hexdigest()
    with pytest.raises(CudaNullSentinelEpochBlocked, match="observation hash mismatch"):
        package.write_cuda_sentinel_epoch(
            tmp_path / "epoch",
            plan=plan,
            runtime=runtime,
            document_embeddings=documents,
            query_embeddings=queries,
            profiles=profiles,
        )
