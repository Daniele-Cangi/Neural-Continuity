from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.stage0_authority import (
    Stage0ControlError,
)
from neural_continuity.m1_diagnostics.stage0_metrics import (
    build_stage0_control_report,
    compare_repeat_control,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _package(root: Path, kind: str, delta: float = 0.0) -> Path:
    root.mkdir()
    source = kind == "m1_onnx_fp32_source_observation"
    observation_name = "source-observations.npz" if source else "target-observations.npz"
    model_name = "teacher-fp32.onnx" if source else "teacher-int8-qdq.onnx"
    (root / model_name).write_bytes(b"model")
    run_ids = np.asarray(["batch-size-0001", "batch-size-0016", "batch-size-0064"])
    batch_sizes = np.asarray([1, 16, 64], dtype=np.int64)
    document_embeddings = np.tile(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        (3, 1, 1),
    )
    query_embeddings = np.zeros((3, 6, 2), dtype=np.float32)
    query_embeddings[:, :, 0] = 1.0
    if delta:
        query_embeddings[0, 0] = np.asarray([1.0, delta], dtype=np.float32)
        query_embeddings[0, 0] /= np.linalg.norm(query_embeddings[0, 0])
    document_ids = np.asarray(["d1", "d2"])
    query_ids = np.asarray([f"q{index}" for index in range(6)])
    query_roles = np.asarray(
        [
            "measurement_null",
            "quantization_calibration",
            "contract_development",
            "validation",
            "frozen_critical",
            "final_holdout",
        ]
    )
    np.savez_compressed(
        root / observation_name,
        run_ids=run_ids,
        batch_sizes=batch_sizes,
        document_ids=document_ids,
        query_ids=query_ids,
        query_roles=query_roles,
        document_embeddings=document_embeddings,
        query_embeddings=query_embeddings,
    )
    qrels = {query_id: ["d1"] for query_id in query_ids.tolist()}
    required_runs = [
        {"run_id": run_id, "batch_size": int(batch_size)}
        for run_id, batch_size in zip(
            run_ids.tolist(),
            batch_sizes.tolist(),
            strict=True,
        )
    ]
    metadata = {
        "dataset_id": "dataset",
        "document_count": 2,
        "query_count": 6,
        "embedding_dimension": 2,
        "embedding_dtype": "float32",
        "output_normalization": "l2_unit_after_encode",
        "query_roles": query_roles.tolist(),
        "qrels": qrels,
        "required_runs": required_runs,
    }
    _write_json(root / "observation-metadata.json", metadata)
    replay = {
        "observation_path": observation_name,
        "metadata_path": "observation-metadata.json",
        "required_runs": required_runs,
        "replay_requires_model_execution": False,
    }
    _write_json(root / "replay-bundle.json", replay)
    identity_name = "source_identity" if source else "candidate_identity"
    manifest = {
        "package_kind": kind,
        "dataset": {"dataset_id": "dataset"},
        "teacher_tokenizer_identity": {"model_id": "teacher"},
        identity_name: {"artifact_sha256": "a" * 64},
        "artifacts": [],
    }
    for name in (
        model_name,
        observation_name,
        "observation-metadata.json",
        "replay-bundle.json",
    ):
        path = root / name
        manifest["artifacts"].append(
            {
                "path": name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    _write_json(root / "evidence-manifest.json", manifest)
    return root / "replay-bundle.json"


def _null_report(limit: float = 0.0) -> dict[str, Any]:
    return {
        "empirical_envelopes": {
            "repeated_inference": {
                "family": "repeated_inference",
                "comparison_count": 2,
                "empirical_maximum": {
                    "document_max_abs_delta": limit,
                    "query_max_abs_delta": limit,
                    "ranking_change_count": 0,
                    "ranking_change_fraction": 0.0,
                    "absolute_metric_delta": {
                        "recall_at_k": 0.0,
                        "mrr_at_k": 0.0,
                        "ndcg_at_k": 0.0,
                    },
                },
                "empirical_minimum": {
                    "document_min_cosine_similarity": 0.0,
                    "query_min_cosine_similarity": 0.0,
                },
            }
        }
    }


def test_repeat_control_passes_bitwise_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _package(
        tmp_path / "baseline",
        "m1_onnx_fp32_source_observation",
    )
    fresh = _package(
        tmp_path / "fresh",
        "m1_onnx_fp32_source_observation",
    )
    monkeypatch.setattr(
        "neural_continuity.m1_diagnostics.stage0_metrics.replay_fp32_source_observation",
        lambda *_args: {"replay_verified": True},
    )

    report = compare_repeat_control(
        baseline,
        fresh,
        "m1_onnx_fp32_source_observation",
        _null_report(),
        top_k=1,
    )

    assert report["outcome"] == "PASS"
    assert all(run["exact_embedding_identity"] for run in report["runs"])


def test_repeat_control_blocks_delta_outside_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _package(
        tmp_path / "baseline",
        "m1_onnx_int8_target_observation",
    )
    fresh = _package(
        tmp_path / "fresh",
        "m1_onnx_int8_target_observation",
        delta=0.1,
    )
    monkeypatch.setattr(
        "neural_continuity.m1_diagnostics.stage0_metrics.replay_int8_target_observation",
        lambda *_args: {"replay_verified": True},
    )

    report = compare_repeat_control(
        baseline,
        fresh,
        "m1_onnx_int8_target_observation",
        _null_report(),
        top_k=1,
    )

    assert report["outcome"] == "BLOCKED"
    assert report["scientific_fail_recorded"] is False


def test_stage0_aggregate_requires_both_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = iter(
        [
            {"outcome": "PASS"},
            {"outcome": "BLOCKED"},
        ]
    )
    monkeypatch.setattr(
        "neural_continuity.m1_diagnostics.stage0_metrics.compare_repeat_control",
        lambda *_args, **_kwargs: next(outcomes),
    )

    report = build_stage0_control_report("a", "b", "c", "d", {})

    assert report["status"] == "BLOCKED"
    assert report["stage_1_execution_started"] is False
    assert report["scientific_decision_recomputed"] is False


def test_repeat_control_rejects_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _package(
        tmp_path / "baseline",
        "m1_onnx_fp32_source_observation",
    )
    fresh = _package(
        tmp_path / "fresh",
        "m1_onnx_fp32_source_observation",
    )
    metadata = fresh.parent / "observation-metadata.json"
    payload = __import__("json").loads(metadata.read_text(encoding="utf-8"))
    payload["dataset_id"] = "other"
    _write_json(metadata, payload)
    manifest_path = fresh.parent / "evidence-manifest.json"
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        if artifact["path"] == "observation-metadata.json":
            artifact["sha256"] = sha256_file(metadata)
            artifact["size_bytes"] = metadata.stat().st_size
    _write_json(manifest_path, manifest)
    monkeypatch.setattr(
        "neural_continuity.m1_diagnostics.stage0_metrics.replay_fp32_source_observation",
        lambda *_args: {"replay_verified": True},
    )

    with pytest.raises(Stage0ControlError) as error:
        compare_repeat_control(
            baseline,
            fresh,
            "m1_onnx_fp32_source_observation",
            _null_report(),
            top_k=1,
        )
    assert error.value.code == "STAGE0_OBSERVATION_IDENTITY_MISMATCH"
