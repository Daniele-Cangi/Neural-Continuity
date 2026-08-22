from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neural_continuity.m1_diagnostics.measurement_null_sentinel_checkpoint import (
    create_sentinel_run,
    inspect_sentinel_checkpoint,
    write_epoch_checkpoint,
)
from neural_continuity.m1_teacher_evidence import TeacherEvidenceError


def _run_plan(epoch_count: int = 2) -> dict[str, object]:
    return {
        "kind": "m1-measurement-null-sentinel-run-plan",
        "version": "1.0.0",
        "phase_id": "tensor_sentinel_preflight",
        "process_epoch_count": epoch_count,
        "runs": [
            {"run_id": "batch_1_primary", "batch_size": 1},
            {"run_id": "batch_16_primary", "batch_size": 16},
            {"run_id": "batch_16_repeat", "batch_size": 16},
            {"run_id": "batch_64_primary", "batch_size": 64},
        ],
        "document_ids": ["doc-1", "doc-2", "doc-3"],
        "query_ids": ["query-1", "query-2"],
    }


def _embeddings() -> tuple[np.ndarray, np.ndarray]:
    documents = np.zeros((4, 3, 4), dtype=np.float32)
    queries = np.zeros((4, 2, 4), dtype=np.float32)
    documents[:, :, 0] = 1.0
    queries[:, :, 1] = 1.0
    return documents, queries


def _runtime(process_id: str) -> dict[str, object]:
    return {
        "model_execution_used": True,
        "onnx_graph_loaded": True,
        "activation_read": False,
        "execution_provider": "CPUExecutionProvider",
        "process_instance_id": process_id,
    }


def _prepared_state(root: Path) -> tuple[str, str]:
    prepared = create_sentinel_run(root, _run_plan())
    root_sha256 = str(prepared["root_manifest_sha256"])
    checkpoint_sha256 = str(prepared["latest_checkpoint_sha256"])
    return root_sha256, checkpoint_sha256


def test_checkpoint_chain_resumes_from_exact_external_hash(tmp_path: Path) -> None:
    root = tmp_path / "sentinel"
    root_sha256, checkpoint_sha256 = _prepared_state(root)
    documents, queries = _embeddings()

    initial = inspect_sentinel_checkpoint(root, root_sha256, checkpoint_sha256)
    first = write_epoch_checkpoint(
        initial,
        document_embeddings=documents,
        query_embeddings=queries,
        runtime_inventory=_runtime("process-1"),
    )
    first_sha256 = str(first["checkpoint_manifest_sha256"])
    resumed = inspect_sentinel_checkpoint(root, root_sha256, first_sha256)
    second = write_epoch_checkpoint(
        resumed,
        document_embeddings=documents,
        query_embeddings=queries,
        runtime_inventory=_runtime("process-2"),
    )
    final = inspect_sentinel_checkpoint(
        root,
        root_sha256,
        str(second["checkpoint_manifest_sha256"]),
    )

    assert final.complete is True
    assert final.completed_epoch_count == 2
    assert final.next_epoch_number is None


def test_checkpoint_rejects_tampered_epoch_artifact(tmp_path: Path) -> None:
    root = tmp_path / "sentinel"
    root_sha256, checkpoint_sha256 = _prepared_state(root)
    documents, queries = _embeddings()
    state = inspect_sentinel_checkpoint(root, root_sha256, checkpoint_sha256)
    result = write_epoch_checkpoint(
        state,
        document_embeddings=documents,
        query_embeddings=queries,
        runtime_inventory=_runtime("process-1"),
    )
    summary_path = root / "epoch-0001" / "epoch-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["scientific_decision"] = "PASS"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(TeacherEvidenceError) as error:
        inspect_sentinel_checkpoint(
            root,
            root_sha256,
            str(result["checkpoint_manifest_sha256"]),
        )

    assert error.value.status == "BLOCKED"


def test_checkpoint_requires_distinct_process_identity(tmp_path: Path) -> None:
    root = tmp_path / "sentinel"
    root_sha256, checkpoint_sha256 = _prepared_state(root)
    documents, queries = _embeddings()
    state = inspect_sentinel_checkpoint(root, root_sha256, checkpoint_sha256)
    first = write_epoch_checkpoint(
        state,
        document_embeddings=documents,
        query_embeddings=queries,
        runtime_inventory=_runtime("same-process"),
    )
    resumed = inspect_sentinel_checkpoint(
        root,
        root_sha256,
        str(first["checkpoint_manifest_sha256"]),
    )
    second = write_epoch_checkpoint(
        resumed,
        document_embeddings=documents,
        query_embeddings=queries,
        runtime_inventory=_runtime("same-process"),
    )

    with pytest.raises(TeacherEvidenceError) as error:
        inspect_sentinel_checkpoint(
            root,
            root_sha256,
            str(second["checkpoint_manifest_sha256"]),
        )

    assert error.value.code == "SENTINEL_PROCESS_RESTART_REQUIRED"
