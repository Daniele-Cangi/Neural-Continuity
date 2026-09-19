from __future__ import annotations

import hashlib
import uuid

import pytest

from neural_continuity.evidence import canonical_json_bytes
from neural_continuity.m1_diagnostics import cuda_null_full_epoch_format as epoch_format
from neural_continuity.m1_diagnostics.cuda_null_full_epoch_format import (
    DATASET_MANIFEST_SHA256,
    DOCUMENT_COUNT,
    RUN_LAYOUT,
    SOURCE_ONNX_SHA256,
    FullEpochFormatBlocked,
    validate_full_epoch_plan,
    validate_full_epoch_runtime,
)

AUTHORITY = "a" * 64
RUNTIME = "b" * 64


def _plan(epoch: int = 1) -> dict[str, object]:
    documents = [f"d{index:05d}" for index in range(DOCUMENT_COUNT)]
    queries = [f"q{index:03d}" for index in range(81)]
    return {
        "kind": "m1_cuda_null_full_corpus_epoch_plan",
        "version": "1.0.0",
        "phase_id": "full_corpus_qualification",
        "epoch_number": epoch,
        "attempt_number": 1,
        "execution_authority_sha256": AUTHORITY,
        "previous_completed_epoch_manifest_sha256": None if epoch == 1 else "c" * 64,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "source_onnx_sha256": SOURCE_ONNX_SHA256,
        "document_ids": documents,
        "query_ids": queries,
        "query_role": "measurement_null",
        "qrels": {query: [documents[index]] for index, query in enumerate(queries)},
        "runs": [{"label": label, "batch_size": size} for label, size in RUN_LAYOUT],
        "qualifying_detection_evidence": True,
        "capture_status": "CAPTURED_NOT_DECIDED",
        "scientific_decision": "NOT_EVALUATED",
        "source_only": True,
        "candidate_or_int8_execution": False,
        "holdout_access": False,
        "operational_tolerance_change": False,
    }


@pytest.fixture(autouse=True)
def _bind_synthetic_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan()
    monkeypatch.setattr(
        epoch_format,
        "DOCUMENT_IDS_SHA256",
        hashlib.sha256(canonical_json_bytes(plan["document_ids"])).hexdigest(),
    )
    monkeypatch.setattr(
        epoch_format,
        "QUERY_IDS_SHA256",
        hashlib.sha256(canonical_json_bytes(plan["query_ids"])).hexdigest(),
    )
    monkeypatch.setattr(
        epoch_format,
        "QRELS_IDENTITY_SHA256",
        hashlib.sha256(canonical_json_bytes(plan["qrels"])).hexdigest(),
    )


def test_full_epoch_plan_accepts_only_frozen_scope() -> None:
    validate_full_epoch_plan(_plan(), AUTHORITY)
    plan = _plan()
    plan["candidate_or_int8_execution"] = True
    with pytest.raises(FullEpochFormatBlocked, match="scope differs"):
        validate_full_epoch_plan(plan, AUTHORITY)


def test_full_epoch_plan_requires_predecessor_after_first_epoch() -> None:
    plan = _plan(2)
    plan["previous_completed_epoch_manifest_sha256"] = None
    with pytest.raises(FullEpochFormatBlocked, match="previous completed epoch"):
        validate_full_epoch_plan(plan, AUTHORITY)


def test_full_epoch_plan_fails_closed_on_missing_qrel() -> None:
    plan = _plan()
    plan["qrels"].pop("q080")
    with pytest.raises(FullEpochFormatBlocked, match="qrels query order differs"):
        validate_full_epoch_plan(plan, AUTHORITY)


def test_full_epoch_plan_rejects_same_size_substituted_population() -> None:
    plan = _plan()
    plan["document_ids"][0] = "substituted-document"
    plan["qrels"]["q000"] = ["substituted-document"]
    with pytest.raises(FullEpochFormatBlocked, match="frozen materialization"):
        validate_full_epoch_plan(plan, AUTHORITY)


def test_full_epoch_runtime_separates_technical_and_scientific_state() -> None:
    runtime = {
        "process_instance_id": str(uuid.uuid4()),
        "runtime_identity_sha256": RUNTIME,
        "session_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "source_only": True,
        "onnx_graph_loaded": True,
        "session_created": True,
        "model_execution_used": True,
        "full_corpus_execution": True,
        "candidate_or_int8_executed": False,
        "holdout_accessed": False,
    }
    validate_full_epoch_runtime(runtime)
    runtime["candidate_or_int8_executed"] = True
    with pytest.raises(FullEpochFormatBlocked, match="runtime scope differs"):
        validate_full_epoch_runtime(runtime)
