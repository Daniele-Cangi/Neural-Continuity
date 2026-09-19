"""Frozen structural schema for one qualifying CUDA full-corpus epoch."""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping
from typing import Any

FORMAT_VERSION = "1.0.0"
EPOCH_COUNT = 120
DOCUMENT_COUNT = 5183
QUERY_COUNT = 81
EMBEDDING_DIMENSION = 384
DATASET_MANIFEST_SHA256 = "0746d98f5e69c6a0ee48ca3f47b342de1d968a877c90df26ffe8f893437fd5de"
SOURCE_ONNX_SHA256 = "5c0d999bd6b5e64e36cad1f61a83ef8e7507d55be49086745780fabb7c648511"
RUN_LAYOUT = (
    ("batch_1_primary", 1),
    ("batch_16_primary", 16),
    ("batch_16_repeat", 16),
    ("batch_64_primary", 64),
)
PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")
ROLES = ("documents", "measurement_null_queries")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class FullEpochFormatBlocked(ValueError):
    """An epoch plan or runtime record differs from the frozen full design."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise FullEpochFormatBlocked(reason)


def _digest(value: Any, label: str) -> None:
    _require(
        isinstance(value, str) and SHA256.fullmatch(value) is not None,
        f"{label} is not a SHA-256 digest",
    )


def _ids(value: Any, count: int, label: str) -> list[str]:
    _require(
        isinstance(value, list)
        and len(value) == count
        and all(isinstance(item, str) and bool(item) for item in value)
        and len(set(value)) == count,
        f"{label} identities are missing or duplicated",
    )
    return value


def validate_full_epoch_plan(plan: Mapping[str, Any], external_authority_sha256: str) -> None:
    expected = {
        "kind",
        "version",
        "phase_id",
        "epoch_number",
        "attempt_number",
        "execution_authority_sha256",
        "previous_completed_epoch_manifest_sha256",
        "dataset_manifest_sha256",
        "source_onnx_sha256",
        "document_ids",
        "query_ids",
        "query_role",
        "qrels",
        "runs",
        "qualifying_detection_evidence",
        "capture_status",
        "scientific_decision",
        "source_only",
        "candidate_or_int8_execution",
        "holdout_access",
        "operational_tolerance_change",
    }
    _digest(external_authority_sha256, "external execution authority")
    _require(set(plan) == expected, "full epoch plan field set differs")
    _require(
        plan["kind"] == "m1_cuda_null_full_corpus_epoch_plan"
        and plan["version"] == FORMAT_VERSION
        and plan["phase_id"] == "full_corpus_qualification"
        and type(plan["epoch_number"]) is int
        and 1 <= plan["epoch_number"] <= EPOCH_COUNT
        and type(plan["attempt_number"]) is int
        and plan["attempt_number"] >= 1
        and plan["execution_authority_sha256"] == external_authority_sha256
        and plan["dataset_manifest_sha256"] == DATASET_MANIFEST_SHA256
        and plan["source_onnx_sha256"] == SOURCE_ONNX_SHA256
        and plan["query_role"] == "measurement_null"
        and plan["runs"]
        == [{"label": label, "batch_size": batch_size} for label, batch_size in RUN_LAYOUT]
        and plan["qualifying_detection_evidence"] is True
        and plan["capture_status"] == "CAPTURED_NOT_DECIDED"
        and plan["scientific_decision"] == "NOT_EVALUATED"
        and plan["source_only"] is True
        and plan["candidate_or_int8_execution"] is False
        and plan["holdout_access"] is False
        and plan["operational_tolerance_change"] is False,
        "full epoch plan scope differs",
    )
    predecessor = plan["previous_completed_epoch_manifest_sha256"]
    if plan["epoch_number"] == 1:
        _require(predecessor is None, "first full epoch has a predecessor")
    else:
        _digest(predecessor, "previous completed epoch manifest")
    documents = _ids(plan["document_ids"], DOCUMENT_COUNT, "document")
    queries = _ids(plan["query_ids"], QUERY_COUNT, "query")
    _require(
        queries == sorted(queries, key=lambda item: item.encode("utf-8")),
        "query identities are not in canonical UTF-8 order",
    )
    qrels = plan["qrels"]
    _require(isinstance(qrels, dict) and list(qrels) == queries, "qrels query order differs")
    document_set = set(documents)
    for query_id in queries:
        relevant = qrels[query_id]
        _require(
            isinstance(relevant, list)
            and bool(relevant)
            and len(set(relevant)) == len(relevant)
            and all(isinstance(item, str) and item in document_set for item in relevant),
            f"qrels differ for query: {query_id}",
        )


def validate_full_epoch_runtime(runtime: Mapping[str, Any]) -> None:
    expected = {
        "process_instance_id",
        "runtime_identity_sha256",
        "session_providers",
        "source_only",
        "onnx_graph_loaded",
        "session_created",
        "model_execution_used",
        "full_corpus_execution",
        "candidate_or_int8_executed",
        "holdout_accessed",
    }
    _require(set(runtime) == expected, "full epoch runtime field set differs")
    process = runtime["process_instance_id"]
    try:
        parsed = uuid.UUID(process)
    except (ValueError, AttributeError) as exc:
        raise FullEpochFormatBlocked("process identity malformed") from exc
    _require(
        str(parsed) == process
        and runtime["session_providers"] == list(PROVIDERS)
        and runtime["source_only"] is True
        and runtime["onnx_graph_loaded"] is True
        and runtime["session_created"] is True
        and runtime["model_execution_used"] is True
        and runtime["full_corpus_execution"] is True
        and runtime["candidate_or_int8_executed"] is False
        and runtime["holdout_accessed"] is False,
        "full epoch runtime scope differs",
    )
    _digest(runtime["runtime_identity_sha256"], "runtime identity")
