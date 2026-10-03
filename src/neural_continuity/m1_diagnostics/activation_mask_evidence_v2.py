"""Hash-bound, model-free replay of reconstructed M1-B v2 attention masks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.activation_analysis_authority import (
    verify_activation_analysis_input,
)
from neural_continuity.m1_diagnostics.authority import DiagnosticPreflightError
from neural_continuity.m1_diagnostics.fidelity_authority import verify_fidelity_authority
from neural_continuity.m1_diagnostics.fidelity_control import (
    BATCH_SIZE,
    CONFIG_SHA256,
    DATASET_ID,
    DATASET_MANIFEST_SHA256,
    QUERY_COUNT,
    ROLE_MEMBERSHIP_SHA256,
    _canonical_queries,
    _token_inputs,
    _verify_dataset_identity,
    _verify_tokenizer,
)

MASK_SCHEMA_VERSION = "m1-b-v2-derived-attention-masks-1"
_TOKENIZER_IDENTITY_FIELDS = (
    "model_id",
    "revision",
    "device",
    "cache_only",
    "snapshot_files",
)


@dataclass(frozen=True)
class VerifiedAttentionMasks:
    artifact_sha256: str
    source_activation_manifest_sha256: str
    masks_by_batch_id: dict[str, np.ndarray]


def _blocked(code: str, message: str) -> DiagnosticPreflightError:
    return DiagnosticPreflightError(status="BLOCKED", code=code, message=message)


def _mask_rows(value: Any, query_count: int) -> np.ndarray:
    if not isinstance(value, list) or len(value) != query_count or not value:
        raise _blocked("MASK_ROWS_INVALID", "Mask rows do not match the declared queries")
    widths = {len(row) for row in value if isinstance(row, list)}
    if (
        len(widths) != 1
        or 0 in widths
        or any(
            not isinstance(row, list)
            or any(type(item) is not int or item not in (0, 1) for item in row)
            for row in value
        )
    ):
        raise _blocked("MASK_ROWS_INVALID", "Attention masks must be rectangular binary arrays")
    return np.asarray(value, dtype=np.int64)


def capture_reconstructed_attention_masks(
    *,
    source_bundle: str | Path,
    source_manifest_sha256: str,
    config_path: str | Path,
    dataset_directory: str | Path,
    instrumentation_directory: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Record the frozen tokenizer's masks without executing either ONNX graph."""
    source = verify_activation_analysis_input(source_bundle, source_manifest_sha256)
    if source.capture_plan.get("batch_size") != BATCH_SIZE:
        raise _blocked("MASK_BATCH_POLICY_MISMATCH", "Capture batch size is not frozen")
    dataset, role_record = _verify_dataset_identity(Path(dataset_directory).resolve())
    instrumentation = verify_fidelity_authority(instrumentation_directory)
    teacher, tokenizer_identity = _verify_tokenizer(Path(config_path).resolve(), instrumentation)
    query_ids, query_texts = _canonical_queries(dataset)
    batches = source.batch_index.get("batches")
    if (
        len(query_ids) != QUERY_COUNT
        or not isinstance(batches, list)
        or role_record.get("membership_sha256") != ROLE_MEMBERSHIP_SHA256
    ):
        raise _blocked("MASK_SOURCE_IDENTITY_MISMATCH", "Frozen query or role identity differs")

    records: list[dict[str, Any]] = []
    for start in range(0, len(query_ids), BATCH_SIZE):
        batch_number = start // BATCH_SIZE
        if batch_number >= len(batches) or not isinstance(batches[batch_number], Mapping):
            raise _blocked("MASK_BATCH_IDENTITY_MISMATCH", "Capture batch declaration is missing")
        declaration = batches[batch_number]
        expected_ids = query_ids[start : start + BATCH_SIZE]
        if declaration.get("query_ids") != expected_ids or not isinstance(
            declaration.get("batch_id"), str
        ):
            raise _blocked("MASK_BATCH_IDENTITY_MISMATCH", "Capture query order differs")
        inputs = _token_inputs(teacher, query_texts[start : start + BATCH_SIZE])
        mask = np.asarray(inputs["attention_mask"])
        if (
            mask.ndim != 2
            or mask.shape[0] != len(expected_ids)
            or mask.shape[1] == 0
            or mask.dtype.kind not in "biu"
            or not np.isin(mask, (0, 1)).all()
        ):
            raise _blocked(
                "MASK_TOKENIZER_OUTPUT_INVALID", "Frozen tokenizer returned an invalid mask"
            )
        records.append(
            {
                "batch_id": declaration["batch_id"],
                "query_ids": expected_ids,
                "attention_mask": mask.astype(np.int64, copy=False).tolist(),
            }
        )
    if len(records) != len(batches):
        raise _blocked("MASK_BATCH_IDENTITY_MISMATCH", "Capture batch set is incomplete")

    payload = {
        "kind": "m1-b-v2-derived-attention-mask-observations",
        "schema_version": MASK_SCHEMA_VERSION,
        "status": "RECONSTRUCTED_INPUTS",
        "source_activation_bundle": str(source.bundle_path),
        "source_activation_manifest_sha256": source.manifest_sha256,
        "source_capture_plan_sha256": sha256_file(source.root / "capture-plan.json"),
        "dataset_id": DATASET_ID,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "role_membership_sha256": ROLE_MEMBERSHIP_SHA256,
        "configuration_sha256": CONFIG_SHA256,
        "tokenizer_identity": {
            key: tokenizer_identity.get(key) for key in _TOKENIZER_IDENTITY_FIELDS
        },
        "query_count": len(query_ids),
        "batch_count": len(records),
        "batches": records,
        "original_capture_mask_equality_verified": False,
        "qualifying_m1_evidence": False,
        "model_execution_used": False,
        "replay_requires_model_execution": False,
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(canonical_json_bytes(payload) + b"\n")
    return {
        "status": "RECONSTRUCTED_INPUTS",
        "artifact_path": str(destination),
        "artifact_sha256": sha256_file(destination),
        "query_count": len(query_ids),
        "batch_count": len(records),
        "qualifying_m1_evidence": False,
        "model_execution_used": False,
    }


def verify_reconstructed_attention_masks(
    artifact_path: str | Path, expected_sha256: str
) -> VerifiedAttentionMasks:
    """Check the pinned artifact and source capture without loading a model."""
    path = Path(artifact_path).resolve()
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise _blocked("MASK_ARTIFACT_HASH_MISMATCH", "Mask artifact hash differs")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _blocked("MASK_ARTIFACT_INVALID", "Mask artifact is not valid JSON") from exc
    if not isinstance(payload, dict) or any(
        payload.get(key) != value
        for key, value in {
            "kind": "m1-b-v2-derived-attention-mask-observations",
            "schema_version": MASK_SCHEMA_VERSION,
            "status": "RECONSTRUCTED_INPUTS",
            "dataset_id": DATASET_ID,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "role_membership_sha256": ROLE_MEMBERSHIP_SHA256,
            "configuration_sha256": CONFIG_SHA256,
            "original_capture_mask_equality_verified": False,
            "qualifying_m1_evidence": False,
            "model_execution_used": False,
            "replay_requires_model_execution": False,
        }.items()
    ):
        raise _blocked("MASK_ARTIFACT_IDENTITY_MISMATCH", "Mask authority identity differs")
    source_bundle = payload.get("source_activation_bundle")
    source_manifest = payload.get("source_activation_manifest_sha256")
    if not isinstance(source_bundle, str) or not isinstance(source_manifest, str):
        raise _blocked("MASK_SOURCE_IDENTITY_MISMATCH", "Source activation identity is missing")
    source = verify_activation_analysis_input(source_bundle, source_manifest)
    if payload.get("source_capture_plan_sha256") != sha256_file(source.root / "capture-plan.json"):
        raise _blocked("MASK_SOURCE_IDENTITY_MISMATCH", "Capture plan hash differs")
    batches = payload.get("batches")
    declarations = source.batch_index.get("batches")
    if (
        not isinstance(batches, list)
        or not isinstance(declarations, list)
        or len(batches) != len(declarations)
        or payload.get("batch_count") != len(declarations)
        or payload.get("query_count") != source.capture_plan.get("query_count")
    ):
        raise _blocked("MASK_BATCH_IDENTITY_MISMATCH", "Mask batch set is incomplete")
    masks: dict[str, np.ndarray] = {}
    for observed, declared in zip(batches, declarations, strict=True):
        if not isinstance(observed, dict) or not isinstance(declared, Mapping):
            raise _blocked("MASK_BATCH_IDENTITY_MISMATCH", "Mask batch record is invalid")
        batch_id = declared.get("batch_id")
        query_ids = declared.get("query_ids")
        if (
            not isinstance(batch_id, str)
            or not isinstance(query_ids, list)
            or observed.get("batch_id") != batch_id
            or observed.get("query_ids") != query_ids
            or batch_id in masks
        ):
            raise _blocked("MASK_BATCH_IDENTITY_MISMATCH", "Mask batch or query IDs differ")
        masks[batch_id] = _mask_rows(observed.get("attention_mask"), len(query_ids))
    return VerifiedAttentionMasks(
        artifact_sha256=expected_sha256,
        source_activation_manifest_sha256=source.manifest_sha256,
        masks_by_batch_id=masks,
    )


def replay_reconstructed_attention_masks(
    artifact_path: str | Path, expected_sha256: str
) -> dict[str, Any]:
    verified = verify_reconstructed_attention_masks(artifact_path, expected_sha256)
    return {
        "status": "PASS",
        "replay_verified": True,
        "artifact_sha256": verified.artifact_sha256,
        "batch_count": len(verified.masks_by_batch_id),
        "qualifying_m1_evidence": False,
        "model_execution_used": False,
    }
