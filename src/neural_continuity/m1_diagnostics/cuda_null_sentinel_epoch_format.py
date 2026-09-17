"""Frozen shape and provider rules for one non-qualifying CUDA sentinel epoch."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from neural_continuity.m1_diagnostics.cuda_null_preflight_readiness import CONFIG_SHA256
from neural_continuity.m1_diagnostics.cuda_null_sentinel_readiness import (
    DOCUMENT_COUNT,
    EMBEDDING_DIMENSION,
    EPOCH_COUNT,
    EPOCH_LAYOUT,
    QUERY_COUNT,
    SOURCE_PREFLIGHT_MANIFEST_SHA256,
)
from neural_continuity.m1_diagnostics.cuda_null_source_preflight_inputs import (
    DATASET_MANIFEST_SHA256,
    SOURCE_ONNX_SHA256,
)

FORMAT_VERSION = "1.0.0"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")
ROLES = ("documents", "measurement_null_queries")


class CudaNullSentinelEpochBlocked(ValueError):
    """An epoch package is malformed or exceeds its non-qualifying scope."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise CudaNullSentinelEpochBlocked(reason)


def _ordered_ids(value: object, count: int, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != count
    ):
        raise CudaNullSentinelEpochBlocked(f"{label} identities are missing or duplicated")
    return value


def validate_epoch_plan(plan: Mapping[str, Any]) -> None:
    expected_keys = {
        "kind",
        "version",
        "phase_id",
        "epoch_number",
        "previous_checkpoint_sha256",
        "config_sha256",
        "source_preflight_manifest_sha256",
        "dataset_manifest_sha256",
        "source_onnx_sha256",
        "document_ids",
        "query_ids",
        "query_role",
        "runs",
        "qualifying_detection_evidence",
        "scientific_decision",
        "full_corpus_execution",
        "int8_execution",
    }
    _require(set(plan) == expected_keys, "epoch plan field set differs from frozen format")
    _require(
        plan["kind"] == "m1_cuda_null_sentinel_epoch_plan"
        and plan["version"] == FORMAT_VERSION
        and plan["phase_id"] == "tensor_sentinel_preflight"
        and type(plan["epoch_number"]) is int
        and 1 <= plan["epoch_number"] <= EPOCH_COUNT
        and isinstance(plan["previous_checkpoint_sha256"], str)
        and SHA256_PATTERN.fullmatch(plan["previous_checkpoint_sha256"]) is not None
        and plan["config_sha256"] == CONFIG_SHA256
        and plan["source_preflight_manifest_sha256"] == SOURCE_PREFLIGHT_MANIFEST_SHA256
        and plan["dataset_manifest_sha256"] == DATASET_MANIFEST_SHA256
        and plan["source_onnx_sha256"] == SOURCE_ONNX_SHA256
        and plan["query_role"] == "measurement_null"
        and plan["runs"] == [{"label": label, "batch_size": batch} for label, batch in EPOCH_LAYOUT]
        and plan["qualifying_detection_evidence"] is False
        and plan["scientific_decision"] == "NOT_EVALUATED"
        and plan["full_corpus_execution"] is False
        and plan["int8_execution"] is False,
        "epoch plan differs from the frozen non-qualifying design",
    )
    _ordered_ids(plan["document_ids"], DOCUMENT_COUNT, "document")
    queries = _ordered_ids(plan["query_ids"], QUERY_COUNT, "query")
    _require(
        queries == sorted(queries, key=lambda item: item.encode("utf-8")),
        "query identities are not in canonical UTF-8 order",
    )


def _validate_profile(
    profile: Mapping[str, Any],
    *,
    label: str,
    role: str,
    batch_size: int,
    observation: np.ndarray,
) -> tuple[int, int]:
    expected_keys = {
        "run_label",
        "role",
        "batch_size",
        "item_count",
        "observation_sha256",
        "provider_event_counts",
        "operator_event_counts",
        "cpu_fallback_operator_types",
        "unclassified_cpu_events",
        "undeclared_providers",
    }
    _require(set(profile) == expected_keys, f"profile field set mismatch: {label}/{role}")
    digest = hashlib.sha256(np.ascontiguousarray(observation).tobytes()).hexdigest()
    _require(
        profile["run_label"] == label
        and profile["role"] == role
        and type(profile["batch_size"]) is int
        and profile["batch_size"] == batch_size
        and type(profile["item_count"]) is int
        and profile["item_count"] == observation.shape[0]
        and profile["observation_sha256"] == digest,
        f"profile identity or observation hash mismatch: {label}/{role}",
    )
    counts = profile["provider_event_counts"]
    operators = profile["operator_event_counts"]
    _require(
        isinstance(counts, dict)
        and isinstance(operators, dict)
        and set(counts) == set(operators)
        and set(counts).issubset(PROVIDERS)
        and type(counts.get(PROVIDERS[0])) is int
        and counts[PROVIDERS[0]] > 0,
        f"CUDA provider activity is missing: {label}/{role}",
    )
    for provider, count in counts.items():
        by_operator = operators[provider]
        _require(
            type(count) is int
            and count > 0
            and isinstance(by_operator, dict)
            and bool(by_operator)
            and all(
                isinstance(name, str)
                and bool(name)
                and name != "UNCLASSIFIED"
                and type(events) is int
                and events > 0
                for name, events in by_operator.items()
            )
            and sum(by_operator.values()) == count,
            f"provider/operator classification mismatch: {label}/{role}",
        )
    _require(
        profile["cpu_fallback_operator_types"] == sorted(operators.get("CPUExecutionProvider", {}))
        and profile["unclassified_cpu_events"] == 0
        and profile["undeclared_providers"] == [],
        f"CPU fallback or undeclared provider mismatch: {label}/{role}",
    )
    return counts[PROVIDERS[0]], counts.get(PROVIDERS[1], 0)


def technical_summary(
    plan: Mapping[str, Any],
    runtime: Mapping[str, Any],
    document_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    profiles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute structural evidence; this does not authenticate GPU execution."""
    validate_epoch_plan(plan)
    _require(
        set(runtime)
        == {
            "process_instance_id",
            "runtime_identity_sha256",
            "session_providers",
            "source_only",
            "onnx_graph_loaded",
            "session_created",
            "model_execution_used",
            "int8_executed",
            "holdout_accessed",
        }
        and isinstance(runtime["process_instance_id"], str)
        and bool(runtime["process_instance_id"])
        and isinstance(runtime["runtime_identity_sha256"], str)
        and SHA256_PATTERN.fullmatch(runtime["runtime_identity_sha256"]) is not None
        and runtime["session_providers"] == list(PROVIDERS)
        and runtime["source_only"] is True
        and runtime["onnx_graph_loaded"] is True
        and runtime["session_created"] is True
        and runtime["model_execution_used"] is True
        and runtime["int8_executed"] is False
        and runtime["holdout_accessed"] is False,
        "runtime inventory claims differ from source-only CUDA scope",
    )
    for array, count, role in (
        (document_embeddings, DOCUMENT_COUNT, ROLES[0]),
        (query_embeddings, QUERY_COUNT, ROLES[1]),
    ):
        _require(
            isinstance(array, np.ndarray)
            and array.dtype == np.dtype("<f4")
            and array.shape == (len(EPOCH_LAYOUT), count, EMBEDDING_DIMENSION)
            and np.isfinite(array).all(),
            f"invalid float32 sentinel observations: {role}",
        )
        norms = np.linalg.norm(array.astype(np.float64), axis=2)
        _require(
            bool(np.isfinite(norms).all() and np.all(np.abs(norms - 1.0) <= 1e-4)),
            f"sentinel observations are not L2 normalized: {role}",
        )
    _require(len(profiles) == len(EPOCH_LAYOUT) * len(ROLES), "sentinel profile set is incomplete")
    event_counts: list[dict[str, Any]] = []
    for run_index, (label, batch_size) in enumerate(EPOCH_LAYOUT):
        for role_index, role in enumerate(ROLES):
            profile = profiles[run_index * len(ROLES) + role_index]
            _require(isinstance(profile, Mapping), "sentinel profile is malformed")
            array = document_embeddings if role == ROLES[0] else query_embeddings
            cuda_events, cpu_events = _validate_profile(
                profile,
                label=label,
                role=role,
                batch_size=batch_size,
                observation=array[run_index],
            )
            event_counts.append(
                {
                    "run_label": label,
                    "role": role,
                    "cuda_operator_events": cuda_events,
                    "cpu_operator_events": cpu_events,
                }
            )
    return {
        "kind": "m1_cuda_null_sentinel_epoch_technical_summary",
        "version": FORMAT_VERSION,
        "epoch_number": plan["epoch_number"],
        "run_count": len(EPOCH_LAYOUT),
        "observation_count": len(profiles),
        "document_count": DOCUMENT_COUNT,
        "query_count": QUERY_COUNT,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "provider_event_counts": event_counts,
        "qualifying_detection_evidence": False,
        "scientific_decision": "NOT_EVALUATED",
        "authority_verified": False,
        "checkpoint_chain_verified": False,
    }
