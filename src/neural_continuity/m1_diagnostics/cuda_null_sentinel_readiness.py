"""Non-executing readiness gate for the proposed CUDA measurement-null sentinel."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from neural_continuity.m1_diagnostics.cuda_null_preflight_readiness import (
    CONFIG_PATH,
    CONFIG_SHA256,
    PROTOCOL_PATH,
    PROTOCOL_SHA256,
    CudaNullPreflightBlocked,
    _pinned_file,
)
from neural_continuity.m1_diagnostics.cuda_null_source_preflight_evidence import (
    replay_source_preflight,
)
from neural_continuity.m1_diagnostics.cuda_null_source_preflight_inputs import (
    DATASET_MANIFEST_SHA256,
    SOURCE_ONNX_SHA256,
)

SOURCE_PREFLIGHT_MANIFEST_SHA256 = (
    "9340a7a52ff6502d9ee6639dc3c15d41325c3521c77629778e44f8d7043bd8d1"
)
EPOCH_COUNT = 120
DOCUMENT_COUNT = 256
QUERY_COUNT = 81
EMBEDDING_DIMENSION = 384
EPOCH_LAYOUT = (
    ("batch_1_primary", 1),
    ("batch_16_primary", 16),
    ("batch_16_repeat", 16),
    ("batch_64_primary", 64),
)
SELECTION_DOMAIN = "neural-continuity:m1:null-extension:v1:document"


class CudaNullSentinelReadinessBlocked(ValueError):
    """The frozen proposal or technical preflight did not verify."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise CudaNullSentinelReadinessBlocked(reason)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise CudaNullSentinelReadinessBlocked(f"{label} is missing or malformed")
    return value


def _verify_frozen_design(config: Mapping[str, Any]) -> None:
    scope = _mapping(config.get("scope"), "scope")
    dataset = _mapping(config.get("dataset_authority"), "dataset authority")
    source = _mapping(config.get("source_authority"), "source authority")
    design = _mapping(config.get("design"), "design")
    sentinel = _mapping(design.get("sentinel"), "sentinel design")
    full = _mapping(design.get("full_corpus"), "full-corpus design")
    freeze = _mapping(config.get("freeze_gate"), "freeze gate")
    expected_layout = [
        {"label": label, "batch_size": batch_size} for label, batch_size in EPOCH_LAYOUT
    ]
    _require(config.get("status") == "DRAFT_NOT_EXECUTABLE", "CUDA protocol is not a draft")
    _require(
        scope.get("source_only") is True
        and scope.get("full_corpus_authorized") is False
        and scope.get("candidate_or_int8_execution_allowed") is False
        and scope.get("holdout_execution_allowed") is False
        and scope.get("operational_tolerance_change_allowed") is False
        and scope.get("scientific_decision") == "NOT_EVALUATED",
        "CUDA proposal scope differs from frozen source-only design",
    )
    _require(
        dataset.get("materialization_manifest_sha256") == DATASET_MANIFEST_SHA256
        and dataset.get("document_count") == 5183
        and dataset.get("measurement_null_query_count") == QUERY_COUNT
        and source.get("onnx_fp32_artifact_sha256") == SOURCE_ONNX_SHA256
        and source.get("embedding_dimension") == EMBEDDING_DIMENSION
        and source.get("normalization") == "l2_unit_after_encode",
        "CUDA proposal input identity differs from frozen source",
    )
    _require(
        sentinel.get("process_epoch_count") == EPOCH_COUNT
        and sentinel.get("document_count") == DOCUMENT_COUNT
        and sentinel.get("selection_domain") == SELECTION_DOMAIN
        and sentinel.get("qualifying_detection_evidence") is False
        and full.get("start_authorized") is False
        and design.get("query_role") == "measurement_null"
        and design.get("query_count") == QUERY_COUNT
        and design.get("independent_process_per_epoch") is True
        and design.get("early_stopping_allowed") is False
        and design.get("adaptive_sample_size_allowed") is False
        and design.get("epoch_layout") == expected_layout
        and design.get("restart_pairs") == "adjacent_disjoint_1_2_through_119_120"
        and design.get("target_percentile") == 0.95
        and design.get("target_confidence") == 0.95
        and design.get("ninety_ninth_percentile_claim_allowed") is False
        and design.get("prediction_interval_language_allowed") is False,
        "CUDA sentinel design differs from frozen proposal",
    )
    _require(
        freeze.get("independent_review_complete") is False
        and freeze.get("fail_closed_replay_verified") is False
        and freeze.get("fresh_source_only_preflight_verified") is False,
        "draft freeze gate unexpectedly grants execution",
    )


def verify_cuda_sentinel_readiness(
    *,
    config_path: Path,
    external_config_sha256: str,
    source_preflight_bundle: Path,
    external_source_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify frozen design and source preflight, but never grant execution."""
    _require(
        external_config_sha256 == CONFIG_SHA256,
        "external CUDA-null config SHA-256 mismatch",
    )
    _require(
        external_source_manifest_sha256 == SOURCE_PREFLIGHT_MANIFEST_SHA256,
        "external source-preflight manifest SHA-256 mismatch",
    )
    try:
        _pinned_file(config_path, CONFIG_PATH, CONFIG_SHA256, "CUDA-null configuration")
        _pinned_file(PROTOCOL_PATH, PROTOCOL_PATH, PROTOCOL_SHA256, "qualification protocol")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (CudaNullPreflightBlocked, OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CudaNullSentinelReadinessBlocked("frozen CUDA proposal is unavailable") from exc
    _verify_frozen_design(_mapping(config, "CUDA proposal"))

    replay = replay_source_preflight(source_preflight_bundle, external_source_manifest_sha256)
    _require(
        replay.get("replay_status") == "PASS"
        and replay.get("technical_preflight_status") == "PASS"
        and replay.get("decision_match") is True
        and replay.get("artifact_manifest_sha256") == external_source_manifest_sha256
        and replay.get("scientific_decision") == "NOT_EVALUATED"
        and replay.get("model_loaded") is False
        and replay.get("onnx_graph_loaded") is False,
        "source-only preflight replay did not verify",
    )
    runs = replay.get("run_timings")
    _require(
        isinstance(runs, list)
        and [(run.get("role"), run.get("batch_size")) for run in runs if isinstance(run, dict)]
        == [
            (role, batch)
            for role in ("documents", "measurement_null_queries")
            for batch in (1, 16, 64)
        ],
        "source-only preflight run coverage differs from frozen scope",
    )
    return {
        "status": "SENTINEL_DESIGN_REVIEW_REQUIRED",
        "config_sha256": external_config_sha256,
        "source_preflight_manifest_sha256": external_source_manifest_sha256,
        "source_preflight_replay_status": "PASS",
        "process_epoch_count": EPOCH_COUNT,
        "document_count": DOCUMENT_COUNT,
        "measurement_null_query_count": QUERY_COUNT,
        "raw_observation_bytes_before_overhead": (
            EPOCH_COUNT
            * len(EPOCH_LAYOUT)
            * (DOCUMENT_COUNT + QUERY_COUNT)
            * EMBEDDING_DIMENSION
            * 4
        ),
        "storage_budget_verified": False,
        "runtime_budget_verified": False,
        "independent_review_complete": False,
        "execution_authorized": False,
        "sentinel_started": False,
        "full_corpus_started": False,
        "int8_executed": False,
        "scientific_decision": "NOT_EVALUATED",
        "model_loaded": False,
        "onnx_graph_loaded": False,
    }
