from __future__ import annotations

from typing import Any

from neural_continuity.m1_diagnostics.measurement_null_extension_authority import (
    MeasurementNullPlanAuthority,
)

PROCESS_EPOCHS = 120
DISJOINT_RESTART_PAIRS = PROCESS_EPOCHS // 2
BATCH_SIZES = (1, 16, 64)
REPEAT_BATCH_SIZE = 16
PASSES_PER_EPOCH = 4
SENTINEL_DOCUMENT_COUNT = 256
ORDER_STATISTIC_PERCENTILE = 0.95
ORDER_STATISTIC_CONFIDENCE = 0.95


def _maximum_order_confidence(percentile: float, independent_count: int) -> float:
    return 1.0 - percentile**independent_count


def build_measurement_null_extension_plan(
    authority: MeasurementNullPlanAuthority,
) -> dict[str, Any]:
    epoch_confidence_95 = _maximum_order_confidence(0.95, PROCESS_EPOCHS)
    restart_confidence_95 = _maximum_order_confidence(0.95, DISJOINT_RESTART_PAIRS)
    restart_confidence_99 = _maximum_order_confidence(0.99, DISJOINT_RESTART_PAIRS)
    return {
        "kind": "m1-measurement-null-extension-plan",
        "version": "1.0.0",
        "status": "PREREGISTERED_NOT_EXECUTED",
        "authority": {
            "runtime_provenance_bundle": str(authority.provenance_bundle),
            "runtime_provenance_manifest_sha256": authority.provenance_manifest_sha256,
            "required_provenance_status": "BLOCKED",
            "required_attribution_status": "INCONCLUSIVE",
            "required_classification": "FROZEN_BATCH_ENVELOPE_DOES_NOT_COVER_CANONICAL_BASELINE",
        },
        "objective": (
            "estimate source-only numerical detection envelopes across repeated inference, "
            "batch-size variation, and independent process restarts"
        ),
        "scope": {
            "milestone": "M1 measurement integrity support",
            "source_model": "verified Transition A ONNX FP32 only",
            "execution_provider": "CPUExecutionProvider",
            "query_role": "measurement_null",
            "document_population": "frozen materialized corpus",
            "candidate_or_int8_execution_allowed": False,
            "holdout_query_access_allowed": False,
            "stage_1_execution_allowed": False,
            "operational_tolerance_change_allowed": False,
            "existing_evidence_mutation_allowed": False,
        },
        "frozen_design": {
            "process_epoch_count": PROCESS_EPOCHS,
            "batch_sizes": list(BATCH_SIZES),
            "repeat_batch_size": REPEAT_BATCH_SIZE,
            "passes_per_epoch": PASSES_PER_EPOCH,
            "passes_per_phase": PROCESS_EPOCHS * PASSES_PER_EPOCH,
            "total_planned_passes": PROCESS_EPOCHS * PASSES_PER_EPOCH * 2,
            "early_stopping_allowed": False,
            "adaptive_sample_size_allowed": False,
            "resume_requires_epoch_manifest_verification": True,
            "runtime_inventory_required_per_epoch": True,
            "independent_process_required_per_epoch": True,
        },
        "phases": [
            {
                "phase_id": "tensor_sentinel_preflight",
                "qualifying_detection_evidence": False,
                "process_epoch_count": PROCESS_EPOCHS,
                "documents": {
                    "count": SENTINEL_DOCUMENT_COUNT,
                    "selection": "first IDs after domain-separated SHA-256 ordering",
                    "selection_domain": "neural-continuity:m1:null-extension:v1:document",
                    "text_or_qrel_dependent_selection": False,
                },
                "queries": "all and only measurement_null query IDs",
                "advance_gate": (
                    "technical integrity only; numerical results cannot gate or " "tune phase 2"
                ),
            },
            {
                "phase_id": "full_corpus_qualification",
                "qualifying_detection_evidence": True,
                "process_epoch_count": PROCESS_EPOCHS,
                "documents": "all frozen document IDs",
                "queries": "all and only measurement_null query IDs",
                "start_condition": (
                    "phase 1 artifacts complete, integrity verified, and no " "execution error"
                ),
            },
        ],
        "epoch_layout": [
            {"run": "batch_1_primary", "batch_size": 1},
            {"run": "batch_16_primary", "batch_size": 16},
            {"run": "batch_16_repeat", "batch_size": 16},
            {"run": "batch_64_primary", "batch_size": 64},
        ],
        "comparison_families": {
            "repeated_inference": {
                "comparison": "batch_16_primary versus batch_16_repeat within each epoch",
                "independent_epoch_maxima": PROCESS_EPOCHS,
                "envelope": "maximum/minimum over preregistered epoch-level extrema",
            },
            "batch_size_variation": {
                "comparisons": ["1 versus 16", "1 versus 64", "16 versus 64"],
                "cluster_unit": "process epoch",
                "independent_epoch_maxima": PROCESS_EPOCHS,
                "envelope": "maximum/minimum over preregistered epoch-level extrema",
            },
            "process_restart_variation": {
                "comparison": "batch_16_primary across disjoint adjacent epoch pairs",
                "pairing": "(1,2), (3,4), ..., (119,120)",
                "independent_pair_count": DISJOINT_RESTART_PAIRS,
                "envelope": "maximum/minimum over preregistered pair-level extrema",
            },
        },
        "metrics": {
            "tensor": [
                "document_max_abs_delta",
                "query_max_abs_delta",
                "document_min_cosine_similarity",
                "query_min_cosine_similarity",
            ],
            "functional_measurement_null_only": [
                "ranking_change_count",
                "ranking_change_fraction",
                "recall_at_k_absolute_delta",
                "mrr_at_k_absolute_delta",
                "ndcg_at_k_absolute_delta",
            ],
            "role_aggregation": "measurement_null only; no cross-role pooling",
        },
        "finite_sample_statement": {
            "method": "nonparametric maximum order statistic",
            "target_percentile": ORDER_STATISTIC_PERCENTILE,
            "target_confidence": ORDER_STATISTIC_CONFIDENCE,
            "epoch_family_confidence_at_95th": epoch_confidence_95,
            "restart_family_confidence_at_95th": restart_confidence_95,
            "restart_family_confidence_at_99th": restart_confidence_99,
            "ninety_fifth_percentile_claim_supported": (
                epoch_confidence_95 >= ORDER_STATISTIC_CONFIDENCE
                and restart_confidence_95 >= ORDER_STATISTIC_CONFIDENCE
            ),
            "ninety_ninth_percentile_claim_supported": False,
            "prediction_interval_language_allowed": False,
            "independence_assumption_must_be_reported": True,
        },
        "decision_semantics": {
            "missing_epoch_or_artifact": "BLOCKED",
            "runtime_identity_mismatch": "BLOCKED",
            "execution_failure": "EXECUTION_ERROR",
            "scientific_fail_from_null_capture_allowed": False,
            "candidate_comparison_allowed_during_capture": False,
            "stage_1_release": (
                "forbidden until full-corpus completion, model-free replay, independent review, "
                "and a separately frozen detection-limit authority"
            ),
        },
        "required_epoch_artifacts": [
            "epoch-plan.json",
            "runtime-inventory.json",
            "raw-observations.npz",
            "epoch-summary.json",
            "artifact-manifest.json",
        ],
        "required_aggregate_artifacts": [
            "null-envelope-report.json",
            "coverage-statement.json",
            "replay-bundle.json",
            "artifact-manifest.json",
        ],
        "execution_started": False,
        "model_execution_used_for_preregistration": False,
        "candidate_or_holdout_result_selected_design": False,
    }
