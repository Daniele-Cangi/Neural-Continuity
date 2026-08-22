from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.m1_diagnostics.runtime_provenance_authority import (
    RuntimeProvenanceAuthority,
    RuntimeProvenanceError,
)
from neural_continuity.m1_diagnostics.runtime_provenance_environment import (
    historical_runtime_coverage,
)
from neural_continuity.m1_diagnostics.stage0_metrics import (
    _observation,
    _package,
    _run_report,
    build_stage0_control_report,
)

FP32_KIND = "m1_onnx_fp32_source_observation"
INT8_KIND = "m1_onnx_int8_target_observation"


def _envelope(null_report: Mapping[str, Any], family: str) -> Mapping[str, Any]:
    envelopes = null_report.get("empirical_envelopes")
    if not isinstance(envelopes, Mapping):
        raise RuntimeProvenanceError(
            "RUNTIME_PROVENANCE_NULL_ENVELOPE_INVALID",
            "measurement-null empirical envelopes are missing",
        )
    envelope = envelopes.get(family)
    if not isinstance(envelope, Mapping) or envelope.get("family") != family:
        raise RuntimeProvenanceError(
            "RUNTIME_PROVENANCE_NULL_ENVELOPE_INVALID",
            f"measurement-null envelope is missing: {family}",
        )
    count = envelope.get("comparison_count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise RuntimeProvenanceError(
            "RUNTIME_PROVENANCE_NULL_ENVELOPE_INVALID",
            f"measurement-null envelope has no observations: {family}",
        )
    if not isinstance(envelope.get("empirical_maximum"), Mapping) or not isinstance(
        envelope.get("empirical_minimum"), Mapping
    ):
        raise RuntimeProvenanceError(
            "RUNTIME_PROVENANCE_NULL_ENVELOPE_INVALID",
            f"measurement-null envelope bounds are invalid: {family}",
        )
    return envelope


def _run_values(values: Mapping[str, np.ndarray]) -> tuple[list[str], list[int]]:
    run_ids = values.get("run_ids")
    batch_sizes = values.get("batch_sizes")
    if not isinstance(run_ids, np.ndarray) or not isinstance(batch_sizes, np.ndarray):
        raise RuntimeProvenanceError(
            "RUNTIME_PROVENANCE_OBSERVATION_INVALID", "observation run identities are missing"
        )
    ids = [str(value) for value in run_ids.tolist()]
    batches = [int(value) for value in batch_sizes.tolist()]
    if len(ids) != len(batches) or len(ids) < 2 or len(ids) != len(set(ids)):
        raise RuntimeProvenanceError(
            "RUNTIME_PROVENANCE_OBSERVATION_INVALID", "observation run identities are invalid"
        )
    return ids, batches


def _intra_package_report(
    bundle: str | Path, kind: str, envelope: Mapping[str, Any], top_k: int
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    manifest, metadata, values = _package(bundle, kind)
    run_ids, batch_sizes = _run_values(values)
    reference = _observation(values, metadata, 0)
    comparisons = [
        _run_report(
            reference,
            _observation(values, metadata, index),
            f"{run_ids[0]}__{run_ids[index]}",
            batch_sizes[index],
            envelope,
            top_k,
        )
        for index in range(1, len(run_ids))
    ]
    return (
        {
            "package_kind": kind,
            "detection_limit_family": "batch_size_variation",
            "reference_run_id": run_ids[0],
            "comparison_count": len(comparisons),
            "comparisons": comparisons,
            "outcome": (
                "PASS"
                if all(report.get("outcome") == "PASS" for report in comparisons)
                else "BLOCKED"
            ),
        },
        manifest,
    )


def _attribution(
    intra_reports: Mapping[str, Mapping[str, Any]],
    cross_epoch_report: Mapping[str, Any],
    historical_coverage: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_blocked = any(
        intra_reports[name].get("outcome") == "BLOCKED"
        for name in ("baseline_fp32", "baseline_int8")
    )
    fresh_blocked = any(
        intra_reports[name].get("outcome") == "BLOCKED" for name in ("fresh_fp32", "fresh_int8")
    )
    controls = cross_epoch_report.get("controls")
    cross_blocked = isinstance(controls, Mapping) and any(
        isinstance(control, Mapping) and control.get("outcome") == "BLOCKED"
        for control in controls.values()
    )
    history_complete = historical_coverage.get("complete") is True
    if baseline_blocked:
        classification = "FROZEN_BATCH_ENVELOPE_DOES_NOT_COVER_CANONICAL_BASELINE"
        reasons = ["canonical_baseline_batch_variation_exceeds_frozen_envelope"]
    elif fresh_blocked:
        classification = "PROCESS_LOCAL_VARIATION_EXCEEDS_FROZEN_ENVELOPE"
        reasons = ["fresh_batch_variation_exceeds_frozen_envelope"]
    elif cross_blocked and not history_complete:
        classification = "CROSS_EPOCH_DRIFT_WITH_INCOMPLETE_RUNTIME_AUTHORITY"
        reasons = [
            "cross_epoch_repeat_controls_blocked",
            "historical_runtime_identity_incomplete",
        ]
    elif cross_blocked:
        classification = "CROSS_EPOCH_DRIFT_DETECTED"
        reasons = ["cross_epoch_repeat_controls_blocked"]
    else:
        classification = "NO_DRIFT_DETECTED"
        reasons = []
    return {
        "status": "INCONCLUSIVE" if reasons else "PASS",
        "classification": classification,
        "reasons": reasons,
        "causal_runtime_claim_made": False,
        "scientific_regression_recorded": False,
    }


def build_runtime_drift_audit(authority: RuntimeProvenanceAuthority) -> dict[str, Any]:
    batch_envelope = _envelope(authority.null_report, "batch_size_variation")
    baseline_fp32, baseline_fp32_manifest = _intra_package_report(
        authority.baseline_fp32_bundle, FP32_KIND, batch_envelope, authority.top_k
    )
    baseline_int8, baseline_int8_manifest = _intra_package_report(
        authority.baseline_int8_bundle, INT8_KIND, batch_envelope, authority.top_k
    )
    fresh_fp32, _ = _intra_package_report(
        authority.fresh_fp32_bundle, FP32_KIND, batch_envelope, authority.top_k
    )
    fresh_int8, _ = _intra_package_report(
        authority.fresh_int8_bundle, INT8_KIND, batch_envelope, authority.top_k
    )
    cross_epoch = build_stage0_control_report(
        authority.baseline_fp32_bundle,
        authority.fresh_fp32_bundle,
        authority.baseline_int8_bundle,
        authority.fresh_int8_bundle,
        authority.null_report,
    )
    if cross_epoch != authority.stage0_report:
        raise RuntimeProvenanceError(
            "RUNTIME_PROVENANCE_STAGE0_REPORT_MISMATCH",
            "recomputed cross-epoch report differs from Stage 0 evidence",
        )
    coverage = historical_runtime_coverage([baseline_fp32_manifest, baseline_int8_manifest])
    intra = {
        "baseline_fp32": baseline_fp32,
        "baseline_int8": baseline_int8,
        "fresh_fp32": fresh_fp32,
        "fresh_int8": fresh_int8,
    }
    attribution = _attribution(intra, cross_epoch, coverage)
    return {
        "kind": "m1-runtime-provenance-drift-audit",
        "version": "1.1.0",
        "status": "BLOCKED" if attribution["status"] != "PASS" else "PASS",
        "attribution": attribution,
        "historical_runtime_coverage": coverage,
        "intra_package_batch_variation": intra,
        "cross_epoch_repeat_controls": cross_epoch,
        "cross_epoch_report_match": True,
        "frozen_transition_b_v1_scientific_decision": "FAIL",
        "stage_1_gate_status": "BLOCKED",
        "stage_1_execution_started": False,
        "model_execution_used": False,
        "onnx_graph_loaded": False,
        "activation_read": False,
        "operational_tolerances_changed": False,
        "detection_limits_changed": False,
    }
