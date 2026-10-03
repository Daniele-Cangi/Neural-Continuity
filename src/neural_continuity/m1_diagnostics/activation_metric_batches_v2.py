"""Adapt verified capture batches to the canonical M1-B v2 metric engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from neural_continuity.m1_diagnostics.activation_analysis_authority import ActivationAnalysisError
from neural_continuity.m1_diagnostics.activation_analysis_metrics import (
    _array_key,
    validate_activation_batch,
    verified_activation_batch_paths,
)
from neural_continuity.m1_diagnostics.activation_metrics_v2 import (
    combine_activation_metric_rows_v2,
    compute_activation_metrics_v2,
)
from neural_continuity.m1_diagnostics.authority import DiagnosticPreflightError


@dataclass(frozen=True)
class ProbeMetricLayout:
    masking: Literal["none", "sequence", "attention_scores"]
    sequence_axis: int | None = None
    feature_axis: int | None = None


def _blocked(code: str, message: str) -> DiagnosticPreflightError:
    return DiagnosticPreflightError(status="BLOCKED", code=code, message=message)


def analyze_captured_activation_metrics_v2(
    root: Path,
    capture_plan: Mapping[str, Any],
    batch_index: Mapping[str, Any],
    layouts_by_probe_id: Mapping[str, ProbeMetricLayout],
    attention_masks_by_batch_id: Mapping[str, np.ndarray] | None = None,
    *,
    declared_shapes_by_probe_id: Mapping[str, tuple[int | str, ...]] | None = None,
) -> dict[str, Any]:
    """Compute canonical metrics from captured arrays, with explicit mask semantics.

    The caller must verify capture integrity and the structural layout authority
    before calling this adapter. A sequence probe without a recorded mask blocks.
    """
    mappings = capture_plan.get("probe_mappings")
    integer_mappings = capture_plan.get("integer_mappings")
    if not isinstance(mappings, list) or not isinstance(integer_mappings, list):
        raise _blocked("METRIC_CAPTURE_PLAN_INVALID", "Capture probe mappings are missing")

    def validated_ids(records: list[Any]) -> list[str]:
        identities: list[str] = []
        for record in records:
            value = record.get("probe_id") if isinstance(record, Mapping) else None
            if not isinstance(value, str) or not value or value in identities:
                raise _blocked("METRIC_CAPTURE_PLAN_INVALID", "Probe identities are invalid")
            identities.append(value)
        return identities

    probe_ids = validated_ids(mappings)
    integer_ids = validated_ids(integer_mappings)
    if (
        len(probe_ids) != len(mappings)
        or len(integer_ids) != len(integer_mappings)
        or any(not isinstance(value, str) or not value for value in probe_ids + integer_ids)
        or len(set(probe_ids)) != len(probe_ids)
        or set(layouts_by_probe_id) != set(probe_ids)
        or any(not isinstance(value, ProbeMetricLayout) for value in layouts_by_probe_id.values())
    ):
        raise _blocked(
            "METRIC_LAYOUT_IDENTITY_MISMATCH", "Metric layouts do not match captured probes"
        )
    masks = attention_masks_by_batch_id or {}
    requires_masks = any(layout.masking != "none" for layout in layouts_by_probe_id.values())
    try:
        batch_paths = verified_activation_batch_paths(root, batch_index)
    except ActivationAnalysisError as exc:
        raise _blocked("METRIC_CAPTURE_BATCH_INVALID", str(exc)) from exc
    expected_batch_ids = {str(record.get("batch_id")) for _, _, record in batch_paths}
    if (requires_masks and set(masks) != expected_batch_ids) or (not requires_masks and masks):
        raise _blocked(
            "METRIC_MASK_SET_MISMATCH", "Mask batch identities are incomplete or undeclared"
        )

    rows_by_probe: dict[str, list[dict[str, Any]]] = {probe_id: [] for probe_id in probe_ids}
    observed_query_ids: list[str] = []
    for floating_path, integer_path, record in batch_paths:
        batch_id = str(record.get("batch_id"))
        try:
            with (
                np.load(floating_path, allow_pickle=False) as floating,
                np.load(integer_path, allow_pickle=False) as integer,
            ):
                query_ids = validate_activation_batch(
                    floating,
                    integer,
                    floating_path,
                    integer_path,
                    record,
                    probe_ids,
                    integer_ids,
                )
                observed_query_ids.extend(query_ids)
                for probe_id in probe_ids:
                    layout = layouts_by_probe_id[probe_id]
                    source = floating[_array_key("source", probe_id)]
                    target = floating[_array_key("target", probe_id)]
                    if declared_shapes_by_probe_id is not None:
                        shape = declared_shapes_by_probe_id.get(probe_id)
                        if (
                            shape is None
                            or len(shape) != source.ndim
                            or any(
                                isinstance(dim, int) and source.shape[axis] != dim
                                for axis, dim in enumerate(shape)
                            )
                        ):
                            raise _blocked(
                                "METRIC_DECLARED_SHAPE_MISMATCH",
                                "Captured tensor disagrees with its frozen graph declaration",
                            )
                    if layout.masking == "none":
                        if (
                            source.ndim != 2
                            or layout.feature_axis != 1
                            or layout.sequence_axis is not None
                        ):
                            raise _blocked(
                                "METRIC_LAYOUT_AMBIGUOUS",
                                "Unmasked probes require a declared two-dimensional feature layout",
                            )
                        mask = None
                    else:
                        mask = masks[batch_id]
                        if layout.masking == "sequence" and layout.sequence_axis is None:
                            raise _blocked(
                                "METRIC_LAYOUT_AMBIGUOUS",
                                "Sequence probe has no structural sequence axis",
                            )
                        if layout.masking == "attention_scores" and (
                            layout.sequence_axis is not None or layout.feature_axis is not None
                        ):
                            raise _blocked(
                                "METRIC_LAYOUT_AMBIGUOUS",
                                "Attention-score layout has conflicting axes",
                            )
                    result = compute_activation_metrics_v2(
                        probe_id=probe_id,
                        query_ids=query_ids,
                        source=source,
                        target=target,
                        attention_mask=mask,
                        sequence_axis=layout.sequence_axis,
                        feature_axis=layout.feature_axis,
                        attention_scores=layout.masking == "attention_scores",
                    )
                    rows_by_probe[probe_id].extend(result["per_query"])
        except (OSError, ValueError, KeyError, ActivationAnalysisError) as exc:
            raise _blocked("METRIC_CAPTURE_BATCH_INVALID", str(exc)) from exc

    if len(observed_query_ids) != capture_plan.get("query_count") or len(
        set(observed_query_ids)
    ) != len(observed_query_ids):
        raise _blocked(
            "METRIC_QUERY_IDENTITY_MISMATCH", "Captured query identities are incomplete or repeated"
        )
    return {
        "kind": "m1-b-v2-canonical-activation-metrics",
        "status": "METRICS_COMPUTED",
        "probe_count": len(probe_ids),
        "query_count": len(observed_query_ids),
        "probes": [
            combine_activation_metric_rows_v2(probe_id=probe_id, rows=rows_by_probe[probe_id])
            for probe_id in probe_ids
        ],
        "model_execution_used": False,
        "scientific_decision_recomputed": False,
    }
