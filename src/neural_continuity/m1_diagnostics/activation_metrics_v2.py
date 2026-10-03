"""Per-query M1-B v2 activation metrics and model-free metric replay."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.m1_diagnostics.authority import DiagnosticPreflightError
from neural_continuity.m1_diagnostics.metric_arithmetic import (
    binary64_hex,
    cosine_similarity,
    l2_norm,
    pairwise_mean,
)

CANONICAL_METRIC_ALGORITHM_VERSION = "m1-b-v2-canonical-metrics-1"


def _blocked(code: str, message: str) -> DiagnosticPreflightError:
    return DiagnosticPreflightError(status="BLOCKED", code=code, message=message)


@lru_cache(maxsize=1)
def canonical_metric_implementation_sha256() -> str:
    paths = (Path(__file__), Path(__file__).with_name("metric_arithmetic.py"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("ascii"))
        content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _canonical_arrays(
    source: np.ndarray, target: np.ndarray, query_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[int, ...]]:
    left = np.asarray(source)
    right = np.asarray(target)
    if left.shape != right.shape or left.ndim < 1:
        raise _blocked(
            "METRIC_LOGICAL_SHAPE_MISMATCH", "Paired tensors must have identical non-scalar shapes"
        )
    if left.dtype.kind not in "fiu" or right.dtype.kind not in "fiu":
        raise _blocked(
            "METRIC_DTYPE_UNSUPPORTED", "Activation tensors must contain real numeric values"
        )
    if len(query_ids) != left.shape[0] or any(
        not isinstance(value, str) or not value for value in query_ids
    ):
        raise _blocked("METRIC_QUERY_IDENTITY_INVALID", "Query IDs must identify every batch row")
    if len(set(query_ids)) != len(query_ids):
        raise _blocked("METRIC_QUERY_IDENTITY_DUPLICATED", "Query IDs must be unique")
    order = tuple(sorted(range(len(query_ids)), key=lambda index: query_ids[index]))
    canonical_ids = tuple(query_ids[index] for index in order)
    with np.errstate(over="ignore", invalid="ignore"):
        left64 = np.array(left, dtype=np.dtype("<f8"), order="C", copy=True)[list(order)]
        right64 = np.array(right, dtype=np.dtype("<f8"), order="C", copy=True)[list(order)]
    if not np.isfinite(left64).all() or not np.isfinite(right64).all():
        raise _blocked("METRIC_INPUT_NONFINITE", "Activation values must be finite before masking")
    return left64, right64, canonical_ids, order


def _valid_mask(
    shape: tuple[int, ...],
    attention_mask: np.ndarray | None,
    sequence_axis: int | None,
    attention_scores: bool,
    batch_order: tuple[int, ...],
) -> np.ndarray:
    if attention_mask is None:
        if sequence_axis is not None or attention_scores:
            raise _blocked(
                "METRIC_ATTENTION_MASK_MISSING",
                "Sequence metrics require the frozen attention mask",
            )
        return np.ones(shape, dtype=np.bool_)
    mask = np.asarray(attention_mask)
    if mask.ndim != 2 or mask.shape[0] != shape[0] or mask.dtype.kind not in "biu":
        raise _blocked(
            "METRIC_ATTENTION_MASK_INVALID",
            "Attention mask must be a binary [batch, sequence] array",
        )
    if not np.isin(mask, (0, 1)).all():
        raise _blocked(
            "METRIC_ATTENTION_MASK_INVALID", "Attention mask values must be exactly zero or one"
        )
    mask = mask[list(batch_order)].astype(np.bool_, copy=False)
    if attention_scores:
        if (
            sequence_axis is not None
            or len(shape) != 4
            or mask.shape[1] != shape[2]
            or mask.shape[1] != shape[3]
        ):
            raise _blocked(
                "METRIC_ATTENTION_SCORE_MASK_AMBIGUOUS",
                "Attention-score tensors require shape [batch, heads, query, key]",
            )
        return np.broadcast_to(mask[:, None, :, None] & mask[:, None, None, :], shape)
    if sequence_axis is None:
        raise _blocked(
            "METRIC_ATTENTION_MASK_AMBIGUOUS",
            "A mask was provided without a structural sequence axis",
        )
    axis = sequence_axis % len(shape)
    if axis == 0 or mask.shape[1] != shape[axis]:
        raise _blocked(
            "METRIC_SEQUENCE_AXIS_INVALID",
            "Structural sequence axis does not match the attention mask",
        )
    reshape = [1] * len(shape)
    reshape[0] = shape[0]
    reshape[axis] = mask.shape[1]
    return np.broadcast_to(mask.reshape(reshape), shape)


def _cosine_values(
    left: np.ndarray,
    right: np.ndarray,
    valid: np.ndarray,
    feature_axis: int | None,
) -> list[float]:
    if feature_axis is None:
        return [cosine_similarity(left[valid].ravel(order="C"), right[valid].ravel(order="C"))]
    axis = feature_axis % (left.ndim + 1)
    if axis == 0 or axis != left.ndim:
        raise _blocked(
            "METRIC_FEATURE_AXIS_INVALID",
            "The structural feature axis must be the final non-batch axis",
        )
    width = left.shape[-1]
    left_rows = left.reshape(-1, width)
    right_rows = right.reshape(-1, width)
    valid_rows = valid.reshape(-1, width)
    if not np.equal(valid_rows, valid_rows[:, :1]).all():
        raise _blocked("METRIC_FEATURE_MASK_AMBIGUOUS", "A feature vector is only partially masked")
    return [
        cosine_similarity(source_row, target_row)
        for source_row, target_row, row_valid in zip(
            left_rows, right_rows, valid_rows[:, 0], strict=True
        )
        if row_valid
    ]


def _aggregate(values: Sequence[str]) -> dict[str, str]:
    decoded = [float.fromhex(value) for value in values]
    return {
        "minimum": binary64_hex(min(decoded)),
        "maximum": binary64_hex(max(decoded)),
        "mean": binary64_hex(pairwise_mean(decoded)),
    }


def compute_activation_metrics_v2(
    *,
    probe_id: str,
    query_ids: Sequence[str],
    source: np.ndarray,
    target: np.ndarray,
    attention_mask: np.ndarray | None = None,
    sequence_axis: int | None = None,
    feature_axis: int | None = None,
    attention_scores: bool = False,
) -> dict[str, Any]:
    """Compute frozen per-query metrics; structural axes must be supplied by inventory."""
    if not probe_id:
        raise _blocked("METRIC_PROBE_ID_INVALID", "Probe ID must be nonempty")
    if attention_scores and feature_axis is not None:
        raise _blocked(
            "METRIC_ATTENTION_SCORE_FEATURE_AXIS_AMBIGUOUS",
            "Attention-score cosine uses the protocol's masked flattened rule",
        )
    left, right, canonical_ids, order = _canonical_arrays(source, target, query_ids)
    valid = _valid_mask(left.shape, attention_mask, sequence_axis, attention_scores, order)
    if feature_axis is not None and (left.ndim < 2 or feature_axis % left.ndim != left.ndim - 1):
        raise _blocked(
            "METRIC_FEATURE_AXIS_INVALID", "Feature axis must be the final non-batch axis"
        )

    rows: list[dict[str, Any]] = []
    for index, query_id in enumerate(canonical_ids):
        query_left = left[index]
        query_right = right[index]
        query_valid = valid[index]
        source_values = [float(value) for value in query_left[query_valid].ravel(order="C")]
        target_values = [float(value) for value in query_right[query_valid].ravel(order="C")]
        if not source_values:
            raise _blocked("METRIC_EMPTY_VALID_SET", "A query has no valid activation elements")
        deltas = [
            target_value - source_value
            for source_value, target_value in zip(source_values, target_values, strict=True)
        ]
        absolute_deltas = [abs(value) for value in deltas]
        if not all(np.isfinite(value) for value in deltas):
            raise _blocked(
                "METRIC_ARITHMETIC_NONFINITE", "Activation subtraction became non-finite"
            )
        delta_norm = l2_norm(deltas)
        source_norm = l2_norm(source_values)
        target_norm = l2_norm(target_values)
        if source_norm == 0.0 and target_norm == 0.0:
            relative_l2: str | None = binary64_hex(0.0)
            relative_status = "FINITE"
        elif source_norm == 0.0:
            relative_l2 = None
            relative_status = "UNBOUNDED_ZERO_REFERENCE"
        else:
            relative_l2 = binary64_hex(delta_norm / source_norm)
            relative_status = "FINITE"
        denominator = max(source_norm, target_norm)
        symmetric_l2 = 0.0 if denominator == 0.0 else delta_norm / denominator
        cosines = _cosine_values(
            query_left,
            query_right,
            query_valid,
            feature_axis,
        )
        if not cosines:
            raise _blocked("METRIC_EMPTY_VALID_SET", "A query has no valid cosine vectors")
        rows.append(
            {
                "query_id": query_id,
                "valid_element_count": len(source_values),
                "finite_source_count": len(source_values),
                "finite_target_count": len(target_values),
                "maximum_absolute_delta": binary64_hex(max(absolute_deltas)),
                "mean_absolute_delta": binary64_hex(pairwise_mean(absolute_deltas)),
                "relative_l2_error": relative_l2,
                "relative_l2_status": relative_status,
                "minimum_cosine_similarity": binary64_hex(min(cosines)),
                "mean_cosine_similarity": binary64_hex(pairwise_mean(cosines)),
                "symmetric_l2": binary64_hex(symmetric_l2),
            }
        )

    return combine_activation_metric_rows_v2(probe_id=probe_id, rows=rows)


def combine_activation_metric_rows_v2(
    *, probe_id: str, rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Assemble canonical aggregates after per-batch metric computation."""
    if not probe_id or not rows:
        raise _blocked("METRIC_ROWS_INVALID", "Probe ID and metric rows must be nonempty")
    ordered_rows = sorted(rows, key=lambda row: row["query_id"])
    query_ids = [row["query_id"] for row in ordered_rows]
    if any(not isinstance(value, str) or not value for value in query_ids) or len(
        set(query_ids)
    ) != len(query_ids):
        raise _blocked("METRIC_QUERY_IDENTITY_INVALID", "Metric rows have invalid query identities")
    relative_values = [row["relative_l2_error"] for row in ordered_rows]
    relative_summary = (
        {"status": "UNBOUNDED_ZERO_REFERENCE", "minimum": None, "maximum": None, "mean": None}
        if any(value is None for value in relative_values)
        else {"status": "FINITE", **_aggregate(relative_values)}
    )
    return {
        "algorithm_version": CANONICAL_METRIC_ALGORITHM_VERSION,
        "implementation_sha256": canonical_metric_implementation_sha256(),
        "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__},
        "probe_id": probe_id,
        "query_ids": query_ids,
        "per_query": ordered_rows,
        "aggregates": {
            "maximum_absolute_delta": _aggregate(
                [row["maximum_absolute_delta"] for row in ordered_rows]
            ),
            "mean_absolute_delta": _aggregate([row["mean_absolute_delta"] for row in ordered_rows]),
            "relative_l2_error": relative_summary,
            "minimum_cosine_similarity": _aggregate(
                [row["minimum_cosine_similarity"] for row in ordered_rows]
            ),
            "mean_cosine_similarity": _aggregate(
                [row["mean_cosine_similarity"] for row in ordered_rows]
            ),
            "symmetric_l2": _aggregate([row["symmetric_l2"] for row in ordered_rows]),
            "finite_source_count": sum(row["finite_source_count"] for row in ordered_rows),
            "finite_target_count": sum(row["finite_target_count"] for row in ordered_rows),
        },
        "model_execution_used": False,
    }


def replay_activation_metrics_v2(
    *,
    recorded_metrics: dict[str, Any],
    declared_algorithm_version: str,
    declared_implementation_sha256: str,
    **observations: Any,
) -> dict[str, Any]:
    """Recompute from recorded arrays and fail closed on identity or bitwise mismatch."""
    actual_hash = canonical_metric_implementation_sha256()
    if declared_algorithm_version != CANONICAL_METRIC_ALGORITHM_VERSION:
        raise _blocked(
            "METRIC_REPLAY_ALGORITHM_MISMATCH", "Declared metric algorithm version is unsupported"
        )
    if declared_implementation_sha256 != actual_hash:
        raise _blocked(
            "METRIC_REPLAY_IMPLEMENTATION_MISMATCH", "Canonical metric implementation hash differs"
        )
    recomputed = compute_activation_metrics_v2(**observations)
    if recomputed["implementation_sha256"] != declared_implementation_sha256:
        raise _blocked(
            "METRIC_REPLAY_IMPLEMENTATION_MISMATCH", "Recomputed implementation identity differs"
        )
    expected_json = json.dumps(
        recorded_metrics, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    actual_json = json.dumps(recomputed, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if expected_json != actual_json:
        raise _blocked(
            "METRIC_REPLAY_RESULT_MISMATCH",
            "Recorded metric artifact differs from model-free recomputation",
        )
    return {
        "status": "PASS",
        "metrics": recomputed,
        "status_match": True,
        "model_execution_used": False,
    }
