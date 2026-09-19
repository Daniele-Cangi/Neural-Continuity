"""Model-free pair comparison for replayed CUDA full-corpus source runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.m1_diagnostics.cuda_null_full_retrieval_replay import (
    _load_pinned_array,
    replay_full_retrieval,
)

_ROLE_COUNTS = {"documents": 5183, "measurement_null_queries": 81}
_BLOCK_ROWS = 64
_METRICS = ("recall_at_k", "mrr_at_k", "ndcg_at_k")
_COMPARISON_FIELDS = {
    "left_run_label",
    "right_run_label",
    "document_max_abs_delta",
    "query_max_abs_delta",
    "document_min_cosine_similarity",
    "query_min_cosine_similarity",
    "ranking_change_count",
    "ranking_change_fraction",
    "recall_at_k_absolute_delta",
    "mrr_at_k_absolute_delta",
    "ndcg_at_k_absolute_delta",
}


class FullComparisonReplayBlocked(ValueError):
    """A retained pair comparison differs from model-free recomputation."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise FullComparisonReplayBlocked(reason)


def _tensor_extrema(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    _require(left.shape == right.shape, "paired observation shapes differ")
    maximum = 0.0
    minimum = 1.0
    for start in range(0, left.shape[0], _BLOCK_ROWS):
        left_block = np.asarray(left[start : start + _BLOCK_ROWS], dtype=np.float64)
        right_block = np.asarray(right[start : start + _BLOCK_ROWS], dtype=np.float64)
        maximum = max(maximum, float(np.max(np.abs(left_block - right_block))))
        denominator = np.linalg.norm(left_block, axis=1) * np.linalg.norm(right_block, axis=1)
        _require(bool(np.isfinite(denominator).all() and np.all(denominator > 0)), "invalid norm")
        cosine = np.sum(left_block * right_block, axis=1) / denominator
        _require(bool(np.isfinite(cosine).all()), "non-finite cosine similarity")
        minimum = min(minimum, float(np.min(cosine)))
    return maximum, minimum


def recompute_full_comparison(
    left_root: Path,
    right_root: Path,
    *,
    left_run_label: str,
    right_run_label: str,
    document_ids: Sequence[str],
    query_ids: Sequence[str],
    qrels: Mapping[str, Sequence[str]],
    left_document_record: dict[str, Any],
    left_query_record: dict[str, Any],
    right_document_record: dict[str, Any],
    right_query_record: dict[str, Any],
    left_rankings: list[dict[str, Any]],
    left_metrics: dict[str, Any],
    right_rankings: list[dict[str, Any]],
    right_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Recompute tensor and measurement-null functional differences."""
    replay_full_retrieval(
        left_root,
        run_label=left_run_label,
        document_ids=document_ids,
        query_ids=query_ids,
        qrels=qrels,
        document_record=left_document_record,
        query_record=left_query_record,
        recorded_rankings=left_rankings,
        recorded_metrics=left_metrics,
    )
    replay_full_retrieval(
        right_root,
        run_label=right_run_label,
        document_ids=document_ids,
        query_ids=query_ids,
        qrels=qrels,
        document_record=right_document_record,
        query_record=right_query_record,
        recorded_rankings=right_rankings,
        recorded_metrics=right_metrics,
    )
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for role, count, left_record, right_record in (
        ("documents", _ROLE_COUNTS["documents"], left_document_record, right_document_record),
        (
            "measurement_null_queries",
            _ROLE_COUNTS["measurement_null_queries"],
            left_query_record,
            right_query_record,
        ),
    ):
        arrays[role] = (
            _load_pinned_array(
                left_root / f"observations/{left_run_label}/{role}.npy",
                left_record["array_sha256"],
                count,
            ),
            _load_pinned_array(
                right_root / f"observations/{right_run_label}/{role}.npy",
                right_record["array_sha256"],
                count,
            ),
        )
    document_max, document_cosine = _tensor_extrema(*arrays["documents"])
    query_max, query_cosine = _tensor_extrema(*arrays["measurement_null_queries"])
    _require(
        [row.get("query_id") for row in left_rankings]
        == [row.get("query_id") for row in right_rankings],
        "ranking query order differs",
    )
    changes = sum(
        left["ranked_document_ids"] != right["ranked_document_ids"]
        for left, right in zip(left_rankings, right_rankings, strict=True)
    )
    left_values = left_metrics.get("metrics")
    right_values = right_metrics.get("metrics")
    if (
        not isinstance(left_values, dict)
        or not isinstance(right_values, dict)
        or set(left_values) != set(_METRICS)
        or set(right_values) != set(_METRICS)
    ):
        raise FullComparisonReplayBlocked("retrieval metric set differs")
    return {
        "left_run_label": left_run_label,
        "right_run_label": right_run_label,
        "document_max_abs_delta": document_max,
        "query_max_abs_delta": query_max,
        "document_min_cosine_similarity": document_cosine,
        "query_min_cosine_similarity": query_cosine,
        "ranking_change_count": changes,
        "ranking_change_fraction": changes / len(query_ids),
        **{
            f"{name}_absolute_delta": abs(float(left_values[name]) - float(right_values[name]))
            for name in _METRICS
        },
    }


def replay_full_comparison(
    left_root: Path,
    right_root: Path,
    *,
    recorded_comparison: dict[str, Any],
    **inputs: Any,
) -> dict[str, Any]:
    """Require exact agreement with a retained comparison record."""
    _require(
        isinstance(recorded_comparison, dict) and set(recorded_comparison) == _COMPARISON_FIELDS,
        "comparison schema differs",
    )
    observed = recompute_full_comparison(left_root, right_root, **inputs)
    _require(recorded_comparison == observed, "recorded comparison differs")
    return {
        "status": "FULL_COMPARISON_REPLAY_PASS",
        "comparison": observed,
        "model_loaded": False,
        "scientific_decision": "NOT_EVALUATED",
        "execution_authorized": False,
    }
