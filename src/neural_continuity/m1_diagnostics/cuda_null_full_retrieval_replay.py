"""Canonical measurement-null retrieval replay from verified full-corpus arrays."""

from __future__ import annotations

import hashlib
import io
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.evidence import canonical_json_bytes
from neural_continuity.m1_diagnostics.cuda_null_full_epoch_format import (
    DOCUMENT_IDS_SHA256,
    QRELS_IDENTITY_SHA256,
    QUERY_IDS_SHA256,
)
from neural_continuity.m1_diagnostics.cuda_null_full_raw_replay import replay_full_raw_run

_TOP_K = 10
_DOCUMENT_COUNT = 5183
_QUERY_COUNT = 81


class FullRetrievalReplayBlocked(ValueError):
    """A declared ranking, metric, or frozen retrieval identity differs."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise FullRetrievalReplayBlocked(reason)


def _validate_inputs(
    document_ids: Sequence[str],
    query_ids: Sequence[str],
    qrels: Mapping[str, Sequence[str]],
) -> None:
    _require(
        len(document_ids) == _DOCUMENT_COUNT
        and len(query_ids) == _QUERY_COUNT
        and len(set(document_ids)) == len(document_ids)
        and len(set(query_ids)) == len(query_ids)
        and all(isinstance(item, str) and bool(item) for item in (*document_ids, *query_ids)),
        "frozen retrieval IDs are incomplete or duplicated",
    )
    _require(set(qrels) == set(query_ids), "measurement-null qrels identities differ")
    available = set(document_ids)
    for query_id in query_ids:
        relevant = qrels[query_id]
        _require(
            isinstance(relevant, list | tuple)
            and bool(relevant)
            and len(set(relevant)) == len(relevant)
            and all(isinstance(item, str) and item in available for item in relevant),
            f"measurement-null qrels invalid: {query_id}",
        )
    identities = (
        (list(document_ids), DOCUMENT_IDS_SHA256, "document identities"),
        (list(query_ids), QUERY_IDS_SHA256, "query identities"),
        ({key: list(qrels[key]) for key in query_ids}, QRELS_IDENTITY_SHA256, "qrels identities"),
    )
    for value, expected_sha256, label in identities:
        observed = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
        _require(observed == expected_sha256, f"{label} differ from frozen materialization")


def _load_pinned_array(path: Path, expected_sha256: str, row_count: int) -> np.ndarray:
    """Hash and decode one bounded byte snapshot, never reopen by path for scoring."""
    maximum_bytes = row_count * 384 * 4 + 1024 * 1024
    try:
        with path.open("rb") as stream:
            raw = stream.read(maximum_bytes + 1)
    except OSError as exc:
        raise FullRetrievalReplayBlocked("raw observation cannot be opened") from exc
    _require(len(raw) <= maximum_bytes, "raw observation exceeds memory bound")
    _require(hashlib.sha256(raw).hexdigest() == expected_sha256, "raw observation changed")
    try:
        array = np.load(io.BytesIO(raw), allow_pickle=False)
    except (OSError, ValueError, TypeError) as exc:
        raise FullRetrievalReplayBlocked("pinned observation cannot be decoded") from exc
    _require(
        isinstance(array, np.ndarray)
        and array.dtype == np.dtype("<f4")
        and array.shape == (row_count, 384)
        and bool(array.flags.c_contiguous),
        "pinned observation shape or dtype differs",
    )
    return array


def recompute_full_retrieval(
    root: Path,
    *,
    run_label: str,
    document_ids: Sequence[str],
    query_ids: Sequence[str],
    qrels: Mapping[str, Sequence[str]],
    document_record: dict[str, Any],
    query_record: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay raw integrity, then rank every query with the frozen tie rule."""
    _validate_inputs(document_ids, query_ids, qrels)
    replay_full_raw_run(
        root,
        run_label=run_label,
        role="documents",
        ordered_ids=document_ids,
        record=document_record,
    )
    replay_full_raw_run(
        root,
        run_label=run_label,
        role="measurement_null_queries",
        ordered_ids=query_ids,
        record=query_record,
    )
    documents = _load_pinned_array(
        root / f"observations/{run_label}/documents.npy",
        document_record["array_sha256"],
        _DOCUMENT_COUNT,
    )
    queries = _load_pinned_array(
        root / f"observations/{run_label}/measurement_null_queries.npy",
        query_record["array_sha256"],
        _QUERY_COUNT,
    )
    document_array = np.asarray(document_ids)
    rankings: list[dict[str, Any]] = []
    values: list[dict[str, float]] = []
    for index, query_id in enumerate(query_ids):
        scores = documents @ queries[index]
        _require(bool(np.isfinite(scores).all()), "retrieval scores are non-finite")
        order = np.lexsort((document_array, -scores))[:_TOP_K]
        ranked = [str(document_array[position]) for position in order]
        relevant = set(qrels[query_id])
        hits = [rank + 1 for rank, document_id in enumerate(ranked) if document_id in relevant]
        recall = len(hits) / len(relevant)
        reciprocal_rank = 1.0 / hits[0] if hits else 0.0
        dcg = sum(1.0 / math.log2(rank + 1) for rank in hits)
        ideal_count = min(len(relevant), _TOP_K)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
        rankings.append(
            {
                "query_id": query_id,
                "role": "measurement_null",
                "ranked_document_ids": ranked,
            }
        )
        values.append({"recall_at_k": recall, "mrr_at_k": reciprocal_rank, "ndcg_at_k": ndcg})
    metrics = {
        "top_k": _TOP_K,
        "role": "measurement_null",
        "query_count": _QUERY_COUNT,
        "metrics": {
            name: float(sum(row[name] for row in values) / len(values))
            for name in ("recall_at_k", "mrr_at_k", "ndcg_at_k")
        },
    }
    return rankings, metrics


def replay_full_retrieval(
    root: Path,
    *,
    run_label: str,
    document_ids: Sequence[str],
    query_ids: Sequence[str],
    qrels: Mapping[str, Sequence[str]],
    document_record: dict[str, Any],
    query_record: dict[str, Any],
    recorded_rankings: list[dict[str, Any]],
    recorded_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Compare retained rankings and metrics with model-free recomputation."""
    rankings, metrics = recompute_full_retrieval(
        root,
        run_label=run_label,
        document_ids=document_ids,
        query_ids=query_ids,
        qrels=qrels,
        document_record=document_record,
        query_record=query_record,
    )
    _require(recorded_rankings == rankings, "recorded rankings differ from raw observations")
    _require(recorded_metrics == metrics, "recorded metrics differ from raw observations")
    return {
        "status": "FULL_RETRIEVAL_REPLAY_PASS",
        "run_label": run_label,
        "measurement_null_query_count": _QUERY_COUNT,
        "top_k": _TOP_K,
        "model_loaded": False,
        "scientific_decision": "NOT_EVALUATED",
        "execution_authorized": False,
    }
