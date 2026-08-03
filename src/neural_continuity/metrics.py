from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from .datasets import RetrievalFixture
from .observations import ModelObservation


def _reciprocal_rank(ranking: list[str], relevant: list[str]) -> float:
    relevant_set = set(relevant)
    for idx, doc_id in enumerate(ranking, start=1):
        if doc_id in relevant_set:
            return 1.0 / idx
    return 0.0


def compute_functional_metrics(
    source: ModelObservation,
    candidate: ModelObservation,
    fixture: RetrievalFixture,
    top_k_values: list[int],
) -> dict[str, Any]:
    relevant_by_query = {q.query_id: set(q.relevant_document_ids) for q in fixture.queries}
    recall_hits = {k: 0 for k in top_k_values}
    reciprocal_ranks: dict[str, float] = {}
    per_query_top: dict[str, str | None] = {}
    regressions = {"source_correct_candidate_wrong": [], "other": []}

    for query in fixture.queries:
        qid = query.query_id
        source_top = source.query_results[qid].ranked_documents
        candidate_top = candidate.query_results[qid].ranked_documents
        top_id = candidate_top[0] if candidate_top else None
        per_query_top[qid] = top_id

        source_correct = bool(source_top and source_top[0] in relevant_by_query[qid])
        candidate_correct = bool(top_id and top_id in relevant_by_query[qid])
        if source_correct and not candidate_correct:
            regressions["source_correct_candidate_wrong"].append(qid)
        elif source_correct != candidate_correct:
            regressions["other"].append(qid)

        rr = _reciprocal_rank(candidate_top, query.relevant_document_ids)
        reciprocal_ranks[qid] = rr

        for k in top_k_values:
            if any(doc_id in relevant_by_query[qid] for doc_id in candidate_top[:k]):
                recall_hits[k] += 1

    query_count = len(fixture.queries)
    recall = {f"recall_at_{k}": recall_hits[k] / query_count for k in top_k_values}
    mean_rr = float(np.mean(list(reciprocal_ranks.values()))) if reciprocal_ranks else 0.0

    return {
        "per_query_top_result": per_query_top,
        "recall": recall,
        "mean_reciprocal_rank": mean_rr,
        "reciprocal_ranks": reciprocal_ranks,
        "regressions": regressions,
        "sample_count": query_count,
    }


def _query_rank_map(ranking: list[str], all_docs: list[str]) -> list[int]:
    pos = {doc_id: idx for idx, doc_id in enumerate(ranking)}
    return [pos.get(doc_id, len(all_docs)) for doc_id in all_docs]


def _changed_neighbors(source_top_k: list[str], candidate_top_k: list[str]) -> list[str]:
    return sorted(set(source_top_k).symmetric_difference(set(candidate_top_k)))


def compute_topology_metrics(
    source: ModelObservation,
    candidate: ModelObservation,
    topology_k: int = 5,
) -> dict[str, Any]:
    drifts: list[float] = []
    overlaps: list[float] = []
    correlations: list[float] = []
    changed_by_query: dict[str, list[str]] = {}

    for query in source.query_results:
        src_vec = np.asarray(source.query_embeddings[query], dtype=np.float32)
        cand_vec = np.asarray(candidate.query_embeddings[query], dtype=np.float32)
        if np.linalg.norm(src_vec) == 0 or np.linalg.norm(cand_vec) == 0:
            drift = 0.0
        else:
            drift = 1.0 - float(
                np.dot(src_vec, cand_vec) / (np.linalg.norm(src_vec) * np.linalg.norm(cand_vec))
            )
        if math.isnan(drift):
            drift = 0.0
        drifts.append(float(drift))

        source_rank = source.query_results[query].ranked_documents
        candidate_rank = candidate.query_results[query].ranked_documents
        src_top = source_rank[:topology_k]
        cand_top = candidate_rank[:topology_k]
        overlaps.append(
            len(set(src_top).intersection(set(cand_top))) / topology_k if topology_k else 1.0
        )
        changed_by_query[query] = _changed_neighbors(src_top, cand_top)

        all_docs = sorted(set(source_rank) | set(candidate_rank))
        source_positions = _query_rank_map(source_rank, all_docs)
        candidate_positions = _query_rank_map(candidate_rank, all_docs)
        if len(all_docs) <= 1:
            correlations.append(1.0)
        else:
            corr, _ = spearmanr(source_positions, candidate_positions)
            correlations.append(float(corr) if not math.isnan(corr) else 1.0)

    return {
        "paired_cosine_drift": float(np.mean(drifts)) if drifts else 0.0,
        "nearest_neighbour_overlap_at_k": float(np.mean(overlaps)) if overlaps else 0.0,
        "rank_correlation": float(np.mean(correlations)) if correlations else 1.0,
        "changed_nearest_neighbours": changed_by_query,
        "count_changed_nearest_neighbours": {qid: len(v) for qid, v in changed_by_query.items()},
        "sample_count": len(source.query_results),
    }


def compute_system_metrics(
    _source: ModelObservation, candidate: ModelObservation
) -> dict[str, Any]:
    return {
        "latency_p50_ms": float(candidate.system_metrics["latency_p50_ms"]),
        "latency_p95_ms": float(candidate.system_metrics["latency_p95_ms"]),
        "throughput_queries_per_sec": float(candidate.system_metrics["throughput_queries_per_sec"]),
        "process_rss_peak_bytes": int(candidate.system_metrics["process_rss_peak_bytes"]),
        "cuda_peak_allocated_bytes": candidate.system_metrics["cuda_peak_allocated_bytes"],
        "embedding_byte_size": int(candidate.system_metrics["embedding_bytes"]),
        "sample_count": int(candidate.system_metrics["num_queries"]),
        "batch_size": int(candidate.system_metrics["batch_size"]),
        "hardware": str(candidate.system_metrics["hardware"]),
    }


def compare_observations(
    source: ModelObservation,
    candidate: ModelObservation,
    fixture: RetrievalFixture,
    topology_k: int = 5,
) -> dict[str, Any]:
    top_k = [1, 5]
    source_functional = compute_functional_metrics(source, source, fixture, top_k)
    candidate_functional = compute_functional_metrics(source, candidate, fixture, top_k)
    source_topology = compute_topology_metrics(source, source, topology_k=topology_k)
    candidate_topology = compute_topology_metrics(source, candidate, topology_k=topology_k)
    source_system = compute_system_metrics(source, source)
    candidate_system = compute_system_metrics(source, candidate)

    source_metrics = {
        "recall_at_1": source_functional["recall"]["recall_at_1"],
        "recall_at_5": source_functional["recall"]["recall_at_5"],
        "mean_reciprocal_rank": source_functional["mean_reciprocal_rank"],
        "paired_cosine_drift": source_topology["paired_cosine_drift"],
        "nearest_neighbour_overlap_at_k": source_topology["nearest_neighbour_overlap_at_k"],
        "rank_correlation": source_topology["rank_correlation"],
        "latency_p50_ms": source_system["latency_p50_ms"],
        "latency_p95_ms": source_system["latency_p95_ms"],
        "throughput_queries_per_sec": source_system["throughput_queries_per_sec"],
        "process_rss_peak_mb": source_system["process_rss_peak_bytes"] / 1024**2,
    }

    candidate_metrics = {
        "recall_at_1": candidate_functional["recall"]["recall_at_1"],
        "recall_at_5": candidate_functional["recall"]["recall_at_5"],
        "mean_reciprocal_rank": candidate_functional["mean_reciprocal_rank"],
        "paired_cosine_drift": candidate_topology["paired_cosine_drift"],
        "nearest_neighbour_overlap_at_k": candidate_topology["nearest_neighbour_overlap_at_k"],
        "rank_correlation": candidate_topology["rank_correlation"],
        "latency_p50_ms": candidate_system["latency_p50_ms"],
        "latency_p95_ms": candidate_system["latency_p95_ms"],
        "throughput_queries_per_sec": candidate_system["throughput_queries_per_sec"],
        "process_rss_peak_mb": candidate_system["process_rss_peak_bytes"] / 1024**2,
    }

    metric_deltas = {
        key: abs(candidate_metrics[key] - source_metrics[key]) for key in source_metrics
    }

    return {
        "source": source.model_id,
        "candidate": candidate.model_id,
        "functional": candidate_functional,
        "topology": candidate_topology,
        "system": candidate_system,
        "source_metrics": source_metrics,
        "candidate_metrics": candidate_metrics,
        "metric_deltas": metric_deltas,
        "regressions": candidate_functional["regressions"],
        "affected_samples": {
            "source_correct_candidate_wrong": candidate_functional["regressions"][
                "source_correct_candidate_wrong"
            ],
            "other": candidate_functional["regressions"]["other"],
            "changed_nearest_neighbours": sorted(
                {
                    qid
                    for qid, changed in candidate_topology["changed_nearest_neighbours"].items()
                    if changed
                }
            ),
        },
        "sample_count": len(fixture.queries),
        "changed_nearest_neighbours": candidate_topology["changed_nearest_neighbours"],
    }
