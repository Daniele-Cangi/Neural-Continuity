from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.stats import spearmanr

from .bootstrap import bootstrap_ci
from .datasets import RetrievalFixture
from .observations import ModelObservation

MetricOrientation = Literal[
    "higher_is_better",
    "lower_is_better",
    "two_sided_stability",
    "informational_only",
]


@dataclass(frozen=True)
class MetricPolicy:
    metric_id: str
    family: str
    orientation: MetricOrientation
    may_block_promotion: bool
    minimum_null_observations: int
    minimum_candidate_sample_size: int
    comparison_method: str


METRIC_POLICIES_VERSION = "1.0.0"

METRIC_POLICIES: list[MetricPolicy] = [
    MetricPolicy(
        metric_id="recall_at_1",
        family="functional",
        orientation="higher_is_better",
        may_block_promotion=True,
        minimum_null_observations=2,
        minimum_candidate_sample_size=2,
        comparison_method="query_bootstrap",
    ),
    MetricPolicy(
        metric_id="recall_at_5",
        family="functional",
        orientation="higher_is_better",
        may_block_promotion=True,
        minimum_null_observations=2,
        minimum_candidate_sample_size=2,
        comparison_method="query_bootstrap",
    ),
    MetricPolicy(
        metric_id="mean_reciprocal_rank",
        family="functional",
        orientation="higher_is_better",
        may_block_promotion=True,
        minimum_null_observations=2,
        minimum_candidate_sample_size=2,
        comparison_method="query_bootstrap",
    ),
    MetricPolicy(
        metric_id="paired_cosine_drift",
        family="topology",
        orientation="lower_is_better",
        may_block_promotion=True,
        minimum_null_observations=2,
        minimum_candidate_sample_size=2,
        comparison_method="query_bootstrap",
    ),
    MetricPolicy(
        metric_id="nearest_neighbour_overlap_at_k",
        family="topology",
        orientation="higher_is_better",
        may_block_promotion=True,
        minimum_null_observations=2,
        minimum_candidate_sample_size=2,
        comparison_method="query_bootstrap",
    ),
    MetricPolicy(
        metric_id="rank_correlation",
        family="topology",
        orientation="higher_is_better",
        may_block_promotion=True,
        minimum_null_observations=2,
        minimum_candidate_sample_size=2,
        comparison_method="query_bootstrap",
    ),
    MetricPolicy(
        metric_id="latency_p50_ms",
        family="system",
        orientation="lower_is_better",
        may_block_promotion=False,
        minimum_null_observations=2,
        minimum_candidate_sample_size=1,
        comparison_method="query_bootstrap",
    ),
    MetricPolicy(
        metric_id="latency_p95_ms",
        family="system",
        orientation="lower_is_better",
        may_block_promotion=False,
        minimum_null_observations=2,
        minimum_candidate_sample_size=1,
        comparison_method="query_bootstrap",
    ),
    MetricPolicy(
        metric_id="throughput_queries_per_sec",
        family="system",
        orientation="higher_is_better",
        may_block_promotion=False,
        minimum_null_observations=2,
        minimum_candidate_sample_size=1,
        comparison_method="query_bootstrap",
    ),
]

REQUIRED_METRICS = [policy.metric_id for policy in METRIC_POLICIES]
POLICY_BY_ID = {policy.metric_id: policy for policy in METRIC_POLICIES}


def load_metric_policies_from_mapping(mapping: Mapping[str, Any]) -> list[MetricPolicy]:
    raw_policies = mapping.get("metric_policies")
    if not isinstance(raw_policies, list):
        raise ValueError("metric policies payload must include metric_policies list")

    loaded: list[MetricPolicy] = []
    for entry in raw_policies:
        if not isinstance(entry, dict):
            raise ValueError("invalid metric policy entry")
        loaded.append(
            MetricPolicy(
                metric_id=str(entry["metric_id"]),
                family=str(entry["family"]),
                orientation=entry["orientation"],
                may_block_promotion=bool(entry["may_block_promotion"]),
                minimum_null_observations=int(entry["minimum_null_observations"]),
                minimum_candidate_sample_size=int(entry["minimum_candidate_sample_size"]),
                comparison_method=str(entry["comparison_method"]),
            )
        )
    return loaded


def load_metric_policies_from_path(path: Path) -> tuple[list[MetricPolicy], str]:
    if not path.exists():
        return METRIC_POLICIES, "1.0.0"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("metric policies payload must be a mapping")
    policies = load_metric_policies_from_mapping(payload)
    version = str(payload.get("version", "1.0.0"))
    return policies, version


def metric_policies_payload(policies: Sequence[MetricPolicy]) -> list[dict[str, Any]]:
    return [
        {
            "metric_id": policy.metric_id,
            "family": policy.family,
            "orientation": policy.orientation,
            "may_block_promotion": policy.may_block_promotion,
            "minimum_null_observations": policy.minimum_null_observations,
            "minimum_candidate_sample_size": policy.minimum_candidate_sample_size,
            "comparison_method": policy.comparison_method,
        }
        for policy in policies
    ]


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
    source_recall_hits = {k: 0 for k in top_k_values}
    candidate_recall_hits = {k: 0 for k in top_k_values}
    source_rr_by_query: dict[str, float] = {}
    candidate_rr_by_query: dict[str, float] = {}

    source_top_at_1: dict[str, float] = {}
    source_top_at_5: dict[str, float] = {}
    candidate_top_at_1: dict[str, float] = {}
    candidate_top_at_5: dict[str, float] = {}

    regressions: dict[str, list[str]] = {
        "source_correct_candidate_wrong": [],
        "other": [],
    }
    top_k_max = max(top_k_values)

    for query in fixture.queries:
        qid = query.query_id
        source_top = source.query_results[qid].ranked_documents
        candidate_top = candidate.query_results[qid].ranked_documents

        source_top_id = source_top[0] if source_top else None
        candidate_top_id = candidate_top[0] if candidate_top else None
        source_correct = bool(source_top_id and source_top_id in relevant_by_query[qid])
        candidate_correct = bool(candidate_top_id and candidate_top_id in relevant_by_query[qid])

        if source_correct and not candidate_correct:
            regressions["source_correct_candidate_wrong"].append(qid)
        elif source_correct != candidate_correct:
            regressions["other"].append(qid)

        source_rr = _reciprocal_rank(source_top, query.relevant_document_ids)
        candidate_rr = _reciprocal_rank(candidate_top, query.relevant_document_ids)
        source_rr_by_query[qid] = source_rr
        candidate_rr_by_query[qid] = candidate_rr

        source_recall_1 = 1.0 if source_top_id in relevant_by_query[qid] else 0.0
        candidate_recall_1 = 1.0 if candidate_top_id in relevant_by_query[qid] else 0.0
        source_top_at_1[qid] = source_recall_1
        candidate_top_at_1[qid] = candidate_recall_1

        source_recall_5 = (
            1.0
            if any(doc_id in relevant_by_query[qid] for doc_id in source_top[:top_k_max])
            else 0.0
        )
        candidate_recall_5 = (
            1.0
            if any(doc_id in relevant_by_query[qid] for doc_id in candidate_top[:top_k_max])
            else 0.0
        )
        source_top_at_5[qid] = source_recall_5
        candidate_top_at_5[qid] = candidate_recall_5

        for k in top_k_values:
            if any(doc_id in relevant_by_query[qid] for doc_id in source_top[:k]):
                source_recall_hits[k] += 1
            if any(doc_id in relevant_by_query[qid] for doc_id in candidate_top[:k]):
                candidate_recall_hits[k] += 1

    query_count = len(fixture.queries)
    source_recall = {f"recall_at_{k}": source_recall_hits[k] / query_count for k in top_k_values}
    candidate_recall = {
        f"recall_at_{k}": candidate_recall_hits[k] / query_count for k in top_k_values
    }

    source_mean_rr = (
        float(np.mean(list(source_rr_by_query.values()))) if source_rr_by_query else 0.0
    )
    candidate_mean_rr = (
        float(np.mean(list(candidate_rr_by_query.values()))) if candidate_rr_by_query else 0.0
    )

    return {
        "source_recall_by_query_at_1": source_top_at_1,
        "source_recall_by_query_at_5": source_top_at_5,
        "candidate_recall_by_query_at_1": candidate_top_at_1,
        "candidate_recall_by_query_at_5": candidate_top_at_5,
        "source_reciprocal_ranks": source_rr_by_query,
        "candidate_reciprocal_ranks": candidate_rr_by_query,
        "source_recall": source_recall,
        "candidate_recall": candidate_recall,
        "source_mean_reciprocal_rank": source_mean_rr,
        "candidate_mean_reciprocal_rank": candidate_mean_rr,
        "regressions": regressions,
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
    per_query_drift: dict[str, float] = {}
    per_query_overlap: dict[str, float] = {}
    per_query_correlation: dict[str, float] = {}

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
        per_query_drift[query] = float(drift)

        source_rank = source.query_results[query].ranked_documents
        candidate_rank = candidate.query_results[query].ranked_documents
        src_top = source_rank[:topology_k]
        cand_top = candidate_rank[:topology_k]
        overlap = len(set(src_top).intersection(set(cand_top))) / topology_k if topology_k else 1.0
        overlaps.append(overlap)
        per_query_overlap[query] = float(overlap)

        changed_by_query[query] = _changed_neighbors(src_top, cand_top)

        all_docs = sorted(set(source_rank) | set(candidate_rank))
        source_positions = _query_rank_map(source_rank, all_docs)
        candidate_positions = _query_rank_map(candidate_rank, all_docs)
        if len(all_docs) <= 1:
            corr = 1.0
        else:
            corr, _ = spearmanr(source_positions, candidate_positions)
            corr = float(corr) if not math.isnan(corr) else 1.0
        correlations.append(float(corr))
        per_query_correlation[query] = float(corr)

    return {
        "paired_cosine_drift": float(np.mean(drifts)) if drifts else 0.0,
        "nearest_neighbour_overlap_at_k": float(np.mean(overlaps)) if overlaps else 0.0,
        "rank_correlation": float(np.mean(correlations)) if correlations else 1.0,
        "count_changed_nearest_neighbours": {qid: len(v) for qid, v in changed_by_query.items()},
        "changed_nearest_neighbours": changed_by_query,
        "per_query_drift": per_query_drift,
        "per_query_overlap": per_query_overlap,
        "per_query_rank_correlation": per_query_correlation,
    }


def compute_system_metrics(
    _source: ModelObservation, candidate: ModelObservation
) -> dict[str, Any]:
    latencies = candidate.system_metrics.get("latencies_by_query_ms", {})
    values = np.asarray(list(latencies.values()), dtype=float)
    latency_p50 = float(np.quantile(values, 0.5)) if len(values) else 0.0
    latency_p95 = float(np.quantile(values, 0.95)) if len(values) else 0.0
    return {
        "latency_p50_ms": latency_p50,
        "latency_p95_ms": latency_p95,
        "throughput_queries_per_sec": float(candidate.system_metrics["throughput_queries_per_sec"]),
        "process_rss_peak_bytes": int(candidate.system_metrics["process_rss_peak_bytes"]),
        "cuda_peak_allocated_bytes": candidate.system_metrics["cuda_peak_allocated_bytes"],
        "embedding_byte_size": int(candidate.system_metrics["embedding_bytes"]),
        "sample_count": int(candidate.system_metrics["num_queries"]),
        "batch_size": int(candidate.system_metrics["batch_size"]),
        "hardware": str(candidate.system_metrics["hardware"]),
        "latencies_by_query_ms": latencies,
    }


def _metric_value_pairs(
    source_functional: dict[str, Any],
    candidate_functional: dict[str, Any],
    source_topology: dict[str, Any],
    candidate_topology: dict[str, Any],
    source_system: dict[str, Any],
    candidate_system: dict[str, Any],
    policy: MetricPolicy,
) -> tuple[list[float], list[float], dict[str, float], dict[str, float]]:
    if policy.metric_id == "recall_at_1":
        return (
            [float(v) for v in source_functional["source_recall_by_query_at_1"].values()],
            [float(v) for v in candidate_functional["candidate_recall_by_query_at_1"].values()],
            source_functional["source_recall_by_query_at_1"],
            candidate_functional["candidate_recall_by_query_at_1"],
        )
    if policy.metric_id == "recall_at_5":
        return (
            [float(v) for v in source_functional["source_recall_by_query_at_5"].values()],
            [float(v) for v in candidate_functional["candidate_recall_by_query_at_5"].values()],
            source_functional["source_recall_by_query_at_5"],
            candidate_functional["candidate_recall_by_query_at_5"],
        )
    if policy.metric_id == "mean_reciprocal_rank":
        return (
            [float(v) for v in source_functional["source_reciprocal_ranks"].values()],
            [float(v) for v in candidate_functional["candidate_reciprocal_ranks"].values()],
            source_functional["source_reciprocal_ranks"],
            candidate_functional["candidate_reciprocal_ranks"],
        )
    if policy.metric_id == "paired_cosine_drift":
        return (
            [float(v) for v in source_topology["per_query_drift"].values()],
            [float(v) for v in candidate_topology["per_query_drift"].values()],
            source_topology["per_query_drift"],
            candidate_topology["per_query_drift"],
        )
    if policy.metric_id == "nearest_neighbour_overlap_at_k":
        return (
            [float(v) for v in source_topology["per_query_overlap"].values()],
            [float(v) for v in candidate_topology["per_query_overlap"].values()],
            source_topology["per_query_overlap"],
            candidate_topology["per_query_overlap"],
        )
    if policy.metric_id == "rank_correlation":
        return (
            [float(v) for v in source_topology["per_query_rank_correlation"].values()],
            [float(v) for v in candidate_topology["per_query_rank_correlation"].values()],
            source_topology["per_query_rank_correlation"],
            candidate_topology["per_query_rank_correlation"],
        )
    if policy.metric_id == "latency_p50_ms":
        source_p50 = float(source_system["latency_p50_ms"])
        candidate_p50 = float(candidate_system["latency_p50_ms"])
        return (
            [source_p50],
            [candidate_p50],
            {"single": source_p50},
            {"single": candidate_p50},
        )
    if policy.metric_id == "latency_p95_ms":
        source_p95 = float(source_system["latency_p95_ms"])
        candidate_p95 = float(candidate_system["latency_p95_ms"])
        return (
            [source_p95],
            [candidate_p95],
            {"single": source_p95},
            {"single": candidate_p95},
        )
    if policy.metric_id == "throughput_queries_per_sec":
        source_tput = float(source_system["throughput_queries_per_sec"])
        candidate_tput = float(candidate_system["throughput_queries_per_sec"])
        return (
            [source_tput],
            [candidate_tput],
            {"single": source_tput},
            {"single": candidate_tput},
        )

    raise ValueError(f"unsupported metric: {policy.metric_id}")


def _metric_delta_and_uncertainty(
    source_functional: dict[str, Any],
    candidate_functional: dict[str, Any],
    source_topology: dict[str, Any],
    candidate_topology: dict[str, Any],
    source_system: dict[str, Any],
    candidate_system: dict[str, Any],
    policy: MetricPolicy,
    *,
    seed: int,
    bootstrap_samples: int,
    confidence_level: float,
) -> tuple[float, dict[str, Any], list[str]]:
    source_values, candidate_values, source_series, candidate_series = _metric_value_pairs(
        source_functional,
        candidate_functional,
        source_topology,
        candidate_topology,
        source_system,
        candidate_system,
        policy,
    )

    if len(source_values) != len(candidate_values):
        raise ValueError("source and candidate metric samples must be aligned")

    deltas = [candidate_values[i] - source_values[i] for i in range(len(source_values))]
    metric_delta = float(np.mean(deltas)) if deltas else 0.0

    status = "sufficient"
    if len(candidate_values) < policy.minimum_candidate_sample_size:
        status = "insufficient"

    lower = upper = None
    if status == "sufficient":
        if policy.comparison_method != "query_bootstrap":
            raise ValueError(f"unsupported comparison method: {policy.comparison_method}")
        _, lower, upper = bootstrap_ci(
            deltas,
            sample_count=bootstrap_samples,
            confidence=confidence_level,
            seed=seed,
        )

    return (
        metric_delta,
        {
            "estimate": metric_delta,
            "lower_bound": float(lower) if lower is not None else None,
            "upper_bound": float(upper) if upper is not None else None,
            "sample_count": len(candidate_values),
            "comparison_method": policy.comparison_method,
            "status": status,
            "seed": seed,
            "bootstrap_samples": bootstrap_samples,
            "confidence_level": confidence_level,
            "source_values": source_series,
            "candidate_values": candidate_series,
        },
        list(source_series.keys()),
    )


def compare_observations(
    source: ModelObservation,
    candidate: ModelObservation,
    fixture: RetrievalFixture,
    topology_k: int = 5,
    *,
    metric_bootstrap_samples: int = 200,
    metric_bootstrap_seed: int = 11,
    confidence_level: float = 0.99,
    metric_policies: list[MetricPolicy] | None = None,
) -> dict[str, Any]:
    top_k = [1, 5]
    source_functional = compute_functional_metrics(source, source, fixture, top_k)
    candidate_functional = compute_functional_metrics(source, candidate, fixture, top_k)
    source_topology = compute_topology_metrics(source, source, topology_k=topology_k)
    candidate_topology = compute_topology_metrics(source, candidate, topology_k=topology_k)
    source_system = compute_system_metrics(source, source)
    candidate_system = compute_system_metrics(source, candidate)

    source_metrics = {
        "recall_at_1": source_functional["source_recall"]["recall_at_1"],
        "recall_at_5": source_functional["source_recall"]["recall_at_5"],
        "mean_reciprocal_rank": source_functional["source_mean_reciprocal_rank"],
        "paired_cosine_drift": source_topology["paired_cosine_drift"],
        "nearest_neighbour_overlap_at_k": source_topology["nearest_neighbour_overlap_at_k"],
        "rank_correlation": source_topology["rank_correlation"],
        "latency_p50_ms": source_system["latency_p50_ms"],
        "latency_p95_ms": source_system["latency_p95_ms"],
        "throughput_queries_per_sec": source_system["throughput_queries_per_sec"],
        "process_rss_peak_mb": source_system["process_rss_peak_bytes"] / 1024**2,
    }

    candidate_metrics = {
        "recall_at_1": candidate_functional["candidate_recall"]["recall_at_1"],
        "recall_at_5": candidate_functional["candidate_recall"]["recall_at_5"],
        "mean_reciprocal_rank": candidate_functional["candidate_mean_reciprocal_rank"],
        "paired_cosine_drift": candidate_topology["paired_cosine_drift"],
        "nearest_neighbour_overlap_at_k": candidate_topology["nearest_neighbour_overlap_at_k"],
        "rank_correlation": candidate_topology["rank_correlation"],
        "latency_p50_ms": candidate_system["latency_p50_ms"],
        "latency_p95_ms": candidate_system["latency_p95_ms"],
        "throughput_queries_per_sec": candidate_system["throughput_queries_per_sec"],
        "process_rss_peak_mb": candidate_system["process_rss_peak_bytes"] / 1024**2,
    }

    metric_deltas: dict[str, float] = {}
    metric_uncertainty: dict[str, dict[str, Any]] = {}
    policies = metric_policies or METRIC_POLICIES

    for idx, policy in enumerate(policies):
        estimate, uncertainty, _ = _metric_delta_and_uncertainty(
            source_functional,
            candidate_functional,
            source_topology,
            candidate_topology,
            source_system,
            candidate_system,
            policy,
            seed=metric_bootstrap_seed + idx,
            bootstrap_samples=metric_bootstrap_samples,
            confidence_level=confidence_level,
        )
        metric_deltas[policy.metric_id] = estimate
        metric_uncertainty[policy.metric_id] = uncertainty

    return {
        "source": source.model_id,
        "candidate": candidate.model_id,
        "functional": candidate_functional,
        "topology": candidate_topology,
        "system": candidate_system,
        "source_metrics": source_metrics,
        "candidate_metrics": candidate_metrics,
        "metric_deltas": metric_deltas,
        "metric_uncertainty": metric_uncertainty,
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
        "metric_policies": [
            {
                "metric_id": p.metric_id,
                "orientation": p.orientation,
                "family": p.family,
                "may_block_promotion": p.may_block_promotion,
                "minimum_null_observations": p.minimum_null_observations,
                "minimum_candidate_sample_size": p.minimum_candidate_sample_size,
                "comparison_method": p.comparison_method,
            }
            for p in policies
        ],
    }
