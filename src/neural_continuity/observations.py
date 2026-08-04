from __future__ import annotations

import os
import platform
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import psutil

try:
    torch: Any = __import__("torch")
except ModuleNotFoundError:  # pragma: no cover - exercised in environments without torch

    class _NoCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def reset_peak_memory_stats(*_args, **_kwargs) -> None:
            return None

        @staticmethod
        def max_memory_allocated(*_args, **_kwargs) -> int:
            return 0

    class _NoTorch:
        cuda = _NoCuda()

    torch = _NoTorch()

from .datasets import RetrievalFixture
from .models import EmbeddingModel


@dataclass(frozen=True)
class QueryObservation:
    query_id: str
    ranked_documents: list[str]
    score_by_document: dict[str, float]


@dataclass(frozen=True)
class ModelObservation:
    run_id: str
    run_label: str
    batch_size: int
    model_id: str
    model_manifest: dict[str, Any]
    query_results: dict[str, QueryObservation]
    query_embeddings: dict[str, list[float]]
    document_embeddings: dict[str, dict[str, list[float]]]
    query_latencies_ms: dict[str, float]
    system_metrics: dict[str, Any]


def chunked(values: Sequence[Any], batch_size: int) -> list[list[Any]]:
    return [list(values[i : i + batch_size]) for i in range(0, len(values), batch_size)]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _sort_candidates(scores: dict[str, float]) -> list[str]:
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def evaluate_model(
    model: EmbeddingModel,
    fixture: RetrievalFixture,
    batch_size: int,
    run_label: str,
    model_manifest: dict[str, Any],
) -> ModelObservation:
    process = psutil.Process(os.getpid())
    start_rss = process.memory_info().rss
    mem_samples = [start_rss]
    query_times: dict[str, float] = {}
    query_results: dict[str, QueryObservation] = {}
    query_embeddings: dict[str, list[float]] = {}
    document_embeddings: dict[str, dict[str, list[float]]] = {}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for query in fixture.queries:
        q_start = time.perf_counter()
        query_vec = np.asarray(
            model.encode([query.query], batch_size=max(1, batch_size))[0], dtype=np.float32
        )
        query_embeddings[query.query_id] = query_vec.tolist()

        doc_texts = [doc.text for doc in query.candidate_documents]
        doc_vectors = np.asarray(
            model.encode(doc_texts, batch_size=max(1, batch_size)), dtype=np.float32
        )
        scores = {
            doc.document_id: _cosine_similarity(query_vec, doc_vectors[idx])
            for idx, doc in enumerate(query.candidate_documents)
        }
        ranked = _sort_candidates(scores)
        query_results[query.query_id] = QueryObservation(
            query_id=query.query_id,
            ranked_documents=ranked,
            score_by_document=scores,
        )
        document_embeddings[query.query_id] = {
            doc.document_id: doc_vectors[idx].tolist()
            for idx, doc in enumerate(query.candidate_documents)
        }
        query_times[query.query_id] = (time.perf_counter() - q_start) * 1000.0
        mem_samples.append(process.memory_info().rss)

    elapsed = (time.perf_counter() - start) * 1000.0
    latencies = np.asarray(list(query_times.values()), dtype=float)
    latency_p50 = float(np.quantile(latencies, 0.5)) if len(latencies) else 0.0
    latency_p95 = float(np.quantile(latencies, 0.95)) if len(latencies) else 0.0
    throughput = len(query_times) / (elapsed / 1000.0) if elapsed > 0 else 0.0

    cuda_peak = None
    if torch.cuda.is_available():
        cuda_peak = int(torch.cuda.max_memory_allocated())

    embedding_bytes = 0
    embedding_bytes += sum(len(v) * 4 for v in query_embeddings.values())
    embedding_bytes += sum(
        len(vec) * 4 for docs in document_embeddings.values() for vec in docs.values()
    )

    system_metrics = {
        "latencies_by_query_ms": query_times,
        "latency_p50_ms": latency_p50,
        "latency_p95_ms": latency_p95,
        "throughput_queries_per_sec": throughput,
        "process_rss_peak_bytes": int(max(mem_samples) - min(mem_samples)),
        "cuda_peak_allocated_bytes": cuda_peak,
        "embedding_bytes": embedding_bytes,
        "num_queries": len(query_times),
        "batch_size": batch_size,
        "hardware": "cuda" if torch.cuda.is_available() else "cpu",
        "elapsed_ms": elapsed,
        "platform": platform.platform(),
    }
    return ModelObservation(
        run_id=str(uuid.uuid4()),
        run_label=run_label,
        batch_size=batch_size,
        model_id=getattr(model, "model_id", "unknown"),
        model_manifest=model_manifest,
        query_results=query_results,
        query_embeddings=query_embeddings,
        document_embeddings=document_embeddings,
        query_latencies_ms=query_times,
        system_metrics=system_metrics,
    )


def observation_to_rows(observation: ModelObservation) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query in observation.query_results.values():
        for rank, doc_id in enumerate(query.ranked_documents, start=1):
            rows.append(
                {
                    "run_id": observation.run_id,
                    "run_label": observation.run_label,
                    "query_id": query.query_id,
                    "doc_id": doc_id,
                    "score": float(query.score_by_document[doc_id]),
                    "rank": rank,
                }
            )
    return rows


def observation_to_manifest(observation: ModelObservation) -> dict[str, Any]:
    return {
        "run_id": observation.run_id,
        "run_label": observation.run_label,
        "batch_size": observation.batch_size,
        "model_id": observation.model_id,
        "model_manifest": observation.model_manifest,
        "query_results": {
            query_id: {
                "query_id": row.query_id,
                "ranked_documents": row.ranked_documents,
                "score_by_document": row.score_by_document,
            }
            for query_id, row in observation.query_results.items()
        },
        "query_embeddings": observation.query_embeddings,
        "document_embeddings": observation.document_embeddings,
        "query_latencies_ms": observation.query_latencies_ms,
        "system_metrics": observation.system_metrics,
    }


def observation_from_manifest(payload: Mapping[str, Any]) -> ModelObservation:
    queries = {
        query_id: QueryObservation(
            query_id=query_id,
            ranked_documents=list(data["ranked_documents"]),
            score_by_document={
                doc_id: float(score) for doc_id, score in data["score_by_document"].items()
            },
        )
        for query_id, data in payload.get("query_results", {}).items()
    }
    return ModelObservation(
        run_id=str(payload["run_id"]),
        run_label=str(payload["run_label"]),
        batch_size=int(payload["batch_size"]),
        model_id=str(payload["model_id"]),
        model_manifest=dict(payload.get("model_manifest", {})),
        query_results=queries,
        query_embeddings={
            query_id: list(vector)
            for query_id, vector in payload.get("query_embeddings", {}).items()
        },
        document_embeddings={
            query_id: {doc_id: list(vector) for doc_id, vector in doc_map.items()}
            for query_id, doc_map in payload.get("document_embeddings", {}).items()
        },
        query_latencies_ms={
            query_id: float(value)
            for query_id, value in payload.get("query_latencies_ms", {}).items()
        },
        system_metrics=dict(payload.get("system_metrics", {})),
    )


def save_raw_observations_parquet(
    observations: list[ModelObservation],
    path,
) -> None:
    rows: list[dict[str, Any]] = []
    for observation in observations:
        rows.extend(observation_to_rows(observation))
    pd.DataFrame(rows).to_parquet(path, index=False)
