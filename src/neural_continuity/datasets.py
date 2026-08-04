from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CandidateDocument:
    document_id: str
    text: str


@dataclass(frozen=True)
class RetrievalQuery:
    query_id: str
    query: str
    candidate_documents: list[CandidateDocument]
    relevant_document_ids: list[str]


@dataclass(frozen=True)
class RetrievalFixture:
    fixture_id: str
    name: str
    description: str
    queries: list[RetrievalQuery]


def _canonical_payload(fixture: RetrievalFixture) -> dict[str, Any]:
    return {
        "fixture_id": fixture.fixture_id,
        "name": fixture.name,
        "description": fixture.description,
        "queries": [
            {
                "query_id": q.query_id,
                "query": q.query,
                "candidate_documents": [
                    {"document_id": d.document_id, "text": d.text} for d in q.candidate_documents
                ],
                "relevant_document_ids": list(q.relevant_document_ids),
            }
            for q in fixture.queries
        ],
    }


def fixture_identity(fixture: RetrievalFixture) -> str:
    payload = json.dumps(_canonical_payload(fixture), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _decode_query(raw: dict[str, Any]) -> RetrievalQuery:
    return RetrievalQuery(
        query_id=str(raw["query_id"]),
        query=str(raw["query"]),
        candidate_documents=[
            CandidateDocument(
                document_id=str(doc["document_id"]),
                text=str(doc["text"]),
            )
            for doc in raw["candidate_documents"]
        ],
        relevant_document_ids=[str(rid) for rid in raw["relevant_document_ids"]],
    )


def fixture_payload(fixture: RetrievalFixture) -> dict[str, Any]:
    return _canonical_payload(fixture)


def fixture_from_payload(payload: Mapping[str, Any]) -> RetrievalFixture:
    queries = payload.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("fixture payload must include a non-empty query list")
    decoded_queries = [_decode_query(query) for query in queries]
    return RetrievalFixture(
        fixture_id=str(payload.get("fixture_id", "")),
        name=str(payload.get("name", "unnamed")),
        description=str(payload.get("description", "")),
        queries=decoded_queries,
    )


def load_retrieval_fixture(path: str | Path) -> RetrievalFixture:
    payload = Path(path)
    with payload.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    queries_raw = raw["queries"]
    if not isinstance(queries_raw, list) or not queries_raw:
        raise ValueError("fixture must include non-empty queries")

    queries = [_decode_query(raw_query) for raw_query in queries_raw]
    return RetrievalFixture(
        fixture_id=str(raw["fixture_id"]),
        name=str(raw.get("name", "unnamed")),
        description=str(raw.get("description", "")),
        queries=queries,
    )
