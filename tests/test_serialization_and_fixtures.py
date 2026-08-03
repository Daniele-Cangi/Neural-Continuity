from __future__ import annotations

from pathlib import Path

from neural_continuity.datasets import (
    CandidateDocument,
    RetrievalFixture,
    RetrievalQuery,
    fixture_identity,
)
from neural_continuity.evidence import canonical_json_bytes, sha256_file


def test_canonical_serialization_stable():
    payload = {"b": 2, "a": 1, "nested": {"z": 3, "y": 2}}
    first = canonical_json_bytes(payload)
    second = canonical_json_bytes(payload)
    assert first == second


def test_artifact_hash_detects_mutation(tmp_path: Path):
    p = tmp_path / "artifact.txt"
    p.write_text("first", encoding="utf-8")
    h1 = sha256_file(p)
    p.write_text("second", encoding="utf-8")
    h2 = sha256_file(p)
    assert h1 != h2


def test_fixture_identity_stable(tmp_path: Path):
    fixture = RetrievalFixture(
        fixture_id="fixture-stable",
        name="stable",
        description="stable",
        queries=[
            RetrievalQuery(
                query_id="q1",
                query="q",
                candidate_documents=[
                    CandidateDocument(document_id="d1", text="t1"),
                ],
                relevant_document_ids=["d1"],
            )
        ],
    )
    first = fixture_identity(fixture)
    second = fixture_identity(fixture)
    assert first == second
