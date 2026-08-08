from __future__ import annotations

import numpy as np
import pytest

from neural_continuity.m1_b import pure_comparison
from neural_continuity.m1_teacher_evidence import ROLE_ORDER, TeacherEvidenceError


def _values(target: bool = False) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    query_ids = np.asarray([f"{role}-q" for role in ROLE_ORDER])
    query_roles = np.asarray(list(ROLE_ORDER))
    document_embeddings = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    query_embeddings = np.asarray([[[1.0, 0.0]] * len(query_ids)], dtype=np.float32)
    if target:
        query_embeddings[0, -1] = [0.0, 1.0]
    metadata: dict[str, object] = {
        "dataset_id": "fixture",
        "qrels": {query_id: ["d1"] for query_id in query_ids.tolist()},
    }
    values = {
        "run_ids": np.asarray(["batch-size-0001"]),
        "batch_sizes": np.asarray([1]),
        "document_ids": np.asarray(["d1", "d2"]),
        "query_ids": query_ids,
        "query_roles": query_roles,
        "document_embeddings": document_embeddings,
        "query_embeddings": query_embeddings,
    }
    return metadata, values


def test_comparison_reports_functional_ranking_and_role_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _values()
    target = _values(target=True)
    monkeypatch.setattr(
        pure_comparison,
        "verify_paired_observation_compatibility",
        lambda *_: {"status": "PASS"},
    )
    monkeypatch.setattr(
        pure_comparison,
        "_observations",
        lambda path: source if str(path) == "source" else target,
    )

    result = pure_comparison.compare_paired_observations("source", "target", top_k=2)

    assert result["comparison_state"] == "CAPTURED_NOT_DECIDED"
    assert result["transition_b_decision"] == "NOT_EVALUATED"
    assert result["runs"][0]["functional"]["query_max_abs_delta"] == 1.0
    assert result["runs"][0]["roles"]["final_holdout"]["ranking_change_count"] == 1


def test_comparison_blocks_incompatible_embedding_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _values()
    target_metadata, target_values = _values()
    target_values["query_embeddings"] = target_values["query_embeddings"][:, :-1]
    monkeypatch.setattr(
        pure_comparison,
        "verify_paired_observation_compatibility",
        lambda *_: {"status": "PASS"},
    )
    monkeypatch.setattr(
        pure_comparison,
        "_observations",
        lambda path: source if str(path) == "source" else (target_metadata, target_values),
    )

    with pytest.raises(TeacherEvidenceError) as error:
        pure_comparison.compare_paired_observations("source", "target")

    assert error.value.status == "BLOCKED"
    assert error.value.code == "COMPARISON_OBSERVATION_INVALID"
