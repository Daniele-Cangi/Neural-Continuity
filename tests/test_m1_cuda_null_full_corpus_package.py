from __future__ import annotations

from pathlib import Path

import pytest

from neural_continuity.m1_diagnostics import cuda_null_full_corpus_package as package

AUTHORITY = "a" * 64
MANIFEST = "b" * 64


def _comparison(left: str, right: str) -> dict[str, object]:
    return {
        "left_run_label": left,
        "right_run_label": right,
        "document_max_abs_delta": 0.0,
        "query_max_abs_delta": 0.0,
        "document_min_cosine_similarity": 1.0,
        "query_min_cosine_similarity": 1.0,
        "ranking_change_count": 0,
        "ranking_change_fraction": 0.0,
        "recall_at_k_absolute_delta": 0.0,
        "mrr_at_k_absolute_delta": 0.0,
        "ndcg_at_k_absolute_delta": 0.0,
    }


def test_recompute_builds_all_preregistered_units(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repeated = _comparison("batch_16_primary", "batch_16_repeat")
    batches = [
        _comparison("batch_1_primary", "batch_16_primary"),
        _comparison("batch_1_primary", "batch_64_primary"),
        _comparison("batch_16_primary", "batch_64_primary"),
    ]
    plan = {"document_ids": ["d1"], "query_ids": ["q1"], "qrels": {"q1": ["d1"]}}
    records = [
        {"run_label": "batch_16_primary", "role": "documents"},
        {"run_label": "batch_16_primary", "role": "measurement_null_queries"},
    ]

    monkeypatch.setattr(
        package,
        "replay_full_epoch_package",
        lambda bundle, *_args: {
            "replay_status": "PASS",
            "epoch_number": int(bundle.parent.name.removeprefix("epoch-")),
        },
    )

    def read_json(path: Path):
        if path.name == "epoch-plan.json":
            return plan
        if path.name == "within-epoch-comparisons.json":
            return [
                {"comparison": repeated},
                *({"comparison": comparison} for comparison in batches),
            ]
        if path.name == "run-records.json":
            return records
        if path.name == "rankings.json":
            return {"batch_16_primary": []}
        if path.name == "metrics.json":
            return {"batch_16_primary": {}}
        raise AssertionError(path)

    monkeypatch.setattr(package, "_read_json", read_json)
    monkeypatch.setattr(
        package,
        "recompute_full_comparison",
        lambda *_args, **_kwargs: _comparison("batch_16_primary", "batch_16_primary"),
    )
    declarations = [{"epoch": epoch, "manifest_sha256": MANIFEST} for epoch in range(1, 121)]
    result = package._recompute(tmp_path, AUTHORITY, declarations)
    assert {family: len(units) for family, units in result["units"].items()} == {
        "repeated_inference": 120,
        "batch_size_variation": 120,
        "process_restart_variation": 60,
    }
    assert result["units"]["process_restart_variation"][-1]["unit_number"] == 60


def test_recompute_rejects_incomplete_epoch_coverage(tmp_path: Path) -> None:
    with pytest.raises(package.FullCorpusPackageBlocked, match="coverage incomplete"):
        package._recompute(tmp_path, AUTHORITY, [])
