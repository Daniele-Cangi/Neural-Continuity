from __future__ import annotations

import pytest

from neural_continuity.m1_diagnostics.cuda_null_full_extrema import (
    FullExtremaBlocked,
    aggregate_family_extrema,
    batch_epoch_unit,
    single_comparison_unit,
)


def _comparison(left: str, right: str, value: float) -> dict[str, object]:
    return {
        "left_run_label": left,
        "right_run_label": right,
        "document_max_abs_delta": value,
        "query_max_abs_delta": value / 2,
        "document_min_cosine_similarity": 1.0 - value,
        "query_min_cosine_similarity": 1.0 - value / 2,
        "ranking_change_count": int(value > 0),
        "ranking_change_fraction": value,
        "recall_at_k_absolute_delta": value,
        "mrr_at_k_absolute_delta": value,
        "ndcg_at_k_absolute_delta": value,
    }


def test_batch_unit_uses_all_three_preregistered_pairs() -> None:
    comparisons = [
        _comparison("batch_1_primary", "batch_16_primary", 0.1),
        _comparison("batch_1_primary", "batch_64_primary", 0.3),
        _comparison("batch_16_primary", "batch_64_primary", 0.2),
    ]
    unit = batch_epoch_unit(1, comparisons)
    assert unit["comparison_count"] == 3
    assert unit["maximum"]["document_max_abs_delta"] == 0.3
    assert unit["minimum"]["document_min_cosine_similarity"] == 0.7


def test_batch_unit_fails_closed_when_pair_is_missing() -> None:
    with pytest.raises(FullExtremaBlocked, match="batch comparison set differs"):
        batch_epoch_unit(
            1,
            [
                _comparison("batch_1_primary", "batch_16_primary", 0.1),
                _comparison("batch_1_primary", "batch_64_primary", 0.3),
            ],
        )


def test_family_aggregation_requires_exact_ordered_coverage() -> None:
    comparison = _comparison("batch_16_primary", "batch_16_repeat", 0.1)
    units = [
        single_comparison_unit("repeated_inference", epoch, comparison) for epoch in range(1, 121)
    ]
    report = aggregate_family_extrema("repeated_inference", units)
    assert report["unit_count"] == 120
    assert report["scientific_decision"] == "NOT_EVALUATED"
    with pytest.raises(FullExtremaBlocked, match="coverage incomplete"):
        aggregate_family_extrema("repeated_inference", units[:-1])
