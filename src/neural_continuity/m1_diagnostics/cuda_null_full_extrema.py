"""Preregistered CUDA full-corpus comparison units and extrema aggregation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

_EPOCHS = 120
_FAMILY_COUNTS = {
    "repeated_inference": _EPOCHS,
    "batch_size_variation": _EPOCHS,
    "process_restart_variation": _EPOCHS // 2,
}
_MAXIMUM_FIELDS = (
    "document_max_abs_delta",
    "query_max_abs_delta",
    "ranking_change_count",
    "ranking_change_fraction",
    "recall_at_k_absolute_delta",
    "mrr_at_k_absolute_delta",
    "ndcg_at_k_absolute_delta",
)
_MINIMUM_FIELDS = ("document_min_cosine_similarity", "query_min_cosine_similarity")
_VALUE_FIELDS = set(_MAXIMUM_FIELDS) | set(_MINIMUM_FIELDS)


class FullExtremaBlocked(ValueError):
    """Comparison units do not match the preregistered CUDA design."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise FullExtremaBlocked(reason)


def batch_epoch_unit(epoch: int, comparisons: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collapse the three fixed batch comparisons to one epoch-level unit."""
    _require(type(epoch) is int and 1 <= epoch <= _EPOCHS, "batch unit epoch invalid")
    expected_pairs = {
        ("batch_1_primary", "batch_16_primary"),
        ("batch_1_primary", "batch_64_primary"),
        ("batch_16_primary", "batch_64_primary"),
    }
    _require(
        len(comparisons) == 3
        and {(item.get("left_run_label"), item.get("right_run_label")) for item in comparisons}
        == expected_pairs,
        "batch comparison set differs",
    )
    return _unit("batch_size_variation", epoch, comparisons)


def single_comparison_unit(
    family: str,
    unit_number: int,
    comparison: dict[str, Any],
) -> dict[str, Any]:
    """Build one repeated-inference or disjoint restart unit."""
    _require(family in {"repeated_inference", "process_restart_variation"}, "family invalid")
    expected = _FAMILY_COUNTS[family]
    _require(type(unit_number) is int and 1 <= unit_number <= expected, "unit number invalid")
    return _unit(family, unit_number, [comparison])


def _unit(family: str, number: int, comparisons: Sequence[dict[str, Any]]) -> dict[str, Any]:
    _require(
        all(isinstance(item, dict) and _VALUE_FIELDS.issubset(item) for item in comparisons),
        "comparison metric set incomplete",
    )
    maxima = {field: max(item[field] for item in comparisons) for field in _MAXIMUM_FIELDS}
    minima = {field: min(item[field] for item in comparisons) for field in _MINIMUM_FIELDS}
    _require(
        all(type(value) in (int, float) and value >= 0 for value in maxima.values())
        and all(type(value) in (int, float) and -1.0 <= value <= 1.0 for value in minima.values()),
        "comparison extrema invalid",
    )
    return {
        "family": family,
        "unit_number": number,
        "comparison_count": len(comparisons),
        "maximum": maxima,
        "minimum": minima,
    }


def aggregate_family_extrema(family: str, units: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate exact preregistered units without percentile interpolation."""
    _require(family in _FAMILY_COUNTS, "comparison family invalid")
    expected = _FAMILY_COUNTS[family]
    _require(len(units) == expected, "comparison family coverage incomplete")
    for number, unit in enumerate(units, start=1):
        _require(
            isinstance(unit, dict)
            and unit.get("family") == family
            and unit.get("unit_number") == number
            and set(unit.get("maximum", {})) == set(_MAXIMUM_FIELDS)
            and set(unit.get("minimum", {})) == set(_MINIMUM_FIELDS),
            "comparison unit order or schema differs",
        )
    return {
        "family": family,
        "unit_count": expected,
        "maximum": {
            field: max(unit["maximum"][field] for unit in units) for field in _MAXIMUM_FIELDS
        },
        "minimum": {
            field: min(unit["minimum"][field] for unit in units) for field in _MINIMUM_FIELDS
        },
        "order_statistic": "observed_maximum_or_minimum",
        "target_percentile": 0.95,
        "target_confidence": 0.95,
        "scientific_decision": "NOT_EVALUATED",
    }
