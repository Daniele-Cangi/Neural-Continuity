from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from neural_continuity.m1_diagnostics.structural_cluster_analysis import (
    analyze_structural_clusters,
)
from neural_continuity.m1_diagnostics.structural_cluster_authority import (
    StructuralClusterError,
)


def _probe(
    order: int,
    classification: str,
    basis: str,
    families: list[str],
    *,
    relative_l2: float | None = 0.1,
    bitwise_rate: float = 0.5,
    integer_rate: float | None = 0.0,
) -> dict[str, Any]:
    differing = 0 if classification == "BITWISE_EQUAL" else 1
    return {
        "probe_id": f"probe-{order:04d}",
        "probe_order": order,
        "target_tensor_basis": basis,
        "structural_families": families,
        "floating": {
            "classification": classification,
            "differing_value_count": differing,
            "bitwise_difference_rate": 0.0 if differing == 0 else bitwise_rate,
            "relative_l2_error": relative_l2,
        },
        "integer_dtype_extremes": (
            {
                "classification": (
                    "DTYPE_EXTREME_VALUES_OBSERVED"
                    if integer_rate and integer_rate > 0.0
                    else "NO_DTYPE_EXTREME_VALUES"
                ),
                "dtype_extreme_rate": integer_rate,
            }
            if integer_rate is not None
            else None
        ),
    }


def _diagnostics() -> Mapping[str, Any]:
    return {
        "probes": [
            _probe(1, "BITWISE_EQUAL", "direct", ["A"]),
            _probe(2, "FINITE_BITWISE_DRIFT", "direct", ["A"], relative_l2=0.2),
            _probe(
                3,
                "FINITE_BITWISE_DRIFT",
                "direct",
                ["A", "B"],
                relative_l2=0.4,
                integer_rate=0.25,
            ),
            _probe(4, "FINITE_BITWISE_DRIFT", "direct", ["C"], relative_l2=0.9),
            _probe(5, "NONFINITE_OBSERVED", "direct", ["C"], relative_l2=None),
            _probe(6, "NONFINITE_OBSERVED", "direct", ["C"], relative_l2=None),
            _probe(7, "BITWISE_EQUAL", "direct", ["C"]),
        ]
    }


def test_structural_clusters_use_only_declared_structural_boundaries() -> None:
    result = analyze_structural_clusters(_diagnostics())

    assert result["cluster_count"] == 3
    assert result["summary"]["finite_cluster_count"] == 2
    assert result["summary"]["nonfinite_cluster_count"] == 1
    first = result["clusters"][0]
    assert first["probe_ids"] == ["probe-0002", "probe-0003"]
    assert first["shared_structural_families"] == ["A"]
    assert first["peak_relative_l2_error"]["probe_id"] == "probe-0003"
    assert first["first_following_bitwise_equal_recovery"]["probe_id"] == "probe-0007"


def test_nonfinite_clusters_are_not_numerically_ranked_with_finite_clusters() -> None:
    result = analyze_structural_clusters(_diagnostics())

    summary = result["summary"]
    assert summary["ranked_finite_cluster_ids"] == ["finite-0002", "finite-0001"]
    assert summary["nonfinite_cluster_ids_by_onset"] == ["nonfinite-0001"]
    assert summary["nonfinite_ranked_with_finite"] is False
    nonfinite = result["clusters"][2]
    assert nonfinite["peak_relative_l2_error"] is None


def test_bitwise_equal_probe_has_no_cluster_membership() -> None:
    result = analyze_structural_clusters(_diagnostics())

    memberships = {item["probe_id"]: item for item in result["membership"]}
    assert memberships["probe-0001"]["cluster_id"] is None
    assert memberships["probe-0007"]["cluster_id"] is None


def test_structural_clusters_fail_closed_on_noncontiguous_probe_order() -> None:
    diagnostics = dict(_diagnostics())
    probes = list(diagnostics["probes"])
    probes[2] = dict(probes[2], probe_order=9)
    diagnostics["probes"] = probes

    with pytest.raises(StructuralClusterError) as error:
        analyze_structural_clusters(diagnostics)
    assert error.value.code == "STRUCTURAL_CLUSTER_PROBE_ORDER_MISMATCH"
