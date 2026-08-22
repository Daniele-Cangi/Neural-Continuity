from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from neural_continuity.m1_diagnostics.structural_cluster_authority import (
    StructuralClusterError,
)

BITWISE_EQUAL = "BITWISE_EQUAL"
FINITE_DRIFT = "FINITE_BITWISE_DRIFT"
NONFINITE = "NONFINITE_OBSERVED"
ALLOWED_CLASSIFICATIONS = frozenset({BITWISE_EQUAL, FINITE_DRIFT, NONFINITE})


def _finite_number(value: Any, field: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_PROBE_INVALID",
            f"{field} must be a finite number",
        )
    result = float(value)
    if not math.isfinite(result):
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_PROBE_INVALID",
            f"{field} must be a finite number",
        )
    return result


def _validated_probes(probe_diagnostics: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_probes = probe_diagnostics.get("probes")
    if not isinstance(raw_probes, list) or not raw_probes:
        raise StructuralClusterError(
            "STRUCTURAL_CLUSTER_PROBE_INVALID",
            "probe diagnostics are missing",
        )
    probes: list[Mapping[str, Any]] = []
    identities: set[str] = set()
    for expected_order, raw in enumerate(raw_probes, start=1):
        if not isinstance(raw, Mapping):
            raise StructuralClusterError(
                "STRUCTURAL_CLUSTER_PROBE_INVALID",
                "probe record must be an object",
            )
        probe_id = raw.get("probe_id")
        if (
            not isinstance(probe_id, str)
            or not probe_id
            or probe_id in identities
            or raw.get("probe_order") != expected_order
        ):
            raise StructuralClusterError(
                "STRUCTURAL_CLUSTER_PROBE_ORDER_MISMATCH",
                "probe identities or contiguous order are invalid",
            )
        basis = raw.get("target_tensor_basis")
        families = raw.get("structural_families")
        floating = raw.get("floating")
        if (
            not isinstance(basis, str)
            or not basis
            or not isinstance(families, list)
            or not families
            or not all(isinstance(value, str) and value for value in families)
            or len(set(families)) != len(families)
            or not isinstance(floating, Mapping)
        ):
            raise StructuralClusterError(
                "STRUCTURAL_CLUSTER_PROBE_INVALID",
                f"structural metadata is invalid for {probe_id}",
            )
        classification = floating.get("classification")
        if classification not in ALLOWED_CLASSIFICATIONS:
            raise StructuralClusterError(
                "STRUCTURAL_CLUSTER_CLASSIFICATION_INVALID",
                f"unsupported classification for {probe_id}",
            )
        difference_rate = _finite_number(
            floating.get("bitwise_difference_rate"),
            f"{probe_id}.bitwise_difference_rate",
        )
        if difference_rate is None or not 0.0 <= difference_rate <= 1.0:
            raise StructuralClusterError(
                "STRUCTURAL_CLUSTER_PROBE_INVALID",
                f"bitwise difference rate is outside [0, 1] for {probe_id}",
            )
        differing_count = floating.get("differing_value_count")
        if not isinstance(differing_count, int) or differing_count < 0:
            raise StructuralClusterError(
                "STRUCTURAL_CLUSTER_PROBE_INVALID",
                f"differing value count is invalid for {probe_id}",
            )
        if classification == BITWISE_EQUAL and differing_count != 0:
            raise StructuralClusterError(
                "STRUCTURAL_CLUSTER_CLASSIFICATION_INVALID",
                f"bitwise-equal probe reports differing values: {probe_id}",
            )
        if classification == FINITE_DRIFT and differing_count == 0:
            raise StructuralClusterError(
                "STRUCTURAL_CLUSTER_CLASSIFICATION_INVALID",
                f"finite-drift probe reports no differing values: {probe_id}",
            )
        _finite_number(
            floating.get("relative_l2_error"),
            f"{probe_id}.relative_l2_error",
            optional=True,
        )
        integer = raw.get("integer_dtype_extremes")
        if integer is not None:
            if not isinstance(integer, Mapping):
                raise StructuralClusterError(
                    "STRUCTURAL_CLUSTER_PROBE_INVALID",
                    f"integer proxy is invalid for {probe_id}",
                )
            rate = _finite_number(
                integer.get("dtype_extreme_rate"),
                f"{probe_id}.dtype_extreme_rate",
                optional=True,
            )
            if rate is not None and not 0.0 <= rate <= 1.0:
                raise StructuralClusterError(
                    "STRUCTURAL_CLUSTER_PROBE_INVALID",
                    f"integer extreme rate is outside [0, 1] for {probe_id}",
                )
        identities.add(probe_id)
        probes.append(raw)
    return probes


def _compatible(previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    if previous["floating"]["classification"] != current["floating"]["classification"]:
        return False
    if previous["target_tensor_basis"] != current["target_tensor_basis"]:
        return False
    previous_families = set(previous["structural_families"])
    current_families = set(current["structural_families"])
    return bool(previous_families & current_families)


def _metric_peak(
    members: list[Mapping[str, Any]],
    field: str,
) -> dict[str, Any] | None:
    candidates: list[tuple[float, int, Mapping[str, Any]]] = []
    for member in members:
        value = member["floating"].get(field)
        if value is not None:
            candidates.append((float(value), int(member["probe_order"]), member))
    if not candidates:
        return None
    value, _, probe = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
    return {
        "probe_id": probe["probe_id"],
        "probe_order": probe["probe_order"],
        "value": value,
    }


def _integer_proxy(members: list[Mapping[str, Any]]) -> dict[str, Any]:
    comparable: list[tuple[float, Mapping[str, Any]]] = []
    extreme_count = 0
    for member in members:
        integer = member.get("integer_dtype_extremes")
        if not isinstance(integer, Mapping) or integer.get("dtype_extreme_rate") is None:
            continue
        rate = float(integer["dtype_extreme_rate"])
        comparable.append((rate, member))
        if integer.get("classification") == "DTYPE_EXTREME_VALUES_OBSERVED":
            extreme_count += 1
    peak = None
    if comparable:
        rate, member = sorted(
            comparable,
            key=lambda item: (-item[0], int(item[1]["probe_order"])),
        )[0]
        peak = {
            "probe_id": member["probe_id"],
            "probe_order": member["probe_order"],
            "value": rate,
        }
    return {
        "comparable_probe_count": len(comparable),
        "extreme_observed_probe_count": extreme_count,
        "mean_dtype_extreme_rate": (
            sum(rate for rate, _ in comparable) / len(comparable) if comparable else None
        ),
        "maximum_dtype_extreme_rate": peak,
        "causal_interpretation": "NOT_ESTABLISHED",
    }


def _recovery(
    probes: list[Mapping[str, Any]],
    end_order: int,
) -> dict[str, Any] | None:
    for probe in probes[end_order:]:
        if probe["floating"]["classification"] == BITWISE_EQUAL:
            return {
                "probe_id": probe["probe_id"],
                "probe_order": probe["probe_order"],
            }
    return None


def _cluster_record(
    cluster_id: str,
    members: list[Mapping[str, Any]],
    probes: list[Mapping[str, Any]],
) -> dict[str, Any]:
    classification = str(members[0]["floating"]["classification"])
    family_sets = [set(member["structural_families"]) for member in members]
    return {
        "cluster_id": cluster_id,
        "cluster_type": (
            "FINITE_DRIFT" if classification == FINITE_DRIFT else "NONFINITE_OBSERVED"
        ),
        "classification": classification,
        "start_probe_id": members[0]["probe_id"],
        "start_probe_order": members[0]["probe_order"],
        "end_probe_id": members[-1]["probe_id"],
        "end_probe_order": members[-1]["probe_order"],
        "member_count": len(members),
        "probe_ids": [member["probe_id"] for member in members],
        "target_tensor_basis": members[0]["target_tensor_basis"],
        "structural_families": sorted(set().union(*family_sets)),
        "shared_structural_families": sorted(set.intersection(*family_sets)),
        "peak_relative_l2_error": (
            _metric_peak(members, "relative_l2_error") if classification == FINITE_DRIFT else None
        ),
        "maximum_bitwise_difference_rate": _metric_peak(
            members,
            "bitwise_difference_rate",
        ),
        "integer_dtype_extreme_proxy": _integer_proxy(members),
        "first_following_bitwise_equal_recovery": _recovery(
            probes,
            int(members[-1]["probe_order"]),
        ),
        "causal_interpretation": "NOT_ESTABLISHED",
    }


def analyze_structural_clusters(
    probe_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    probes = _validated_probes(probe_diagnostics)
    raw_clusters: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    for probe in probes:
        if probe["floating"]["classification"] == BITWISE_EQUAL:
            if current:
                raw_clusters.append(current)
                current = []
            continue
        if current and not _compatible(current[-1], probe):
            raw_clusters.append(current)
            current = []
        current.append(probe)
    if current:
        raw_clusters.append(current)

    counters = {"FINITE_DRIFT": 0, "NONFINITE_OBSERVED": 0}
    clusters: list[dict[str, Any]] = []
    membership_ids: dict[str, str] = {}
    for members in raw_clusters:
        cluster_type = (
            "FINITE_DRIFT"
            if members[0]["floating"]["classification"] == FINITE_DRIFT
            else "NONFINITE_OBSERVED"
        )
        counters[cluster_type] += 1
        prefix = "finite" if cluster_type == "FINITE_DRIFT" else "nonfinite"
        cluster_id = f"{prefix}-{counters[cluster_type]:04d}"
        cluster = _cluster_record(cluster_id, members, probes)
        clusters.append(cluster)
        membership_ids.update({str(member["probe_id"]): cluster_id for member in members})

    finite_clusters = [cluster for cluster in clusters if cluster["cluster_type"] == "FINITE_DRIFT"]
    nonfinite_clusters = [
        cluster for cluster in clusters if cluster["cluster_type"] == "NONFINITE_OBSERVED"
    ]

    def finite_rank_key(cluster: Mapping[str, Any]) -> tuple[float, float, int]:
        relative = cluster["peak_relative_l2_error"]
        relative_value = float(relative["value"]) if relative is not None else -1.0
        bitwise = cluster["maximum_bitwise_difference_rate"]
        bitwise_value = float(bitwise["value"]) if bitwise is not None else -1.0
        return (-relative_value, -bitwise_value, int(cluster["start_probe_order"]))

    ranked_finite = sorted(finite_clusters, key=finite_rank_key)
    membership = [
        {
            "probe_id": probe["probe_id"],
            "probe_order": probe["probe_order"],
            "classification": probe["floating"]["classification"],
            "cluster_id": membership_ids.get(str(probe["probe_id"])),
            "target_tensor_basis": probe["target_tensor_basis"],
            "structural_families": sorted(probe["structural_families"]),
        }
        for probe in probes
    ]
    equal_count = sum(probe["floating"]["classification"] == BITWISE_EQUAL for probe in probes)
    largest_finite = (
        sorted(
            finite_clusters,
            key=lambda cluster: (
                -int(cluster["member_count"]),
                int(cluster["start_probe_order"]),
            ),
        )[0]["cluster_id"]
        if finite_clusters
        else None
    )
    return {
        "probe_count": len(probes),
        "cluster_count": len(clusters),
        "membership": membership,
        "clusters": clusters,
        "summary": {
            "bitwise_equal_probe_count": equal_count,
            "finite_drift_probe_count": sum(
                probe["floating"]["classification"] == FINITE_DRIFT for probe in probes
            ),
            "nonfinite_probe_count": sum(
                probe["floating"]["classification"] == NONFINITE for probe in probes
            ),
            "finite_cluster_count": len(finite_clusters),
            "nonfinite_cluster_count": len(nonfinite_clusters),
            "first_finite_cluster_id": (
                finite_clusters[0]["cluster_id"] if finite_clusters else None
            ),
            "first_nonfinite_cluster_id": (
                nonfinite_clusters[0]["cluster_id"] if nonfinite_clusters else None
            ),
            "largest_finite_cluster_id": largest_finite,
            "finite_cluster_ranking_basis": (
                "peak_relative_l2_then_bitwise_difference_rate_then_onset"
            ),
            "ranked_finite_cluster_ids": [cluster["cluster_id"] for cluster in ranked_finite],
            "nonfinite_cluster_ids_by_onset": [
                cluster["cluster_id"]
                for cluster in sorted(
                    nonfinite_clusters,
                    key=lambda cluster: int(cluster["start_probe_order"]),
                )
            ],
            "nonfinite_ranked_with_finite": False,
        },
    }
