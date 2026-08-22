from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from itertools import combinations
from typing import Any

from neural_continuity.m1_diagnostics.causal_plan_authority import (
    CausalPlanError,
)

ALLOWED_CLUSTER_TYPES = frozenset({"FINITE_DRIFT", "NONFINITE_OBSERVED"})


def _utf8(values: set[str] | list[str]) -> list[str]:
    return sorted(values, key=lambda value: value.encode("utf-8"))


def _validated_clusters(
    cluster_report: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    raw_clusters = cluster_report.get("clusters")
    if not isinstance(raw_clusters, list) or not raw_clusters:
        raise CausalPlanError(
            "CAUSAL_PLAN_CLUSTER_INVALID",
            "structural clusters are missing",
        )
    clusters: list[Mapping[str, Any]] = []
    identities: set[str] = set()
    previous_onset = 0
    for raw in raw_clusters:
        if not isinstance(raw, Mapping):
            raise CausalPlanError(
                "CAUSAL_PLAN_CLUSTER_INVALID",
                "cluster record must be an object",
            )
        cluster_id = raw.get("cluster_id")
        cluster_type = raw.get("cluster_type")
        onset = raw.get("start_probe_order")
        families = raw.get("structural_families")
        basis = raw.get("target_tensor_basis")
        if (
            not isinstance(cluster_id, str)
            or not cluster_id
            or cluster_id in identities
            or cluster_type not in ALLOWED_CLUSTER_TYPES
            or not isinstance(onset, int)
            or onset <= previous_onset
            or not isinstance(basis, str)
            or not basis
            or not isinstance(families, list)
            or not families
            or not all(isinstance(value, str) and value for value in families)
            or len(set(families)) != len(families)
            or raw.get("causal_interpretation") != "NOT_ESTABLISHED"
        ):
            raise CausalPlanError(
                "CAUSAL_PLAN_CLUSTER_INVALID",
                "cluster identity, order or structural metadata is invalid",
            )
        identities.add(cluster_id)
        previous_onset = onset
        clusters.append(raw)
    return clusters


def _exposures(
    clusters: list[Mapping[str, Any]],
) -> tuple[dict[str, list[Mapping[str, Any]]], dict[tuple[str, str], list[Mapping[str, Any]]]]:
    family_exposure: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    pair_exposure: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for cluster in clusters:
        families = _utf8(set(cluster["structural_families"]))
        for family in families:
            family_exposure[family].append(cluster)
        for pair in combinations(families, 2):
            pair_exposure[pair].append(cluster)
    return dict(family_exposure), dict(pair_exposure)


def _cluster_ids(
    clusters: list[Mapping[str, Any]],
    cluster_type: str,
) -> list[str]:
    return [
        str(cluster["cluster_id"])
        for cluster in clusters
        if cluster["cluster_type"] == cluster_type
    ]


def build_causal_design(
    cluster_report: Mapping[str, Any],
) -> dict[str, Any]:
    clusters = _validated_clusters(cluster_report)
    family_exposure, pair_exposure = _exposures(clusters)
    family_names = _utf8(set(family_exposure))
    pair_names = sorted(
        pair_exposure,
        key=lambda pair: (
            pair[0].encode("utf-8"),
            pair[1].encode("utf-8"),
        ),
    )

    family_hypotheses: list[dict[str, Any]] = []
    single_interventions: list[dict[str, Any]] = []
    for index, family in enumerate(family_names, start=1):
        exposed = family_exposure[family]
        hypothesis_id = f"family-hypothesis-{index:04d}"
        intervention_id = f"single-family-holdout-{index:04d}"
        family_hypotheses.append(
            {
                "hypothesis_id": hypothesis_id,
                "hypothesis_type": "SINGLE_FAMILY_PRECISION_HOLDOUT",
                "structural_family": family,
                "finite_cluster_ids": _cluster_ids(exposed, "FINITE_DRIFT"),
                "nonfinite_cluster_ids": _cluster_ids(
                    exposed,
                    "NONFINITE_OBSERVED",
                ),
                "predicted_direction": ("lower_drift_or_nonfinite_incidence_in_exposed_clusters"),
                "hypothesis_status": "NOT_TESTED",
                "decision_states": [
                    "SUPPORTED",
                    "NOT_SUPPORTED",
                    "INCONCLUSIVE",
                ],
                "causal_claim_made": False,
            }
        )
        single_interventions.append(
            {
                "intervention_id": intervention_id,
                "hypothesis_id": hypothesis_id,
                "stage": 1,
                "intervention_type": "HOLD_STRUCTURAL_FAMILY_FP32",
                "structural_families": [family],
                "exposed_cluster_ids": [str(cluster["cluster_id"]) for cluster in exposed],
                "frozen_int8_candidate_mutated": False,
                "derived_diagnostic_candidate_required": True,
                "execution_status": "NOT_AUTHORIZED",
            }
        )

    interaction_hypotheses: list[dict[str, Any]] = []
    pair_interventions: list[dict[str, Any]] = []
    for index, pair in enumerate(pair_names, start=1):
        exposed = pair_exposure[pair]
        hypothesis_id = f"interaction-hypothesis-{index:04d}"
        intervention_id = f"pair-holdout-{index:04d}"
        interaction_hypotheses.append(
            {
                "hypothesis_id": hypothesis_id,
                "hypothesis_type": "PAIRWISE_PRECISION_INTERACTION",
                "structural_families": list(pair),
                "finite_cluster_ids": _cluster_ids(exposed, "FINITE_DRIFT"),
                "nonfinite_cluster_ids": _cluster_ids(
                    exposed,
                    "NONFINITE_OBSERVED",
                ),
                "hypothesis_status": "DEFERRED_PRE_REGISTERED",
                "activation_condition": ("all_single_family_controls_complete_and_interpretable"),
                "decision_states": [
                    "SUPPORTED",
                    "NOT_SUPPORTED",
                    "INCONCLUSIVE",
                ],
                "causal_claim_made": False,
            }
        )
        pair_interventions.append(
            {
                "intervention_id": intervention_id,
                "hypothesis_id": hypothesis_id,
                "stage": 2,
                "intervention_type": "HOLD_STRUCTURAL_FAMILY_PAIR_FP32",
                "structural_families": list(pair),
                "exposed_cluster_ids": [str(cluster["cluster_id"]) for cluster in exposed],
                "frozen_int8_candidate_mutated": False,
                "derived_diagnostic_candidate_required": True,
                "execution_status": "DEFERRED_NOT_AUTHORIZED",
            }
        )

    controls = [
        {
            "control_id": "control-0001",
            "control_type": "FROZEN_INT8_EXACT_REPLAY",
            "purpose": "execution_and_observation_repeatability",
            "required_outcome": "WITHIN_VERIFIED_MEASUREMENT_DETECTION_LIMIT",
            "execution_status": "NOT_AUTHORIZED",
        },
        {
            "control_id": "control-0002",
            "control_type": "VERIFIED_ONNX_FP32_REFERENCE",
            "purpose": "positive_reference_restoration",
            "required_outcome": "REFERENCE_IDENTITY_REPRODUCED",
            "execution_status": "NOT_AUTHORIZED",
        },
    ]
    return {
        "cluster_count": len(clusters),
        "observed_structural_families": family_names,
        "family_hypotheses": family_hypotheses,
        "interaction_hypotheses": interaction_hypotheses,
        "controls": controls,
        "single_family_interventions": single_interventions,
        "pair_interventions": pair_interventions,
        "summary": {
            "family_hypothesis_count": len(family_hypotheses),
            "interaction_hypothesis_count": len(interaction_hypotheses),
            "control_count": len(controls),
            "stage_1_intervention_count": len(single_interventions),
            "stage_2_intervention_count": len(pair_interventions),
            "all_clusters_included": True,
            "cluster_selection_cutoff_used": False,
            "nonfinite_ranked_with_finite": False,
            "causal_claim_made": False,
        },
    }
