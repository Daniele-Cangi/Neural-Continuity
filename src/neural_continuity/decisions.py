from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import FAIL, INCONCLUSIVE, PASS
from .metrics import POLICY_BY_ID, MetricPolicy


@dataclass(frozen=True)
class DecisionReason:
    category: str
    metric: str
    message: str
    affected_sample_ids: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContinuityDecision:
    status: str
    reasons: list[DecisionReason]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": [
                {
                    "category": reason.category,
                    "metric": reason.metric,
                    "message": reason.message,
                    "affected_sample_ids": reason.affected_sample_ids,
                    "details": reason.details,
                }
                for reason in self.reasons
            ],
        }


def _policy_lookup(
    required_metrics: list[str],
    metric_policies: list[MetricPolicy] | None,
) -> dict[str, MetricPolicy]:
    if metric_policies is None:
        return {name: POLICY_BY_ID[name] for name in required_metrics if name in POLICY_BY_ID}
    return {
        policy.metric_id: policy
        for policy in metric_policies
        if policy.metric_id in required_metrics or not required_metrics
    }


def _interval_decision(
    *,
    policy: MetricPolicy,
    envelope_lower: float,
    envelope_upper: float,
    observed_lower: float | None,
    observed_upper: float | None,
) -> tuple[str | None, dict[str, Any]]:
    if observed_lower is None or observed_upper is None:
        return "insufficient_candidate_evidence", {"status": "insufficient"}

    if policy.orientation == "higher_is_better":
        if observed_upper < envelope_lower:
            return (
                "metric_outside_authorized_region",
                {"direction": "lower_worse", "hard": policy.may_block_promotion},
            )
        if observed_lower < envelope_lower < observed_upper:
            return (
                "metric_interval_overlaps_authorized_boundary",
                {"direction": "boundary", "hard": policy.may_block_promotion},
            )
        return None, {}

    if policy.orientation == "lower_is_better":
        if observed_lower > envelope_upper:
            return (
                "metric_outside_authorized_region",
                {"direction": "higher_worse", "hard": policy.may_block_promotion},
            )
        if observed_lower < envelope_upper < observed_upper:
            return (
                "metric_interval_overlaps_authorized_boundary",
                {"direction": "boundary", "hard": policy.may_block_promotion},
            )
        return None, {}

    if policy.orientation == "two_sided_stability":
        if observed_upper < envelope_lower or observed_lower > envelope_upper:
            return (
                "metric_outside_authorized_region",
                {"direction": "outside", "hard": policy.may_block_promotion},
            )
        if (observed_lower < envelope_lower < observed_upper) or (
            observed_lower < envelope_upper < observed_upper
        ):
            return (
                "metric_interval_overlaps_authorized_boundary",
                {"direction": "boundary", "hard": policy.may_block_promotion},
            )
        return None, {}

    # informational_only
    return None, {}


def evaluate_comparison(
    comparison: dict[str, Any],
    envelopes: dict[str, dict[str, Any]],
    *,
    required_metrics: list[str],
    metric_policies: list[MetricPolicy] | None = None,
) -> ContinuityDecision:
    required = list(dict.fromkeys(required_metrics))
    reasons: list[DecisionReason] = []

    metric_deltas = comparison.get("metric_deltas", {})
    uncertainty = comparison.get("metric_uncertainty", {})
    policies = _policy_lookup(required, metric_policies)

    missing = [metric for metric in required if metric not in metric_deltas]
    if missing:
        reasons.append(
            DecisionReason(
                category="missing_evidence",
                metric=",".join(missing),
                message="Missing required metric values for this comparison.",
                details={"missing_metrics": missing},
            )
        )

    for metric in required:
        policy = policies.get(metric)
        if policy is None:
            continue

        envelope = envelopes.get(metric)
        if not envelope:
            reasons.append(
                DecisionReason(
                    category="missing_null_envelope",
                    metric=metric,
                    message="No null envelope available for this metric.",
                    details={"metric": metric},
                )
            )
            continue
        if envelope.get("status") != "complete":
            reasons.append(
                DecisionReason(
                    category="insufficient_null_evidence",
                    metric=metric,
                    message="Null evidence is insufficient for this metric.",
                    details={
                        "status": envelope.get("status"),
                        "sample_count": envelope.get("sample_count"),
                    },
                )
            )
            continue

        metric_uncertainty = uncertainty.get(metric)
        if not metric_uncertainty:
            reasons.append(
                DecisionReason(
                    category="missing_evidence",
                    metric=metric,
                    message="Missing metric uncertainty for this comparison.",
                    details={"metric": metric},
                )
            )
            continue

        category, detail = _interval_decision(
            policy=policy,
            envelope_lower=float(envelope.get("lower_bound", float("nan"))),
            envelope_upper=float(envelope.get("upper_bound", float("nan"))),
            observed_lower=metric_uncertainty.get("lower_bound"),
            observed_upper=metric_uncertainty.get("upper_bound"),
        )
        if category is None:
            continue

        reasons.append(
            DecisionReason(
                category=category,
                metric=metric,
                message=(
                    "metric interval "
                    f"[{metric_uncertainty.get('lower_bound')}, "
                    f"{metric_uncertainty.get('upper_bound')}] "
                    f"is not fully inside null interval "
                    f"[{envelope.get('lower_bound')}, {envelope.get('upper_bound')}]."
                ),
                details={
                    "null_interval": [envelope.get("lower_bound"), envelope.get("upper_bound")],
                    "candidate_interval": [
                        metric_uncertainty.get("lower_bound"),
                        metric_uncertainty.get("upper_bound"),
                    ],
                    "interval_detail": detail,
                    "policy": {
                        "metric_id": metric,
                        "may_block_promotion": policy.may_block_promotion,
                        "minimum_candidate_sample_size": policy.minimum_candidate_sample_size,
                        "minimum_null_observations": policy.minimum_null_observations,
                    },
                },
            )
        )

    if any(r.category in {"missing_evidence", "insufficient_candidate_evidence"} for r in reasons):
        return ContinuityDecision(status=INCONCLUSIVE, reasons=reasons)
    if any(r.category in {"insufficient_null_evidence", "missing_null_envelope"} for r in reasons):
        return ContinuityDecision(status=INCONCLUSIVE, reasons=reasons)

    hard_outside = any(
        r.category == "metric_outside_authorized_region"
        and bool(r.details.get("interval_detail", {}).get("hard", False))
        for r in reasons
    )
    hard_boundary_overlap = any(
        r.category == "metric_interval_overlaps_authorized_boundary"
        and bool(r.details.get("interval_detail", {}).get("hard", False))
        for r in reasons
    )
    if hard_boundary_overlap:
        return ContinuityDecision(status=INCONCLUSIVE, reasons=reasons)
    if hard_outside:
        return ContinuityDecision(status=FAIL, reasons=reasons)

    for regression in comparison.get("regressions", {}).get("source_correct_candidate_wrong", []):
        reasons.append(
            DecisionReason(
                category="frozen_set_regression",
                metric="source_correct_candidate_wrong",
                message="Primary regression detected: source correct / candidate wrong.",
                affected_sample_ids=[regression],
            )
        )

    if any(r.category == "frozen_set_regression" for r in reasons):
        return ContinuityDecision(status=FAIL, reasons=reasons)

    return ContinuityDecision(status=PASS, reasons=reasons)
