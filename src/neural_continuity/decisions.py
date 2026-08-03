from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from . import FAIL, INCONCLUSIVE, PASS


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
            "reasons": [dataclasses.asdict(reason) for reason in self.reasons],
        }


def evaluate_comparison(
    comparison: dict[str, Any],
    envelopes: dict[str, dict[str, Any]],
    *,
    required_metrics: Iterable[str],
    require_boundary_inconclusive: bool = False,
) -> ContinuityDecision:
    metric_deltas = comparison.get("metric_deltas", {})
    reasons: list[DecisionReason] = []

    missing = [metric for metric in required_metrics if metric not in metric_deltas]
    if missing:
        reasons.append(
            DecisionReason(
                category="missing_evidence",
                metric=",".join(missing),
                message="Missing required metric values for this comparison.",
                affected_sample_ids=[],
                details={"missing_metrics": missing},
            )
        )

    sample_count = int(comparison.get("sample_count", 0))
    if sample_count < 2:
        reasons.append(
            DecisionReason(
                category="insufficient_sample",
                metric="sample_count",
                message="Sample count is below the required threshold for robust evidence.",
                details={"observed": sample_count},
            )
        )

    for regression in comparison.get("regressions", {}).get("source_correct_candidate_wrong", []):
        reasons.append(
            DecisionReason(
                category="frozen_set_regression",
                metric="source_correct_candidate_wrong",
                message=("Primary regression detected: source correct / candidate wrong."),
                affected_sample_ids=[regression],
            )
        )

    for metric, delta in metric_deltas.items():
        envelope = envelopes.get(metric)
        if not envelope:
            continue
        upper = float(envelope["upper_bound"])
        if delta > upper:
            reasons.append(
                DecisionReason(
                    category="envelope_exceeded",
                    metric=metric,
                    message=f"delta {delta:.6f} exceeded null envelope upper {upper:.6f}",
                    affected_sample_ids=comparison.get("affected_samples", {}).get(
                        "source_correct_candidate_wrong", []
                    ),
                    details={"delta": delta, "upper": upper},
                )
            )

    if reasons:
        if any(
            r.category in {"missing_evidence", "insufficient_sample", "boundary_overlap"}
            for r in reasons
        ):
            return ContinuityDecision(status=INCONCLUSIVE, reasons=reasons)
        return ContinuityDecision(status=FAIL, reasons=reasons)

    if require_boundary_inconclusive:
        reasons.append(
            DecisionReason(
                category="boundary_overlap",
                metric="boundary_case",
                message="Explicit synthetic boundary control is treated as inconclusive.",
            )
        )
        return ContinuityDecision(status=INCONCLUSIVE, reasons=reasons)

    if reasons:
        return ContinuityDecision(status=FAIL, reasons=reasons)

    return ContinuityDecision(status=PASS, reasons=[])
