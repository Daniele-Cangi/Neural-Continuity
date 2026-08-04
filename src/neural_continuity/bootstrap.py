from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MetricEnvelope:
    metric: str
    method: str
    seed: int
    bootstrap_samples: int
    sample_count: int
    confidence_level: float
    status: str
    noise_source_counts: dict[str, int]
    raw_null_values: list[float]
    observed_null_distribution: list[float] | None
    source_envelopes: dict[str, dict[str, Any]]
    lower_bound: float
    upper_bound: float
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "method": self.method,
            "seed": self.seed,
            "bootstrap_samples": self.bootstrap_samples,
            "sample_count": self.sample_count,
            "confidence_level": self.confidence_level,
            "status": self.status,
            "noise_source_counts": self.noise_source_counts,
            "raw_null_values": self.raw_null_values,
            "observed_null_distribution": self.observed_null_distribution,
            "source_envelopes": self.source_envelopes,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "details": self.details,
        }


def bootstrap_ci(
    values: Sequence[float], *, sample_count: int, confidence: float, seed: int
) -> tuple[list[float], float | None, float | None]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return [], None, None
    if sample_count <= 1:
        mean = float(np.mean(arr))
        return [mean], mean, mean

    rng = np.random.default_rng(seed)
    boot = []
    n = len(arr)
    for _ in range(sample_count):
        draw = rng.choice(arr, size=n, replace=True)
        boot.append(float(np.mean(draw)))
    alpha = max(0.0, 1.0 - confidence)
    lower = float(np.quantile(boot, alpha / 2))
    upper = float(np.quantile(boot, 1 - alpha / 2))
    return boot, lower, upper


def prediction_interval(values: Sequence[float], *, confidence: float) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("cannot compute prediction interval of empty distribution")
    alpha = max(0.0, 1.0 - confidence)
    lower = float(np.quantile(arr, alpha / 2))
    upper = float(np.quantile(arr, 1 - alpha / 2))
    return lower, upper


def build_envelopes(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    metric_names: Sequence[str] | None = None,
    metric_policies: Sequence[Mapping[str, Any]] | None = None,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    if metric_names is None and metric_policies is None:
        raise ValueError("metric_names or metric_policies must be provided")
    if metric_policies is None and metric_names is None:
        raise ValueError("metric_names or metric_policies must be provided")
    if metric_policies is None:
        metric_policies = [
            {"metric": metric, "minimum_null_observations": 1}
            for metric in metric_names or []
        ]

    resolved_metrics: dict[str, Mapping[str, Any]] = {}
    for policy in metric_policies:
        metric = str(policy.get("metric", policy.get("metric_id", "")))
        if not metric:
            continue
        resolved_metrics[metric] = policy
    if metric_names is not None:
        for metric in metric_names:
            if metric not in resolved_metrics:
                resolved_metrics[metric] = {"metric": metric, "minimum_null_observations": 1}

    deltas_by_metric: dict[str, dict[str, list[float]]] = {
        metric: {} for metric in resolved_metrics
    }
    for comparison in comparisons:
        deltas = comparison.get("metric_deltas", {})
        source = str(comparison.get("noise_source", "unknown"))
        for metric in resolved_metrics:
            value = deltas.get(metric)
            if value is not None and not math.isnan(value):
                deltas_by_metric[metric].setdefault(source, []).append(float(value))

    envelopes: dict[str, dict[str, Any]] = {}
    for idx, (metric, values) in enumerate(deltas_by_metric.items()):
        source_counts: dict[str, int] = {}
        values_only: list[float] = []
        for source_name, source_values in values.items():
            source_counts[source_name] = len(source_values)
            values_only.extend(source_values)

        metric_policy = resolved_metrics[metric]
        required_null = int(metric_policy.get("minimum_null_observations", 1))
        if len(values_only) < required_null:
            envelopes[metric] = MetricEnvelope(
                metric=metric,
                method="prediction_interval",
                seed=seed + idx + 1,
                bootstrap_samples=bootstrap_samples,
                sample_count=len(values_only),
                confidence_level=confidence_level,
                status="insufficient",
                noise_source_counts=source_counts,
                raw_null_values=values_only,
                observed_null_distribution=None,
                source_envelopes={},
                lower_bound=math.nan,
                upper_bound=math.nan,
                details={"required_null_observations": required_null},
            ).as_dict()
            continue

        source_envelopes: dict[str, dict[str, Any]] = {}
        complete_sources: list[tuple[str, float, float]] = []
        for source_name, source_values in values.items():
            if len(source_values) < required_null:
                source_envelopes[source_name] = {
                    "status": "insufficient",
                    "sample_count": len(source_values),
                    "lower_bound": math.nan,
                    "upper_bound": math.nan,
                }
                continue

            lower_bound, upper_bound = prediction_interval(
                source_values, confidence=confidence_level
            )
            source_envelopes[source_name] = {
                "status": "complete",
                "sample_count": len(source_values),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }
            complete_sources.append((source_name, lower_bound, upper_bound))

        lower = (
            min(lower_bound for _, lower_bound, _ in complete_sources)
            if complete_sources
            else math.nan
        )
        upper = (
            max(upper_bound for _, _, upper_bound in complete_sources)
            if complete_sources
            else math.nan
        )
        envelopes[metric] = MetricEnvelope(
            metric=metric,
            method="prediction_interval",
            seed=seed + idx + 1,
            bootstrap_samples=bootstrap_samples,
            sample_count=len(values_only),
            confidence_level=confidence_level,
            status="complete" if complete_sources else "insufficient",
            noise_source_counts=source_counts,
            raw_null_values=values_only,
            observed_null_distribution=sorted(values_only),
            source_envelopes=source_envelopes,
            lower_bound=float(lower) if lower is not None else math.nan,
            upper_bound=float(upper) if upper is not None else math.nan,
            details={
                "required_null_observations": required_null,
                "confidence_interval_style": "empirical_tolerance",
                "source_bounds": {
                    source_name: {
                        "count": envelope["sample_count"],
                        "status": envelope["status"],
                    }
                    for source_name, envelope in source_envelopes.items()
                },
            },
        ).as_dict()
    return envelopes
