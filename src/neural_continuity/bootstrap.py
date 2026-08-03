from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MetricEnvelope:
    metric: str
    seed: int
    sample_count: int
    confidence_level: float
    observed_null_distribution: list[float]
    lower_bound: float
    upper_bound: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "seed": self.seed,
            "sample_count": self.sample_count,
            "confidence_level": self.confidence_level,
            "observed_null_distribution": self.observed_null_distribution,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }


def bootstrap_ci(
    values: Sequence[float], *, sample_count: int, confidence: float, seed: int
) -> tuple[list[float], float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        arr = np.array([0.0], dtype=float)

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


def build_envelopes(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    metric_names: Sequence[str],
    bootstrap_samples: int,
    confidence_level: float,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    deltas_by_metric: dict[str, list[float]] = {metric: [] for metric in metric_names}
    for comparison in comparisons:
        deltas = comparison.get("metric_deltas", {})
        for metric in metric_names:
            value = deltas.get(metric)
            if value is not None and not math.isnan(value):
                deltas_by_metric[metric].append(float(abs(value)))

    envelopes: dict[str, dict[str, Any]] = {}
    for idx, (metric, values) in enumerate(deltas_by_metric.items()):
        observed, lower, upper = bootstrap_ci(
            values if values else [0.0],
            sample_count=bootstrap_samples,
            confidence=confidence_level,
            seed=seed + idx + 1,
        )
        envelopes[metric] = MetricEnvelope(
            metric=metric,
            seed=seed + idx + 1,
            sample_count=len(values),
            confidence_level=confidence_level,
            observed_null_distribution=observed,
            lower_bound=lower,
            upper_bound=upper,
        ).as_dict()
    return envelopes
