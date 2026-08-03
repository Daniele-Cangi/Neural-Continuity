from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NoiseRecord:
    metric: str
    source: str
    value: float


def extract_metric_series(comparisons: list[dict[str, Any]]) -> dict[str, list[float]]:
    metric_series: dict[str, list[float]] = {}
    for comparison in comparisons:
        for metric, delta in comparison.get("metric_deltas", {}).items():
            metric_series.setdefault(metric, []).append(float(abs(delta)))
    return metric_series


def noise_sources_for_null(
    repeat_index: int,
    run_label: str,
    batch_size: int,
    base_batch_size: int,
) -> str:
    if repeat_index == 0 and batch_size == base_batch_size:
        return "exact_base"
    if repeat_index == 0:
        return "batch_size"
    if batch_size == base_batch_size:
        return "repeated_inference"
    return "runtime_and_batch"
