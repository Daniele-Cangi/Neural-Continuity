from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.m1_diagnostics.activation_analysis_authority import (
    ActivationAnalysisError,
)

ProgressCallback = Callable[[Mapping[str, Any]], None]


@dataclass
class FloatingAccumulator:
    value_count: int = 0
    finite_pair_count: int = 0
    differing_value_count: int = 0
    source_nonfinite_count: int = 0
    target_nonfinite_count: int = 0
    sum_abs_delta: float = 0.0
    sum_squared_delta: float = 0.0
    source_sum_squared: float = 0.0
    target_sum_squared: float = 0.0
    source_target_dot: float = 0.0
    max_abs_delta: float = 0.0
    source_dtypes: set[str] = field(default_factory=set)
    target_dtypes: set[str] = field(default_factory=set)

    def update(self, source: np.ndarray, target: np.ndarray, probe_id: str) -> None:
        source_array = np.asarray(source)
        target_array = np.asarray(target)
        if source_array.shape != target_array.shape:
            raise ActivationAnalysisError(
                "ACTIVATION_PAIR_SHAPE_MISMATCH",
                f"source and target shapes differ for {probe_id}",
            )
        source_dtype = str(source_array.dtype)
        target_dtype = str(target_array.dtype)
        supported = np.issubdtype(source_array.dtype, np.number) or np.issubdtype(
            source_array.dtype, np.bool_
        )
        if source_array.dtype != target_array.dtype or not supported:
            raise ActivationAnalysisError(
                "ACTIVATION_DTYPE_INVALID",
                f"paired probe dtype is unsupported or differs: {probe_id}",
            )
        if self.source_dtypes and source_dtype not in self.source_dtypes:
            raise ActivationAnalysisError(
                "ACTIVATION_DTYPE_CHURN",
                f"paired probe dtype changes across batches: {probe_id}",
            )
        self.source_dtypes.add(source_dtype)
        self.target_dtypes.add(target_dtype)
        self.value_count += int(source_array.size)
        self.differing_value_count += int(np.count_nonzero(source_array != target_array))
        source_finite = np.isfinite(source_array)
        target_finite = np.isfinite(target_array)
        self.source_nonfinite_count += int(source_array.size - np.count_nonzero(source_finite))
        self.target_nonfinite_count += int(target_array.size - np.count_nonzero(target_finite))
        paired = source_finite & target_finite
        paired_count = int(np.count_nonzero(paired))
        self.finite_pair_count += paired_count
        if paired_count == 0:
            return
        source_values = source_array[paired].astype(np.float64, copy=False)
        target_values = target_array[paired].astype(np.float64, copy=False)
        delta = target_values - source_values
        absolute = np.abs(delta)
        self.sum_abs_delta += float(np.sum(absolute, dtype=np.float64))
        self.sum_squared_delta += float(np.sum(delta * delta, dtype=np.float64))
        self.source_sum_squared += float(np.sum(source_values * source_values, dtype=np.float64))
        self.target_sum_squared += float(np.sum(target_values * target_values, dtype=np.float64))
        self.source_target_dot += float(np.sum(source_values * target_values, dtype=np.float64))
        self.max_abs_delta = max(self.max_abs_delta, float(np.max(absolute)))

    def finish(self) -> dict[str, Any]:
        finite = self.finite_pair_count
        rmse = math.sqrt(self.sum_squared_delta / finite) if finite else None
        mean_abs = self.sum_abs_delta / finite if finite else None
        relative_l2 = (
            math.sqrt(self.sum_squared_delta / self.source_sum_squared)
            if self.source_sum_squared > 0.0
            else None
        )
        cosine_denominator = math.sqrt(self.source_sum_squared * self.target_sum_squared)
        cosine = self.source_target_dot / cosine_denominator if cosine_denominator > 0.0 else None
        nonfinite = self.source_nonfinite_count + self.target_nonfinite_count
        dtype_name = next(iter(self.source_dtypes), None)
        dtype_kind = np.dtype(dtype_name).kind if dtype_name is not None else None
        if dtype_kind == "f":
            metric_domain: str | None = "FLOATING_NUMERIC"
        elif dtype_kind in {"i", "u"}:
            metric_domain = "INTEGER_NUMERIC"
        elif dtype_kind == "b":
            metric_domain = "BOOLEAN_NUMERIC"
        else:
            metric_domain = None
        if nonfinite:
            classification = "NONFINITE_OBSERVED"
        elif self.differing_value_count:
            classification = "FINITE_BITWISE_DRIFT"
        else:
            classification = "BITWISE_EQUAL"
        return {
            "classification": classification,
            "value_count": self.value_count,
            "finite_pair_count": finite,
            "differing_value_count": self.differing_value_count,
            "bitwise_difference_rate": (
                self.differing_value_count / self.value_count if self.value_count else None
            ),
            "metric_domain": metric_domain,
            "source_nonfinite_count": self.source_nonfinite_count,
            "target_nonfinite_count": self.target_nonfinite_count,
            "source_dtypes": sorted(self.source_dtypes),
            "target_dtypes": sorted(self.target_dtypes),
            "mean_abs_delta": mean_abs,
            "rmse": rmse,
            "relative_l2_error": relative_l2,
            "max_abs_delta": self.max_abs_delta,
            "cosine_similarity": cosine,
        }


@dataclass
class IntegerAccumulator:
    value_count: int = 0
    dtype_name: str | None = None
    dtype_min: int | None = None
    dtype_max: int | None = None
    observed_min: int | None = None
    observed_max: int | None = None
    dtype_min_count: int = 0
    dtype_max_count: int = 0

    def update(self, values: np.ndarray, probe_id: str) -> None:
        array = np.asarray(values)
        if not np.issubdtype(array.dtype, np.integer) or array.dtype.itemsize != 1:
            raise ActivationAnalysisError(
                "INTEGER_ACTIVATION_DTYPE_INVALID",
                f"integer probe is not an 8-bit integer: {probe_id}",
            )
        dtype_name = str(array.dtype)
        limits = np.iinfo(array.dtype)
        if self.dtype_name is not None and self.dtype_name != dtype_name:
            raise ActivationAnalysisError(
                "INTEGER_ACTIVATION_DTYPE_CHURN",
                f"integer dtype changes across batches: {probe_id}",
            )
        self.dtype_name = dtype_name
        self.dtype_min = int(limits.min)
        self.dtype_max = int(limits.max)
        self.value_count += int(array.size)
        if array.size:
            current_min = int(np.min(array))
            current_max = int(np.max(array))
            self.observed_min = (
                current_min if self.observed_min is None else min(self.observed_min, current_min)
            )
            self.observed_max = (
                current_max if self.observed_max is None else max(self.observed_max, current_max)
            )
            self.dtype_min_count += int(np.count_nonzero(array == limits.min))
            self.dtype_max_count += int(np.count_nonzero(array == limits.max))

    def finish(self) -> dict[str, Any]:
        extreme_count = self.dtype_min_count + self.dtype_max_count
        rate = extreme_count / self.value_count if self.value_count else None
        return {
            "classification": (
                "DTYPE_EXTREME_VALUES_OBSERVED" if extreme_count else "NO_DTYPE_EXTREME_VALUES"
            ),
            "value_count": self.value_count,
            "dtype": self.dtype_name,
            "dtype_min": self.dtype_min,
            "dtype_max": self.dtype_max,
            "observed_min": self.observed_min,
            "observed_max": self.observed_max,
            "dtype_min_count": self.dtype_min_count,
            "dtype_max_count": self.dtype_max_count,
            "dtype_extreme_count": extreme_count,
            "dtype_extreme_rate": rate,
            "causal_interpretation": "NOT_ESTABLISHED",
        }


def _array_key(prefix: str, probe_id: str) -> str:
    return f"{prefix}__{probe_id.replace('-', '_')}"


def _query_ids(archive: Any, batch_name: str) -> list[str]:
    try:
        values = archive["query_ids"]
    except KeyError as exc:
        raise ActivationAnalysisError(
            "QUERY_IDENTITIES_MISSING",
            f"query identities are missing from {batch_name}",
        ) from exc
    if values.ndim != 1:
        raise ActivationAnalysisError(
            "QUERY_IDENTITIES_INVALID",
            f"query identities are not one-dimensional: {batch_name}",
        )
    return [str(value) for value in values.tolist()]


def _pearson(pairs: Sequence[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    values = np.asarray(pairs, dtype=np.float64)
    left = values[:, 0] - np.mean(values[:, 0], dtype=np.float64)
    right = values[:, 1] - np.mean(values[:, 1], dtype=np.float64)
    denominator = math.sqrt(
        float(np.sum(left * left, dtype=np.float64))
        * float(np.sum(right * right, dtype=np.float64))
    )
    if denominator == 0.0:
        return None
    return float(np.sum(left * right, dtype=np.float64) / denominator)


def analyze_activation_batches(
    root: Path,
    capture_plan: Mapping[str, Any],
    batch_index: Mapping[str, Any],
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    probe_mappings = capture_plan["probe_mappings"]
    integer_mappings = capture_plan["integer_mappings"]
    batch_records = batch_index.get("batches")
    if not isinstance(batch_records, list):
        raise ActivationAnalysisError(
            "ACTIVATION_BATCH_INDEX_INVALID", "activation batch records are missing"
        )
    floating_files = sorted(root.glob("batch-*-floating.npz"))
    integer_files = sorted(root.glob("batch-*-integer.npz"))
    expected_batches = len(batch_records)
    if len(floating_files) != expected_batches or len(integer_files) != expected_batches:
        raise ActivationAnalysisError(
            "ACTIVATION_BATCH_SET_INCOMPLETE",
            "activation batch files are incomplete",
        )
    probe_ids = [str(mapping["probe_id"]) for mapping in probe_mappings]
    integer_ids = [str(mapping["probe_id"]) for mapping in integer_mappings]
    floating_accumulators = {probe_id: FloatingAccumulator() for probe_id in probe_ids}
    integer_accumulators = {probe_id: IntegerAccumulator() for probe_id in integer_ids}
    observed_queries: list[str] = []
    expected_floating_keys = {"query_ids"}
    for probe_id in probe_ids:
        expected_floating_keys.add(_array_key("source", probe_id))
        expected_floating_keys.add(_array_key("target", probe_id))
    expected_integer_keys = {"query_ids"} | {
        _array_key("target_integer", probe_id) for probe_id in integer_ids
    }

    for batch_number, (floating_path, integer_path, batch_record) in enumerate(
        zip(floating_files, integer_files, batch_records, strict=True), start=1
    ):
        with (
            np.load(floating_path, allow_pickle=False) as floating,
            np.load(integer_path, allow_pickle=False) as integer,
        ):
            if set(floating.files) != expected_floating_keys:
                raise ActivationAnalysisError(
                    "FLOATING_BATCH_SCHEMA_MISMATCH",
                    f"floating batch schema differs: {floating_path.name}",
                )
            if set(integer.files) != expected_integer_keys:
                raise ActivationAnalysisError(
                    "INTEGER_BATCH_SCHEMA_MISMATCH",
                    f"integer batch schema differs: {integer_path.name}",
                )
            floating_queries = _query_ids(floating, floating_path.name)
            integer_queries = _query_ids(integer, integer_path.name)
            expected_queries = [str(value) for value in batch_record.get("query_ids", [])]
            if (
                floating_path.name != batch_record.get("floating_path")
                or integer_path.name != batch_record.get("integer_path")
                or floating_queries != integer_queries
                or floating_queries != expected_queries
            ):
                raise ActivationAnalysisError(
                    "BATCH_QUERY_IDENTITY_MISMATCH",
                    f"batch files or query identities differ: {floating_path.name}",
                )
            observed_queries.extend(floating_queries)
            for probe_id in probe_ids:
                floating_accumulators[probe_id].update(
                    floating[_array_key("source", probe_id)],
                    floating[_array_key("target", probe_id)],
                    probe_id,
                )
            for probe_id in integer_ids:
                integer_accumulators[probe_id].update(
                    integer[_array_key("target_integer", probe_id)],
                    probe_id,
                )
        if progress is not None:
            progress(
                {
                    "event": "activation_analysis_batch_complete",
                    "batch": batch_number,
                    "batch_count": expected_batches,
                    "query_count": len(floating_queries),
                }
            )
    if len(observed_queries) != int(capture_plan["query_count"]):
        raise ActivationAnalysisError(
            "QUERY_ORDER_MISMATCH",
            "observed query count differs from the capture plan",
        )

    records: list[dict[str, Any]] = []
    classifications: Counter[str] = Counter()
    correlation_pairs: list[tuple[float, float]] = []
    for order, mapping in enumerate(probe_mappings, start=1):
        probe_id = str(mapping["probe_id"])
        floating_summary = floating_accumulators[probe_id].finish()
        integer_summary = (
            integer_accumulators[probe_id].finish() if probe_id in integer_accumulators else None
        )
        classifications[floating_summary["classification"]] += 1
        if (
            integer_summary is not None
            and floating_summary["relative_l2_error"] is not None
            and integer_summary["dtype_extreme_rate"] is not None
        ):
            correlation_pairs.append(
                (
                    float(floating_summary["relative_l2_error"]),
                    float(integer_summary["dtype_extreme_rate"]),
                )
            )
        records.append(
            {
                "probe_id": probe_id,
                "probe_order": order,
                "target_tensor_basis": mapping.get("target_tensor_basis"),
                "structural_families": list(mapping.get("structural_families", [])),
                "floating": floating_summary,
                "integer_dtype_extremes": integer_summary,
            }
        )

    def ranking_key(record: Mapping[str, Any]) -> tuple[float, float, int]:
        floating = record["floating"]
        relative = floating["relative_l2_error"]
        return (
            -(float(relative) if relative is not None else -1.0),
            -float(floating["max_abs_delta"]),
            int(record["probe_order"]),
        )

    ranked = sorted(records, key=ranking_key)
    for rank, record in enumerate(ranked, start=1):
        record["divergence_rank"] = rank
    first_divergence = next(
        (
            {
                "probe_id": record["probe_id"],
                "probe_order": record["probe_order"],
                "target_tensor_basis": record["target_tensor_basis"],
                "structural_families": record["structural_families"],
            }
            for record in records
            if record["floating"]["differing_value_count"] > 0
        ),
        None,
    )
    return {
        "kind": "m1-diagnostic-activation-analysis",
        "status": "COMPLETE",
        "batch_count": expected_batches,
        "query_count": len(observed_queries),
        "probe_count": len(records),
        "integer_probe_count": len(integer_ids),
        "probes": records,
        "summary": {
            "classification_counts": dict(sorted(classifications.items())),
            "first_bitwise_divergence": first_divergence,
            "ranking_basis": "relative_l2_error_then_max_abs_delta",
            "ranked_probe_ids": [record["probe_id"] for record in ranked],
            "integer_extreme_correlation": {
                "pair_count": len(correlation_pairs),
                "pearson_relative_l2_vs_dtype_extreme_rate": _pearson(correlation_pairs),
                "interpretation": "DESCRIPTIVE_ASSOCIATION_NOT_CAUSATION",
            },
        },
    }
