from __future__ import annotations

import numpy as np

from neural_continuity.m1_diagnostics.activation_metrics_v2 import (
    CANONICAL_METRIC_ALGORITHM_VERSION,
    compute_activation_metrics_v2,
    replay_activation_metrics_v2,
)
from neural_continuity.m1_diagnostics.metric_arithmetic import pairwise_sum


def test_pairwise_sum_uses_fixed_adjacent_tree() -> None:
    assert pairwise_sum((1e16, 1.0, -1e16, 1.0)) == 0.0


def test_metrics_sort_queries_and_exclude_masked_sequence_positions() -> None:
    result = compute_activation_metrics_v2(
        probe_id="probe-0001",
        query_ids=("q2", "q1"),
        source=np.array([[[1.0], [1.0], [999.0]], [[0.0], [777.0], [888.0]]]),
        target=np.array([[[2.0], [2.0], [-999.0]], [[0.0], [-777.0], [-888.0]]]),
        attention_mask=np.array([[1, 1, 0], [1, 0, 0]], dtype=np.int64),
        sequence_axis=1,
        feature_axis=2,
    )

    assert result["query_ids"] == ["q1", "q2"]
    assert result["per_query"][0]["valid_element_count"] == 1
    assert result["per_query"][0]["maximum_absolute_delta"] == "0x0.0p+0"
    assert result["per_query"][1]["valid_element_count"] == 2
    assert result["per_query"][1]["maximum_absolute_delta"] == "0x1.0000000000000p+0"


def test_zero_reference_uses_unbounded_marker_and_replays_model_free() -> None:
    observations = {
        "probe_id": "probe-0002",
        "query_ids": ("q1",),
        "source": np.array([[0.0]]),
        "target": np.array([[1.0]]),
    }
    result = compute_activation_metrics_v2(**observations)
    assert result["per_query"][0]["relative_l2_error"] is None
    assert result["per_query"][0]["relative_l2_status"] == "UNBOUNDED_ZERO_REFERENCE"

    replay = replay_activation_metrics_v2(
        recorded_metrics=result,
        declared_algorithm_version=CANONICAL_METRIC_ALGORITHM_VERSION,
        declared_implementation_sha256=result["implementation_sha256"],
        **observations,
    )
    assert replay["status"] == "PASS"
    assert replay["model_execution_used"] is False
