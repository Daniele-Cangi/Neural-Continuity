from pathlib import Path

import numpy as np
import pytest

from neural_continuity.m1_diagnostics.activation_metric_batches_v2 import (
    ProbeMetricLayout,
    analyze_captured_activation_metrics_v2,
)
from neural_continuity.m1_diagnostics.authority import DiagnosticPreflightError


def _capture(root: Path, source: np.ndarray, target: np.ndarray) -> tuple[dict, dict]:
    query_ids = np.asarray(["q-2", "q-1"])
    np.savez_compressed(
        root / "batch-0001-floating.npz",
        query_ids=query_ids,
        source__probe_0001=source,
        target__probe_0001=target,
    )
    np.savez_compressed(root / "batch-0001-integer.npz", query_ids=query_ids)
    return (
        {
            "query_count": 2,
            "probe_mappings": [{"probe_id": "probe-0001"}],
            "integer_mappings": [],
        },
        {
            "batches": [
                {
                    "batch_id": "batch-0001",
                    "floating_path": "batch-0001-floating.npz",
                    "integer_path": "batch-0001-integer.npz",
                    "query_ids": ["q-2", "q-1"],
                }
            ]
        },
    )


def test_joins_canonical_query_metrics_across_verified_capture_schema(tmp_path: Path) -> None:
    plan, index = _capture(
        tmp_path,
        np.asarray([[1.0, 2.0], [0.0, 0.0]], dtype=np.float32),
        np.asarray([[1.0, 2.0], [1.0, 0.0]], dtype=np.float32),
    )

    result = analyze_captured_activation_metrics_v2(
        tmp_path,
        plan,
        index,
        {"probe-0001": ProbeMetricLayout(masking="none", feature_axis=1)},
    )

    assert result["status"] == "METRICS_COMPUTED"
    assert result["probes"][0]["query_ids"] == ["q-1", "q-2"]
    assert result["probes"][0]["per_query"][0]["relative_l2_error"] is None
    assert (
        result["probes"][0]["aggregates"]["relative_l2_error"]["status"]
        == "UNBOUNDED_ZERO_REFERENCE"
    )


def test_sequence_probe_requires_recorded_mask(tmp_path: Path) -> None:
    source = np.zeros((2, 3, 2), dtype=np.float32)
    target = source.copy()
    target[:, 2, :] = 9.0
    plan, index = _capture(tmp_path, source, target)
    layout = {"probe-0001": ProbeMetricLayout(masking="sequence", sequence_axis=1, feature_axis=2)}

    with pytest.raises(DiagnosticPreflightError):
        analyze_captured_activation_metrics_v2(tmp_path, plan, index, layout)

    result = analyze_captured_activation_metrics_v2(
        tmp_path,
        plan,
        index,
        layout,
        {"batch-0001": np.asarray([[1, 1, 0], [1, 1, 0]], dtype=np.int64)},
    )
    assert result["probes"][0]["per_query"][0]["maximum_absolute_delta"] == float(0).hex()
