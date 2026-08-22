from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neural_continuity.m1_diagnostics.activation_evidence import (
    finalize_activation_package,
    prepare_capture_package,
    replay_activation_capture,
    write_activation_batch,
)
from neural_continuity.m1_diagnostics.fidelity_authority import FidelityGateError


def _plan() -> dict[str, object]:
    return {
        "kind": "test_capture_plan",
        "query_count": 2,
        "probe_mappings": [
            {
                "probe_id": "probe-0001",
                "target_tensor_basis": "post_quantize_dequantize_output",
            },
            {
                "probe_id": "probe-0002",
                "target_tensor_basis": "direct_compute_output",
            },
        ],
        "integer_mappings": [{"probe_id": "probe-0001"}],
    }


def _capture(output: Path) -> dict[str, object]:
    build_root = output.parent / f".{output.name}.building"
    build_root.mkdir()
    (build_root / "target-integer-capture.onnx").write_bytes(b"derived-graph")
    prepare_capture_package(
        build_root,
        _plan(),
        {
            "status": "PASS",
            "derivative_final_output_fidelity": "PASS",
            "activations_read_before_preflight": False,
        },
    )
    record = write_activation_batch(
        build_root,
        "batch-0001",
        ["q-001", "q-002"],
        _plan()["probe_mappings"],
        _plan()["integer_mappings"],
        [
            np.asarray([[1.0], [2.0]], dtype=np.float32),
            np.asarray(2, dtype=np.int64),
        ],
        [
            np.asarray([[1.0], [2.0]], dtype=np.float32),
            np.asarray(2, dtype=np.int64),
        ],
        [np.asarray([[0, 255], [4, 5]], dtype=np.uint8)],
    )
    return finalize_activation_package(
        build_root,
        output,
        {
            "batch_size": 16,
            "batch_count": 1,
            "query_count": 2,
            "batches": [record],
        },
    )


def test_activation_package_is_deterministic_and_model_free_replayable(
    tmp_path: Path,
) -> None:
    first = _capture(tmp_path / "first")
    second = _capture(tmp_path / "second")
    assert first["artifact_manifest_sha256"] == second["artifact_manifest_sha256"]
    replay = replay_activation_capture(
        tmp_path / "first" / "replay-bundle.json",
        str(first["artifact_manifest_sha256"]),
    )
    assert replay["status"] == "COMPLETE"
    assert replay["replay_verified"] is True
    assert replay["summary_match"] is True
    assert replay["model_execution_used"] is False
    assert replay["floating_probe_count"] == 2
    assert replay["integer_probe_count"] == 1


def test_activation_pair_shape_mismatch_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FidelityGateError) as error:
        write_activation_batch(
            tmp_path,
            "batch-0001",
            ["q-001", "q-002"],
            [{"probe_id": "probe-0001", "target_tensor_basis": "direct_compute_output"}],
            [],
            [np.zeros((2, 3), dtype=np.float32)],
            [np.zeros((2, 4), dtype=np.float32)],
            [],
        )
    assert error.value.code == "PAIRED_ACTIVATION_SHAPE_MISMATCH"
    assert error.value.status == "BLOCKED"


def test_activation_replay_fails_closed_for_missing_batch(tmp_path: Path) -> None:
    captured = _capture(tmp_path / "missing")
    (tmp_path / "missing" / "batch-0001-integer.npz").unlink()
    with pytest.raises(FidelityGateError) as error:
        replay_activation_capture(
            tmp_path / "missing" / "replay-bundle.json",
            str(captured["artifact_manifest_sha256"]),
        )
    assert error.value.code == "ARTIFACT_MISSING"
    assert error.value.status == "BLOCKED"
