from pathlib import Path

import numpy as np
import pytest

from neural_continuity.m1_diagnostics.activation_evidence import (
    finalize_activation_package,
    prepare_capture_package,
    write_activation_batch,
)
from neural_continuity.m1_diagnostics.activation_inputs import INPUT_CAPTURE_VERSION
from neural_continuity.m1_diagnostics.activation_metric_replay_v2 import (
    analyze_recorded_activation_metrics_v2,
    replay_recorded_activation_metrics_v2,
)
from neural_continuity.m1_diagnostics.authority import DiagnosticPreflightError
from neural_continuity.m1_diagnostics.fidelity_authority import FidelityGateError
from neural_continuity.m1_diagnostics.graph_inventory import GraphInventory, TensorInventory
from neural_continuity.m1_diagnostics.metric_layouts import derive_probe_metric_layouts
from neural_continuity.m1_diagnostics.probe_plan import ProbePair, ProbePlan


def _package(tmp_path: Path, *, bind_layout: bool = True, width: int = 2) -> tuple:
    graphs = []
    for role, digest, name in (
        ("onnx_fp32_source", "source", "left"),
        ("onnx_int8_candidate", "target", "right"),
    ):
        graphs.append(
            GraphInventory(
                role=role,
                graph_sha256=digest,
                ir_version=10,
                opsets=(("", 17),),
                inputs=(TensorInventory("mask", "INT64", ("B", "S")),),
                outputs=(TensorInventory(name, "FLOAT", ("B", "S", 2)),),
                initializer_count=0,
                op_counts=(),
                nodes=(),
            )
        )
    source, target = graphs
    probe = ProbePair("p1", 0, 0, "", "Identity", 0, "left", "right", "direct", ())
    plan = ProbePlan("source", "target", (probe,))
    layouts = derive_probe_metric_layouts(
        source, target, plan, source_mask_input_position=0, target_mask_input_position=0
    )
    capture_plan = {
        "query_count": 2,
        "probe_mappings": [probe.to_dict()],
        "integer_mappings": [],
        "probe_plan_sha256": plan.sha256,
        "input_capture_version": INPUT_CAPTURE_VERSION,
    }
    if bind_layout:
        capture_plan["metric_layout_sha256"] = layouts["layout_sha256"]
    build = tmp_path / "build"
    build.mkdir()
    prepare_capture_package(
        build,
        capture_plan,
        {
            "status": "PASS",
            "derivative_final_output_fidelity": "PASS",
            "activations_read_before_preflight": False,
        },
    )
    left = np.zeros((2, 3, width), dtype=np.float32)
    right = left.copy()
    right[:, 2, :] = 100.0
    record = write_activation_batch(
        build,
        "batch-0001",
        ["q1", "q2"],
        [probe.to_dict()],
        [],
        [left],
        [right],
        [],
        token_inputs={
            "input_ids": np.asarray([[7, 8, 0], [9, 10, 0]], dtype=np.int64),
            "attention_mask": np.asarray([[1, 1, 0], [1, 1, 0]], dtype=np.int64),
            "token_type_ids": np.zeros((2, 3), dtype=np.int64),
        },
    )
    root = tmp_path / "capture"
    result = finalize_activation_package(build, root, {"batches": [record]})
    return root, result["artifact_manifest_sha256"], layouts, source, target, plan


def _analyze(package: tuple) -> dict:
    return analyze_recorded_activation_metrics_v2(
        *package, source_mask_input_position=0, target_mask_input_position=0
    )


def test_paired_package_replays_metrics_using_actual_mask(tmp_path: Path) -> None:
    package = _package(tmp_path)
    result = _analyze(package)
    assert result["actual_input_batches_verified"] == 1
    assert result["diagnostic_complete"] is False
    for row in result["probes"][0]["per_query"]:
        assert row["maximum_absolute_delta"] == float(0).hex()
    assert replay_recorded_activation_metrics_v2(
        result, *package, source_mask_input_position=0, target_mask_input_position=0
    ) == {"replay_verified": True, "model_execution_used": False}


def test_missing_precapture_layout_binding_blocks(tmp_path: Path) -> None:
    with pytest.raises(DiagnosticPreflightError) as error:
        _analyze(_package(tmp_path, bind_layout=False))
    assert error.value.status == "BLOCKED"


def test_actual_width_must_match_declared_graph_width(tmp_path: Path) -> None:
    with pytest.raises(DiagnosticPreflightError) as error:
        _analyze(_package(tmp_path, width=3))
    assert error.value.code == "METRIC_DECLARED_SHAPE_MISMATCH"


def test_missing_actual_inputs_blocks_integration(tmp_path: Path) -> None:
    package = _package(tmp_path)
    (package[0] / "batch-0001-inputs.npz").unlink()
    with pytest.raises(FidelityGateError) as error:
        _analyze(package)
    assert error.value.status == "BLOCKED"


def test_changed_metric_report_blocks_replay(tmp_path: Path) -> None:
    package = _package(tmp_path)
    report = _analyze(package)
    report["actual_input_batches_verified"] = 9
    with pytest.raises(DiagnosticPreflightError):
        replay_recorded_activation_metrics_v2(
            report, *package, source_mask_input_position=0, target_mask_input_position=0
        )
