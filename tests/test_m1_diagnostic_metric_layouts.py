from dataclasses import replace

import pytest

from neural_continuity.m1_diagnostics.authority import DiagnosticPreflightError
from neural_continuity.m1_diagnostics.graph_inventory import GraphInventory, TensorInventory
from neural_continuity.m1_diagnostics.metric_layouts import (
    derive_probe_metric_layouts,
    replay_probe_metric_layouts,
)
from neural_continuity.m1_diagnostics.probe_plan import ProbePair, ProbePlan


def _fixtures(shape: tuple[int | str, ...]) -> tuple[GraphInventory, GraphInventory, ProbePlan]:
    source = GraphInventory(
        role="onnx_fp32_source",
        graph_sha256="source",
        ir_version=10,
        opsets=(("", 17),),
        inputs=(TensorInventory("mask-a", "INT64", ("batch", "sequence")),),
        outputs=(),
        initializer_count=0,
        op_counts=(),
        nodes=(),
        value_info=(TensorInventory("tensor-a", "FLOAT", shape),),
    )
    target_shape = tuple(
        "other-batch" if dim == "batch" else "other-sequence" if dim == "sequence" else dim
        for dim in shape
    )
    target = replace(
        source,
        role="onnx_int8_candidate",
        graph_sha256="target",
        inputs=(TensorInventory("mask-b", "INT64", ("other-batch", "other-sequence")),),
        value_info=(TensorInventory("tensor-b", "FLOAT", target_shape),),
    )
    probe = ProbePair(
        "p1", 0, 0, "", "Identity", 0, "tensor-a", "tensor-b", "direct_compute_output", ()
    )
    return source, target, ProbePlan("source", "target", (probe,))


def _derive(source: GraphInventory, target: GraphInventory, plan: ProbePlan) -> dict:
    return derive_probe_metric_layouts(
        source, target, plan, source_mask_input_position=0, target_mask_input_position=0
    )


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        (("batch", "sequence", 8), ("sequence", 1, 2)),
        (("batch", 8, "sequence"), ("sequence", 2, None)),
        (("batch", 2, "sequence", "sequence"), ("attention_scores", None, None)),
        (("batch", 8), ("none", None, 1)),
    ],
)
def test_layouts_derive_only_from_declared_dimensions(shape: tuple, expected: tuple) -> None:
    source, target, plan = _fixtures(shape)
    result = _derive(source, target, plan)
    layout = result["layouts"]["p1"]
    assert (layout["masking"], layout["sequence_axis"], layout["feature_axis"]) == expected
    assert len(result["layout_sha256"]) == 64
    assert result["model_execution_used"] is False
    assert result["execution_authorized"] is False
    assert result == _derive(source, target, plan)
    assert replay_probe_metric_layouts(
        result, source, target, plan, source_mask_input_position=0, target_mask_input_position=0
    ) == {"replay_verified": True, "model_execution_used": False}


@pytest.mark.parametrize(
    "shape",
    [
        (2, "sequence", 8),
        ("batch", "?", 8),
        ("batch", "sequence", "sequence"),
        ("batch", 2, 8),
        ("batch", 0),
        ("batch", "batch", 8),
    ],
)
def test_ambiguous_dimensions_block(shape: tuple) -> None:
    source, target, plan = _fixtures(shape)
    with pytest.raises(DiagnosticPreflightError) as error:
        _derive(source, target, plan)
    assert error.value.status == "BLOCKED"


def test_missing_probe_shape_blocks() -> None:
    source, target, plan = _fixtures(("batch", "sequence", 8))
    with pytest.raises(DiagnosticPreflightError):
        _derive(replace(source, value_info=()), target, plan)


def test_paired_shape_mismatch_blocks() -> None:
    source, target, plan = _fixtures(("batch", "sequence", 8))
    target = replace(
        target,
        value_info=(TensorInventory("tensor-b", "FLOAT", ("other-batch", "other-sequence", 9)),),
    )
    with pytest.raises(DiagnosticPreflightError):
        _derive(source, target, plan)


def test_probe_plan_identity_mismatch_blocks() -> None:
    source, target, plan = _fixtures(("batch", 8))
    with pytest.raises(DiagnosticPreflightError):
        _derive(source, target, replace(plan, source_graph_sha256="different"))


def test_altered_layout_fails_model_free_replay() -> None:
    source, target, plan = _fixtures(("batch", "sequence", 8))
    recorded = _derive(source, target, plan)
    recorded["layouts"]["p1"]["sequence_axis"] = 2
    with pytest.raises(DiagnosticPreflightError) as error:
        replay_probe_metric_layouts(
            recorded,
            source,
            target,
            plan,
            source_mask_input_position=0,
            target_mask_input_position=0,
        )
    assert error.value.status == "BLOCKED"
