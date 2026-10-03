from __future__ import annotations

from neural_continuity.m1_diagnostics.graph_inventory import (
    GraphInventory,
    NodeInventory,
    TensorInventory,
)
from neural_continuity.m1_diagnostics.probe_plan import build_probe_plan


def _node(
    index: int,
    name: str,
    op_type: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    families: tuple[str, ...],
) -> NodeInventory:
    return NodeInventory(index, name, op_type, "", inputs, outputs, families, True)


def _inventory(role: str, nodes: tuple[NodeInventory, ...]) -> GraphInventory:
    return GraphInventory(
        role=role,  # type: ignore[arg-type]
        graph_sha256=role,
        ir_version=10,
        opsets=(("", 17),),
        inputs=(TensorInventory("input", "FLOAT", (1,)),),
        outputs=(TensorInventory("output", "FLOAT", (1,)),),
        initializer_count=0,
        op_counts=(),
        nodes=nodes,
    )


def test_probe_plan_uses_uniform_structural_lineage_and_post_qdq_output() -> None:
    source = _inventory(
        "onnx_fp32_source",
        (
            _node(
                0,
                "compute",
                "MatMul",
                ("input",),
                ("output",),
                ("ATTENTION_OR_MATMUL",),
            ),
        ),
    )
    target = _inventory(
        "onnx_int8_candidate",
        (
            _node(0, "compute", "MatMul", ("input",), ("output",), ("QUANTIZED_COMPUTE",)),
            _node(
                1,
                "quant",
                "QuantizeLinear",
                ("output",),
                ("quantized",),
                ("QUANTIZATION_BOUNDARY",),
            ),
            _node(
                2,
                "dequant",
                "DequantizeLinear",
                ("quantized",),
                ("dequantized",),
                ("QUANTIZATION_BOUNDARY",),
            ),
        ),
    )

    plan = build_probe_plan(source, target)

    assert len(plan.probes) == 1
    assert plan.probes[0].source_tensor == "output"
    assert plan.probes[0].target_tensor == "dequantized"
    assert plan.probes[0].target_tensor_basis == "post_quantize_dequantize_output"
    assert len(plan.sha256) == 64


def test_probe_plan_pairs_structurally_equivalent_nodes_with_different_names() -> None:
    source = _inventory(
        "onnx_fp32_source",
        (
            _node(
                0,
                "source_compute",
                "MatMul",
                ("input", "source_weight"),
                ("source_activation",),
                ("ATTENTION_OR_MATMUL",),
            ),
            _node(
                1,
                "source_output",
                "Identity",
                ("source_activation",),
                ("output",),
                ("FINAL_OUTPUT",),
            ),
        ),
    )
    target = _inventory(
        "onnx_int8_candidate",
        (
            _node(
                0,
                "candidate_compute",
                "QLinearMatMul",
                (
                    "input",
                    "a_scale",
                    "a_zero",
                    "candidate_weight",
                    "b_scale",
                    "b_zero",
                    "y_scale",
                    "y_zero",
                ),
                ("candidate_quantized",),
                ("QUANTIZED_COMPUTE",),
            ),
            _node(
                1,
                "candidate_dequantize",
                "DequantizeLinear",
                ("candidate_quantized", "y_scale", "y_zero"),
                ("candidate_activation",),
                ("QUANTIZATION_BOUNDARY",),
            ),
            _node(
                2,
                "candidate_output",
                "Identity",
                ("candidate_activation",),
                ("output",),
                ("FINAL_OUTPUT",),
            ),
        ),
    )

    plan = build_probe_plan(source, target)
    compute_probe = next(probe for probe in plan.probes if probe.source_node_index == 0)

    assert compute_probe.target_node_index == 0
    assert compute_probe.source_tensor == "source_activation"
    assert compute_probe.target_tensor == "candidate_activation"
    assert compute_probe.target_tensor_basis == "post_dequantize_output"


def test_probe_plan_blocks_quantized_output_without_float_lineage() -> None:
    from neural_continuity.m1_diagnostics.authority import DiagnosticPreflightError

    source = _inventory(
        "onnx_fp32_source",
        (
            _node(
                0,
                "source_compute",
                "MatMul",
                ("input", "source_weight"),
                ("output",),
                ("ATTENTION_OR_MATMUL",),
            ),
        ),
    )
    target = _inventory(
        "onnx_int8_candidate",
        (
            _node(
                0,
                "candidate_compute",
                "QLinearMatMul",
                (
                    "input",
                    "a_scale",
                    "a_zero",
                    "candidate_weight",
                    "b_scale",
                    "b_zero",
                    "y_scale",
                    "y_zero",
                ),
                ("output",),
                ("QUANTIZED_COMPUTE",),
            ),
        ),
    )

    try:
        build_probe_plan(source, target)
    except DiagnosticPreflightError as exc:
        assert exc.status == "BLOCKED"
        assert exc.code == "POST_QDQ_FLOAT_LINEAGE_MISSING"
    else:
        raise AssertionError("quantized integer output cannot be treated as a float activation")


def test_probe_plan_blocks_ambiguous_structural_classes() -> None:
    from neural_continuity.m1_diagnostics.authority import DiagnosticPreflightError

    source = _inventory(
        "onnx_fp32_source",
        (
            _node(
                0,
                "source_a",
                "MatMul",
                ("input", "weight_a"),
                ("a",),
                ("ATTENTION_OR_MATMUL",),
            ),
            _node(
                1,
                "source_b",
                "MatMul",
                ("input", "weight_b"),
                ("b",),
                ("ATTENTION_OR_MATMUL",),
            ),
            _node(
                2,
                "source_output",
                "Identity",
                ("input",),
                ("output",),
                ("FINAL_OUTPUT",),
            ),
        ),
    )
    target = _inventory(
        "onnx_int8_candidate",
        (
            _node(
                0,
                "target_a",
                "MatMulInteger",
                ("input", "weight_a"),
                ("a",),
                ("QUANTIZED_COMPUTE",),
            ),
            _node(
                1,
                "target_b",
                "MatMulInteger",
                ("input", "weight_b"),
                ("b",),
                ("QUANTIZED_COMPUTE",),
            ),
            _node(
                2,
                "target_output",
                "Identity",
                ("input",),
                ("output",),
                ("FINAL_OUTPUT",),
            ),
        ),
    )

    try:
        build_probe_plan(source, target)
    except DiagnosticPreflightError as exc:
        assert exc.status == "BLOCKED"
        assert exc.code == "PROBE_LINEAGE_NOT_BIJECTIVE"
    else:
        raise AssertionError("duplicate structural classes must fail closed")
