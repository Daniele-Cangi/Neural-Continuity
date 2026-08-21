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
