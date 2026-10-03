from __future__ import annotations

from neural_continuity.m1_diagnostics.graph_inventory import (
    GraphInventory,
    NodeInventory,
    TensorInventory,
)
from neural_continuity.m1_diagnostics.graph_predecessors import (
    ProbePredecessor,
    graph_predecessor_edges,
    localize_dominant_onset,
)
from neural_continuity.m1_diagnostics.probe_plan import ProbePair


def _node(
    index: int, op_type: str, inputs: tuple[str, ...], outputs: tuple[str, ...]
) -> NodeInventory:
    return NodeInventory(index, f"node-{index}", op_type, "", inputs, outputs, (), True)


def test_graph_predecessors_traverse_unprobed_nodes_by_tensor_lineage() -> None:
    graph = GraphInventory(
        role="onnx_fp32_source",
        graph_sha256="source",
        ir_version=10,
        opsets=(("", 17),),
        inputs=(TensorInventory("input", "FLOAT", (1,)),),
        outputs=(TensorInventory("output", "FLOAT", (1,)),),
        initializer_count=0,
        op_counts=(),
        nodes=(
            _node(0, "MatMul", ("input",), ("probe-a",)),
            _node(1, "Identity", ("probe-a",), ("middle",)),
            _node(2, "ReduceMean", ("middle",), ("probe-b",)),
            _node(3, "Identity", ("probe-b",), ("output",)),
        ),
    )
    probes = (
        ProbePair("p1", 0, 0, "a", "MatMul", 0, "probe-a", "target-a", "direct", ()),
        ProbePair("p2", 2, 2, "b", "ReduceMean", 0, "probe-b", "target-b", "direct", ()),
    )

    assert graph_predecessor_edges(graph, probes) == (
        ProbePredecessor("p1", "input-0", None, "graph-input:p1:input-0"),
        ProbePredecessor("p2", "input-0/node-1-input-0", "p1"),
    )


def test_dominant_onset_preserves_maximum_growth_ties() -> None:
    result = localize_dominant_onset(
        probe_ids=("p1", "p2", "p3"),
        predecessor_edges=(
            ProbePredecessor("p2", "input-0", "p1"),
            ProbePredecessor("p3", "input-0", "p1"),
        ),
        probe_scores_hex={
            "p1": "0x0.0p+0",
            "p2": "0x1.0000000000000p-1",
            "p3": "0x1.0000000000000p-1",
        },
    )

    assert result["maximum_growth"] == "0x1.0000000000000p-1"
    assert result["dominant_onset_boundary_probe_ids"] == ["p2", "p3"]
    assert result["localization_status"] == "LOCALIZED"
