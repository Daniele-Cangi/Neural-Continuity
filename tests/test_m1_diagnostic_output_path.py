from __future__ import annotations

from types import SimpleNamespace

from neural_continuity.m1_diagnostics.graph_inventory import _output_path_indices


def _node(op_type: str, inputs: tuple[str, ...], outputs: tuple[str, ...]) -> SimpleNamespace:
    return SimpleNamespace(op_type=op_type, input=inputs, output=outputs)


def test_output_path_starts_at_first_reduction_and_adds_nearest_norm() -> None:
    nodes = [
        _node("MatMul", ("input", "weight"), ("computed",)),
        _node("LayerNormalization", ("computed",), ("normalized",)),
        _node("ReduceMean", ("normalized",), ("pooled",)),
        _node("ReduceMax", ("pooled",), ("reduced_again",)),
        _node("Gemm", ("reduced_again", "projection"), ("embedding",)),
    ]

    assert _output_path_indices(nodes, {0, 1, 2, 3, 4}) == {1, 2, 3, 4}


def test_output_path_stops_normalization_search_at_attention_boundary() -> None:
    nodes = [
        _node("LayerNormalization", ("input",), ("normalized",)),
        _node("MatMul", ("normalized", "weight"), ("attended",)),
        _node("ReduceMean", ("attended",), ("pooled",)),
    ]

    assert _output_path_indices(nodes, {0, 1, 2}) == {2}
