from types import SimpleNamespace

from neural_continuity.m1_diagnostics.normalization_patterns import (
    find_structural_normalizations,
)


def _node(op_type: str, inputs: tuple[str, ...], output: str) -> SimpleNamespace:
    return SimpleNamespace(op_type=op_type, input=inputs, output=(output,))


def test_recognizes_mean_centered_variance_and_affine_tail_without_names() -> None:
    nodes = [
        _node("ReduceMean", ("x",), "mean"),
        _node("Sub", ("x", "mean"), "centered"),
        _node("Mul", ("centered", "centered"), "squared"),
        _node("ReduceMean", ("squared",), "variance"),
        _node("Add", ("variance", "epsilon"), "stabilized"),
        _node("Sqrt", ("stabilized",), "std"),
        _node("Div", ("centered", "std"), "normalized"),
        _node("Mul", ("normalized", "scale"), "scaled"),
        _node("Add", ("scaled", "bias"), "result"),
    ]

    matches = find_structural_normalizations(nodes, {"epsilon", "scale", "bias"})

    assert len(matches) == 1
    assert matches[0]["rule_id"] == "mean_centered_variance_reduce_mean_sqrt_v1"
    assert matches[0]["input_tensor"] == "x"
    assert matches[0]["output_tensor"] == "result"
    assert matches[0]["boundary_node_index"] == 8
    assert matches[0]["node_indices"] == list(range(9))


def test_recognizes_rms_norm_only_when_divisor_uses_same_input_lineage() -> None:
    nodes = [
        _node("Mul", ("x", "x"), "squared"),
        _node("ReduceMean", ("squared",), "variance"),
        _node("Add", ("variance", "epsilon"), "stabilized"),
        _node("Sqrt", ("stabilized",), "rms"),
        _node("Div", ("x", "rms"), "normalized"),
    ]

    matches = find_structural_normalizations(nodes, {"epsilon"})

    assert len(matches) == 1
    assert matches[0]["rule_id"] == "rms_norm_reduce_mean_sqrt_v1"
    assert matches[0]["input_tensor"] == "x"
    assert matches[0]["output_tensor"] == "normalized"
    assert matches[0]["node_indices"] == list(range(5))


def test_does_not_match_variance_from_a_different_input() -> None:
    nodes = [
        _node("Mul", ("other", "other"), "squared"),
        _node("ReduceMean", ("squared",), "variance"),
        _node("Sqrt", ("variance",), "rms"),
        _node("Div", ("x", "rms"), "normalized"),
    ]

    assert find_structural_normalizations(nodes, set()) == ()
