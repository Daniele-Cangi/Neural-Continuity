from __future__ import annotations

from types import SimpleNamespace

from neural_continuity.m1_diagnostics.graph_inventory import _output_ancestors


def _node(inputs: list[str], outputs: list[str]) -> SimpleNamespace:
    return SimpleNamespace(input=inputs, output=outputs)


def test_output_ancestors_follow_dag_predecessors_without_name_exceptions() -> None:
    nodes = [
        _node(["input"], ["branch_a"]),
        _node(["input"], ["unrelated"]),
        _node(["branch_a"], ["branch_b"]),
        _node(["branch_b"], ["output"]),
    ]

    assert _output_ancestors(nodes, {"output"}) == {0, 2, 3}
