"""Conservative structural recognition for decomposed normalization graphs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def find_structural_normalizations(
    nodes: list[Any], initializer_names: set[str]
) -> tuple[dict[str, object], ...]:
    """Recognize exact RMSNorm and mean-centered LayerNorm operator patterns.

    Recognition is based on tensor provenance and operator structure only. Unknown
    decompositions are intentionally left unmatched instead of being inferred from
    node names, tensor names, or model-specific conventions.
    """
    producer_by_tensor = {
        str(output): index for index, node in enumerate(nodes) for output in node.output if output
    }
    consumers_by_tensor: dict[str, list[int]] = defaultdict(list)
    for index, node in enumerate(nodes):
        for value in node.input:
            if value:
                consumers_by_tensor[str(value)].append(index)

    matches: dict[tuple[object, ...], dict[str, object]] = {}
    for divide_index, divide in enumerate(nodes):
        if str(divide.op_type) != "Div":
            continue
        inputs = _nonempty_inputs(divide)
        if len(inputs) != 2:
            continue
        numerator, denominator = inputs
        denominator_nodes = _denominator_lineage(
            denominator, nodes, producer_by_tensor, initializer_names
        )
        if denominator_nodes is None:
            continue
        reduction_index, variance_input, denominator_path = denominator_nodes
        square_index = producer_by_tensor.get(variance_input)
        if square_index is None or str(nodes[square_index].op_type) != "Mul":
            continue
        square_inputs = _nonempty_inputs(nodes[square_index])
        if len(square_inputs) != 2 or square_inputs[0] != square_inputs[1]:
            continue

        source_tensor = square_inputs[0]
        pattern_indices = {square_index, reduction_index, divide_index}
        pattern_indices.update(denominator_path)
        source_producer = producer_by_tensor.get(source_tensor)

        if numerator != source_tensor:
            continue
        rule_id = "rms_norm_reduce_mean_sqrt_v1"
        if source_producer is not None and str(nodes[source_producer].op_type) == "Sub":
            centered_inputs = _nonempty_inputs(nodes[source_producer])
            if len(centered_inputs) != 2 or centered_inputs[1] not in producer_by_tensor:
                continue
            mean_index = producer_by_tensor[centered_inputs[1]]
            mean_node = nodes[mean_index]
            mean_inputs = _nonempty_inputs(mean_node)
            if (
                str(mean_node.op_type) != "ReduceMean"
                or len(mean_inputs) != 1
                or mean_inputs[0] != centered_inputs[0]
                or numerator != source_tensor
            ):
                continue
            rule_id = "mean_centered_variance_reduce_mean_sqrt_v1"
            source_tensor = centered_inputs[0]
            pattern_indices.update((source_producer, mean_index))
        output_tensor, boundary_index, affine_indices = _affine_tail(
            divide_index, nodes, consumers_by_tensor, initializer_names
        )
        pattern_indices.update(affine_indices)
        key = (rule_id, tuple(sorted(pattern_indices)), source_tensor, output_tensor)
        matches[key] = {
            "rule_id": rule_id,
            "node_indices": list(sorted(pattern_indices)),
            "boundary_node_index": boundary_index,
            "input_tensor": source_tensor,
            "output_tensor": output_tensor,
        }

    return tuple(matches[key] for key in sorted(matches, key=repr))


def _denominator_lineage(
    tensor: str,
    nodes: list[Any],
    producer_by_tensor: dict[str, int],
    initializer_names: set[str],
) -> tuple[int, str, set[int]] | None:
    sqrt_index = producer_by_tensor.get(tensor)
    if sqrt_index is None or str(nodes[sqrt_index].op_type) != "Sqrt":
        return None
    sqrt_inputs = _nonempty_inputs(nodes[sqrt_index])
    if len(sqrt_inputs) != 1:
        return None
    variance_tensor = sqrt_inputs[0]
    add_index = producer_by_tensor.get(variance_tensor)
    path_indices = {sqrt_index}
    if add_index is not None and str(nodes[add_index].op_type) == "Add":
        add_inputs = _nonempty_inputs(nodes[add_index])
        if len(add_inputs) != 2:
            return None
        reduction_tensors = [
            value
            for value in add_inputs
            if (producer := producer_by_tensor.get(value)) is not None
            and str(nodes[producer].op_type) == "ReduceMean"
        ]
        static_inputs = [
            value
            for value in add_inputs
            if value in initializer_names
            or (
                (producer := producer_by_tensor.get(value)) is not None
                and str(nodes[producer].op_type) == "Constant"
            )
        ]
        if len(reduction_tensors) != 1 or len(static_inputs) != 1:
            return None
        variance_tensor = reduction_tensors[0]
        path_indices.add(add_index)

    reduction_index = producer_by_tensor.get(variance_tensor)
    if reduction_index is None or str(nodes[reduction_index].op_type) != "ReduceMean":
        return None
    reduction_inputs = _nonempty_inputs(nodes[reduction_index])
    if len(reduction_inputs) != 1:
        return None
    path_indices.add(reduction_index)
    return reduction_index, reduction_inputs[0], path_indices


def _affine_tail(
    start_index: int,
    nodes: list[Any],
    consumers_by_tensor: dict[str, list[int]],
    initializer_names: set[str],
) -> tuple[str, int, set[int]]:
    current_index = start_index
    current_tensor = str(nodes[current_index].output[0])
    affine_indices: set[int] = set()
    while True:
        consumers = sorted(set(consumers_by_tensor.get(current_tensor, ())))
        if len(consumers) != 1:
            break
        candidate_index = consumers[0]
        candidate = nodes[candidate_index]
        if str(candidate.op_type) not in {"Mul", "Add"}:
            break
        inputs = _nonempty_inputs(candidate)
        activation_inputs = [value for value in inputs if value == current_tensor]
        static_inputs = [value for value in inputs if value in initializer_names]
        if len(inputs) != 2 or len(activation_inputs) != 1 or len(static_inputs) != 1:
            break
        outputs = [str(value) for value in candidate.output if value]
        if len(outputs) != 1:
            break
        current_index = candidate_index
        current_tensor = outputs[0]
        affine_indices.add(candidate_index)
    return current_tensor, current_index, affine_indices


def _nonempty_inputs(node: Any) -> list[str]:
    return [str(value) for value in node.input if value]
