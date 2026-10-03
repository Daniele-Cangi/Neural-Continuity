"""Deterministic structural inventory for authority-verified ONNX graphs."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

from neural_continuity.m1_diagnostics.authority import (
    DiagnosticPreflightError,
    VerifiedAuthority,
    VerifiedAuthoritySet,
)

GraphRole = Literal["onnx_fp32_source", "onnx_int8_candidate"]

_QUANTIZATION_OPS = frozenset(
    {
        "QuantizeLinear",
        "DequantizeLinear",
        "QLinearMatMul",
        "QLinearConv",
        "MatMulInteger",
        "ConvInteger",
    }
)
_NORMALIZATION_OPS = frozenset(
    {
        "BatchNormalization",
        "InstanceNormalization",
        "LayerNormalization",
        "LpNormalization",
    }
)
_ATTENTION_COMPUTE_OPS = frozenset(
    {
        "Attention",
        "MultiHeadAttention",
        "MatMul",
        "MatMulInteger",
        "QLinearMatMul",
        "Gemm",
    }
)
_OUTPUT_PATH_REDUCTIONS = frozenset(
    {
        "ReduceMean",
        "ReduceSum",
        "ReduceMax",
        "GlobalAveragePool",
        "GlobalMaxPool",
    }
)
_FAMILY_ORDER = (
    "QUANTIZATION_BOUNDARY",
    "QUANTIZED_COMPUTE",
    "NORMALIZATION",
    "ATTENTION_OR_MATMUL",
    "OUTPUT_PATH",
    "FINAL_OUTPUT",
)


@dataclass(frozen=True)
class LoadedOnnxGraph:
    role: GraphRole
    authority: VerifiedAuthority
    model: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class TensorInventory:
    name: str
    data_type: str
    shape: tuple[int | str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "data_type": self.data_type, "shape": list(self.shape)}


@dataclass(frozen=True)
class NodeInventory:
    index: int
    name: str
    op_type: str
    domain: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    structural_families: tuple[str, ...]
    is_output_ancestor: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "op_type": self.op_type,
            "domain": self.domain,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "structural_families": list(self.structural_families),
            "is_output_ancestor": self.is_output_ancestor,
        }


@dataclass(frozen=True)
class GraphInventory:
    role: GraphRole
    graph_sha256: str
    ir_version: int
    opsets: tuple[tuple[str, int], ...]
    inputs: tuple[TensorInventory, ...]
    outputs: tuple[TensorInventory, ...]
    initializer_count: int
    op_counts: tuple[tuple[str, int], ...]
    nodes: tuple[NodeInventory, ...]
    structural_rules: tuple[dict[str, object], ...] = ()
    value_info: tuple[TensorInventory, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "m1_transition_b_v2_static_graph_inventory",
            "role": self.role,
            "graph_sha256": self.graph_sha256,
            "ir_version": self.ir_version,
            "opsets": [{"domain": domain, "version": version} for domain, version in self.opsets],
            "inputs": [value.to_dict() for value in self.inputs],
            "outputs": [value.to_dict() for value in self.outputs],
            "initializer_count": self.initializer_count,
            "node_count": len(self.nodes),
            "op_counts": {op_type: count for op_type, count in self.op_counts},
            "nodes": [node.to_dict() for node in self.nodes],
            "structural_rules": [dict(rule) for rule in self.structural_rules],
            "value_info": [value.to_dict() for value in self.value_info],
            "inventory_basis": "structural_onnx_dag",
            "model_execution_used": False,
        }


def load_verified_graph(authorities: VerifiedAuthoritySet, role: GraphRole) -> LoadedOnnxGraph:
    """Load a graph only after the complete frozen authority set is verified."""

    authorities.assert_complete()
    authority = authorities.authority_for(role)
    try:
        import onnx
    except ImportError as exc:
        raise DiagnosticPreflightError(
            status="EXECUTION_ERROR",
            code="ONNX_DEPENDENCY_MISSING",
            message="The onnx package is required for static graph inspection",
        ) from exc

    try:
        model = onnx.load(str(authority.path), load_external_data=False)
        onnx.checker.check_model(model)
    except Exception as exc:
        raise DiagnosticPreflightError(
            status="EXECUTION_ERROR",
            code="ONNX_STATIC_LOAD_FAILED",
            message=f"Authority-verified graph could not be parsed: {role}",
            details={"role": role, "path": str(authority.path), "error": str(exc)},
        ) from exc
    return LoadedOnnxGraph(role=role, authority=authority, model=model)


def _tensor_inventory(value: Any, onnx_module: Any) -> TensorInventory:
    tensor_type = value.type.tensor_type
    try:
        data_type = str(onnx_module.TensorProto.DataType.Name(tensor_type.elem_type))
    except ValueError:
        data_type = f"UNKNOWN_{tensor_type.elem_type}"
    shape: list[int | str] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            shape.append(str(dimension.dim_param))
        else:
            shape.append("?")
    return TensorInventory(name=str(value.name), data_type=data_type, shape=tuple(shape))


def _output_ancestors(nodes: list[Any], graph_outputs: set[str]) -> set[int]:
    producer_by_tensor = {
        output: index for index, node in enumerate(nodes) for output in node.output if output
    }
    ancestors: set[int] = set()
    pending = list(sorted(graph_outputs))
    visited_tensors: set[str] = set()
    while pending:
        tensor = pending.pop()
        if not tensor or tensor in visited_tensors:
            continue
        visited_tensors.add(tensor)
        producer_index = producer_by_tensor.get(tensor)
        if producer_index is None:
            continue
        ancestors.add(producer_index)
        pending.extend(str(value) for value in nodes[producer_index].input if value)
    return ancestors


def build_graph_inventory(graph: LoadedOnnxGraph) -> GraphInventory:
    """Describe graph structure without selecting candidate-specific exceptions."""

    import onnx

    from .normalization_patterns import find_structural_normalizations

    nodes = list(graph.model.graph.node)
    output_names = {str(value.name) for value in graph.model.graph.output}
    output_ancestors = _output_ancestors(nodes, output_names)
    structural_rules = find_structural_normalizations(
        nodes,
        {str(value.name) for value in graph.model.graph.initializer},
    )
    structural_norm_indices: set[int] = set()
    for rule in structural_rules:
        boundary_index = rule["boundary_node_index"]
        if isinstance(boundary_index, bool) or not isinstance(boundary_index, int):
            raise ValueError("Structural normalization boundary must be an integer node index")
        structural_norm_indices.add(boundary_index)
    output_path_indices = _output_path_indices(nodes, output_ancestors, structural_norm_indices)
    producer_op_by_tensor = {
        str(output): str(node.op_type) for node in nodes for output in node.output if output
    }
    consumer_ops_by_tensor: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        for node_input in node.input:
            if node_input:
                consumer_ops_by_tensor[str(node_input)].append(str(node.op_type))

    inventory_nodes: list[NodeInventory] = []
    for index, node in enumerate(nodes):
        op_type = str(node.op_type)
        inputs = tuple(str(value) for value in node.input)
        outputs = tuple(str(value) for value in node.output)
        families: set[str] = set()
        if op_type in _QUANTIZATION_OPS:
            families.add("QUANTIZATION_BOUNDARY")
        has_dequantized_input = any(
            producer_op_by_tensor.get(value) == "DequantizeLinear" for value in inputs if value
        )
        has_quantized_output = any(
            "QuantizeLinear" in consumer_ops_by_tensor.get(value, []) for value in outputs if value
        )
        if op_type not in {"QuantizeLinear", "DequantizeLinear"} and (
            has_dequantized_input or has_quantized_output or op_type in _QUANTIZATION_OPS
        ):
            families.add("QUANTIZED_COMPUTE")
        if op_type in _NORMALIZATION_OPS:
            families.add("NORMALIZATION")
        if index in structural_norm_indices:
            families.add("NORMALIZATION")
        if op_type in _ATTENTION_COMPUTE_OPS:
            families.add("ATTENTION_OR_MATMUL")
        if index in output_path_indices:
            families.add("OUTPUT_PATH")
        if any(value in output_names for value in outputs):
            families.add("FINAL_OUTPUT")
        inventory_nodes.append(
            NodeInventory(
                index=index,
                name=str(node.name),
                op_type=op_type,
                domain=str(node.domain),
                inputs=inputs,
                outputs=outputs,
                structural_families=tuple(family for family in _FAMILY_ORDER if family in families),
                is_output_ancestor=index in output_ancestors,
            )
        )

    op_counts = Counter(node.op_type for node in inventory_nodes)
    return GraphInventory(
        role=graph.role,
        graph_sha256=graph.authority.sha256,
        ir_version=int(graph.model.ir_version),
        opsets=tuple(
            sorted((str(opset.domain), int(opset.version)) for opset in graph.model.opset_import)
        ),
        inputs=tuple(_tensor_inventory(value, onnx) for value in graph.model.graph.input),
        outputs=tuple(_tensor_inventory(value, onnx) for value in graph.model.graph.output),
        initializer_count=len(graph.model.graph.initializer),
        op_counts=tuple(sorted(op_counts.items())),
        nodes=tuple(inventory_nodes),
        structural_rules=tuple(structural_rules),
        value_info=tuple(_tensor_inventory(value, onnx) for value in graph.model.graph.value_info),
    )


def _output_path_indices(
    nodes: list[Any],
    output_ancestors: set[int],
    structural_norm_indices: set[int] | None = None,
) -> set[int]:
    """Label reductions on the final-output path and their nearest input norms."""
    producer_by_tensor = {
        str(output): index for index, node in enumerate(nodes) for output in node.output if output
    }
    consumers_by_node: dict[int, set[int]] = defaultdict(set)
    for consumer_index, node in enumerate(nodes):
        for value in node.input:
            producer_index = producer_by_tensor.get(str(value))
            if producer_index is not None:
                consumers_by_node[producer_index].add(consumer_index)

    reduction_indices = {
        index for index in output_ancestors if str(nodes[index].op_type) in _OUTPUT_PATH_REDUCTIONS
    }
    if not reduction_indices:
        return set()

    has_reduction_free_path: dict[int, bool] = {}
    first_reductions: set[int] = set()
    for index, node in enumerate(nodes):
        clean_input_path = False
        for value in node.input:
            tensor_name = str(value)
            if not tensor_name:
                continue
            producer_index = producer_by_tensor.get(tensor_name)
            if (
                producer_index is None
                or has_reduction_free_path.get(producer_index, False)
                and (str(nodes[producer_index].op_type) not in _OUTPUT_PATH_REDUCTIONS)
            ):
                clean_input_path = True
        has_reduction_free_path[index] = clean_input_path
        if index in reduction_indices and clean_input_path:
            first_reductions.add(index)

    if not first_reductions:
        from neural_continuity.m1_diagnostics.authority import DiagnosticPreflightError

        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="OUTPUT_PATH_REDUCTION_LINEAGE_INVALID",
            message="Output reduction ancestry has no topological first reduction",
        )

    output_path: set[int] = set()
    pending = list(sorted(first_reductions))
    while pending:
        index = pending.pop()
        if index in output_path or index not in output_ancestors:
            continue
        output_path.add(index)
        pending.extend(sorted(consumers_by_node.get(index, ()), reverse=True))

    normalizations = {
        index for index, node in enumerate(nodes) if str(node.op_type) in _NORMALIZATION_OPS
    }
    normalizations.update(structural_norm_indices or ())
    attention_barriers = _ATTENTION_COMPUTE_OPS
    for reduction_index in first_reductions:
        backward = [
            producer_by_tensor[str(value)]
            for value in nodes[reduction_index].input
            if str(value) in producer_by_tensor
        ]
        visited: set[int] = set()
        while backward:
            index = backward.pop()
            if index in visited:
                continue
            visited.add(index)
            op_type = str(nodes[index].op_type)
            if index in normalizations:
                output_path.add(index)
                continue
            if op_type in attention_barriers:
                continue
            backward.extend(
                producer_by_tensor[str(value)]
                for value in nodes[index].input
                if str(value) in producer_by_tensor
            )
    return output_path
