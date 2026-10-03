"""Structural, deterministic pairing plan for later diagnostic activation probes."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass

from neural_continuity.m1_diagnostics.authority import DiagnosticPreflightError
from neural_continuity.m1_diagnostics.graph_inventory import GraphInventory, NodeInventory

_RELEVANT_FAMILIES = frozenset(
    {
        "QUANTIZED_COMPUTE",
        "NORMALIZATION",
        "ATTENTION_OR_MATMUL",
        "OUTPUT_PATH",
        "FINAL_OUTPUT",
    }
)
_TRANSPARENT_OPS = frozenset({"QuantizeLinear", "DequantizeLinear"})


@dataclass(frozen=True)
class ProbePair:
    probe_id: str
    source_node_index: int
    target_node_index: int
    node_name: str
    op_type: str
    output_position: int
    source_tensor: str
    target_tensor: str
    target_tensor_basis: str
    structural_families: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "probe_id": self.probe_id,
            "source_node_index": self.source_node_index,
            "target_node_index": self.target_node_index,
            "node_name": self.node_name,
            "op_type": self.op_type,
            "output_position": self.output_position,
            "source_tensor": self.source_tensor,
            "target_tensor": self.target_tensor,
            "target_tensor_basis": self.target_tensor_basis,
            "structural_families": list(self.structural_families),
        }


@dataclass(frozen=True)
class ProbePlan:
    source_graph_sha256: str
    target_graph_sha256: str
    probes: tuple[ProbePair, ...]

    def _payload(self) -> dict[str, object]:
        return {
            "kind": "m1_transition_b_v2_static_probe_plan",
            "version": 1,
            "source_graph_sha256": self.source_graph_sha256,
            "target_graph_sha256": self.target_graph_sha256,
            "lineage_rule": "unique_joint_structural_fingerprint_v2",
            "selection_rule": "structural_family_membership",
            "probes": [probe.to_dict() for probe in self.probes],
            "model_execution_used": False,
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self._payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, object]:
        payload = self._payload()
        payload["probe_count"] = len(self.probes)
        payload["probe_plan_sha256"] = self.sha256
        return payload


def _semantic_op_type(op_type: str) -> str:
    if op_type in {"MatMulInteger", "QLinearMatMul"}:
        return "MatMul"
    if op_type in {"ConvInteger", "QLinearConv"}:
        return "Conv"
    if op_type.startswith("QLinear"):
        return op_type.removeprefix("QLinear")
    return op_type


def _semantic_input_slots(node: NodeInventory) -> tuple[tuple[int, int], ...]:
    if node.op_type == "QLinearMatMul":
        if len(node.inputs) < 4:
            raise DiagnosticPreflightError(
                status="BLOCKED",
                code="QUANTIZED_OPERATOR_INPUTS_INVALID",
                message="QLinearMatMul lacks its declared data inputs",
            )
        return ((0, 0), (1, 3))
    if node.op_type == "QLinearConv":
        if len(node.inputs) < 8:
            raise DiagnosticPreflightError(
                status="BLOCKED",
                code="QUANTIZED_OPERATOR_INPUTS_INVALID",
                message="QLinearConv lacks its declared data inputs",
            )
        slots = [(0, 0), (1, 3)]
        if len(node.inputs) > 8 and node.inputs[8]:
            slots.append((2, 8))
        return tuple(slots)
    if node.op_type == "MatMulInteger":
        return ((0, 0), (1, 1))
    if node.op_type == "ConvInteger":
        return ((0, 0), (1, 1))
    return tuple((position, position) for position, value in enumerate(node.inputs) if value)


def _graph_structure(
    inventory: GraphInventory,
) -> tuple[
    dict[int, NodeInventory],
    dict[int, tuple[object, ...]],
    dict[int, tuple[tuple[int, int | None, int | None, str, int | None], ...]],
    dict[int, tuple[int, ...]],
]:
    nodes = {node.index: node for node in inventory.nodes if node.op_type not in _TRANSPARENT_OPS}
    producers: dict[str, NodeInventory] = {}
    for node in inventory.nodes:
        for tensor in node.outputs:
            if not tensor:
                continue
            if tensor in producers:
                raise DiagnosticPreflightError(
                    status="BLOCKED",
                    code="GRAPH_TENSOR_PRODUCER_AMBIGUOUS",
                    message="A graph tensor has more than one producer",
                    details={"tensor": tensor},
                )
            producers[tensor] = node
    graph_inputs = {tensor.name: position for position, tensor in enumerate(inventory.inputs)}

    def trace(tensor: str) -> tuple[int | None, int | None, str, int | None]:
        visited: set[str] = set()
        while tensor in producers and producers[tensor].op_type in _TRANSPARENT_OPS:
            if tensor in visited:
                raise DiagnosticPreflightError(
                    status="BLOCKED",
                    code="QUANTIZATION_LINEAGE_CYCLE",
                    message="Transparent quantization lineage contains a cycle",
                )
            visited.add(tensor)
            wrapper = producers[tensor]
            if not wrapper.inputs or not wrapper.inputs[0]:
                raise DiagnosticPreflightError(
                    status="BLOCKED",
                    code="QUANTIZATION_LINEAGE_INPUT_MISSING",
                    message="QuantizeLinear or DequantizeLinear has no data input",
                )
            tensor = wrapper.inputs[0]
        producer = producers.get(tensor)
        if producer is not None and producer.index in nodes:
            return producer.index, producer.outputs.index(tensor), "node", None
        if tensor in graph_inputs:
            return None, None, "graph_input", graph_inputs[tensor]
        return None, None, "static_input", None

    input_refs: dict[int, tuple[tuple[int, int | None, int | None, str, int | None], ...]] = {}
    base: dict[int, tuple[object, ...]] = {}
    output_positions: dict[int, list[int]] = {index: [] for index in nodes}

    for node in nodes.values():
        refs: list[tuple[int, int | None, int | None, str, int | None]] = []
        for semantic_position, raw_position in _semantic_input_slots(node):
            producer_index, producer_output, boundary, boundary_position = trace(
                node.inputs[raw_position]
            )
            refs.append(
                (semantic_position, producer_index, producer_output, boundary, boundary_position)
            )
        input_refs[node.index] = tuple(refs)
        boundaries = tuple(
            (position, boundary, boundary_position)
            for position, _, _, boundary, boundary_position in refs
            if boundary != "node"
        )
        base[node.index] = (
            _semantic_op_type(node.op_type),
            node.domain,
            len(refs),
            len(node.outputs),
            boundaries,
        )

    for output_position, graph_output in enumerate(inventory.outputs):
        producer_index, producer_output, boundary, _ = trace(graph_output.name)
        if producer_index is not None and producer_output is not None and boundary == "node":
            output_positions[producer_index].append(output_position)

    return (
        nodes,
        base,
        input_refs,
        {index: tuple(values) for index, values in output_positions.items()},
    )


def _joint_structural_colors(
    source: GraphInventory, target: GraphInventory
) -> tuple[dict[int, int], dict[int, int]]:
    source_nodes, source_base, source_inputs, source_outputs = _graph_structure(source)
    target_nodes, target_base, target_inputs, target_outputs = _graph_structure(target)
    bases = {("source", index): value for index, value in source_base.items()}
    bases.update({("target", index): value for index, value in target_base.items()})
    colors_by_refinement: dict[tuple[str, int], int] = {}

    def assign(signatures: dict[tuple[str, int], tuple[object, ...]]) -> dict[tuple[str, int], int]:
        identifiers = {
            signature: color
            for color, signature in enumerate(sorted(set(signatures.values()), key=repr))
        }
        return {key: identifiers[signature] for key, signature in signatures.items()}

    colors_by_refinement = assign(bases)
    node_count = len(source_nodes) + len(target_nodes)
    for _ in range(node_count + 1):
        signatures: dict[tuple[str, int], tuple[object, ...]] = {}
        for role, nodes, input_refs, output_positions in (
            ("source", source_nodes, source_inputs, source_outputs),
            ("target", target_nodes, target_inputs, target_outputs),
        ):
            for index in nodes:
                incoming = tuple(
                    (
                        input_position,
                        (
                            colors_by_refinement[(role, producer_index)]
                            if producer_index is not None
                            else None
                        ),
                        producer_output,
                        boundary,
                        boundary_position,
                    )
                    for (
                        input_position,
                        producer_index,
                        producer_output,
                        boundary,
                        boundary_position,
                    ) in input_refs[index]
                )
                outgoing = tuple(
                    sorted(
                        (
                            producer_output,
                            consumer_position,
                            colors_by_refinement[(role, consumer_index)],
                        )
                        for consumer_index, refs in input_refs.items()
                        for (
                            consumer_position,
                            producer_index,
                            producer_output,
                            _boundary,
                            _boundary_position,
                        ) in refs
                        if producer_index == index and producer_output is not None
                    )
                )
                signatures[(role, index)] = (
                    colors_by_refinement[(role, index)],
                    bases[(role, index)],
                    incoming,
                    outgoing,
                    output_positions[index],
                )
        refined = assign(signatures)
        # Refinement only splits classes because the previous color is included.
        if len(set(refined.values())) == len(set(colors_by_refinement.values())):
            colors_by_refinement = refined
            break
        colors_by_refinement = refined
    else:
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="STRUCTURAL_FINGERPRINT_DID_NOT_CONVERGE",
            message="Joint structural fingerprint refinement did not converge",
        )

    return (
        {index: colors_by_refinement[("source", index)] for index in source_nodes},
        {index: colors_by_refinement[("target", index)] for index in target_nodes},
    )


def _post_qdq_tensor(
    output_tensor: str,
    consumers: dict[str, list[NodeInventory]],
    target_node: NodeInventory,
) -> tuple[str, str]:
    direct = {
        value
        for node in consumers.get(output_tensor, [])
        if node.op_type == "DequantizeLinear"
        for value in node.outputs
        if value
    }
    chains = set(direct)
    quantizers = [
        node for node in consumers.get(output_tensor, []) if node.op_type == "QuantizeLinear"
    ]
    for quantizer in quantizers:
        for quantized_output in quantizer.outputs:
            for dequantizer in consumers.get(quantized_output, []):
                if dequantizer.op_type == "DequantizeLinear":
                    chains.update(value for value in dequantizer.outputs if value)
    if len(chains) > 1:
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="AMBIGUOUS_POST_QDQ_LINEAGE",
            message="A target output has multiple post-QDQ lineage candidates",
            details={"target_output": output_tensor, "candidate_count": len(chains)},
        )
    if direct:
        return next(iter(direct)), "post_dequantize_output"
    if chains:
        return next(iter(chains)), "post_quantize_dequantize_output"
    if target_node.op_type.startswith("QLinear") or target_node.op_type in {
        "MatMulInteger",
        "ConvInteger",
    }:
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="POST_QDQ_FLOAT_LINEAGE_MISSING",
            message="A quantized compute output has no unique floating-point dequantized tensor",
        )
    return output_tensor, "direct_compute_output"


def build_probe_plan(source: GraphInventory, target: GraphInventory) -> ProbePlan:
    """Build paired probes from unique joint structural fingerprints."""

    if source.role != "onnx_fp32_source" or target.role != "onnx_int8_candidate":
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="GRAPH_ROLE_MISMATCH",
            message="Probe planning requires the frozen FP32 source and INT8 target roles",
        )
    if tuple(value.name for value in source.inputs) != tuple(value.name for value in target.inputs):
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="GRAPH_INPUT_IDENTITY_MISMATCH",
            message="Source and target graph input identities differ",
        )
    if tuple(value.name for value in source.outputs) != tuple(
        value.name for value in target.outputs
    ):
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="GRAPH_OUTPUT_IDENTITY_MISMATCH",
            message="Source and target graph output identities differ",
        )

    source_nodes = {node.index: node for node in source.nodes}
    target_nodes = {node.index: node for node in target.nodes}
    source_colors, target_colors = _joint_structural_colors(source, target)
    consumers: dict[str, list[NodeInventory]] = defaultdict(list)
    for node in target.nodes:
        for node_input in node.inputs:
            if node_input:
                consumers[node_input].append(node)

    selected_colors = {
        source_colors[node.index]
        for node in source.nodes
        if _RELEVANT_FAMILIES.intersection(node.structural_families)
    }
    selected_colors.update(
        target_colors[node.index]
        for node in target.nodes
        if _RELEVANT_FAMILIES.intersection(node.structural_families)
    )
    source_by_color: dict[int, list[NodeInventory]] = defaultdict(list)
    target_by_color: dict[int, list[NodeInventory]] = defaultdict(list)
    for index, color in source_colors.items():
        source_by_color[color].append(source_nodes[index])
    for index, color in target_colors.items():
        target_by_color[color].append(target_nodes[index])

    invalid = [
        color
        for color in selected_colors
        if len(source_by_color.get(color, ())) != 1 or len(target_by_color.get(color, ())) != 1
    ]
    if invalid:
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="PROBE_LINEAGE_NOT_BIJECTIVE",
            message="A selected structural class lacks a unique source-target counterpart",
            details={
                "invalid_class_count": len(invalid),
                "classes": [
                    {
                        "source_count": len(source_by_color.get(color, ())),
                        "target_count": len(target_by_color.get(color, ())),
                    }
                    for color in sorted(invalid)
                ],
            },
        )

    pairs: list[ProbePair] = []
    for color in sorted(selected_colors):
        source_node = source_by_color[color][0]
        target_node = target_by_color[color][0]
        if len(source_node.outputs) != len(target_node.outputs):
            raise DiagnosticPreflightError(
                status="BLOCKED",
                code="PROBE_OUTPUT_ARITY_MISMATCH",
                message="Paired source-target nodes have different output arity",
                details={"structural_class": color},
            )
        families = tuple(
            sorted(
                _RELEVANT_FAMILIES.intersection(
                    set(source_node.structural_families) | set(target_node.structural_families)
                )
            )
        )
        for output_position, (source_tensor, target_tensor) in enumerate(
            zip(source_node.outputs, target_node.outputs, strict=True)
        ):
            if not source_tensor or not target_tensor:
                raise DiagnosticPreflightError(
                    status="BLOCKED",
                    code="PROBE_OUTPUT_IDENTITY_MISSING",
                    message="A selected probe output is unnamed",
                    details={"structural_class": color},
                )
            target_probe_tensor, target_basis = _post_qdq_tensor(
                target_tensor, consumers, target_node
            )
            pairs.append(
                ProbePair(
                    probe_id=f"probe-{len(pairs) + 1:04d}",
                    source_node_index=source_node.index,
                    target_node_index=target_node.index,
                    node_name=source_node.name,
                    op_type=source_node.op_type,
                    output_position=output_position,
                    source_tensor=source_tensor,
                    target_tensor=target_probe_tensor,
                    target_tensor_basis=target_basis,
                    structural_families=families,
                )
            )

    if not pairs:
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="PROBE_PLAN_EMPTY",
            message="Structural selection produced no paired probe points",
        )
    return ProbePlan(
        source_graph_sha256=source.graph_sha256,
        target_graph_sha256=target.graph_sha256,
        probes=tuple(pairs),
    )
