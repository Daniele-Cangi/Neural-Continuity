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
        "OUTPUT_AGGREGATION",
        "FINAL_OUTPUT",
    }
)


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
            "lineage_rule": "unique_exact_node_identity_and_operator",
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


def _unique_identity_index(
    nodes: tuple[NodeInventory, ...],
) -> dict[tuple[str, str], NodeInventory]:
    indexed: dict[tuple[str, str], NodeInventory] = {}
    duplicates: list[dict[str, str]] = []
    for node in nodes:
        key = (node.name, node.op_type)
        if not node.name:
            continue
        if key in indexed:
            duplicates.append({"name": node.name, "op_type": node.op_type})
        indexed[key] = node
    if duplicates:
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="AMBIGUOUS_NODE_IDENTITY",
            message="Graph contains duplicate node identity and operator pairs",
            details={"duplicates": duplicates},
        )
    return indexed


def _post_qdq_tensor(
    output_tensor: str,
    consumers: dict[str, list[NodeInventory]],
) -> tuple[str, str]:
    quantizers = [
        node for node in consumers.get(output_tensor, []) if node.op_type == "QuantizeLinear"
    ]
    chains: list[str] = []
    for quantizer in quantizers:
        for quantized_output in quantizer.outputs:
            for dequantizer in consumers.get(quantized_output, []):
                if dequantizer.op_type == "DequantizeLinear":
                    chains.extend(value for value in dequantizer.outputs if value)
    unique_chains = sorted(set(chains))
    if len(unique_chains) > 1:
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="AMBIGUOUS_POST_QDQ_LINEAGE",
            message="A target output has multiple post-QDQ lineage candidates",
            details={"target_output": output_tensor, "candidate_count": len(unique_chains)},
        )
    if unique_chains:
        return unique_chains[0], "post_quantize_dequantize_output"
    return output_tensor, "direct_compute_output"


def build_probe_plan(source: GraphInventory, target: GraphInventory) -> ProbePlan:
    """Build paired probes using uniform graph identity and adjacency rules."""

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

    source_index = _unique_identity_index(source.nodes)
    target_index = _unique_identity_index(target.nodes)
    consumers: dict[str, list[NodeInventory]] = defaultdict(list)
    for node in target.nodes:
        for node_input in node.inputs:
            if node_input:
                consumers[node_input].append(node)

    selected_keys = {
        (node.name, node.op_type)
        for node in source.nodes
        if node.name and _RELEVANT_FAMILIES.intersection(node.structural_families)
    }
    selected_keys.update(
        (node.name, node.op_type)
        for node in target.nodes
        if node.name and "QUANTIZED_COMPUTE" in node.structural_families
    )

    missing = sorted(
        key for key in selected_keys if key not in source_index or key not in target_index
    )
    if missing:
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="PROBE_LINEAGE_NOT_BIJECTIVE",
            message="A structurally selected probe node has no unique source-target counterpart",
            details={
                "missing_count": len(missing),
                "missing": [{"name": name, "op_type": op_type} for name, op_type in missing],
            },
        )

    pairs: list[ProbePair] = []
    ordered_keys = sorted(selected_keys, key=lambda key: source_index[key].index)
    for key in ordered_keys:
        source_node = source_index[key]
        target_node = target_index[key]
        if len(source_node.outputs) != len(target_node.outputs):
            raise DiagnosticPreflightError(
                status="BLOCKED",
                code="PROBE_OUTPUT_ARITY_MISMATCH",
                message="Paired source-target nodes have different output arity",
                details={"name": key[0], "op_type": key[1]},
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
                    details={"name": key[0], "op_type": key[1]},
                )
            target_probe_tensor, target_basis = _post_qdq_tensor(target_tensor, consumers)
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
