"""Model-free probe-predecessor and dominant-onset derivation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from neural_continuity.m1_diagnostics.authority import DiagnosticPreflightError
from neural_continuity.m1_diagnostics.graph_inventory import GraphInventory, NodeInventory
from neural_continuity.m1_diagnostics.metric_arithmetic import binary64_bits, binary64_hex
from neural_continuity.m1_diagnostics.probe_plan import ProbePair


def _blocked(code: str, message: str) -> DiagnosticPreflightError:
    return DiagnosticPreflightError(status="BLOCKED", code=code, message=message)


@dataclass(frozen=True, order=True)
class ProbePredecessor:
    child_probe_id: str
    lineage_path_id: str
    parent_probe_id: str | None = None
    synthetic_predecessor_id: str | None = None


def graph_predecessor_edges(
    source: GraphInventory, probes: Sequence[ProbePair]
) -> tuple[ProbePredecessor, ...]:
    """Map each probe to its nearest probed ancestors using source tensor lineage."""
    if source.role != "onnx_fp32_source" or not probes:
        raise _blocked(
            "PROBE_PREDECESSOR_AUTHORITY_INVALID",
            "A source graph and nonempty probe plan are required",
        )
    nodes = {node.index: node for node in source.nodes}
    producer_by_tensor: dict[str, NodeInventory] = {}
    for node in source.nodes:
        for tensor in node.outputs:
            if not tensor:
                continue
            if tensor in producer_by_tensor:
                raise _blocked(
                    "GRAPH_TENSOR_PRODUCER_AMBIGUOUS", "A graph tensor has more than one producer"
                )
            producer_by_tensor[tensor] = node

    probe_by_id: dict[str, ProbePair] = {}
    probe_by_tensor: dict[str, str] = {}
    probes_by_node: dict[int, list[str]] = defaultdict(list)
    for probe in probes:
        if not probe.probe_id or probe.probe_id in probe_by_id:
            raise _blocked("PROBE_IDENTITY_DUPLICATED", "Probe IDs must be unique and nonempty")
        probe_node = nodes.get(probe.source_node_index)
        if probe_node is None or probe.source_tensor not in probe_node.outputs:
            raise _blocked(
                "PROBE_SOURCE_LINEAGE_INVALID",
                "Probe source tensor is not produced by its declared node",
            )
        if probe_node.outputs.index(probe.source_tensor) != probe.output_position:
            raise _blocked(
                "PROBE_SOURCE_LINEAGE_INVALID",
                "Probe source output position does not match tensor lineage",
            )
        if probe.source_tensor in probe_by_tensor:
            raise _blocked(
                "PROBE_SOURCE_LINEAGE_AMBIGUOUS", "Multiple probes claim the same source tensor"
            )
        probe_by_id[probe.probe_id] = probe
        probe_by_tensor[probe.source_tensor] = probe.probe_id
        probes_by_node[probe.source_node_index].append(probe.probe_id)

    edges: set[ProbePredecessor] = set()
    for child in probes:
        child_node = nodes[child.source_node_index]
        pending: list[tuple[str, str, frozenset[str]]] = [
            (value, f"input-{position}", frozenset())
            for position, value in enumerate(child_node.inputs)
            if value
        ]
        if not pending:
            edges.add(
                ProbePredecessor(
                    child_probe_id=child.probe_id,
                    lineage_path_id="constant-origin",
                    synthetic_predecessor_id=f"graph-input:{child.probe_id}:constant-origin",
                )
            )
        while pending:
            tensor, path_id, visited = pending.pop()
            if tensor in visited:
                raise _blocked("PROBE_GRAPH_CYCLE", "Probe lineage contains a graph cycle")
            path_visited = visited | {tensor}
            producer = producer_by_tensor.get(tensor)
            if producer is None:
                edges.add(
                    ProbePredecessor(
                        child_probe_id=child.probe_id,
                        lineage_path_id=path_id,
                        synthetic_predecessor_id=f"graph-input:{child.probe_id}:{path_id}",
                    )
                )
                continue
            if producer.index >= child.source_node_index:
                raise _blocked(
                    "PROBE_GRAPH_NOT_TOPOLOGICAL",
                    "Probe predecessor traversal encountered a non-topological edge",
                )
            parent_id = probe_by_tensor.get(tensor)
            if parent_id is not None:
                if parent_id != child.probe_id:
                    edges.add(
                        ProbePredecessor(
                            child_probe_id=child.probe_id,
                            lineage_path_id=path_id,
                            parent_probe_id=parent_id,
                        )
                    )
                continue
            if producer.index in probes_by_node:
                raise _blocked(
                    "PROBE_PREDECESSOR_BOUNDARY_MISSING",
                    "A selected predecessor node lacks a probe for the traversed output",
                )
            input_branches = [
                (value, position) for position, value in enumerate(producer.inputs) if value
            ]
            if not input_branches:
                edges.add(
                    ProbePredecessor(
                        child_probe_id=child.probe_id,
                        lineage_path_id=path_id,
                        synthetic_predecessor_id=f"graph-input:{child.probe_id}:{path_id}",
                    )
                )
            else:
                pending.extend(
                    (
                        value,
                        f"{path_id}/node-{producer.index}-input-{position}",
                        path_visited,
                    )
                    for value, position in input_branches
                )

    return tuple(sorted(edges))


def localize_dominant_onset(
    *,
    probe_ids: Sequence[str],
    predecessor_edges: Sequence[ProbePredecessor],
    probe_scores_hex: Mapping[str, str],
) -> dict[str, Any]:
    """Return all edges tied at the largest nonnegative symmetric-L2 growth."""
    ordered_ids = tuple(sorted(probe_ids))
    if not ordered_ids or len(set(ordered_ids)) != len(ordered_ids):
        raise _blocked("PROBE_IDENTITY_INVALID", "Onset localization requires unique probe IDs")
    if set(probe_scores_hex) != set(ordered_ids):
        raise _blocked(
            "PROBE_SCORE_SET_MISMATCH", "Every planned probe must have exactly one replayed score"
        )

    scores: dict[str, float] = {}
    for probe_id in ordered_ids:
        encoded = probe_scores_hex[probe_id]
        try:
            score = float.fromhex(encoded)
        except (TypeError, ValueError) as exc:
            raise _blocked(
                "PROBE_SCORE_INVALID", "Probe scores must be hexadecimal binary64 values"
            ) from exc
        if not math.isfinite(score) or binary64_hex(score) != encoded:
            raise _blocked(
                "PROBE_SCORE_INVALID", "Probe scores must be finite canonical hexadecimal values"
            )
        scores[probe_id] = score

    edges = tuple(sorted(set(predecessor_edges)))
    for edge in edges:
        if edge.child_probe_id not in scores:
            raise _blocked(
                "PROBE_PREDECESSOR_EDGE_INVALID",
                "Predecessor edge references an invalid child probe",
            )
        if edge.parent_probe_id is not None and (
            edge.parent_probe_id not in scores or edge.parent_probe_id == edge.child_probe_id
        ):
            raise _blocked(
                "PROBE_PREDECESSOR_EDGE_INVALID",
                "Predecessor edge references an invalid parent probe",
            )
        if (edge.parent_probe_id is None) == (edge.synthetic_predecessor_id is None):
            raise _blocked(
                "PROBE_PREDECESSOR_EDGE_INVALID",
                "Each edge requires exactly one real or synthetic predecessor",
            )
        if not edge.lineage_path_id:
            raise _blocked(
                "PROBE_PREDECESSOR_EDGE_INVALID",
                "Each predecessor edge requires a lineage path identity",
            )

    scored_edges: list[dict[str, str]] = []
    numeric_growths: list[float] = []
    for edge in edges:
        child = edge.child_probe_id
        parent_score = 0.0 if edge.parent_probe_id is None else scores[edge.parent_probe_id]
        difference = scores[child] - parent_score
        if not math.isfinite(difference):
            raise _blocked("METRIC_ARITHMETIC_NONFINITE", "Probe-score growth became non-finite")
        growth = max(0.0, difference)
        numeric_growths.append(growth)
        item: dict[str, str] = {
            "child_probe_id": child,
            "lineage_path_id": edge.lineage_path_id,
            "parent_score": binary64_hex(parent_score),
            "child_score": binary64_hex(scores[child]),
            "growth": binary64_hex(growth),
        }
        if edge.parent_probe_id is not None:
            item["parent_probe_id"] = edge.parent_probe_id
        else:
            item["synthetic_predecessor_id"] = str(edge.synthetic_predecessor_id)
        scored_edges.append(item)

    maximum_growth = max(numeric_growths, default=0.0)
    maximum_bits = binary64_bits(maximum_growth)
    dominant_edges = [
        edge
        for edge, growth in zip(scored_edges, numeric_growths, strict=True)
        if growth == maximum_growth and binary64_bits(growth) == maximum_bits
    ]
    dominant_boundaries = sorted({edge["child_probe_id"] for edge in dominant_edges})
    return {
        "probe_scores": {probe_id: binary64_hex(scores[probe_id]) for probe_id in ordered_ids},
        "predecessor_edges": scored_edges,
        "maximum_growth": binary64_hex(maximum_growth),
        "dominant_onset_edges": dominant_edges,
        "dominant_onset_boundary_probe_ids": dominant_boundaries,
        "localization_status": "UNRESOLVED_ZERO_GROWTH" if maximum_growth == 0.0 else "LOCALIZED",
        "model_execution_used": False,
    }
