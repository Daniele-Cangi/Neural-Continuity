"""Derive metric axes from declared graph dimensions, never activation values."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any

from neural_continuity.evidence import canonical_json_bytes
from neural_continuity.m1_diagnostics.activation_metric_batches_v2 import ProbeMetricLayout
from neural_continuity.m1_diagnostics.authority import DiagnosticPreflightError
from neural_continuity.m1_diagnostics.graph_inventory import GraphInventory, TensorInventory
from neural_continuity.m1_diagnostics.probe_plan import ProbePlan


def _blocked(message: str) -> DiagnosticPreflightError:
    return DiagnosticPreflightError(
        status="BLOCKED", code="METRIC_LAYOUT_AMBIGUOUS", message=message
    )


def _dimensions(
    graph: GraphInventory, mask_input_position: int
) -> tuple[str, str, dict[str, TensorInventory]]:
    if not 0 <= mask_input_position < len(graph.inputs):
        raise _blocked("The frozen mask input position is absent")
    mask = graph.inputs[mask_input_position]
    if mask.data_type != "INT64" or len(mask.shape) != 2:
        raise _blocked("The mask input must declare two INT64 dimensions")
    batch, sequence = mask.shape
    if (
        not isinstance(batch, str)
        or not isinstance(sequence, str)
        or batch in ("", "?")
        or sequence in ("", "?")
        or batch == sequence
    ):
        raise _blocked("Distinct symbolic batch and sequence identities are required")
    tensors: dict[str, TensorInventory] = {}
    for tensor in (*graph.inputs, *graph.outputs, *graph.value_info):
        previous = tensors.get(tensor.name)
        if previous is not None and previous != tensor:
            raise _blocked("Conflicting tensor declarations")
        tensors[tensor.name] = tensor
    return batch, sequence, tensors


def _layout(
    tensor: TensorInventory, batch: str, sequence: str
) -> tuple[ProbeMetricLayout, tuple[int | str, ...]]:
    shape = tensor.shape
    if (
        len(shape) < 2
        or shape[0] != batch
        or shape.count(batch) != 1
        or any(
            isinstance(dim, bool)
            or isinstance(dim, int)
            and dim <= 0
            or isinstance(dim, str)
            and dim in ("", "?")
            for dim in shape
        )
    ):
        raise _blocked("Batch identity or tensor dimensions are ambiguous")
    axes = [axis for axis, dim in enumerate(shape) if dim == sequence]
    if len(shape) == 4 and axes == [2, 3]:
        layout = ProbeMetricLayout(masking="attention_scores")
    elif len(axes) == 1:
        axis = axes[0]
        feature = len(shape) - 1 if axis != len(shape) - 1 else None
        layout = ProbeMetricLayout(masking="sequence", sequence_axis=axis, feature_axis=feature)
    elif not axes and len(shape) == 2:
        layout = ProbeMetricLayout(masking="none", feature_axis=1)
    else:
        raise _blocked("No unique supported valid-element mask can be established")
    normalized = tuple(
        "$batch" if dim == batch else "$sequence" if dim == sequence else dim for dim in shape
    )
    return layout, normalized


def derive_probe_metric_layouts(
    source: GraphInventory,
    target: GraphInventory,
    plan: ProbePlan,
    *,
    source_mask_input_position: int,
    target_mask_input_position: int,
) -> dict[str, Any]:
    """Mask input positions must come from the frozen input-role binding.

    Missing declared shapes block; this function does not infer shapes, load
    graphs, inspect observations, or grant execution authorization.
    """
    if (
        source.role != "onnx_fp32_source"
        or target.role != "onnx_int8_candidate"
        or plan.source_graph_sha256 != source.graph_sha256
        or plan.target_graph_sha256 != target.graph_sha256
        or not plan.probes
    ):
        raise _blocked("Graph identities do not match the probe plan")
    source_batch, source_sequence, source_tensors = _dimensions(source, source_mask_input_position)
    target_batch, target_sequence, target_tensors = _dimensions(target, target_mask_input_position)
    layouts: dict[str, dict[str, Any]] = {}
    for probe in plan.probes:
        left = source_tensors.get(probe.source_tensor)
        right = target_tensors.get(probe.target_tensor)
        if left is None or right is None or probe.probe_id in layouts:
            raise _blocked("Every unique probe requires paired declared tensor shapes")
        left_layout, left_shape = _layout(left, source_batch, source_sequence)
        right_layout, right_shape = _layout(right, target_batch, target_sequence)
        if (
            left_layout != right_layout
            or left_shape != right_shape
            or left.data_type != right.data_type
        ):
            raise _blocked("Paired graph declarations disagree on shape, dtype or layout")
        layouts[probe.probe_id] = asdict(left_layout)
    payload: dict[str, Any] = {
        "rule": "declared_graph_dimensions_v1",
        "probe_plan_sha256": plan.sha256,
        "source_graph_sha256": source.graph_sha256,
        "target_graph_sha256": target.graph_sha256,
        "source_mask_input_position": source_mask_input_position,
        "target_mask_input_position": target_mask_input_position,
        "layouts": layouts,
        "model_execution_used": False,
        "execution_authorized": False,
    }
    payload["layout_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def replay_probe_metric_layouts(
    recorded: dict[str, Any],
    source: GraphInventory,
    target: GraphInventory,
    plan: ProbePlan,
    *,
    source_mask_input_position: int,
    target_mask_input_position: int,
) -> dict[str, Any]:
    recomputed = derive_probe_metric_layouts(
        source,
        target,
        plan,
        source_mask_input_position=source_mask_input_position,
        target_mask_input_position=target_mask_input_position,
    )
    if canonical_json_bytes(recorded) != canonical_json_bytes(recomputed):
        raise _blocked("Recorded metric layouts do not match structural replay")
    return {"replay_verified": True, "model_execution_used": False}
