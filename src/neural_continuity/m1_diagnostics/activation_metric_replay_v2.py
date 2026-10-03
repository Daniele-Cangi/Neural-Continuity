"""Join capture integrity, structural layouts and canonical model-free metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes
from neural_continuity.m1_diagnostics.activation_evidence import (
    _load_json,
    replay_activation_capture,
)
from neural_continuity.m1_diagnostics.activation_inputs import (
    INPUT_CAPTURE_VERSION,
    load_token_input_batch,
)
from neural_continuity.m1_diagnostics.activation_metric_batches_v2 import (
    ProbeMetricLayout,
    analyze_captured_activation_metrics_v2,
)
from neural_continuity.m1_diagnostics.authority import DiagnosticPreflightError
from neural_continuity.m1_diagnostics.graph_inventory import GraphInventory
from neural_continuity.m1_diagnostics.metric_layouts import replay_probe_metric_layouts
from neural_continuity.m1_diagnostics.probe_plan import ProbePlan


def _blocked(message: str) -> DiagnosticPreflightError:
    return DiagnosticPreflightError(
        status="BLOCKED", code="METRIC_CAPTURE_BINDING_INVALID", message=message
    )


def analyze_recorded_activation_metrics_v2(
    root: Path,
    expected_manifest_sha256: str,
    recorded_layouts: dict[str, Any],
    source: GraphInventory,
    target: GraphInventory,
    plan: ProbePlan,
    *,
    source_mask_input_position: int,
    target_mask_input_position: int,
) -> dict[str, Any]:
    """Inputs inventories and role bindings must already be authority-verified.

    This joins software replay components, not the full diagnostic authority or
    H1-H4 package. It never emits diagnostic COMPLETE or authorizes execution.
    """
    root = root.resolve()
    replay_probe_metric_layouts(
        recorded_layouts,
        source,
        target,
        plan,
        source_mask_input_position=source_mask_input_position,
        target_mask_input_position=target_mask_input_position,
    )
    replay_activation_capture(root / "replay-bundle.json", expected_manifest_sha256)
    capture_plan = _load_json(root / "capture-plan.json", "CAPTURE_PLAN_INVALID")
    index = _load_json(root / "batch-index.json", "BATCH_INDEX_INVALID")
    if (
        capture_plan.get("input_capture_version") != INPUT_CAPTURE_VERSION
        or capture_plan.get("metric_layout_sha256") != recorded_layouts["layout_sha256"]
        or capture_plan.get("probe_plan_sha256") != plan.sha256
    ):
        raise _blocked("Capture did not freeze these structural layouts and actual inputs")
    mappings = capture_plan.get("probe_mappings")
    expected = {
        probe.probe_id: (probe.source_tensor, probe.target_tensor, probe.target_tensor_basis)
        for probe in plan.probes
    }
    if (
        not isinstance(mappings, list)
        or [item.get("probe_id") for item in mappings] != [probe.probe_id for probe in plan.probes]
        or any(
            (item.get("source_tensor"), item.get("target_tensor"), item.get("target_tensor_basis"))
            != expected[item["probe_id"]]
            for item in mappings
        )
    ):
        raise _blocked("Capture probe lineage or ordering differs from the frozen plan")
    manifest = _load_json(root / "artifact-manifest.json", "MANIFEST_INVALID")
    paths = {entry["path"] for entry in manifest["artifacts"]}
    required = {
        "capture-plan.json",
        "capture-preflight.json",
        "batch-index.json",
        "activation-report.json",
        "replay-bundle.json",
    }
    batches = index["batches"]
    for record in batches:
        required.update(record[name] for name in ("floating_path", "integer_path", "inputs_path"))
    if not required.issubset(paths):
        raise _blocked("Required metric replay artifacts are absent from the pinned manifest")
    masks = {
        record["batch_id"]: load_token_input_batch(root, record)["attention_mask"]
        for record in batches
    }
    layouts = {
        probe_id: ProbeMetricLayout(**value)
        for probe_id, value in recorded_layouts["layouts"].items()
    }
    tensors = {value.name: value for value in (*source.inputs, *source.outputs, *source.value_info)}
    shapes = {probe.probe_id: tensors[probe.source_tensor].shape for probe in plan.probes}
    needs_masks = any(layout.masking != "none" for layout in layouts.values())
    result = analyze_captured_activation_metrics_v2(
        root,
        capture_plan,
        index,
        layouts,
        masks if needs_masks else None,
        declared_shapes_by_probe_id=shapes,
    )
    result.update(
        {
            "source_manifest_sha256": expected_manifest_sha256,
            "metric_layout_sha256": recorded_layouts["layout_sha256"],
            "actual_input_batches_verified": len(batches),
            "diagnostic_complete": False,
        }
    )
    return result


def replay_recorded_activation_metrics_v2(
    recorded_report: dict[str, Any], *args: Any, **kwargs: Any
) -> dict[str, bool]:
    recomputed = analyze_recorded_activation_metrics_v2(*args, **kwargs)
    if canonical_json_bytes(recorded_report) != canonical_json_bytes(recomputed):
        raise _blocked("Recorded canonical metrics do not match model-free replay")
    return {"replay_verified": True, "model_execution_used": False}
