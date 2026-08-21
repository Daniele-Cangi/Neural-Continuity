"""Derive probe-output ONNX copies without executing either frozen model."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from neural_continuity.m1_diagnostics.authority import DiagnosticPreflightError
from neural_continuity.m1_diagnostics.static_package import VerifiedStaticPreflight

InstrumentationRole = Literal["source", "target"]


@dataclass(frozen=True)
class InstrumentedGraph:
    role: InstrumentationRole
    path: Path
    sha256: str
    size_bytes: int
    frozen_graph_sha256: str
    original_outputs: tuple[str, ...]
    probe_outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "path": self.path.name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "frozen_graph_sha256": self.frozen_graph_sha256,
            "original_outputs": list(self.original_outputs),
            "probe_outputs": list(self.probe_outputs),
            "probe_output_count": len(self.probe_outputs),
        }


@dataclass(frozen=True)
class InstrumentationResult:
    source: InstrumentedGraph
    target: InstrumentedGraph
    probe_mappings: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "m1_transition_b_v2_instrumentation_result",
            "status": "READY_FOR_FIDELITY_CONTROL",
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "probe_count": len(self.probe_mappings),
            "probe_mappings": list(self.probe_mappings),
            "frozen_models_overwritten": False,
            "onnx_runtime_session_created": False,
            "activations_read": False,
            "model_execution_used": False,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_mappings(plan: dict[str, object]) -> tuple[dict[str, str], ...]:
    raw_probes = plan.get("probes")
    if not isinstance(raw_probes, list):
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="STATIC_PROBE_PLAN_INVALID",
            message="Verified probe plan has no probe list",
        )
    mappings: list[dict[str, str]] = []
    for raw_probe in raw_probes:
        probe = cast(dict[str, object], raw_probe)
        probe_id = probe.get("probe_id")
        source_tensor = probe.get("source_tensor")
        target_tensor = probe.get("target_tensor")
        identity_values = (probe_id, source_tensor, target_tensor)
        if not all(isinstance(value, str) and value for value in identity_values):
            raise DiagnosticPreflightError(
                status="BLOCKED",
                code="STATIC_PROBE_PLAN_INVALID",
                message="Verified probe mapping is incomplete",
            )
        mappings.append(
            {
                "probe_id": cast(str, probe_id),
                "source_tensor": cast(str, source_tensor),
                "target_tensor": cast(str, target_tensor),
            }
        )
    return tuple(mappings)


def _value_info_by_name(model: Any) -> dict[str, Any]:
    values = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    return {str(value.name): value for value in values if value.name}


def _derive_graph(
    *,
    role: InstrumentationRole,
    frozen_path: Path,
    frozen_sha256: str,
    probe_outputs: tuple[str, ...],
    output_path: Path,
) -> InstrumentedGraph:
    try:
        import onnx
    except ImportError as exc:
        raise DiagnosticPreflightError(
            status="EXECUTION_ERROR",
            code="ONNX_DEPENDENCY_MISSING",
            message="The onnx package is required for graph instrumentation",
        ) from exc

    before_sha256 = _sha256_file(frozen_path)
    if before_sha256 != frozen_sha256:
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="FROZEN_GRAPH_CHANGED",
            message="Frozen graph changed after authority verification",
            details={"role": role},
        )
    try:
        model = onnx.load(str(frozen_path), load_external_data=False)
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=False)
    except Exception as exc:
        raise DiagnosticPreflightError(
            status="EXECUTION_ERROR",
            code="ONNX_INSTRUMENTATION_INFERENCE_FAILED",
            message="Static ONNX shape inference failed during instrumentation",
            details={"role": role, "error": str(exc)},
        ) from exc

    original_outputs = tuple(str(value.name) for value in model.graph.output)
    existing_outputs = set(original_outputs)
    value_info = _value_info_by_name(inferred)
    producer_tensors = {
        str(output) for node in model.graph.node for output in node.output if output
    }
    added: list[str] = []
    seen: set[str] = set()
    for tensor_name in probe_outputs:
        if tensor_name in seen:
            continue
        seen.add(tensor_name)
        if tensor_name in existing_outputs:
            continue
        if tensor_name not in producer_tensors or tensor_name not in value_info:
            raise DiagnosticPreflightError(
                status="BLOCKED",
                code="PROBE_TENSOR_NOT_INSTRUMENTABLE",
                message="Planned probe tensor has no unique typed graph value",
                details={"role": role, "tensor": tensor_name},
            )
        model.graph.output.append(copy.deepcopy(value_info[tensor_name]))
        added.append(tensor_name)

    try:
        onnx.checker.check_model(model)
        onnx.save_model(model, str(output_path))
        onnx.checker.check_model(onnx.load(str(output_path), load_external_data=False))
    except Exception as exc:
        raise DiagnosticPreflightError(
            status="EXECUTION_ERROR",
            code="ONNX_INSTRUMENTED_GRAPH_INVALID",
            message="Derived instrumented graph failed structural validation",
            details={"role": role, "error": str(exc)},
        ) from exc
    if _sha256_file(frozen_path) != frozen_sha256:
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="FROZEN_GRAPH_OVERWRITE_DETECTED",
            message="Frozen graph bytes changed during instrumentation",
            details={"role": role},
        )
    return InstrumentedGraph(
        role=role,
        path=output_path,
        sha256=_sha256_file(output_path),
        size_bytes=output_path.stat().st_size,
        frozen_graph_sha256=frozen_sha256,
        original_outputs=original_outputs,
        probe_outputs=tuple(added),
    )


def derive_instrumented_graphs(
    verified: VerifiedStaticPreflight, output_directory: Path
) -> InstrumentationResult:
    """Create derived graph copies only after static and frozen authority checks."""

    verified.authorities.assert_complete()
    mappings = _probe_mappings(verified.probe_plan)
    source_authority = verified.authorities.authority_for("onnx_fp32_source")
    target_authority = verified.authorities.authority_for("onnx_int8_candidate")
    output_directory.mkdir(parents=True, exist_ok=True)
    source = _derive_graph(
        role="source",
        frozen_path=source_authority.path,
        frozen_sha256=source_authority.sha256,
        probe_outputs=tuple(mapping["source_tensor"] for mapping in mappings),
        output_path=output_directory / "source-instrumented.onnx",
    )
    target = _derive_graph(
        role="target",
        frozen_path=target_authority.path,
        frozen_sha256=target_authority.sha256,
        probe_outputs=tuple(mapping["target_tensor"] for mapping in mappings),
        output_path=output_directory / "target-instrumented.onnx",
    )
    return InstrumentationResult(source=source, target=target, probe_mappings=mappings)
