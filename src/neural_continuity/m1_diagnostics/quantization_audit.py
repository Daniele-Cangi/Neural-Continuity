"""Read-only structural audit of ONNX QDQ quantization parameters."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from neural_continuity.m1_diagnostics.graph_inventory import LoadedOnnxGraph


@dataclass(frozen=True)
class ParameterSummary:
    tensor_name: str
    source_kind: str
    data_type: str
    shape: tuple[int, ...]
    element_count: int
    value_sha256: str
    minimum: float | int | str
    maximum: float | int | str
    all_finite: bool
    all_positive: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "tensor_name": self.tensor_name,
            "source_kind": self.source_kind,
            "data_type": self.data_type,
            "shape": list(self.shape),
            "element_count": self.element_count,
            "value_sha256": self.value_sha256,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "all_finite": self.all_finite,
            "all_positive": self.all_positive,
        }


@dataclass(frozen=True)
class QuantizationParameterRecord:
    node_index: int
    node_name: str
    op_type: str
    axis: int
    scale: ParameterSummary | None
    zero_point: ParameterSummary | None
    zero_point_implicit_default: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "node_index": self.node_index,
            "node_name": self.node_name,
            "op_type": self.op_type,
            "axis": self.axis,
            "scale": None if self.scale is None else self.scale.to_dict(),
            "zero_point": None if self.zero_point is None else self.zero_point.to_dict(),
            "zero_point_implicit_default": self.zero_point_implicit_default,
        }


@dataclass(frozen=True)
class AuditFinding:
    code: str
    classification: str
    node_index: int
    node_name: str
    op_type: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "classification": self.classification,
            "node_index": self.node_index,
            "node_name": self.node_name,
            "op_type": self.op_type,
            "message": self.message,
        }


@dataclass(frozen=True)
class QuantizationAudit:
    graph_sha256: str
    records: tuple[QuantizationParameterRecord, ...]
    findings: tuple[AuditFinding, ...]

    @property
    def status(self) -> str:
        return "BLOCKED" if self.blocking_findings else "COMPLETE"

    @property
    def integrity_status(self) -> str:
        return "BLOCKED" if self.blocking_findings else "PASS"

    @property
    def blocking_findings(self) -> tuple[AuditFinding, ...]:
        return tuple(
            finding for finding in self.findings if finding.classification == "INTEGRITY_ERROR"
        )

    @property
    def diagnostic_anomalies(self) -> tuple[AuditFinding, ...]:
        return tuple(
            finding for finding in self.findings if finding.classification == "DIAGNOSTIC_ANOMALY"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "m1_transition_b_v2_quantization_parameter_audit",
            "status": self.status,
            "integrity_status": self.integrity_status,
            "graph_sha256": self.graph_sha256,
            "audited_node_count": len(self.records),
            "records": [record.to_dict() for record in self.records],
            "findings": [finding.to_dict() for finding in self.findings],
            "blocking_finding_count": len(self.blocking_findings),
            "diagnostic_anomaly_count": len(self.diagnostic_anomalies),
            "rules_basis": "onnx_qdq_parameter_semantics",
            "read_only": True,
            "model_execution_used": False,
        }


def _constant_arrays(model: Any) -> dict[str, tuple[np.ndarray[Any, Any], str]]:
    import onnx

    arrays: dict[str, tuple[np.ndarray[Any, Any], str]] = {}
    for initializer in model.graph.initializer:
        arrays[str(initializer.name)] = (
            np.asarray(onnx.numpy_helper.to_array(initializer)),
            "initializer",
        )
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        for attribute in node.attribute:
            if attribute.name == "value" and attribute.HasField("t"):
                arrays[str(node.output[0])] = (
                    np.asarray(onnx.numpy_helper.to_array(attribute.t)),
                    "constant_node",
                )
    return arrays


def _summarize(tensor_name: str, value: np.ndarray[Any, Any], source_kind: str) -> ParameterSummary:
    contiguous = np.ascontiguousarray(value)
    finite = np.isfinite(contiguous)
    is_floating = np.issubdtype(contiguous.dtype, np.floating)
    minimum = _json_number(np.min(contiguous), is_floating=is_floating)
    maximum = _json_number(np.max(contiguous), is_floating=is_floating)
    return ParameterSummary(
        tensor_name=tensor_name,
        source_kind=source_kind,
        data_type=str(contiguous.dtype),
        shape=tuple(int(dimension) for dimension in contiguous.shape),
        element_count=int(contiguous.size),
        value_sha256=hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
        minimum=minimum,
        maximum=maximum,
        all_finite=bool(np.all(finite)),
        all_positive=bool(np.all(contiguous > 0)),
    )


def _json_number(value: Any, *, is_floating: bool) -> float | int | str:
    if not is_floating:
        return int(value)
    number = float(value)
    if np.isnan(number):
        return "NaN"
    if np.isposinf(number):
        return "Infinity"
    if np.isneginf(number):
        return "-Infinity"
    return number


def _axis(node: Any) -> int:
    for attribute in node.attribute:
        if attribute.name == "axis":
            return int(attribute.i)
    return 1


def audit_quantization_parameters(graph: LoadedOnnxGraph) -> QuantizationAudit:
    """Audit QDQ parameters without evaluating the graph or any activation."""

    constants = _constant_arrays(graph.model)
    records: list[QuantizationParameterRecord] = []
    findings: list[AuditFinding] = []
    for index, node in enumerate(graph.model.graph.node):
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        node_name = str(node.name)
        op_type = str(node.op_type)
        scale_name = str(node.input[1]) if len(node.input) > 1 else ""
        scale_value = constants.get(scale_name)
        scale = (
            None if scale_value is None else _summarize(scale_name, scale_value[0], scale_value[1])
        )
        zero_name = str(node.input[2]) if len(node.input) > 2 and node.input[2] else ""
        zero_value = constants.get(zero_name) if zero_name else None
        zero_point = (
            None if zero_value is None else _summarize(zero_name, zero_value[0], zero_value[1])
        )
        records.append(
            QuantizationParameterRecord(
                node_index=index,
                node_name=node_name,
                op_type=op_type,
                axis=_axis(node),
                scale=scale,
                zero_point=zero_point,
                zero_point_implicit_default=not zero_name,
            )
        )

        finding_specs: list[tuple[str, str, str]] = []

        if scale is None:
            finding_specs.append(
                (
                    "SCALE_NOT_STATIC",
                    "INTEGRITY_ERROR",
                    "Scale is not resolvable from static ONNX constants",
                )
            )
        else:
            if scale.element_count == 0:
                finding_specs.append(("SCALE_EMPTY", "INTEGRITY_ERROR", "Scale contains no values"))
            if not scale.data_type.startswith("float"):
                finding_specs.append(
                    (
                        "SCALE_NOT_FLOAT",
                        "INTEGRITY_ERROR",
                        "Scale does not use a floating-point type",
                    )
                )
            if not scale.all_finite:
                finding_specs.append(
                    (
                        "SCALE_NOT_FINITE",
                        "DIAGNOSTIC_ANOMALY",
                        "Scale contains a non-finite value",
                    )
                )
            if not scale.all_positive:
                finding_specs.append(
                    (
                        "SCALE_NOT_POSITIVE",
                        "DIAGNOSTIC_ANOMALY",
                        "Scale contains a non-positive value",
                    )
                )
        if zero_name and zero_point is None:
            finding_specs.append(
                (
                    "ZERO_POINT_NOT_STATIC",
                    "INTEGRITY_ERROR",
                    "Declared zero-point is not resolvable from static ONNX constants",
                )
            )
        if zero_point is not None:
            if not np.issubdtype(np.dtype(zero_point.data_type), np.integer):
                finding_specs.append(
                    (
                        "ZERO_POINT_NOT_INTEGER",
                        "INTEGRITY_ERROR",
                        "Zero-point does not use an integer type",
                    )
                )
            if scale is not None and zero_point.shape != scale.shape:
                finding_specs.append(
                    (
                        "PARAMETER_SHAPE_MISMATCH",
                        "INTEGRITY_ERROR",
                        "Scale and zero-point shapes differ",
                    )
                )
        findings.extend(
            AuditFinding(
                code=code,
                classification=classification,
                node_index=index,
                node_name=node_name,
                op_type=op_type,
                message=message,
            )
            for code, classification, message in finding_specs
        )

    return QuantizationAudit(
        graph_sha256=graph.authority.sha256,
        records=tuple(records),
        findings=tuple(findings),
    )
