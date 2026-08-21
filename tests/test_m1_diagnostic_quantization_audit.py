from __future__ import annotations

from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from neural_continuity.m1_diagnostics.authority import VerifiedAuthority
from neural_continuity.m1_diagnostics.graph_inventory import LoadedOnnxGraph
from neural_continuity.m1_diagnostics.quantization_audit import (
    audit_quantization_parameters,
)


def _graph(scale: np.ndarray, zero_point: np.ndarray) -> LoadedOnnxGraph:
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("QuantizeLinear", ["x", "scale", "zero"], ["y"])],
            "audit",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.UINT8, [1])],
            [
                numpy_helper.from_array(scale, name="scale"),
                numpy_helper.from_array(zero_point, name="zero"),
            ],
        )
    )
    authority = VerifiedAuthority(
        role="onnx_int8_candidate", path=Path("model.onnx"), sha256="a" * 64, size_bytes=1
    )
    return LoadedOnnxGraph(role="onnx_int8_candidate", authority=authority, model=model)


def test_quantization_parameter_audit_accepts_valid_static_qdq_parameters() -> None:
    audit = audit_quantization_parameters(
        _graph(np.asarray(0.125, dtype=np.float32), np.asarray(0, dtype=np.uint8))
    )

    assert audit.status == "COMPLETE"
    assert audit.integrity_status == "PASS"
    assert len(audit.records) == 1
    assert not audit.findings


def test_quantization_parameter_audit_records_non_positive_scale_as_anomaly() -> None:
    audit = audit_quantization_parameters(
        _graph(np.asarray(0.0, dtype=np.float32), np.asarray(0, dtype=np.uint8))
    )

    assert audit.status == "COMPLETE"
    assert audit.integrity_status == "PASS"
    assert {finding.code for finding in audit.findings} == {"SCALE_NOT_POSITIVE"}
    assert {finding.classification for finding in audit.findings} == {"DIAGNOSTIC_ANOMALY"}


def test_quantization_parameter_audit_serializes_non_finite_scale_strictly() -> None:
    audit = audit_quantization_parameters(
        _graph(np.asarray(np.inf, dtype=np.float32), np.asarray(0, dtype=np.uint8))
    )

    assert audit.status == "COMPLETE"
    assert audit.records[0].scale is not None
    assert audit.records[0].scale.minimum == "Infinity"
    assert audit.diagnostic_anomalies[0].code == "SCALE_NOT_FINITE"
