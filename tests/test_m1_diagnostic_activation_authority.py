from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from neural_continuity.m1_diagnostics.activation_authority import (
    POST_QDQ_BASIS,
    VerifiedActivationAuthority,
    derive_target_integer_capture_graph,
)
from neural_continuity.m1_diagnostics.fidelity_authority import (
    VerifiedFidelityAuthority,
)


def test_integer_capture_graph_is_derived_from_dequantize_lineage(tmp_path: Path) -> None:
    source_path = tmp_path / "target.onnx"
    graph = helper.make_graph(
        [
            helper.make_node(
                "QuantizeLinear",
                ["x", "scale", "zero_point"],
                ["quantized"],
                name="arbitrary_quantizer",
            ),
            helper.make_node(
                "DequantizeLinear",
                ["quantized", "scale", "zero_point"],
                ["dequantized"],
                name="arbitrary_dequantizer",
            ),
        ],
        "structural-test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 2])],
        [helper.make_tensor_value_info("dequantized", TensorProto.FLOAT, ["batch", 2])],
        [
            numpy_helper.from_array(np.asarray(0.1, dtype=np.float32), name="scale"),
            numpy_helper.from_array(np.asarray(128, dtype=np.uint8), name="zero_point"),
        ],
    )
    onnx.save(helper.make_model(graph), source_path)
    instrumentation = VerifiedFidelityAuthority(
        instrumentation_root=tmp_path,
        instrumentation_manifest_sha256="instrumentation",
        static_manifest_sha256="static",
        source_original_path=source_path,
        source_instrumented_path=source_path,
        target_original_path=source_path,
        target_instrumented_path=source_path,
        authority_records=(),
    )
    authority = VerifiedActivationAuthority(
        fidelity_root=tmp_path,
        fidelity_manifest_sha256="fidelity",
        instrumentation=instrumentation,
        instrumentation_plan={},
        probe_plan={
            "probes": [
                {
                    "probe_id": "probe-0001",
                    "target_tensor": "dequantized",
                    "target_tensor_basis": POST_QDQ_BASIS,
                }
            ]
        },
        quantization_audit={
            "records": [
                {
                    "node_index": 0,
                    "node_name": "arbitrary_quantizer",
                    "op_type": "QuantizeLinear",
                    "scale": {"tensor_name": "scale", "value_sha256": "scale-hash"},
                    "zero_point": {
                        "tensor_name": "zero_point",
                        "value_sha256": "zero-point-hash",
                    },
                    "zero_point_implicit_default": False,
                    "axis": None,
                },
                {
                    "node_index": 1,
                    "node_name": "arbitrary_dequantizer",
                    "op_type": "DequantizeLinear",
                    "scale": {"tensor_name": "scale", "value_sha256": "scale-hash"},
                    "zero_point": {
                        "tensor_name": "zero_point",
                        "value_sha256": "zero-point-hash",
                    },
                    "zero_point_implicit_default": False,
                    "axis": None,
                },
                {
                    "node_index": 99,
                    "node_name": "unrelated_shared_parameter_dequantizer",
                    "op_type": "DequantizeLinear",
                    "scale": {"tensor_name": "scale", "value_sha256": "scale-hash"},
                    "zero_point": {
                        "tensor_name": "zero_point",
                        "value_sha256": "zero-point-hash",
                    },
                    "zero_point_implicit_default": False,
                    "axis": None,
                },
            ]
        },
        static_root=tmp_path,
    )

    output_path = tmp_path / "derived.onnx"
    result = derive_target_integer_capture_graph(authority, output_path)

    assert result["integer_probe_count"] == 1
    assert result["node_name_exceptions"] == []
    assert result["node_index_exceptions"] == []
    assert result["tensor_name_exceptions"] == []
    derived = onnx.load(output_path)
    assert {value.name for value in derived.graph.output} == {
        "dequantized",
        "quantized",
    }
