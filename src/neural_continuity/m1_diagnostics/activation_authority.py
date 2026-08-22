from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural_continuity.evidence import sha256_file
from neural_continuity.m1_diagnostics.fidelity_authority import (
    INSTRUMENTATION_MANIFEST_SHA256,
    PROBE_PLAN_SHA256,
    STATIC_MANIFEST_SHA256,
    FidelityGateError,
    VerifiedFidelityAuthority,
    verify_artifact_manifest,
    verify_fidelity_authority,
)
from neural_continuity.m1_diagnostics.fidelity_control import (
    CONFIG_SHA256,
    DATASET_ID,
    DATASET_MANIFEST_SHA256,
    ROLE,
    ROLE_MEMBERSHIP_SHA256,
)
from neural_continuity.m1_diagnostics.fidelity_evidence import replay_fidelity

FIDELITY_MANIFEST_SHA256 = "5f941050084364541a6412121de4544646a53c9ce900363416b222596cff8f31"
POST_QDQ_BASIS = "post_quantize_dequantize_output"
DIRECT_BASIS = "direct_compute_output"
EXPECTED_PROBE_COUNT = 283
EXPECTED_INTEGER_PROBE_COUNT = 248
EXPECTED_DIRECT_PROBE_COUNT = 35


@dataclass(frozen=True)
class VerifiedActivationAuthority:
    fidelity_root: Path
    fidelity_manifest_sha256: str
    instrumentation: VerifiedFidelityAuthority
    instrumentation_plan: dict[str, Any]
    probe_plan: dict[str, Any]
    quantization_audit: dict[str, Any]
    static_root: Path


def _load_json(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FidelityGateError(code, f"cannot load JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FidelityGateError(code, f"JSON artifact is not an object: {path}")
    return value


def _verify_fidelity_identity(root: Path) -> None:
    verify_artifact_manifest(root, FIDELITY_MANIFEST_SHA256)
    replay = replay_fidelity(root / "replay-bundle.json", FIDELITY_MANIFEST_SHA256)
    if (
        replay.get("status") != "COMPLETE"
        or replay.get("fidelity_status") != "PASS"
        or replay.get("replay_verified") is not True
        or replay.get("model_execution_used") is not False
    ):
        raise FidelityGateError("FIDELITY_AUTHORITY_INVALID", "final-output fidelity did not pass")
    authority = _load_json(root / "fidelity-authority.json", "FIDELITY_AUTHORITY_INVALID")
    dataset = authority.get("dataset")
    if (
        authority.get("status") != "PASS"
        or authority.get("instrumentation_manifest_sha256") != INSTRUMENTATION_MANIFEST_SHA256
        or authority.get("static_manifest_sha256") != STATIC_MANIFEST_SHA256
        or authority.get("configuration_sha256") != CONFIG_SHA256
        or authority.get("frozen_authority_count") != 8
        or not isinstance(dataset, Mapping)
        or dataset.get("dataset_id") != DATASET_ID
        or dataset.get("materialization_manifest_sha256") != DATASET_MANIFEST_SHA256
        or dataset.get("role") != ROLE
        or dataset.get("role_membership_sha256") != ROLE_MEMBERSHIP_SHA256
        or dataset.get("query_count") != 364
    ):
        raise FidelityGateError("FIDELITY_AUTHORITY_INVALID", "fidelity identity is not frozen")


def _probe_projection(probe: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "probe_id": probe.get("probe_id"),
        "source_tensor": probe.get("source_tensor"),
        "target_tensor": probe.get("target_tensor"),
    }


def verify_activation_authority(
    fidelity_directory: str | Path,
    instrumentation_directory: str | Path,
) -> VerifiedActivationAuthority:
    fidelity_root = Path(fidelity_directory).resolve()
    _verify_fidelity_identity(fidelity_root)
    instrumentation = verify_fidelity_authority(instrumentation_directory)

    instrumentation_authority = _load_json(
        instrumentation.instrumentation_root / "instrumentation-authority.json",
        "INSTRUMENTATION_AUTHORITY_INVALID",
    )
    static_preflight = instrumentation_authority.get("static_preflight")
    if not isinstance(static_preflight, Mapping):
        raise FidelityGateError(
            "INSTRUMENTATION_AUTHORITY_INVALID", "static preflight authority is missing"
        )
    static_root_value = static_preflight.get("package_directory")
    if not isinstance(static_root_value, str):
        raise FidelityGateError(
            "INSTRUMENTATION_AUTHORITY_INVALID", "static package directory is missing"
        )
    static_root = Path(static_root_value).resolve()
    probe_plan_path = static_root / "probe-plan.json"
    probe_plan = _load_json(probe_plan_path, "PROBE_PLAN_INVALID")
    if probe_plan.get("probe_plan_sha256") != PROBE_PLAN_SHA256:
        raise FidelityGateError("PROBE_PLAN_HASH_MISMATCH", "frozen probe plan hash mismatch")
    audit = _load_json(
        static_root / "quantization-parameter-audit.json",
        "QUANTIZATION_AUDIT_INVALID",
    )
    instrumentation_plan = _load_json(
        instrumentation.instrumentation_root / "instrumentation-plan.json",
        "INSTRUMENTATION_PLAN_INVALID",
    )

    probes = probe_plan.get("probes")
    mappings = instrumentation_plan.get("probe_mappings")
    if (
        probe_plan.get("probe_count") != EXPECTED_PROBE_COUNT
        or instrumentation_plan.get("probe_count") != EXPECTED_PROBE_COUNT
        or not isinstance(probes, list)
        or not isinstance(mappings, list)
        or len(probes) != EXPECTED_PROBE_COUNT
        or len(mappings) != EXPECTED_PROBE_COUNT
    ):
        raise FidelityGateError("PROBE_PLAN_INVALID", "probe plan is incomplete")
    if [_probe_projection(probe) for probe in probes] != mappings:
        raise FidelityGateError(
            "PROBE_PLAN_MISMATCH", "instrumentation mappings differ from frozen probes"
        )
    probe_ids = [probe.get("probe_id") for probe in probes if isinstance(probe, Mapping)]
    if len(probe_ids) != EXPECTED_PROBE_COUNT or len(set(probe_ids)) != EXPECTED_PROBE_COUNT:
        raise FidelityGateError("PROBE_PLAN_INVALID", "probe IDs are invalid or duplicated")
    basis_counts = {
        basis: sum(
            isinstance(probe, Mapping) and probe.get("target_tensor_basis") == basis
            for probe in probes
        )
        for basis in (POST_QDQ_BASIS, DIRECT_BASIS)
    }
    if basis_counts != {
        POST_QDQ_BASIS: EXPECTED_INTEGER_PROBE_COUNT,
        DIRECT_BASIS: EXPECTED_DIRECT_PROBE_COUNT,
    }:
        raise FidelityGateError("PROBE_PLAN_INVALID", "probe basis counts are not authoritative")
    if (
        audit.get("status") != "COMPLETE"
        or audit.get("integrity_status") != "PASS"
        or audit.get("read_only") is not True
        or audit.get("model_execution_used") is not False
        or audit.get("audited_node_count") != 597
        or audit.get("blocking_finding_count") != 0
        or audit.get("diagnostic_anomaly_count") != 24
        or audit.get("rules_basis") != "onnx_qdq_parameter_semantics"
        or not isinstance(audit.get("records"), list)
    ):
        raise FidelityGateError(
            "QUANTIZATION_AUDIT_INVALID", "quantization parameter audit is not authoritative"
        )
    return VerifiedActivationAuthority(
        fidelity_root=fidelity_root,
        fidelity_manifest_sha256=FIDELITY_MANIFEST_SHA256,
        instrumentation=instrumentation,
        instrumentation_plan=instrumentation_plan,
        probe_plan=probe_plan,
        quantization_audit=audit,
        static_root=static_root,
    )


def _parameter_tensor(record: Mapping[str, Any], field: str) -> Any:
    parameter = record.get(field)
    return parameter.get("tensor_name") if isinstance(parameter, Mapping) else None


def _matching_audit_records(
    records: list[Any],
    *,
    node_index: int,
    node: Any,
    scale_tensor: str,
    zero_point_tensor: str | None,
) -> list[Mapping[str, Any]]:
    return [
        record
        for record in records
        if isinstance(record, Mapping)
        and record.get("node_index") == node_index
        and record.get("node_name") == node.name
        and record.get("op_type") == node.op_type
        and _parameter_tensor(record, "scale") == scale_tensor
        and (
            _parameter_tensor(record, "zero_point") == zero_point_tensor
            or (zero_point_tensor is None and record.get("zero_point_implicit_default") is True)
        )
    ]


def derive_target_integer_capture_graph(
    authority: VerifiedActivationAuthority,
    output_path: str | Path,
) -> dict[str, Any]:
    try:
        import onnx
    except ModuleNotFoundError as exc:
        raise FidelityGateError(
            "ONNX_DEPENDENCY_MISSING", "onnx is unavailable", "EXECUTION_ERROR"
        ) from exc

    target_path = authority.instrumentation.target_instrumented_path
    frozen_hash_before = sha256_file(target_path)
    try:
        model = onnx.load(str(target_path), load_external_data=False)
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception as exc:
        raise FidelityGateError(
            "TARGET_CAPTURE_GRAPH_INVALID",
            f"cannot load or infer target graph: {exc}",
            "EXECUTION_ERROR",
        ) from exc

    producers: dict[str, tuple[int, Any]] = {}
    for node_index, node in enumerate(inferred.graph.node):
        for tensor_name in node.output:
            if tensor_name in producers:
                raise FidelityGateError(
                    "TARGET_CAPTURE_GRAPH_INVALID", "target tensor has multiple producers"
                )
            producers[tensor_name] = (node_index, node)
    value_info = {
        value.name: value
        for value in (
            list(inferred.graph.input)
            + list(inferred.graph.value_info)
            + list(inferred.graph.output)
        )
    }
    audit_records = authority.quantization_audit["records"]
    existing_outputs = {value.name for value in inferred.graph.output}
    integer_names: set[str] = set()
    integer_mappings: list[dict[str, Any]] = []

    for probe in authority.probe_plan["probes"]:
        if probe["target_tensor_basis"] == DIRECT_BASIS:
            continue
        target_tensor = probe["target_tensor"]
        producer_entry = producers.get(target_tensor)
        if producer_entry is None:
            raise FidelityGateError(
                "INTEGER_LINEAGE_INVALID",
                f"post-QDQ probe lacks a DequantizeLinear producer: {probe['probe_id']}",
            )
        producer_index, producer = producer_entry
        if producer.op_type != "DequantizeLinear" or len(producer.input) < 2:
            raise FidelityGateError(
                "INTEGER_LINEAGE_INVALID",
                f"post-QDQ probe lacks a DequantizeLinear producer: {probe['probe_id']}",
            )
        integer_tensor = producer.input[0]
        scale_tensor = producer.input[1]
        zero_point_tensor = producer.input[2] if len(producer.input) > 2 else None
        quantize_entry = producers.get(integer_tensor)
        if quantize_entry is None:
            raise FidelityGateError(
                "INTEGER_LINEAGE_INVALID",
                f"integer probe lacks a QuantizeLinear producer: {probe['probe_id']}",
            )
        quantize_index, quantize = quantize_entry
        quantize_zero_point = quantize.input[2] if len(quantize.input) > 2 else None
        if (
            quantize.op_type != "QuantizeLinear"
            or len(quantize.input) < 2
            or quantize.input[1] != scale_tensor
            or quantize_zero_point != zero_point_tensor
        ):
            raise FidelityGateError(
                "QUANTIZATION_LINEAGE_INVALID",
                f"probe does not form a parameter-identical QDQ chain: {probe['probe_id']}",
            )
        quantize_matches = _matching_audit_records(
            audit_records,
            node_index=quantize_index,
            node=quantize,
            scale_tensor=scale_tensor,
            zero_point_tensor=zero_point_tensor,
        )
        dequantize_matches = _matching_audit_records(
            audit_records,
            node_index=producer_index,
            node=producer,
            scale_tensor=scale_tensor,
            zero_point_tensor=zero_point_tensor,
        )
        if len(quantize_matches) != 1 or len(dequantize_matches) != 1:
            raise FidelityGateError(
                "QUANTIZATION_LINEAGE_INVALID",
                f"probe has no unique audited QDQ lineage: {probe['probe_id']}",
            )
        quantize_record = quantize_matches[0]
        dequantize_record = dequantize_matches[0]
        if (
            quantize_record["scale"]["value_sha256"] != dequantize_record["scale"]["value_sha256"]
            or quantize_record["zero_point"]["value_sha256"]
            != dequantize_record["zero_point"]["value_sha256"]
        ):
            raise FidelityGateError(
                "QUANTIZATION_LINEAGE_INVALID",
                f"audited QDQ parameter values differ: {probe['probe_id']}",
            )
        if not integer_tensor or integer_tensor in integer_names:
            raise FidelityGateError(
                "INTEGER_LINEAGE_INVALID", "integer probe tensors are empty or duplicated"
            )
        tensor_info = value_info.get(integer_tensor)
        if tensor_info is None:
            raise FidelityGateError(
                "INTEGER_LINEAGE_INVALID",
                f"integer tensor type is unavailable: {probe['probe_id']}",
            )
        integer_names.add(integer_tensor)
        if integer_tensor not in existing_outputs:
            inferred.graph.output.append(copy.deepcopy(tensor_info))
            existing_outputs.add(integer_tensor)
        integer_mappings.append(
            {
                "probe_id": probe["probe_id"],
                "target_dequantized_tensor": target_tensor,
                "target_integer_tensor": integer_tensor,
                "lineage_operator": "DequantizeLinear",
                "scale_value_sha256": quantize_record["scale"]["value_sha256"],
                "zero_point_value_sha256": quantize_record["zero_point"]["value_sha256"],
                "quantization_axis": quantize_record.get("axis"),
            }
        )

    expected_integer_count = sum(
        probe["target_tensor_basis"] == POST_QDQ_BASIS for probe in authority.probe_plan["probes"]
    )
    if len(integer_mappings) != expected_integer_count:
        raise FidelityGateError("INTEGER_LINEAGE_INVALID", "integer capture mapping count mismatch")
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        onnx.checker.check_model(inferred)
        onnx.save(inferred, str(destination))
    except Exception as exc:
        raise FidelityGateError(
            "TARGET_CAPTURE_GRAPH_INVALID",
            f"cannot save target capture graph: {exc}",
            "EXECUTION_ERROR",
        ) from exc
    if sha256_file(target_path) != frozen_hash_before:
        raise FidelityGateError("MODEL_MUTATION_DETECTED", "frozen target graph changed")
    return {
        "target_capture_graph_path": str(destination),
        "target_capture_graph_sha256": sha256_file(destination),
        "integer_probe_count": len(integer_mappings),
        "integer_mappings": integer_mappings,
        "derivation_rule": "DequantizeLinear_input_0_for_each_post_QDQ_probe",
        "node_name_exceptions": [],
        "node_index_exceptions": [],
        "tensor_name_exceptions": [],
    }
