from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.activation_authority import (
    VerifiedActivationAuthority,
    derive_target_integer_capture_graph,
    verify_activation_authority,
)
from neural_continuity.m1_diagnostics.activation_evidence import (
    finalize_activation_package,
    prepare_capture_package,
    write_activation_batch,
)
from neural_continuity.m1_diagnostics.fidelity_authority import FidelityGateError
from neural_continuity.m1_diagnostics.fidelity_control import (
    BATCH_SIZE,
    EXECUTION_PROVIDER,
    ROLE,
    _canonical_queries,
    _run_final_outputs,
    _token_inputs,
    _verify_dataset_identity,
    _verify_models_unchanged,
    _verify_tokenizer,
)

ProgressCallback = Callable[[Mapping[str, Any]], None]


def _activation_session(model_path: Path) -> Any:
    try:
        import onnxruntime
    except ModuleNotFoundError as exc:
        raise FidelityGateError(
            "ONNX_DEPENDENCY_MISSING", "onnxruntime is unavailable", "EXECUTION_ERROR"
        ) from exc
    if EXECUTION_PROVIDER not in onnxruntime.get_available_providers():
        raise FidelityGateError(
            "ONNX_PROVIDER_MISSING", "CPUExecutionProvider is unavailable", "EXECUTION_ERROR"
        )
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    try:
        session = onnxruntime.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=[EXECUTION_PROVIDER],
        )
    except Exception as exc:
        raise FidelityGateError(
            "ONNX_SESSION_FAILED",
            f"cannot create activation session: {exc}",
            "EXECUTION_ERROR",
        ) from exc
    if session.get_providers() != [EXECUTION_PROVIDER]:
        raise FidelityGateError("ONNX_PROVIDER_INVALID", "activation session is not CPU-only")
    if {value.name for value in session.get_inputs()} != {
        "input_ids",
        "attention_mask",
        "token_type_ids",
    }:
        raise FidelityGateError("ONNX_IO_INVALID", "activation graph input identity mismatch")
    return session


def _run_requested(
    session: Any,
    output_names: Sequence[str],
    inputs: Mapping[str, np.ndarray],
    label: str,
) -> list[np.ndarray]:
    available = {value.name for value in session.get_outputs()}
    if len(output_names) != len(set(output_names)) or not set(output_names).issubset(available):
        raise FidelityGateError("ONNX_IO_INVALID", f"requested outputs unavailable: {label}")
    try:
        values = session.run(list(output_names), dict(inputs))
    except Exception as exc:
        raise FidelityGateError(
            "ONNX_INFERENCE_FAILED",
            f"activation execution failed for {label}: {exc}",
            "EXECUTION_ERROR",
        ) from exc
    if len(values) != len(output_names):
        raise FidelityGateError("ACTIVATION_COUNT_MISMATCH", f"output count mismatch: {label}")
    return [np.ascontiguousarray(value) for value in values]


def _probe_mappings(authority: VerifiedActivationAuthority) -> list[dict[str, Any]]:
    return [
        {
            "probe_id": probe["probe_id"],
            "source_tensor": probe["source_tensor"],
            "target_tensor": probe["target_tensor"],
            "target_tensor_basis": probe["target_tensor_basis"],
            "structural_families": probe["structural_families"],
        }
        for probe in authority.probe_plan["probes"]
    ]


def _capture_plan(
    authority: VerifiedActivationAuthority,
    derived: Mapping[str, Any],
    query_ids: Sequence[str],
) -> dict[str, Any]:
    query_identity_sha256 = hashlib.sha256(
        canonical_json_bytes({"query_ids": list(query_ids)})
    ).hexdigest()
    return {
        "kind": "m1_transition_b_v2_activation_capture_plan",
        "version": "1.0.0",
        "status": "FROZEN_BEFORE_CAPTURE",
        "fidelity_manifest_sha256": authority.fidelity_manifest_sha256,
        "instrumentation_manifest_sha256": (
            authority.instrumentation.instrumentation_manifest_sha256
        ),
        "static_manifest_sha256": authority.instrumentation.static_manifest_sha256,
        "probe_plan_sha256": authority.probe_plan["probe_plan_sha256"],
        "probe_plan_artifact_sha256": sha256_file(authority.static_root / "probe-plan.json"),
        "quantization_audit_sha256": sha256_file(
            authority.static_root / "quantization-parameter-audit.json"
        ),
        "source_instrumented_sha256": sha256_file(
            authority.instrumentation.source_instrumented_path
        ),
        "target_instrumented_sha256": sha256_file(
            authority.instrumentation.target_instrumented_path
        ),
        "target_capture_graph_sha256": derived["target_capture_graph_sha256"],
        "dataset_role": ROLE,
        "query_count": len(query_ids),
        "query_identity_sha256": query_identity_sha256,
        "query_order": "query_id_utf8_byte_order",
        "batch_size": BATCH_SIZE,
        "execution_provider": EXECUTION_PROVIDER,
        "graph_optimization_level": "ORT_DISABLE_ALL",
        "execution_mode": "ORT_SEQUENTIAL",
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1,
        "probe_count": len(authority.probe_plan["probes"]),
        "integer_probe_count": derived["integer_probe_count"],
        "probe_mappings": _probe_mappings(authority),
        "integer_mappings": derived["integer_mappings"],
        "integer_derivation_rule": derived["derivation_rule"],
        "exceptions": {
            "node_names": derived["node_name_exceptions"],
            "node_indices": derived["node_index_exceptions"],
            "tensor_names": derived["tensor_name_exceptions"],
        },
        "scientific_decision_recomputed": False,
        "frozen_transition_b_v1_scientific_decision": "FAIL",
    }


def capture_activations(
    config_path: str | Path,
    dataset_directory: str | Path,
    instrumentation_directory: str | Path,
    fidelity_directory: str | Path,
    output_directory: str | Path,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    output_path = Path(output_directory).resolve()
    if output_path.exists():
        raise FidelityGateError("OUTPUT_ALREADY_EXISTS", f"output exists: {output_path}")
    if not output_path.parent.is_dir():
        raise FidelityGateError(
            "OUTPUT_PARENT_MISSING", f"output parent missing: {output_path.parent}"
        )

    authority = verify_activation_authority(
        fidelity_directory,
        instrumentation_directory,
    )
    dataset, _ = _verify_dataset_identity(Path(dataset_directory).resolve())
    teacher, _ = _verify_tokenizer(Path(config_path).resolve(), authority.instrumentation)
    query_ids, query_texts = _canonical_queries(dataset)

    build_root = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.building-", dir=output_path.parent)
    )
    try:
        target_capture_path = build_root / "target-integer-capture.onnx"
        derived = derive_target_integer_capture_graph(authority, target_capture_path)

        original_final = _run_final_outputs(
            teacher,
            authority.instrumentation.target_instrumented_path,
            query_texts,
        )
        derived_final = _run_final_outputs(teacher, target_capture_path, query_texts)
        if not np.array_equal(original_final, derived_final):
            raise FidelityGateError(
                "DERIVATIVE_FIDELITY_BLOCKED",
                "target integer-capture derivative changed final embeddings",
            )
        capture_plan = _capture_plan(authority, derived, query_ids)
        preflight = {
            "kind": "m1_transition_b_v2_activation_capture_preflight",
            "status": "PASS",
            "derivative_final_output_fidelity": "PASS",
            "derivative_differing_value_count": 0,
            "derivative_max_abs_delta": 0.0,
            "activations_read_before_preflight": False,
            "onnx_runtime_activation_session_created_before_preflight": False,
        }
        capture_plan_sha256 = prepare_capture_package(
            build_root,
            capture_plan,
            preflight,
        )

        source_session = _activation_session(authority.instrumentation.source_instrumented_path)
        target_session = _activation_session(target_capture_path)
        probe_mappings = capture_plan["probe_mappings"]
        integer_mappings = capture_plan["integer_mappings"]
        source_names = [mapping["source_tensor"] for mapping in probe_mappings]
        target_names = [mapping["target_tensor"] for mapping in probe_mappings]
        integer_names = [mapping["target_integer_tensor"] for mapping in integer_mappings]
        batch_records: list[dict[str, Any]] = []
        total_batches = (len(query_ids) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_number, start in enumerate(
            range(0, len(query_ids), BATCH_SIZE),
            start=1,
        ):
            batch_query_ids = query_ids[start : start + BATCH_SIZE]
            batch_query_texts = query_texts[start : start + BATCH_SIZE]
            inputs = _token_inputs(teacher, batch_query_texts)
            source_values = _run_requested(
                source_session,
                source_names,
                inputs,
                "source_floating",
            )
            target_all = _run_requested(
                target_session,
                target_names + integer_names,
                inputs,
                "target_floating_and_integer",
            )
            target_values = target_all[: len(target_names)]
            integer_values = target_all[len(target_names) :]
            batch_id = f"batch-{batch_number:04d}"
            record = write_activation_batch(
                build_root,
                batch_id,
                batch_query_ids,
                probe_mappings,
                integer_mappings,
                source_values,
                target_values,
                integer_values,
            )
            batch_records.append(record)
            if progress is not None:
                progress(
                    {
                        "event": "activation_batch_captured",
                        "batch": batch_number,
                        "batch_count": total_batches,
                        "query_count": len(batch_query_ids),
                        "floating_size_bytes": record["floating_size_bytes"],
                        "integer_size_bytes": record["integer_size_bytes"],
                    }
                )

        if sha256_file(build_root / "capture-plan.json") != capture_plan_sha256:
            raise FidelityGateError("CAPTURE_PLAN_MUTATED", "capture plan changed during execution")
        if sha256_file(target_capture_path) != derived["target_capture_graph_sha256"]:
            raise FidelityGateError(
                "MODEL_MUTATION_DETECTED", "target capture graph changed during execution"
            )
        _verify_models_unchanged(authority.instrumentation)
        batch_index = {
            "kind": "m1_transition_b_v2_activation_batch_index",
            "batch_size": BATCH_SIZE,
            "batch_count": len(batch_records),
            "query_count": len(query_ids),
            "batches": batch_records,
        }
        return finalize_activation_package(build_root, output_path, batch_index)
    except Exception:
        if build_root.exists():
            shutil.rmtree(build_root)
        raise
