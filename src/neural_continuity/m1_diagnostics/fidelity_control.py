from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.fidelity_authority import (
    EXPECTED_AUTHORITY_HASHES,
    SOURCE_INSTRUMENTED_SHA256,
    TARGET_INSTRUMENTED_SHA256,
    FidelityGateError,
    VerifiedFidelityAuthority,
    verify_fidelity_authority,
)
from neural_continuity.m1_diagnostics.fidelity_evidence import write_fidelity_package
from neural_continuity.m1_teacher_evidence import (
    TeacherEvidenceError,
    _load_config,
    _load_teacher,
    load_materialized_dataset,
)

CONFIG_SHA256 = "c22f142b36d27b6d59087369293b6f5be092fb40380a72e969200686f6969014"
DATASET_MANIFEST_SHA256 = "beab716b9f322478ca3f2efd0e6e93e7d66a2b3483ed098941cd9f2275bcdcc2"
DATASET_ID = "nc-m1-beir-scifact-v1"
ROLE = "contract_development"
ROLE_MEMBERSHIP_SHA256 = "8ebe572a4b582ca3d5aa6dc8ba9c46b14f299a1dba45faab27c3d2646d5387e6"
QUERY_COUNT = 364
BATCH_SIZE = 16
EXECUTION_PROVIDER = "CPUExecutionProvider"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FidelityGateError("IDENTITY_ARTIFACT_INVALID", f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FidelityGateError("IDENTITY_ARTIFACT_INVALID", f"identity is not an object: {path}")
    return value


def _authority_path(authority: VerifiedFidelityAuthority, role: str) -> Path:
    for record in authority.authority_records:
        if record.get("role") == role and isinstance(record.get("path"), str):
            return Path(record["path"]).resolve()
    raise FidelityGateError("AUTHORITY_SET_INVALID", f"authority path missing: {role}")


def _verify_dataset_identity(dataset_root: Path) -> tuple[Any, dict[str, Any]]:
    manifest_path = dataset_root / "materialization-manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != DATASET_MANIFEST_SHA256:
        raise FidelityGateError("DATASET_AUTHORITY_MISMATCH", "dataset manifest hash mismatch")
    manifest = _load_json(manifest_path)
    source_identity = manifest.get("source_identity")
    roles = manifest.get("roles")
    if (
        manifest.get("dataset_id") != DATASET_ID
        or not isinstance(source_identity, Mapping)
        or source_identity.get("materialization_policy_sha256")
        != "445aa58c22faad40ee567d28c98115589cf5811acac9b30e7b7a48f383bf9037"
        or source_identity.get("partition_policy_sha256")
        != "43eb7bd3a805792897de35cebd995d3d5b93931f08fca260f8a8d4aa1883457d"
        or not isinstance(roles, list)
    ):
        raise FidelityGateError("DATASET_AUTHORITY_MISMATCH", "dataset identity mismatch")
    role_records = [
        item for item in roles if isinstance(item, Mapping) and item.get("name") == ROLE
    ]
    if len(role_records) != 1:
        raise FidelityGateError("DATASET_ROLE_INVALID", "contract_development role is not unique")
    role_record = role_records[0]
    if (
        role_record.get("membership_sha256") != ROLE_MEMBERSHIP_SHA256
        or role_record.get("query_count") != QUERY_COUNT
        or role_record.get("upstream_split") != "train"
    ):
        raise FidelityGateError("DATASET_ROLE_INVALID", "contract_development identity mismatch")
    try:
        dataset = load_materialized_dataset(dataset_root)
    except TeacherEvidenceError as exc:
        raise FidelityGateError(exc.code, exc.message, exc.status) from exc
    if dataset.dataset_id != DATASET_ID or ROLE not in dataset.roles:
        raise FidelityGateError("DATASET_ROLE_INVALID", "loaded dataset identity mismatch")
    return dataset, dict(role_record)


def _verify_tokenizer(config_path: Path, authority: VerifiedFidelityAuthority) -> tuple[Any, Any]:
    source_evidence_path = _authority_path(authority, "paired_fp32_evidence")
    if sha256_file(source_evidence_path) != EXPECTED_AUTHORITY_HASHES["paired_fp32_evidence"]:
        raise FidelityGateError("TOKENIZER_AUTHORITY_MISMATCH", "source evidence hash mismatch")
    source_evidence = _load_json(source_evidence_path)
    expected = source_evidence.get("teacher_tokenizer_identity")
    if not isinstance(expected, Mapping):
        raise FidelityGateError("TOKENIZER_AUTHORITY_MISMATCH", "tokenizer identity is missing")
    try:
        config, config_sha256 = _load_config(config_path)
        teacher, observed = _load_teacher(config)
    except TeacherEvidenceError as exc:
        raise FidelityGateError(exc.code, exc.message, exc.status) from exc
    if config_sha256 != CONFIG_SHA256:
        raise FidelityGateError("CONFIG_AUTHORITY_MISMATCH", "parsed configuration hash mismatch")
    for field in ("model_id", "revision", "device", "cache_only", "snapshot_files"):
        if observed.get(field) != expected.get(field):
            raise FidelityGateError(
                "TOKENIZER_AUTHORITY_MISMATCH", f"tokenizer identity mismatch: {field}"
            )
    return teacher, observed


def _canonical_queries(dataset: Any) -> tuple[list[str], list[str]]:
    role_data = dataset.roles[ROLE]
    ordered = sorted(
        zip(role_data.query_ids, role_data.query_texts, strict=True),
        key=lambda item: item[0].encode(),
    )
    query_ids = [query_id for query_id, _ in ordered]
    query_texts = [text for _, text in ordered]
    if len(query_ids) != QUERY_COUNT or len(query_ids) != len(set(query_ids)):
        raise FidelityGateError("DATASET_ROLE_INVALID", "canonical query identity mismatch")
    return query_ids, query_texts


def _token_inputs(teacher: Any, texts: Sequence[str]) -> dict[str, np.ndarray]:
    tokens = teacher.tokenize(list(texts))
    input_ids = tokens.get("input_ids")
    attention_mask = tokens.get("attention_mask")
    if input_ids is None or attention_mask is None:
        raise FidelityGateError(
            "TOKENIZER_OUTPUT_INVALID", "tokenizer lacks authoritative inputs", "EXECUTION_ERROR"
        )
    token_type_ids = tokens.get("token_type_ids")
    if token_type_ids is None:
        token_type_array = np.zeros_like(input_ids.detach().cpu().numpy())
    else:
        token_type_array = token_type_ids.detach().cpu().numpy()
    return {
        "input_ids": input_ids.detach().cpu().numpy().astype(np.int64, copy=False),
        "attention_mask": attention_mask.detach().cpu().numpy().astype(np.int64, copy=False),
        "token_type_ids": np.asarray(token_type_array, dtype=np.int64),
    }


def _run_final_outputs(teacher: Any, model_path: Path, query_texts: Sequence[str]) -> np.ndarray:
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
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    try:
        session = onnxruntime.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=[EXECUTION_PROVIDER],
        )
    except Exception as exc:
        raise FidelityGateError(
            "ONNX_SESSION_FAILED", f"cannot create fidelity session: {exc}", "EXECUTION_ERROR"
        ) from exc
    if session.get_providers() != [EXECUTION_PROVIDER]:
        raise FidelityGateError("ONNX_PROVIDER_INVALID", "session is not CPU-only")
    if {value.name for value in session.get_inputs()} != {
        "input_ids",
        "attention_mask",
        "token_type_ids",
    }:
        raise FidelityGateError("ONNX_IO_INVALID", "model input identity mismatch")
    if "embeddings" not in {value.name for value in session.get_outputs()}:
        raise FidelityGateError("ONNX_IO_INVALID", "final embeddings output is missing")
    batches: list[np.ndarray] = []
    for start in range(0, len(query_texts), BATCH_SIZE):
        inputs = _token_inputs(teacher, query_texts[start : start + BATCH_SIZE])
        try:
            output = session.run(["embeddings"], inputs)[0]
        except Exception as exc:
            raise FidelityGateError(
                "ONNX_INFERENCE_FAILED", f"final-output execution failed: {exc}", "EXECUTION_ERROR"
            ) from exc
        value = np.asarray(output)
        if value.dtype != np.float32 or value.ndim != 2 or not np.isfinite(value).all():
            raise FidelityGateError(
                "ONNX_OUTPUT_INVALID", "final output is not finite float32", "EXECUTION_ERROR"
            )
        batches.append(np.ascontiguousarray(value))
    combined = np.ascontiguousarray(np.concatenate(batches, axis=0), dtype=np.float32)
    if combined.shape[0] != len(query_texts):
        raise FidelityGateError(
            "ONNX_OUTPUT_INVALID", "final output count mismatch", "EXECUTION_ERROR"
        )
    return combined


def _verify_models_unchanged(authority: VerifiedFidelityAuthority) -> None:
    expected = {
        authority.source_original_path: EXPECTED_AUTHORITY_HASHES["onnx_fp32_source"],
        authority.source_instrumented_path: SOURCE_INSTRUMENTED_SHA256,
        authority.target_original_path: EXPECTED_AUTHORITY_HASHES["onnx_int8_candidate"],
        authority.target_instrumented_path: TARGET_INSTRUMENTED_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise FidelityGateError(
                "MODEL_MUTATION_DETECTED", f"model changed during control: {path}"
            )


def capture_fidelity(
    config_path: str | Path,
    dataset_directory: str | Path,
    instrumentation_directory: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    authority = verify_fidelity_authority(instrumentation_directory)
    dataset, role_record = _verify_dataset_identity(Path(dataset_directory).resolve())
    teacher, tokenizer_identity = _verify_tokenizer(Path(config_path).resolve(), authority)
    query_ids, query_texts = _canonical_queries(dataset)

    outputs = {
        "source_original": _run_final_outputs(teacher, authority.source_original_path, query_texts),
        "source_instrumented": _run_final_outputs(
            teacher, authority.source_instrumented_path, query_texts
        ),
        "target_original": _run_final_outputs(teacher, authority.target_original_path, query_texts),
        "target_instrumented": _run_final_outputs(
            teacher, authority.target_instrumented_path, query_texts
        ),
    }
    _verify_models_unchanged(authority)
    query_identity_sha256 = hashlib.sha256(
        canonical_json_bytes({"query_ids": query_ids})
    ).hexdigest()
    authority_record = {
        "instrumentation_manifest_sha256": authority.instrumentation_manifest_sha256,
        "static_manifest_sha256": authority.static_manifest_sha256,
        "frozen_authority_count": len(authority.authority_records),
        "frozen_authorities": list(authority.authority_records),
        "dataset": {
            "dataset_id": DATASET_ID,
            "materialization_manifest_sha256": DATASET_MANIFEST_SHA256,
            "role": ROLE,
            "role_membership_sha256": role_record["membership_sha256"],
            "query_count": len(query_ids),
            "query_identity_sha256": query_identity_sha256,
        },
        "tokenizer": {
            key: tokenizer_identity.get(key)
            for key in ("model_id", "revision", "device", "cache_only", "snapshot_files")
        },
        "configuration_sha256": CONFIG_SHA256,
    }
    execution = {
        "provider": EXECUTION_PROVIDER,
        "batch_size": BATCH_SIZE,
        "role": ROLE,
        "query_order": "query_id_utf8_byte_order",
        "requested_output_names": ["embeddings"],
        "activation_output_requested": False,
        "graph_optimization_level": "ORT_DISABLE_ALL",
        "execution_mode": "ORT_SEQUENTIAL",
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1,
    }
    return write_fidelity_package(
        output_directory,
        query_ids,
        outputs,
        authority_record,
        execution,
    )
