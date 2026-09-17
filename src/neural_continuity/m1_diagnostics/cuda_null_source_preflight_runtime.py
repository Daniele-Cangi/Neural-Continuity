"""Six independent source-only CUDA profiles after fresh frozen authority."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.evidence import canonical_json_bytes
from neural_continuity.m1_b.onnx_source import encode_onnx_source
from neural_continuity.m1_diagnostics.cuda_null_authority import (
    CudaNullAuthorityBlocked,
    _verify_source,
)
from neural_continuity.m1_diagnostics.cuda_null_source_preflight_authority import (
    RUNTIME_IDENTITY_SHA256,
    CudaNullSourcePreflightBlocked,
    verify_source_preflight_authority,
)
from neural_continuity.m1_diagnostics.cuda_null_source_preflight_inputs import (
    DOCUMENT_COUNT,
    EMBEDDING_DIMENSION,
    FROZEN_RUNS,
    MAX_SEQUENCE_LENGTH,
    QUERY_COUNT,
    load_source_preflight_inputs,
)
from neural_continuity.m1_diagnostics.cuda_preflight_runtime import (
    _profile_summary,
    _profiled_session,
)

PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")


def _reverify_teacher_source(transition_a_bundle: Path, snapshot_root: Path) -> None:
    try:
        _verify_source(transition_a_bundle, snapshot_root)
    except (CudaNullAuthorityBlocked, OSError, ValueError) as exc:
        raise CudaNullSourcePreflightBlocked(
            "teacher snapshot or source changed before inference"
        ) from exc


def capture_source_preflight(
    *,
    authorization_spec: Path,
    external_reviewed_spec_sha256: str,
    preflight_spec: Path,
    external_preflight_spec_sha256: str,
    static_bundle: Path,
    external_static_manifest_sha256: str,
    config_path: Path,
    external_config_sha256: str,
    dataset_root: Path,
    transition_a_bundle: Path,
    teacher_snapshot_root: Path,
    cpu_extension_bundle: Path,
    historical_cuda_bundle: Path,
    working_directory: Path,
) -> dict[str, Any]:
    """Verify every authority before importing a teacher or creating a session."""
    runtime: dict[str, Any] = {}
    authority = verify_source_preflight_authority(
        authorization_spec=authorization_spec,
        external_reviewed_spec_sha256=external_reviewed_spec_sha256,
        preflight_spec=preflight_spec,
        external_preflight_spec_sha256=external_preflight_spec_sha256,
        static_bundle=static_bundle,
        external_static_manifest_sha256=external_static_manifest_sha256,
        config_path=config_path,
        external_config_sha256=external_config_sha256,
        dataset_root=dataset_root,
        transition_a_bundle=transition_a_bundle,
        teacher_snapshot_root=teacher_snapshot_root,
        cpu_extension_bundle=cpu_extension_bundle,
        historical_cuda_bundle=historical_cuda_bundle,
        runtime_inventory_out=runtime,
    )
    if (
        authority.get("status") != "SOURCE_ONLY_PREFLIGHT_AUTHORITY_VERIFIED"
        or authority.get("technical_preflight_permission") != "GRANTED_AFTER_REVIEW"
        or authority.get("source_only") is not True
        or authority.get("int8_allowed") is not False
    ):
        raise ValueError("source-only preflight authority did not grant the bounded scope")
    try:
        inputs = load_source_preflight_inputs(
            dataset_root, transition_a_bundle, teacher_snapshot_root
        )
    except (OSError, ValueError) as exc:
        raise CudaNullSourcePreflightBlocked("frozen source inputs did not verify") from exc
    runtime_hash = hashlib.sha256(canonical_json_bytes(runtime) + b"\n").hexdigest()
    if runtime_hash != RUNTIME_IDENTITY_SHA256:
        raise ValueError("runtime identity changed after pre-execution authority")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer

    _reverify_teacher_source(transition_a_bundle, inputs.snapshot_root)
    teacher = SentenceTransformer(str(inputs.snapshot_root), device="cpu")
    _reverify_teacher_source(transition_a_bundle, inputs.snapshot_root)
    teacher.eval()
    if teacher.max_seq_length != MAX_SEQUENCE_LENGTH:
        raise CudaNullSourcePreflightBlocked("teacher tokenizer maximum sequence length mismatch")

    working = Path(working_directory)
    working.mkdir(parents=True, exist_ok=True)
    observations: dict[str, np.ndarray] = {}
    profiles: list[dict[str, Any]] = []
    for role, batch_size in FROZEN_RUNS:
        name = f"{role}_batch_{batch_size}"
        texts = inputs.document_texts if role == "documents" else inputs.query_texts
        expected_count = DOCUMENT_COUNT if role == "documents" else QUERY_COUNT
        session_start = time.perf_counter()
        session = _profiled_session(inputs.source_onnx, working / name, PROVIDERS)
        session_seconds = time.perf_counter() - session_start
        complete = False
        try:
            started = time.perf_counter()
            array = encode_onnx_source(teacher, session, texts, batch_size, name)
            elapsed = time.perf_counter() - started
            complete = True
        finally:
            profile_path = Path(session.end_profiling())
            del session
            if not complete:
                profile_path.unlink(missing_ok=True)
        summary = _profile_summary(profile_path, PROVIDERS)
        if (
            array.dtype != np.float32
            or array.shape != (expected_count, EMBEDDING_DIMENSION)
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"invalid source embedding shape, dtype, or finiteness: {name}")
        norms = np.linalg.norm(array.astype(np.float64), axis=1)
        if not np.isfinite(norms).all() or np.any(np.abs(norms - 1.0) > 1e-4):
            raise ValueError(f"source embeddings are not L2-normalized: {name}")
        if elapsed <= 0:
            raise ValueError(f"non-positive encoding time: {name}")
        normalized = np.ascontiguousarray(array, dtype="<f4")
        observations[name] = normalized
        profiles.append(
            {
                "role": role,
                "batch_size": batch_size,
                "observation_name": name,
                "observation_sha256": hashlib.sha256(normalized.tobytes()).hexdigest(),
                "item_count": expected_count,
                "session_seconds": session_seconds,
                "encode_seconds": elapsed,
                "items_per_second": expected_count / elapsed,
                **summary,
            }
        )
    return {
        "runtime_inventory": runtime,
        "input_identity": {**inputs.identity, "source_preflight_authority": authority},
        "observations": observations,
        "provider_profiles": profiles,
    }
