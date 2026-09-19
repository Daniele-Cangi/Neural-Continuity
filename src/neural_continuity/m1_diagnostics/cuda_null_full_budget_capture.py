"""One non-qualifying full-input timing capture after every authority verifies."""

from __future__ import annotations

import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_b.onnx_source import encode_onnx_source
from neural_continuity.m1_diagnostics.cuda_null_full_budget_authority import (
    DATASET_MANIFEST_SHA256,
    GATE_MANIFEST_SHA256,
    SOURCE_ONNX_SHA256,
    FullBudgetBlocked,
    verify_full_budget_authority,
)
from neural_continuity.m1_diagnostics.cuda_null_full_budget_replay import write_budget_package
from neural_continuity.m1_diagnostics.cuda_null_sentinel_readiness import EPOCH_LAYOUT
from neural_continuity.m1_diagnostics.cuda_null_source_preflight_inputs import (
    CORPUS_SHA256,
    QRELS_SHA256,
    QUERY_SHA256,
    _load_jsonl,
    _verified_file,
    load_source_preflight_inputs,
)
from neural_continuity.m1_diagnostics.cuda_null_source_preflight_runtime import (
    _reverify_teacher_source,
)
from neural_continuity.m1_diagnostics.cuda_preflight_runtime import (
    _profile_summary,
    _profiled_session,
)

PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")


def _identity_and_texts(paths: dict[str, Path]) -> tuple[list[str], list[str], Path, str, str]:
    inputs = load_source_preflight_inputs(
        paths["dataset_root"], paths["transition_a_bundle"], paths["teacher_snapshot_root"]
    )
    root = paths["dataset_root"]
    documents = _load_jsonl(
        _verified_file(root / "corpus.jsonl", CORPUS_SHA256, "corpus"), "document_id", 5183
    )
    queries = _load_jsonl(
        _verified_file(root / "roles" / "measurement_null.queries.jsonl", QUERY_SHA256, "queries"),
        "query_id",
        81,
    )
    _verified_file(root / "roles" / "measurement_null.qrels.tsv", QRELS_SHA256, "qrels")
    if [item[0] for item in queries] != inputs.identity["all_query_ids"]:
        raise FullBudgetBlocked("measurement-null query order differs")
    document_id_sha256 = hashlib.sha256(
        canonical_json_bytes([item[0] for item in documents])
    ).hexdigest()
    query_id_sha256 = hashlib.sha256(
        canonical_json_bytes([item[0] for item in queries])
    ).hexdigest()
    return (
        [item[1] for item in documents],
        [item[1] for item in queries],
        inputs.source_onnx,
        document_id_sha256,
        query_id_sha256,
    )


def _timed_profile(
    teacher: Any, source: Path, texts: list[str], label: str, batch_size: int, scratch: Path
) -> dict[str, Any]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(source), providers=list(PROVIDERS))
    if session.get_providers() != list(PROVIDERS):
        raise FullBudgetBlocked("full-input session provider order differs")
    started = time.perf_counter()
    try:
        embeddings = encode_onnx_source(teacher, session, texts, batch_size, label)
        if embeddings.shape != (len(texts), 384):
            raise FullBudgetBlocked("full-input source embedding shape differs")
        del embeddings
    finally:
        del session
    elapsed = time.perf_counter() - started
    profiled_texts = texts[:64]
    profiled_session = _profiled_session(source, scratch / label, PROVIDERS)
    try:
        sampled_embeddings = encode_onnx_source(
            teacher, profiled_session, profiled_texts, batch_size, f"{label}.profile_sample"
        )
        if sampled_embeddings.shape != (64, 384):
            raise FullBudgetBlocked("sampled provider profile embedding shape differs")
        del sampled_embeddings
    finally:
        profile_path = Path(profiled_session.end_profiling())
        del profiled_session
    profile = _profile_summary(profile_path, PROVIDERS)
    run_label, role = label.rsplit(".", 1)
    return {
        "run_label": run_label,
        "role": role,
        "batch_size": batch_size,
        "item_count": len(texts),
        "profiled_item_count": 64,
        "profile_scope": "canonical_first_64_ids_per_role",
        "full_input_profiled": False,
        "elapsed_seconds": elapsed,
        **profile,
    }


def capture_full_budget(external_spec_sha256: str) -> dict[str, Any]:
    started = time.perf_counter()
    authority = verify_full_budget_authority(external_spec_sha256)
    authority_elapsed = time.perf_counter() - started
    paths = authority.sentinel.paths
    documents, queries, source, document_id_hash, query_id_hash = _identity_and_texts(paths)
    _reverify_teacher_source(paths["transition_a_bundle"], paths["teacher_snapshot_root"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer

    teacher = SentenceTransformer(str(paths["teacher_snapshot_root"]), device="cpu")
    teacher.eval()
    if teacher.max_seq_length != 256:
        raise FullBudgetBlocked("teacher preprocessing length differs")
    scratch = paths["scratch_root"]
    if not scratch.is_dir():
        raise FullBudgetBlocked("declared D scratch root is missing")
    profiles: list[dict[str, Any]] = []
    execution_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="cuda-full-budget-", dir=scratch) as temporary:
        working = Path(temporary)
        for label, batch_size in EPOCH_LAYOUT:
            profiles.append(
                _timed_profile(
                    teacher, source, documents, f"{label}.documents", batch_size, working
                )
            )
            profiles.append(
                _timed_profile(
                    teacher,
                    source,
                    queries,
                    f"{label}.measurement_null_queries",
                    batch_size,
                    working,
                )
            )
    execution_elapsed = time.perf_counter() - execution_started
    record = {
        "kind": "m1_cuda_full_corpus_budget_observation",
        "version": "1.0.0",
        "status": "TECHNICAL_TIMING_CAPTURED_NOT_QUALIFYING",
        "budget_spec_sha256": external_spec_sha256,
        "sentinel_gate_manifest_sha256": GATE_MANIFEST_SHA256,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "source_onnx_sha256": SOURCE_ONNX_SHA256,
        "document_ids_sha256": document_id_hash,
        "query_ids_sha256": query_id_hash,
        "qrels_sha256": sha256_file(paths["dataset_root"] / "roles" / "measurement_null.qrels.tsv"),
        "runtime_identity_sha256": authority.sentinel.runtime_identity_sha256,
        "ordered_providers": list(PROVIDERS),
        "authority_elapsed_seconds": authority_elapsed,
        "execution_elapsed_seconds": execution_elapsed,
        "linear_120_epoch_projection_hours": 120 * (authority_elapsed + execution_elapsed) / 3600,
        "projection_is_guaranteed_upper_bound": False,
        "profiles": profiles,
        "observations_retained": False,
        "rankings_or_metrics_retained": False,
        "qualifying_detection_evidence": False,
        "bounded_full_input_timing_authorized": True,
        "full_corpus_qualification_started": False,
        "full_corpus_qualification_execution_authorized": False,
        "int8_executed": False,
        "holdout_accessed": False,
        "scientific_decision": "NOT_EVALUATED",
    }
    output, manifest_sha256 = write_budget_package(authority.output_directory, record)
    return {
        "status": record["status"],
        "output": str(output),
        "artifact_manifest_sha256": manifest_sha256,
        "authority_elapsed_seconds": authority_elapsed,
        "execution_elapsed_seconds": execution_elapsed,
        "linear_120_epoch_projection_hours": record["linear_120_epoch_projection_hours"],
        "full_corpus_qualification_execution_authorized": False,
    }
