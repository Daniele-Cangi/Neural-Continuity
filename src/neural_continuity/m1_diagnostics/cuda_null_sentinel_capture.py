"""One real, source-only CUDA sentinel epoch in one isolated process."""

from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.m1_b.onnx_source import encode_onnx_source
from neural_continuity.m1_diagnostics.cuda_null_sentinel_epoch_package import (
    replay_cuda_sentinel_epoch,
    write_cuda_sentinel_epoch,
)
from neural_continuity.m1_diagnostics.cuda_null_sentinel_execution_authority import (
    SentinelExecutionBlocked,
    verify_sentinel_execution_authority,
)
from neural_continuity.m1_diagnostics.cuda_null_sentinel_readiness import EPOCH_LAYOUT
from neural_continuity.m1_diagnostics.cuda_null_source_preflight_inputs import (
    CORPUS_SHA256,
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
from neural_continuity.m1_diagnostics.measurement_null_sentinel_authority import (
    _select_sentinel_ids,
)

PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")


def _population(paths: dict[str, Path]) -> tuple[list[str], list[str], list[str], list[str], Path]:
    inputs = load_source_preflight_inputs(
        paths["dataset_root"], paths["transition_a_bundle"], paths["teacher_snapshot_root"]
    )
    root = paths["dataset_root"]
    corpus = _verified_file(root / "corpus.jsonl", CORPUS_SHA256, "corpus")
    query_file = _verified_file(
        root / "roles" / "measurement_null.queries.jsonl", QUERY_SHA256, "queries"
    )
    documents = _load_jsonl(corpus, "document_id", 5183)
    queries = _load_jsonl(query_file, "query_id", 81)
    by_id = dict(documents)
    selected = _select_sentinel_ids([item[0] for item in documents])
    document_ids = list(selected)
    query_ids = [item[0] for item in queries]
    if query_ids != inputs.identity["all_query_ids"]:
        raise SentinelExecutionBlocked("measurement-null query identities changed")
    return (
        document_ids,
        [by_id[item] for item in document_ids],
        query_ids,
        [item[1] for item in queries],
        inputs.source_onnx,
    )


def _encode_profiled(
    teacher: Any, source: Path, texts: list[str], batch_size: int, label: str, scratch: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    session = _profiled_session(source, scratch / label, PROVIDERS)
    try:
        observation = encode_onnx_source(teacher, session, texts, batch_size, label)
    finally:
        profile_path = Path(session.end_profiling())
        del session
    profile = _profile_summary(profile_path, PROVIDERS)
    profile.update(
        {
            "run_label": label.rsplit(".", 1)[0],
            "role": label.rsplit(".", 1)[1],
            "batch_size": batch_size,
            "item_count": len(texts),
            "observation_sha256": hashlib.sha256(observation.tobytes()).hexdigest(),
        }
    )
    return observation, profile


def capture_epoch(
    *, external_authority_sha256: str, epoch_number: int, previous_checkpoint_sha256: str | None
) -> dict[str, Any]:
    """Reverify all authority in this process, then capture exactly one epoch."""
    if type(epoch_number) is not int or not 1 <= epoch_number <= 120:
        raise SentinelExecutionBlocked("epoch number outside frozen range")
    if epoch_number == 1 and previous_checkpoint_sha256 is not None:
        raise SentinelExecutionBlocked("first epoch cannot have a predecessor")
    if epoch_number > 1 and (
        not isinstance(previous_checkpoint_sha256, str) or len(previous_checkpoint_sha256) != 64
    ):
        raise SentinelExecutionBlocked("later epoch requires external predecessor hash")
    authority = verify_sentinel_execution_authority(external_authority_sha256)
    paths = authority.paths
    document_ids, document_texts, query_ids, query_texts, source = _population(paths)
    _reverify_teacher_source(paths["transition_a_bundle"], paths["teacher_snapshot_root"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer

    teacher = SentenceTransformer(str(paths["teacher_snapshot_root"]), device="cpu")
    teacher.eval()
    if teacher.max_seq_length != 256:
        raise SentinelExecutionBlocked("teacher preprocessing length differs from frozen source")
    scratch_root = paths["scratch_root"]
    if not scratch_root.is_dir():
        raise SentinelExecutionBlocked("declared scratch root is absent")
    document_runs: list[np.ndarray] = []
    query_runs: list[np.ndarray] = []
    profiles: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="cuda-sentinel-", dir=scratch_root) as temporary:
        scratch = Path(temporary)
        for run_label, batch_size in EPOCH_LAYOUT:
            documents, document_profile = _encode_profiled(
                teacher,
                source,
                document_texts,
                batch_size,
                f"{run_label}.documents",
                scratch,
            )
            queries, query_profile = _encode_profiled(
                teacher,
                source,
                query_texts,
                batch_size,
                f"{run_label}.measurement_null_queries",
                scratch,
            )
            document_runs.append(documents)
            query_runs.append(queries)
            profiles.extend((document_profile, query_profile))
    plan = {
        "kind": "m1_cuda_null_sentinel_epoch_plan",
        "version": "1.0.0",
        "phase_id": "tensor_sentinel_preflight",
        "epoch_number": epoch_number,
        "previous_checkpoint_sha256": previous_checkpoint_sha256 or external_authority_sha256,
        "config_sha256": "df1a171d93e7fa07909e3f24b552baccd592a7239c5e6a5c4e3f972c475551d3",
        "source_preflight_manifest_sha256": (
            "9340a7a52ff6502d9ee6639dc3c15d41325c3521c77629778e44f8d7043bd8d1"
        ),
        "dataset_manifest_sha256": (
            "0746d98f5e69c6a0ee48ca3f47b342de1d968a877c90df26ffe8f893437fd5de"
        ),
        "source_onnx_sha256": "5c0d999bd6b5e64e36cad1f61a83ef8e7507d55be49086745780fabb7c648511",
        "document_ids": document_ids,
        "query_ids": query_ids,
        "query_role": "measurement_null",
        "runs": [{"label": label, "batch_size": size} for label, size in EPOCH_LAYOUT],
        "qualifying_detection_evidence": False,
        "scientific_decision": "NOT_EVALUATED",
        "full_corpus_execution": False,
        "int8_execution": False,
    }
    runtime = {
        "process_instance_id": uuid.uuid4().hex,
        "runtime_identity_sha256": authority.runtime_identity_sha256,
        "session_providers": list(PROVIDERS),
        "source_only": True,
        "onnx_graph_loaded": True,
        "session_created": True,
        "model_execution_used": True,
        "int8_executed": False,
        "holdout_accessed": False,
    }
    output = paths["output_root"] / f"epoch-{epoch_number:04d}"
    directory, manifest_sha256 = write_cuda_sentinel_epoch(
        output,
        plan=plan,
        runtime=runtime,
        document_embeddings=np.stack(document_runs).astype("<f4", copy=False),
        query_embeddings=np.stack(query_runs).astype("<f4", copy=False),
        profiles=profiles,
    )
    replay = replay_cuda_sentinel_epoch(directory / "replay-bundle.json", manifest_sha256)
    if replay.get("replay_status") != "PASS":
        raise SentinelExecutionBlocked("new epoch package did not replay")
    return {
        "epoch_number": epoch_number,
        "manifest_sha256": manifest_sha256,
        "replay_status": "PASS",
        "process_instance_id": runtime["process_instance_id"],
    }
