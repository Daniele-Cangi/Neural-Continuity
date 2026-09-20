"""Capture exactly one authorized source-only CUDA full-corpus epoch."""

from __future__ import annotations

import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes
from neural_continuity.m1_diagnostics.cuda_null_authority import _measurement_qrels
from neural_continuity.m1_diagnostics.cuda_null_full_comparison_replay import (
    recompute_full_comparison,
)
from neural_continuity.m1_diagnostics.cuda_null_full_epoch_format import (
    DATASET_MANIFEST_SHA256,
    PROVIDERS,
    RUN_LAYOUT,
    SOURCE_ONNX_SHA256,
)
from neural_continuity.m1_diagnostics.cuda_null_full_epoch_package import (
    COMPARISON_PAIRS,
    finalize_full_epoch_package,
    replay_full_epoch_package,
)
from neural_continuity.m1_diagnostics.cuda_null_full_execution_authority import (
    FullCorpusExecutionBlocked,
    verify_full_corpus_execution_authority,
)
from neural_continuity.m1_diagnostics.cuda_null_full_retrieval_replay import (
    recompute_full_retrieval,
)
from neural_continuity.m1_diagnostics.cuda_null_full_segment_capture import (
    _capture_profiled_role,
)
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

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _population(
    paths: dict[str, Path],
) -> tuple[list[str], list[str], list[str], list[str], dict[str, list[str]], Path]:
    inputs = load_source_preflight_inputs(
        paths["dataset_root"], paths["transition_a_bundle"], paths["teacher_snapshot_root"]
    )
    root = paths["dataset_root"]
    documents = _load_jsonl(
        _verified_file(root / "corpus.jsonl", CORPUS_SHA256, "corpus"),
        "document_id",
        5183,
    )
    queries = _load_jsonl(
        _verified_file(
            root / "roles" / "measurement_null.queries.jsonl",
            QUERY_SHA256,
            "queries",
        ),
        "query_id",
        81,
    )
    qrels_path = _verified_file(
        root / "roles" / "measurement_null.qrels.tsv", QRELS_SHA256, "qrels"
    )
    document_ids = [item[0] for item in documents]
    query_ids = [item[0] for item in queries]
    if query_ids != inputs.identity["all_query_ids"]:
        raise FullCorpusExecutionBlocked("measurement-null query identities changed")
    qrels = _measurement_qrels(qrels_path, set(document_ids), set(query_ids))
    return (
        document_ids,
        [item[1] for item in documents],
        query_ids,
        [item[1] for item in queries],
        qrels,
        inputs.source_onnx,
    )


def _validate_attempt(
    epoch_number: int,
    attempt_number: int,
    process_instance_id: str,
    previous_completed_epoch_manifest_sha256: str | None,
) -> None:
    if type(epoch_number) is not int or not 1 <= epoch_number <= 120:
        raise FullCorpusExecutionBlocked("epoch number outside frozen range")
    if type(attempt_number) is not int or attempt_number < 1:
        raise FullCorpusExecutionBlocked("attempt number outside frozen range")
    try:
        process = uuid.UUID(process_instance_id)
    except (ValueError, AttributeError) as exc:
        raise FullCorpusExecutionBlocked("process identity malformed") from exc
    if str(process) != process_instance_id:
        raise FullCorpusExecutionBlocked("process identity is not canonical")
    if epoch_number == 1 and previous_completed_epoch_manifest_sha256 is not None:
        raise FullCorpusExecutionBlocked("first epoch cannot have a predecessor")
    if epoch_number > 1 and (
        not isinstance(previous_completed_epoch_manifest_sha256, str)
        or _SHA256.fullmatch(previous_completed_epoch_manifest_sha256) is None
    ):
        raise FullCorpusExecutionBlocked("later epoch requires completed predecessor")


def capture_full_epoch(
    *,
    external_authority_sha256: str,
    epoch_number: int,
    attempt_number: int,
    process_instance_id: str,
    previous_completed_epoch_manifest_sha256: str | None,
) -> dict[str, Any]:
    """Reverify authority, load the model, capture, seal, and replay one epoch."""
    _validate_attempt(
        epoch_number,
        attempt_number,
        process_instance_id,
        previous_completed_epoch_manifest_sha256,
    )
    authority = verify_full_corpus_execution_authority(external_authority_sha256, resume=True)
    paths = authority.sentinel.paths
    document_ids, document_texts, query_ids, query_texts, qrels, source = _population(paths)
    _reverify_teacher_source(paths["transition_a_bundle"], paths["teacher_snapshot_root"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer

    teacher = SentenceTransformer(str(paths["teacher_snapshot_root"]), device="cpu")
    teacher.eval()
    if teacher.max_seq_length != 256:
        raise FullCorpusExecutionBlocked("teacher preprocessing length differs")

    root = authority.paths["output_root"]
    scratch = authority.paths["scratch_root"]
    output = root / f"epoch-{epoch_number:04d}"
    staging = root / f".epoch-{epoch_number:04d}-attempt-{attempt_number:04d}"
    if output.exists() or staging.exists():
        raise FullCorpusExecutionBlocked("epoch output or attempt staging already exists")
    staging.mkdir()

    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix=f"epoch-{epoch_number:04d}-attempt-{attempt_number:04d}-",
        dir=scratch,
    ) as attempt_scratch:
        for run_label, batch_size in RUN_LAYOUT:
            records.append(
                _capture_profiled_role(
                    teacher,
                    source,
                    ids=document_ids,
                    texts=document_texts,
                    run_label=run_label,
                    role="documents",
                    batch_size=batch_size,
                    staging=staging,
                    scratch=Path(attempt_scratch),
                )
            )
            records.append(
                _capture_profiled_role(
                    teacher,
                    source,
                    ids=query_ids,
                    texts=query_texts,
                    run_label=run_label,
                    role="measurement_null_queries",
                    batch_size=batch_size,
                    staging=staging,
                    scratch=Path(attempt_scratch),
                )
            )

    by_key = {(record["run_label"], record["role"]): record for record in records}
    rankings: dict[str, list[dict[str, Any]]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for run_label, _batch_size in RUN_LAYOUT:
        rankings[run_label], metrics[run_label] = recompute_full_retrieval(
            staging,
            run_label=run_label,
            document_ids=document_ids,
            query_ids=query_ids,
            qrels=qrels,
            document_record=by_key[(run_label, "documents")],
            query_record=by_key[(run_label, "measurement_null_queries")],
        )

    comparisons: list[dict[str, Any]] = []
    for family, left, right in COMPARISON_PAIRS:
        comparison = recompute_full_comparison(
            staging,
            staging,
            left_run_label=left,
            right_run_label=right,
            document_ids=document_ids,
            query_ids=query_ids,
            qrels=qrels,
            left_document_record=by_key[(left, "documents")],
            left_query_record=by_key[(left, "measurement_null_queries")],
            right_document_record=by_key[(right, "documents")],
            right_query_record=by_key[(right, "measurement_null_queries")],
            left_rankings=rankings[left],
            left_metrics=metrics[left],
            right_rankings=rankings[right],
            right_metrics=metrics[right],
        )
        comparisons.append({"family": family, "comparison": comparison})

    plan = {
        "kind": "m1_cuda_null_full_corpus_epoch_plan",
        "version": "1.0.0",
        "phase_id": "full_corpus_qualification",
        "epoch_number": epoch_number,
        "attempt_number": attempt_number,
        "execution_authority_sha256": external_authority_sha256,
        "previous_completed_epoch_manifest_sha256": previous_completed_epoch_manifest_sha256,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "source_onnx_sha256": SOURCE_ONNX_SHA256,
        "document_ids": document_ids,
        "query_ids": query_ids,
        "query_role": "measurement_null",
        "qrels": qrels,
        "runs": [{"label": label, "batch_size": size} for label, size in RUN_LAYOUT],
        "qualifying_detection_evidence": True,
        "capture_status": "CAPTURED_NOT_DECIDED",
        "scientific_decision": "NOT_EVALUATED",
        "source_only": True,
        "candidate_or_int8_execution": False,
        "holdout_access": False,
        "operational_tolerance_change": False,
    }
    runtime = {
        "process_instance_id": process_instance_id,
        "runtime_identity_sha256": authority.sentinel.runtime_identity_sha256,
        "session_providers": list(PROVIDERS),
        "source_only": True,
        "onnx_graph_loaded": True,
        "session_created": True,
        "model_execution_used": True,
        "full_corpus_execution": True,
        "candidate_or_int8_executed": False,
        "holdout_accessed": False,
    }
    for name, value in (
        ("epoch-plan.json", plan),
        ("runtime-inventory.json", runtime),
        ("run-records.json", records),
        ("rankings.json", rankings),
        ("metrics.json", metrics),
        ("within-epoch-comparisons.json", comparisons),
    ):
        _write_json(staging / name, value)
    directory, manifest_sha256 = finalize_full_epoch_package(
        staging,
        output,
        external_authority_sha256=external_authority_sha256,
    )
    replay = replay_full_epoch_package(
        directory / "replay-bundle.json",
        manifest_sha256,
        external_authority_sha256,
    )
    if replay.get("replay_status") != "PASS":
        raise FullCorpusExecutionBlocked("new epoch package replay blocked")
    return {
        "epoch_number": epoch_number,
        "attempt_number": attempt_number,
        "process_instance_id": process_instance_id,
        "manifest_sha256": manifest_sha256,
        "replay_status": "PASS",
        "capture_status": "CAPTURED_NOT_DECIDED",
        "scientific_decision": "NOT_EVALUATED",
    }
