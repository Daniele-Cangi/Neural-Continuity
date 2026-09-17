"""Frozen source-only CUDA preflight inputs, without model or runtime imports."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural_continuity.evidence import sha256_file
from neural_continuity.m1_diagnostics.cuda_null_paths import has_linked_ancestor

DATASET_MANIFEST_SHA256 = "0746d98f5e69c6a0ee48ca3f47b342de1d968a877c90df26ffe8f893437fd5de"
CORPUS_SHA256 = "58c378602a096373e00244657f12c693a5a02a333f893edb59d5349d699a524c"
QUERY_SHA256 = "95ecf07c102b7df46095aeec10feae19725c8b5cbab9f1ea432b9a4c3782fddb"
QRELS_SHA256 = "f41b57315a122b7fee557b94d198c44827676c6c59fdd08169fd46f4070cb3fc"
SOURCE_ONNX_SHA256 = "5c0d999bd6b5e64e36cad1f61a83ef8e7507d55be49086745780fabb7c648511"
TEACHER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DOCUMENT_COUNT = 64
QUERY_COUNT = 64
EMBEDDING_DIMENSION = 384
MAX_SEQUENCE_LENGTH = 256
FROZEN_RUNS = (
    ("documents", 1),
    ("documents", 16),
    ("documents", 64),
    ("measurement_null_queries", 1),
    ("measurement_null_queries", 16),
    ("measurement_null_queries", 64),
)


@dataclass(frozen=True)
class SourcePreflightInputs:
    document_texts: tuple[str, ...]
    query_texts: tuple[str, ...]
    source_onnx: Path
    snapshot_root: Path
    identity: dict[str, Any]


def _safe_path(value: Path, label: str) -> Path:
    if ".." in value.parts:
        raise ValueError(f"{label} traverses a parent")
    path = Path(os.path.abspath(value))
    try:
        linked = has_linked_ancestor(path)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable") from exc
    if linked:
        raise ValueError(f"{label} contains a link or reparse point")
    return path


def _verified_file(path: Path, expected_hash: str, label: str) -> Path:
    file = _safe_path(path, label)
    if not file.is_file() or sha256_file(file) != expected_hash:
        raise ValueError(f"{label} is missing or differs from frozen SHA-256")
    return file


def _load_jsonl(path: Path, identity_key: str, expected_count: int) -> list[tuple[str, str]]:
    rows: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                value = json.loads(line)
                if not isinstance(value, dict) or set(value) != {identity_key, "text"}:
                    raise ValueError(f"invalid {identity_key} record")
                identity, content = value[identity_key], value["text"]
                if (
                    not isinstance(identity, str)
                    or not identity
                    or not isinstance(content, str)
                    or not content
                    or identity in rows
                ):
                    raise ValueError(f"duplicate or malformed {identity_key}")
                rows[identity] = content
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable {identity_key} materialization") from exc
    if len(rows) != expected_count:
        raise ValueError(f"{identity_key} count differs from frozen materialization")
    return sorted(rows.items(), key=lambda item: item[0].encode("utf-8"))


def load_source_preflight_inputs(
    dataset_root: Path, transition_a_bundle: Path, teacher_snapshot_root: Path
) -> SourcePreflightInputs:
    """Select frozen UTF-8 identities only after the live authority has passed."""
    dataset = _safe_path(dataset_root, "dataset root")
    bundle = _safe_path(transition_a_bundle, "Transition A bundle")
    snapshot = _safe_path(teacher_snapshot_root, "teacher snapshot")
    if not snapshot.is_dir() or snapshot.name != TEACHER_REVISION:
        raise ValueError("teacher snapshot revision mismatch")
    _verified_file(
        dataset / "materialization-manifest.json",
        DATASET_MANIFEST_SHA256,
        "canonical materialization manifest",
    )
    corpus = _verified_file(dataset / "corpus.jsonl", CORPUS_SHA256, "corpus")
    queries = _verified_file(
        dataset / "roles" / "measurement_null.queries.jsonl",
        QUERY_SHA256,
        "measurement-null queries",
    )
    _verified_file(
        dataset / "roles" / "measurement_null.qrels.tsv",
        QRELS_SHA256,
        "measurement-null qrels",
    )
    source = _verified_file(bundle.parent / "teacher.onnx", SOURCE_ONNX_SHA256, "ONNX FP32 source")
    documents = _load_jsonl(corpus, "document_id", 5183)
    measurement_queries = _load_jsonl(queries, "query_id", 81)
    selected_documents = documents[:DOCUMENT_COUNT]
    selected_queries = measurement_queries[:QUERY_COUNT]
    identity: dict[str, Any] = {
        "kind": "m1_cuda_null_source_preflight_input_identity",
        "dataset_role": "measurement_null",
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "corpus_sha256": CORPUS_SHA256,
        "queries_sha256": QUERY_SHA256,
        "qrels_sha256": QRELS_SHA256,
        "source_onnx_sha256": SOURCE_ONNX_SHA256,
        "teacher_revision": TEACHER_REVISION,
        "normalization": "l2_unit_after_encode",
        "embedding_dimension": EMBEDDING_DIMENSION,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "all_document_ids": [identity for identity, _ in documents],
        "all_query_ids": [identity for identity, _ in measurement_queries],
        "selected_document_ids": [identity for identity, _ in selected_documents],
        "selected_query_ids": [identity for identity, _ in selected_queries],
    }
    return SourcePreflightInputs(
        tuple(content for _, content in selected_documents),
        tuple(content for _, content in selected_queries),
        source,
        snapshot,
        identity,
    )
