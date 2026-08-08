from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from neural_continuity.evidence import canonical_json_bytes, sha256_file

EVIDENCE_FORMAT_VERSION = "1.0.0"
REPLAY_FORMAT_VERSION = "1.0.0"
ROLE_ORDER = (
    "measurement_null",
    "quantization_calibration",
    "contract_development",
    "validation",
    "frozen_critical",
    "final_holdout",
)


class TeacherEvidenceError(RuntimeError):
    def __init__(self, code: str, message: str, status: str = "BLOCKED") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


@dataclass(frozen=True)
class RoleData:
    name: str
    query_ids: list[str]
    query_texts: list[str]
    relevant_document_ids: dict[str, list[str]]


@dataclass(frozen=True)
class MaterializedDataset:
    dataset_id: str
    manifest_sha256: str
    materialization_policy_sha256: str
    partition_policy_sha256: str
    document_ids: list[str]
    document_texts: list[str]
    roles: dict[str, RoleData]


@dataclass(frozen=True)
class TeacherObservation:
    document_ids: list[str]
    document_embeddings: np.ndarray
    query_ids: list[str]
    query_embeddings: np.ndarray
    query_roles: list[str]
    relevant_document_ids: dict[str, list[str]]


def _fail(code: str, message: str, status: str = "BLOCKED") -> TeacherEvidenceError:
    return TeacherEvidenceError(code, message, status)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail("EVIDENCE_SCHEMA_INVALID", f"{field} must be an object")
    return value


def _require_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise _fail("EVIDENCE_SCHEMA_INVALID", f"{field} must be an array")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail("EVIDENCE_SCHEMA_INVALID", f"{field} must be a non-empty string")
    return value


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _fail("EVIDENCE_SCHEMA_INVALID", f"{field} must be a non-negative integer")
    return value


def _require_sha256(value: Any, field: str) -> str:
    digest = _require_string(value, field).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise _fail("EVIDENCE_SCHEMA_INVALID", f"{field} must be a SHA-256 digest")
    return digest


def _load_json(path: Path, code: str) -> dict[str, Any]:
    if not path.is_file():
        raise _fail(code, f"missing file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _fail(code, f"invalid JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise _fail(code, f"JSON root must be an object: {path}")
    return payload


def _safe_artifact_path(root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise _fail("EVIDENCE_PATH_INVALID", f"unsafe artifact path: {relative_path}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise _fail(
            "EVIDENCE_PATH_INVALID", f"artifact path escapes package: {relative_path}"
        ) from exc
    return resolved


def _verify_artifacts(root: Path, manifest: Mapping[str, Any], field: str) -> None:
    seen_paths: set[str] = set()
    artifacts = _require_list(manifest.get(field), field)
    if not artifacts:
        raise _fail("EVIDENCE_ARTIFACTS_MISSING", f"{field} must not be empty")
    for index, raw_artifact in enumerate(artifacts):
        artifact = _require_mapping(raw_artifact, f"{field}[{index}]")
        relative_path = _require_string(artifact.get("path"), f"{field}[{index}].path")
        expected_size = _require_int(artifact.get("size_bytes"), f"{field}[{index}].size_bytes")
        expected_hash = _require_sha256(artifact.get("sha256"), f"{field}[{index}].sha256")
        if relative_path in seen_paths:
            raise _fail("EVIDENCE_ARTIFACT_DUPLICATE", f"duplicate artifact: {relative_path}")
        seen_paths.add(relative_path)
        artifact_path = _safe_artifact_path(root, relative_path)
        if not artifact_path.is_file():
            raise _fail("EVIDENCE_ARTIFACT_MISSING", f"missing artifact: {relative_path}")
        if artifact_path.stat().st_size != expected_size:
            raise _fail("EVIDENCE_ARTIFACT_SIZE_MISMATCH", f"size mismatch: {relative_path}")
        if sha256_file(artifact_path) != expected_hash:
            raise _fail("EVIDENCE_ARTIFACT_HASH_MISMATCH", f"SHA-256 mismatch: {relative_path}")


def _jsonl_records(path: Path, required_fields: tuple[str, ...]) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise _fail("MATERIALIZED_ARTIFACT_MISSING", f"missing artifact: {path}")
    records: list[Mapping[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise _fail("MATERIALIZED_ARTIFACT_INVALID", f"cannot read {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise _fail("MATERIALIZED_ARTIFACT_INVALID", f"empty record at {path}:{line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _fail(
                "MATERIALIZED_ARTIFACT_INVALID", f"invalid JSON at {path}:{line_number}"
            ) from exc
        if not isinstance(record, Mapping):
            raise _fail(
                "MATERIALIZED_ARTIFACT_INVALID", f"non-object record at {path}:{line_number}"
            )
        for field in required_fields:
            if not isinstance(record.get(field), str):
                raise _fail(
                    "MATERIALIZED_ARTIFACT_INVALID",
                    f"missing string {field} at {path}:{line_number}",
                )
        records.append(record)
    return records


def _membership_sha256(role: str, query_ids: Sequence[str]) -> str:
    payload = f"{role}\n" + "".join(f"{query_id}\n" for query_id in query_ids)
    return _sha256_bytes(payload.encode())


def _load_role_qrels(path: Path, query_ids: set[str]) -> dict[str, list[str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise _fail("MATERIALIZED_ARTIFACT_INVALID", f"cannot read {path}: {exc}") from exc
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if reader.fieldnames != ["query-id", "corpus-id", "score"]:
        raise _fail("MATERIALIZED_ARTIFACT_INVALID", f"invalid qrel header: {path}")
    qrels: dict[str, list[str]] = {query_id: [] for query_id in query_ids}
    seen_pairs: set[tuple[str, str]] = set()
    for line_number, row in enumerate(reader, start=2):
        query_id = row.get("query-id")
        document_id = row.get("corpus-id")
        score_raw = row.get("score")
        if not query_id or not document_id or score_raw is None:
            raise _fail(
                "MATERIALIZED_ARTIFACT_INVALID", f"missing qrel field at {path}:{line_number}"
            )
        try:
            score = float(score_raw)
        except ValueError as exc:
            raise _fail(
                "MATERIALIZED_ARTIFACT_INVALID", f"invalid score at {path}:{line_number}"
            ) from exc
        if not math.isfinite(score) or score <= 0:
            raise _fail("MATERIALIZED_ARTIFACT_INVALID", f"invalid score at {path}:{line_number}")
        if query_id not in qrels:
            raise _fail(
                "MATERIALIZED_ARTIFACT_INVALID", f"qrel query escapes role at {path}:{line_number}"
            )
        pair = (query_id, document_id)
        if pair in seen_pairs:
            raise _fail("MATERIALIZED_ARTIFACT_INVALID", f"duplicate qrel at {path}:{line_number}")
        seen_pairs.add(pair)
        qrels[query_id].append(document_id)
    if any(not document_ids for document_ids in qrels.values()):
        raise _fail("MATERIALIZED_ARTIFACT_INVALID", f"query without positive qrel in {path}")
    return {
        query_id: sorted(document_ids, key=str.encode) for query_id, document_ids in qrels.items()
    }


def load_materialized_dataset(directory: str | Path) -> MaterializedDataset:
    root = Path(directory).resolve()
    manifest_path = root / "materialization-manifest.json"
    manifest = _load_json(manifest_path, "MATERIALIZATION_MANIFEST_INVALID")
    if manifest.get("qualification_state") != "materialized_unqualified":
        raise _fail(
            "MATERIALIZATION_STATE_INVALID", "dataset is not in materialized_unqualified state"
        )
    if manifest.get("qualifying_m1_evidence") is not False:
        raise _fail(
            "MATERIALIZATION_STATE_INVALID", "materialized dataset cannot be qualifying evidence"
        )
    _verify_artifacts(root, manifest, "artifacts")

    source_identity = _require_mapping(manifest.get("source_identity"), "source_identity")
    dataset_id = _require_string(manifest.get("dataset_id"), "dataset_id")
    policy_hash = _require_sha256(
        source_identity.get("materialization_policy_sha256"),
        "source_identity.materialization_policy_sha256",
    )
    partition_hash = _require_sha256(
        source_identity.get("partition_policy_sha256"),
        "source_identity.partition_policy_sha256",
    )
    corpus_records = _jsonl_records(root / "corpus.jsonl", ("document_id", "text"))
    document_ids = [str(record["document_id"]) for record in corpus_records]
    document_texts = [str(record["text"]) for record in corpus_records]
    if not document_ids or len(document_ids) != len(set(document_ids)):
        raise _fail("MATERIALIZED_ARTIFACT_INVALID", "corpus document IDs are empty or duplicated")

    raw_roles = _require_list(manifest.get("roles"), "roles")
    roles: dict[str, RoleData] = {}
    all_query_ids: set[str] = set()
    for index, raw_role in enumerate(raw_roles):
        role = _require_mapping(raw_role, f"roles[{index}]")
        name = _require_string(role.get("name"), f"roles[{index}].name")
        if name not in ROLE_ORDER or name in roles:
            raise _fail("MATERIALIZATION_ROLE_INVALID", f"invalid or duplicate role: {name}")
        query_path = _safe_artifact_path(
            root, _require_string(role.get("queries_artifact"), f"roles[{index}].queries_artifact")
        )
        qrel_path = _safe_artifact_path(
            root, _require_string(role.get("qrels_artifact"), f"roles[{index}].qrels_artifact")
        )
        query_records = _jsonl_records(query_path, ("query_id", "text"))
        query_ids = [str(record["query_id"]) for record in query_records]
        query_texts = [str(record["text"]) for record in query_records]
        if not query_ids or len(query_ids) != len(set(query_ids)):
            raise _fail("MATERIALIZATION_ROLE_INVALID", f"invalid query IDs for role: {name}")
        if len(query_ids) != _require_int(role.get("query_count"), f"roles[{index}].query_count"):
            raise _fail("MATERIALIZATION_ROLE_INVALID", f"query count mismatch for role: {name}")
        if _membership_sha256(name, query_ids) != _require_sha256(
            role.get("membership_sha256"), f"roles[{index}].membership_sha256"
        ):
            raise _fail(
                "MATERIALIZATION_ROLE_INVALID", f"membership hash mismatch for role: {name}"
            )
        if all_query_ids & set(query_ids):
            raise _fail("MATERIALIZATION_ROLE_OVERLAP", f"query IDs overlap at role: {name}")
        all_query_ids.update(query_ids)
        qrels = _load_role_qrels(qrel_path, set(query_ids))
        qrel_count = sum(len(document_ids) for document_ids in qrels.values())
        if qrel_count != _require_int(role.get("qrel_count"), f"roles[{index}].qrel_count"):
            raise _fail("MATERIALIZATION_ROLE_INVALID", f"qrel count mismatch for role: {name}")
        if any(
            document_id not in set(document_ids)
            for values in qrels.values()
            for document_id in values
        ):
            raise _fail(
                "MATERIALIZATION_ROLE_INVALID",
                f"qrel document missing from corpus for role: {name}",
            )
        roles[name] = RoleData(name, query_ids, query_texts, qrels)

    if set(roles) != set(ROLE_ORDER):
        raise _fail(
            "MATERIALIZATION_ROLE_INVALID", "materialized package does not declare all six roles"
        )
    if len(all_query_ids) != _require_int(
        _require_mapping(manifest.get("counts"), "counts").get("queries"), "counts.queries"
    ):
        raise _fail("MATERIALIZATION_ROLE_INVALID", "role membership is not exhaustive")
    return MaterializedDataset(
        dataset_id=dataset_id,
        manifest_sha256=sha256_file(manifest_path),
        materialization_policy_sha256=policy_hash,
        partition_policy_sha256=partition_hash,
        document_ids=document_ids,
        document_texts=document_texts,
        roles=roles,
    )


def _load_config(path: str | Path) -> tuple[dict[str, Any], str]:
    config_path = Path(path)
    if not config_path.is_file():
        raise _fail("TEACHER_CONFIG_MISSING", f"config not found: {config_path}")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise _fail("TEACHER_CONFIG_INVALID", f"cannot load config: {exc}") from exc
    if not isinstance(config, dict):
        raise _fail("TEACHER_CONFIG_INVALID", "config root must be an object")
    _require_mapping(config.get("model"), "model")
    _require_mapping(config.get("evaluation"), "evaluation")
    _require_mapping(config.get("evidence_scope"), "evidence_scope")
    return config, _sha256_bytes(canonical_json_bytes(config))


def _load_teacher(config: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    model_config = _require_mapping(config.get("model"), "model")
    model_id = _require_string(model_config.get("model_id"), "model.model_id")
    revision = _require_string(model_config.get("revision"), "model.revision")
    device = _require_string(model_config.get("device"), "model.device")
    cache_only = model_config.get("cache_only")
    if cache_only is not True:
        raise _fail("TEACHER_CONFIG_INVALID", "model.cache_only must be true")
    if device != "cpu":
        raise _fail("TEACHER_DEVICE_UNVERIFIED", "only the verified CPU path is authorized")
    if model_config.get("output_dtype") != "float32":
        raise _fail("TEACHER_CONFIG_INVALID", "model.output_dtype must be float32")

    try:
        import sentence_transformers
        import torch
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise _fail("TEACHER_DEPENDENCY_MISSING", f"missing dependency: {exc.name}") from exc

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        snapshot_root = Path(
            snapshot_download(repo_id=model_id, revision=revision, local_files_only=True)
        ).resolve()
    except Exception as exc:
        raise _fail(
            "TEACHER_SNAPSHOT_MISSING", f"cached teacher snapshot unavailable: {exc}"
        ) from exc
    if snapshot_root.name != revision:
        raise _fail(
            "TEACHER_REVISION_MISMATCH",
            f"resolved snapshot {snapshot_root.name} does not match frozen revision {revision}",
        )

    files = [path for path in snapshot_root.rglob("*") if path.is_file()]
    if not files:
        raise _fail("TEACHER_SNAPSHOT_MISSING", "cached teacher snapshot has no files")
    inventory = [
        {
            "path": path.relative_to(snapshot_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(files, key=lambda item: item.relative_to(snapshot_root).as_posix())
    ]
    try:
        model = SentenceTransformer(str(snapshot_root), device=device)
        model.eval()
    except Exception as exc:
        raise _fail(
            "TEACHER_LOAD_FAILED", f"cannot load frozen teacher: {exc}", "EXECUTION_ERROR"
        ) from exc
    return model, {
        "model_id": model_id,
        "revision": revision,
        "snapshot_file_count": len(inventory),
        "snapshot_files": inventory,
        "device": device,
        "cache_only": True,
        "encode_normalize_embeddings": False,
        "effective_output_normalization": "l2_unit_after_encode",
        "output_dtype": "float32",
        "torch_version": torch.__version__,
        "sentence_transformers_version": sentence_transformers.__version__,
    }


def _encode(model: Any, texts: Sequence[str], batch_size: int, label: str) -> np.ndarray:
    try:
        values = model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise _fail(
            "TEACHER_ENCODE_FAILED", f"cannot encode {label}: {exc}", "EXECUTION_ERROR"
        ) from exc
    embeddings = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts) or embeddings.shape[1] == 0:
        raise _fail(
            "TEACHER_OUTPUT_INVALID", f"invalid embedding shape for {label}", "EXECUTION_ERROR"
        )
    if not np.isfinite(embeddings).all():
        raise _fail(
            "TEACHER_OUTPUT_INVALID", f"non-finite embedding values for {label}", "EXECUTION_ERROR"
        )
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise _fail(
            "TEACHER_OUTPUT_INVALID", f"non-positive embedding norm for {label}", "EXECUTION_ERROR"
        )
    return np.ascontiguousarray(embeddings / norms[:, np.newaxis], dtype=np.float32)


def _ordered_observation(
    dataset: MaterializedDataset, document_embeddings: np.ndarray, query_embeddings: np.ndarray
) -> TeacherObservation:
    query_ids: list[str] = []
    query_roles: list[str] = []
    relevant_document_ids: dict[str, list[str]] = {}
    query_texts: list[str] = []
    for role in ROLE_ORDER:
        role_data = dataset.roles[role]
        for query_id, query_text in sorted(
            zip(role_data.query_ids, role_data.query_texts, strict=True),
            key=lambda item: item[0].encode(),
        ):
            query_ids.append(query_id)
            query_roles.append(role)
            query_texts.append(query_text)
            relevant_document_ids[query_id] = role_data.relevant_document_ids[query_id]
    if query_embeddings.shape[0] != len(query_texts):
        raise _fail(
            "TEACHER_OUTPUT_INVALID", "query embedding count does not match materialized queries"
        )
    return TeacherObservation(
        document_ids=dataset.document_ids,
        document_embeddings=document_embeddings,
        query_ids=query_ids,
        query_embeddings=query_embeddings,
        query_roles=query_roles,
        relevant_document_ids=relevant_document_ids,
    )


def _validate_observation(observation: TeacherObservation) -> None:
    if not observation.document_ids or len(observation.document_ids) != len(
        set(observation.document_ids)
    ):
        raise _fail("OBSERVATION_INVALID", "document IDs are empty or duplicated")
    if not observation.query_ids or len(observation.query_ids) != len(set(observation.query_ids)):
        raise _fail("OBSERVATION_INVALID", "query IDs are empty or duplicated")
    if len(observation.query_ids) != len(observation.query_roles):
        raise _fail("OBSERVATION_INVALID", "query role count mismatch")
    if (
        observation.document_embeddings.dtype != np.float32
        or observation.query_embeddings.dtype != np.float32
    ):
        raise _fail("OBSERVATION_INVALID", "embedding dtype must be float32")
    if observation.document_embeddings.ndim != 2 or observation.query_embeddings.ndim != 2:
        raise _fail("OBSERVATION_INVALID", "embeddings must be matrices")
    if observation.document_embeddings.shape[0] != len(observation.document_ids):
        raise _fail("OBSERVATION_INVALID", "document embedding count mismatch")
    if observation.query_embeddings.shape[0] != len(observation.query_ids):
        raise _fail("OBSERVATION_INVALID", "query embedding count mismatch")
    if observation.document_embeddings.shape[1] != observation.query_embeddings.shape[1]:
        raise _fail("OBSERVATION_INVALID", "embedding dimensions do not match")
    if (
        not np.isfinite(observation.document_embeddings).all()
        or not np.isfinite(observation.query_embeddings).all()
    ):
        raise _fail("OBSERVATION_INVALID", "embeddings contain non-finite values")
    document_norms = np.linalg.norm(observation.document_embeddings, axis=1)
    query_norms = np.linalg.norm(observation.query_embeddings, axis=1)
    if not np.allclose(document_norms, 1.0, atol=1e-5) or not np.allclose(
        query_norms, 1.0, atol=1e-5
    ):
        raise _fail("OBSERVATION_INVALID", "embeddings are not unit normalized")
    if any(role not in ROLE_ORDER for role in observation.query_roles):
        raise _fail("OBSERVATION_INVALID", "observation contains an unknown role")
    if set(observation.relevant_document_ids) != set(observation.query_ids):
        raise _fail("OBSERVATION_INVALID", "qrels do not match observation queries")
    document_id_set = set(observation.document_ids)
    for query_id, relevant_ids in observation.relevant_document_ids.items():
        if not relevant_ids or len(relevant_ids) != len(set(relevant_ids)):
            raise _fail("OBSERVATION_INVALID", f"invalid qrels for query: {query_id}")
        if any(document_id not in document_id_set for document_id in relevant_ids):
            raise _fail("OBSERVATION_INVALID", f"qrel document missing for query: {query_id}")


def _rank_and_measure(
    observation: TeacherObservation, top_k: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if top_k <= 0 or top_k > len(observation.document_ids):
        raise _fail("EVALUATION_CONFIG_INVALID", "top_k must be in the document range")
    _validate_observation(observation)
    document_ids = np.asarray(observation.document_ids)
    role_rows: dict[str, list[dict[str, float]]] = {role: [] for role in ROLE_ORDER}
    rankings: list[dict[str, Any]] = []
    for index, query_id in enumerate(observation.query_ids):
        scores = observation.document_embeddings @ observation.query_embeddings[index]
        order = np.lexsort((document_ids, -scores))[:top_k]
        ranked_ids = [str(document_ids[position]) for position in order]
        relevant = set(observation.relevant_document_ids[query_id])
        hits = [rank + 1 for rank, document_id in enumerate(ranked_ids) if document_id in relevant]
        recall = len(hits) / len(relevant)
        reciprocal_rank = 1.0 / hits[0] if hits else 0.0
        dcg = sum(1.0 / math.log2(rank + 1) for rank in hits)
        ideal_count = min(len(relevant), top_k)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
        role = observation.query_roles[index]
        role_rows[role].append(
            {"recall_at_k": recall, "mrr_at_k": reciprocal_rank, "ndcg_at_k": ndcg}
        )
        rankings.append(
            {
                "query_id": query_id,
                "role": role,
                "ranked_document_ids": ranked_ids,
                "scores": [float(scores[position]) for position in order],
            }
        )

    metrics: dict[str, Any] = {"top_k": top_k, "roles": {}}
    for role in ROLE_ORDER:
        values = role_rows[role]
        if not values:
            continue
        metrics["roles"][role] = {
            "query_count": len(values),
            "metrics": {
                "recall_at_k": float(sum(item["recall_at_k"] for item in values) / len(values)),
                "mrr_at_k": float(sum(item["mrr_at_k"] for item in values) / len(values)),
                "ndcg_at_k": float(sum(item["ndcg_at_k"] for item in values) / len(values)),
            },
        }
    return rankings, metrics


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def _artifact_entry(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_teacher_evidence_package(
    output_directory: str | Path,
    dataset: MaterializedDataset,
    observation: TeacherObservation,
    teacher_manifest: Mapping[str, Any],
    evidence_scope: Mapping[str, Any],
    config_sha256: str,
    top_k: int,
) -> dict[str, Any]:
    output_path = Path(output_directory).resolve()
    if output_path.exists():
        raise _fail("OUTPUT_ALREADY_EXISTS", f"output already exists: {output_path}")
    _validate_observation(observation)
    rankings, metrics = _rank_and_measure(observation, top_k)
    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.building-", dir=output_path.parent)
    )
    try:
        observation_path = temporary_path / "observations.npz"
        np.savez_compressed(
            observation_path,
            document_ids=np.asarray(observation.document_ids),
            document_embeddings=observation.document_embeddings,
            query_ids=np.asarray(observation.query_ids),
            query_embeddings=observation.query_embeddings,
            query_roles=np.asarray(observation.query_roles),
        )
        rankings_path = temporary_path / "rankings.jsonl"
        _write_bytes(rankings_path, _jsonl_bytes(rankings))
        metrics_path = temporary_path / "teacher-metrics.json"
        _write_bytes(metrics_path, canonical_json_bytes(metrics) + b"\n")
        teacher_manifest_path = temporary_path / "teacher-manifest.json"
        _write_bytes(teacher_manifest_path, canonical_json_bytes(dict(teacher_manifest)) + b"\n")

        replay_bundle = {
            "replay_format_version": REPLAY_FORMAT_VERSION,
            "evidence_scope": dict(evidence_scope),
            "configuration_sha256": config_sha256,
            "dataset": {
                "dataset_id": dataset.dataset_id,
                "materialization_manifest_sha256": dataset.manifest_sha256,
                "materialization_policy_sha256": dataset.materialization_policy_sha256,
                "partition_policy_sha256": dataset.partition_policy_sha256,
            },
            "observation": {
                "path": "observations.npz",
                "embedding_dtype": "float32",
                "embedding_dimension": int(observation.document_embeddings.shape[1]),
                "document_count": len(observation.document_ids),
                "query_count": len(observation.query_ids),
                "output_normalization": "l2_unit_after_encode",
            },
            "evaluation": {
                "top_k": top_k,
                "ranking_path": "rankings.jsonl",
                "metrics_path": "teacher-metrics.json",
                "technical_score_tolerance": 1e-6,
            },
            "qrels": {
                query_id: observation.relevant_document_ids[query_id]
                for query_id in sorted(observation.query_ids, key=str.encode)
            },
            "replay_requires_model_execution": False,
            "transition_decision": "NOT_APPLICABLE",
        }
        bundle_path = temporary_path / "replay-bundle.json"
        _write_bytes(bundle_path, canonical_json_bytes(replay_bundle) + b"\n")

        artifacts = [
            _artifact_entry(temporary_path, path)
            for path in (
                observation_path,
                rankings_path,
                metrics_path,
                teacher_manifest_path,
                bundle_path,
            )
        ]
        artifacts.sort(key=lambda artifact: artifact["path"])
        evidence_manifest = {
            "evidence_format_version": EVIDENCE_FORMAT_VERSION,
            "evidence_status": "CAPTURED_PENDING_REPLAY",
            "qualifying_m1_evidence": bool(evidence_scope.get("qualifying_m1_evidence")),
            "dataset_id": dataset.dataset_id,
            "artifacts": artifacts,
            "integrity": {
                "artifact_hash_algorithm": "SHA-256",
                "missing_evidence_behavior": "BLOCKED",
                "replay_without_model_execution_required": True,
            },
        }
        evidence_manifest_path = temporary_path / "evidence-manifest.json"
        _write_bytes(evidence_manifest_path, canonical_json_bytes(evidence_manifest) + b"\n")
        os.replace(temporary_path, output_path)
        return {
            "output_directory": str(output_path),
            "evidence_manifest_sha256": sha256_file(output_path / "evidence-manifest.json"),
            "dataset_id": dataset.dataset_id,
            "document_count": len(observation.document_ids),
            "query_count": len(observation.query_ids),
            "evidence_status": "CAPTURED_PENDING_REPLAY",
        }
    except Exception:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise


def capture_teacher_evidence(
    config_path: str | Path, dataset_directory: str | Path, output_directory: str | Path
) -> dict[str, Any]:
    config, config_sha256 = _load_config(config_path)
    dataset = load_materialized_dataset(dataset_directory)
    expected_dataset_id = _require_string(
        _require_mapping(config.get("dataset"), "dataset").get("dataset_id"), "dataset.dataset_id"
    )
    if dataset.dataset_id != expected_dataset_id:
        raise _fail(
            "DATASET_ID_MISMATCH",
            f"expected dataset {expected_dataset_id}, got {dataset.dataset_id}",
        )
    evaluation = _require_mapping(config.get("evaluation"), "evaluation")
    top_k = _require_int(evaluation.get("top_k"), "evaluation.top_k")
    batch_size = _require_int(evaluation.get("batch_size"), "evaluation.batch_size")
    if batch_size == 0:
        raise _fail("EVALUATION_CONFIG_INVALID", "evaluation.batch_size must be positive")
    model, teacher_manifest = _load_teacher(config)
    document_embeddings = _encode(model, dataset.document_texts, batch_size, "documents")

    ordered_query_texts: list[str] = []
    for role in ROLE_ORDER:
        role_data = dataset.roles[role]
        ordered_query_texts.extend(
            text
            for _, text in sorted(
                zip(role_data.query_ids, role_data.query_texts, strict=True),
                key=lambda item: item[0].encode(),
            )
        )
    query_embeddings = _encode(model, ordered_query_texts, batch_size, "queries")
    observation = _ordered_observation(dataset, document_embeddings, query_embeddings)
    teacher_manifest = {
        **teacher_manifest,
        "configuration_sha256": config_sha256,
        "dataset_id": dataset.dataset_id,
        "materialization_manifest_sha256": dataset.manifest_sha256,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
    }
    return write_teacher_evidence_package(
        output_directory=output_directory,
        dataset=dataset,
        observation=observation,
        teacher_manifest=teacher_manifest,
        evidence_scope=_require_mapping(config.get("evidence_scope"), "evidence_scope"),
        config_sha256=config_sha256,
        top_k=top_k,
    )


def _load_observation(path: Path, qrels: Mapping[str, Any]) -> TeacherObservation:
    try:
        with np.load(path, allow_pickle=False) as archive:
            document_ids = [str(value) for value in archive["document_ids"].tolist()]
            document_embeddings = np.ascontiguousarray(
                archive["document_embeddings"], dtype=np.float32
            )
            query_ids = [str(value) for value in archive["query_ids"].tolist()]
            query_embeddings = np.ascontiguousarray(archive["query_embeddings"], dtype=np.float32)
            query_roles = [str(value) for value in archive["query_roles"].tolist()]
    except (OSError, KeyError, ValueError) as exc:
        raise _fail("REPLAY_OBSERVATION_INVALID", f"cannot load observations: {exc}") from exc
    relevant_document_ids: dict[str, list[str]] = {}
    for query_id in query_ids:
        values = qrels.get(query_id)
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise _fail("REPLAY_QRELS_INVALID", f"invalid replay qrels for query: {query_id}")
        relevant_document_ids[query_id] = list(values)
    return TeacherObservation(
        document_ids=document_ids,
        document_embeddings=document_embeddings,
        query_ids=query_ids,
        query_embeddings=query_embeddings,
        query_roles=query_roles,
        relevant_document_ids=relevant_document_ids,
    )


def replay_teacher_evidence(bundle_path: str | Path) -> dict[str, Any]:
    bundle_file = Path(bundle_path).resolve()
    root = bundle_file.parent
    bundle = _load_json(bundle_file, "REPLAY_BUNDLE_INVALID")
    if bundle.get("replay_format_version") != REPLAY_FORMAT_VERSION:
        raise _fail("REPLAY_FORMAT_INVALID", "replay bundle format is not authoritative")
    evidence_manifest = _load_json(root / "evidence-manifest.json", "EVIDENCE_MANIFEST_INVALID")
    _verify_artifacts(root, evidence_manifest, "artifacts")
    observation_info = _require_mapping(bundle.get("observation"), "observation")
    evaluation = _require_mapping(bundle.get("evaluation"), "evaluation")
    observation_path = _safe_artifact_path(
        root, _require_string(observation_info.get("path"), "observation.path")
    )
    qrels = _require_mapping(bundle.get("qrels"), "qrels")
    observation = _load_observation(observation_path, qrels)
    _validate_observation(observation)
    if observation.document_embeddings.shape[1] != _require_int(
        observation_info.get("embedding_dimension"), "observation.embedding_dimension"
    ):
        raise _fail("REPLAY_OBSERVATION_INVALID", "embedding dimension mismatch")
    if len(observation.document_ids) != _require_int(
        observation_info.get("document_count"), "observation.document_count"
    ) or len(observation.query_ids) != _require_int(
        observation_info.get("query_count"), "observation.query_count"
    ):
        raise _fail("REPLAY_OBSERVATION_INVALID", "observation count mismatch")
    top_k = _require_int(evaluation.get("top_k"), "evaluation.top_k")
    rankings, metrics = _rank_and_measure(observation, top_k)
    ranking_path = _safe_artifact_path(
        root, _require_string(evaluation.get("ranking_path"), "evaluation.ranking_path")
    )
    metrics_path = _safe_artifact_path(
        root, _require_string(evaluation.get("metrics_path"), "evaluation.metrics_path")
    )
    if ranking_path.read_bytes() != _jsonl_bytes(rankings):
        raise _fail("REPLAY_RANKING_MISMATCH", "replayed rankings do not match captured evidence")
    if metrics_path.read_bytes() != canonical_json_bytes(metrics) + b"\n":
        raise _fail("REPLAY_METRICS_MISMATCH", "replayed metrics do not match captured evidence")
    if bundle.get("replay_requires_model_execution") is not False:
        raise _fail("REPLAY_POLICY_INVALID", "replay must not execute the teacher model")
    return {
        "status": "PASS",
        "replay_verified": True,
        "model_execution_used": False,
        "dataset_id": _require_string(
            _require_mapping(bundle.get("dataset"), "dataset").get("dataset_id"),
            "dataset.dataset_id",
        ),
        "query_count": len(observation.query_ids),
        "document_count": len(observation.document_ids),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture and replay M1 real-teacher evidence without ONNX execution."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture", help="Capture a real-teacher baseline package.")
    capture.add_argument("--config", required=True)
    capture.add_argument("--dataset", required=True)
    capture.add_argument("--output", required=True)
    replay = subparsers.add_parser("replay", help="Replay a captured package without the teacher.")
    replay.add_argument("--bundle", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "capture":
            result = capture_teacher_evidence(args.config, args.dataset, args.output)
        else:
            result = replay_teacher_evidence(args.bundle)
    except TeacherEvidenceError as exc:
        print(
            json.dumps(
                {"status": exc.status, "error": {"code": exc.code, "message": exc.message}},
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
