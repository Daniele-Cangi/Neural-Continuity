from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from neural_continuity.evidence import canonical_json_bytes, sha256_file

MATERIALIZATION_FORMAT_VERSION = "1.0.0"
MATERIALIZER_VERSION = "1.0.0"
REQUIRED_ROLES = {
    "measurement_null",
    "quantization_calibration",
    "contract_development",
    "validation",
    "frozen_critical",
    "final_holdout",
}


class DatasetMaterializationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Qrel:
    query_id: str
    document_id: str
    score: str


@dataclass(frozen=True)
class SourceData:
    documents: dict[str, str]
    queries: dict[str, str]
    qrels: dict[str, list[Qrel]]


def _blocked(code: str, message: str) -> DatasetMaterializationError:
    return DatasetMaterializationError(code, message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _blocked("SOURCE_MANIFEST_INVALID", f"{field} must be an object")
    return value


def _require_sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise _blocked("SOURCE_MANIFEST_INVALID", f"{field} must be an array")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise _blocked("SOURCE_MANIFEST_INVALID", f"{field} must be a non-empty string")
    return value


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _blocked("SOURCE_MANIFEST_INVALID", f"{field} must be a non-negative integer")
    return value


def _require_sha256(value: Any, field: str) -> str:
    digest = _require_string(value, field).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise _blocked("SOURCE_MANIFEST_INVALID", f"{field} must be a SHA-256 digest")
    return digest


def _load_source_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise _blocked("SOURCE_MANIFEST_MISSING", f"source manifest not found: {path}")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _blocked("SOURCE_MANIFEST_INVALID", f"cannot decode source manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise _blocked("SOURCE_MANIFEST_INVALID", "source manifest root must be an object")
    return payload, raw


def _validate_zip_member(name: str) -> PurePosixPath:
    if "\\" in name:
        raise _blocked("SOURCE_ARCHIVE_PATH_INVALID", f"archive member uses backslash: {name}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise _blocked("SOURCE_ARCHIVE_PATH_INVALID", f"unsafe archive member: {name}")
    return path


def _read_source_files(archive_path: Path, source_manifest: Mapping[str, Any]) -> dict[str, bytes]:
    if not archive_path.is_file():
        raise _blocked("SOURCE_ARCHIVE_MISSING", f"source archive not found: {archive_path}")

    source = _require_mapping(source_manifest.get("source"), "source")
    archive = _require_mapping(source.get("archive"), "source.archive")
    expected_archive_hash = _require_sha256(archive.get("sha256"), "source.archive.sha256")
    actual_archive_hash = sha256_file(archive_path)
    if actual_archive_hash != expected_archive_hash:
        raise _blocked(
            "SOURCE_ARCHIVE_HASH_MISMATCH",
            "archive SHA-256 mismatch: "
            f"expected {expected_archive_hash}, got {actual_archive_hash}",
        )

    expected_files: dict[str, tuple[int, str]] = {}
    for index, item in enumerate(_require_sequence(source.get("files"), "source.files")):
        file_entry = _require_mapping(item, f"source.files[{index}]")
        relative_path = _require_string(file_entry.get("path"), f"source.files[{index}].path")
        pure_path = _validate_zip_member(relative_path)
        if len(pure_path.parts) != len(PurePosixPath(relative_path).parts):
            raise _blocked("SOURCE_MANIFEST_INVALID", f"non-canonical source path: {relative_path}")
        if relative_path in expected_files:
            raise _blocked("SOURCE_MANIFEST_INVALID", f"duplicate source path: {relative_path}")
        expected_files[relative_path] = (
            _require_int(file_entry.get("size_bytes"), f"source.files[{index}].size_bytes"),
            _require_sha256(file_entry.get("sha256"), f"source.files[{index}].sha256"),
        )

    if set(expected_files) != {
        "corpus.jsonl",
        "queries.jsonl",
        "qrels/train.tsv",
        "qrels/test.tsv",
    }:
        raise _blocked("SOURCE_MANIFEST_INVALID", "source file declaration is incomplete")

    try:
        with zipfile.ZipFile(archive_path) as archive_file:
            members: dict[str, zipfile.ZipInfo] = {}
            for info in archive_file.infolist():
                path = _validate_zip_member(info.filename)
                if info.is_dir():
                    continue
                normalized = path.as_posix()
                if normalized in members:
                    raise _blocked(
                        "SOURCE_ARCHIVE_DUPLICATE_MEMBER",
                        f"duplicate archive member: {normalized}",
                    )
                members[normalized] = info

            payloads: dict[str, bytes] = {}
            for expected_path, (expected_size, expected_hash) in expected_files.items():
                matches = [
                    info
                    for member_path, info in members.items()
                    if member_path == expected_path or member_path.endswith(f"/{expected_path}")
                ]
                if not matches:
                    raise _blocked(
                        "SOURCE_FILE_MISSING", f"archive is missing declared file: {expected_path}"
                    )
                if len(matches) != 1:
                    raise _blocked(
                        "SOURCE_FILE_AMBIGUOUS",
                        f"archive contains multiple matches for: {expected_path}",
                    )
                payload = archive_file.read(matches[0])
                if len(payload) != expected_size:
                    raise _blocked(
                        "SOURCE_FILE_SIZE_MISMATCH",
                        f"size mismatch for {expected_path}: "
                        f"expected {expected_size}, got {len(payload)}",
                    )
                actual_hash = _sha256_bytes(payload)
                if actual_hash != expected_hash:
                    raise _blocked(
                        "SOURCE_FILE_HASH_MISMATCH",
                        f"SHA-256 mismatch for {expected_path}: expected {expected_hash}, "
                        f"got {actual_hash}",
                    )
                payloads[expected_path] = payload
            return payloads
    except zipfile.BadZipFile as exc:
        raise _blocked("SOURCE_ARCHIVE_INVALID", f"invalid ZIP archive: {exc}") from exc
    except OSError as exc:
        raise _blocked("SOURCE_ARCHIVE_UNREADABLE", f"cannot read ZIP archive: {exc}") from exc


def _jsonl_records(payload: bytes, source_name: str) -> list[Mapping[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _blocked("SOURCE_ENCODING_INVALID", f"{source_name} is not UTF-8") from exc

    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise _blocked(
                "SOURCE_RECORD_INVALID", f"{source_name}:{line_number} is an empty record"
            )
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _blocked(
                "SOURCE_RECORD_INVALID", f"{source_name}:{line_number} is invalid JSON"
            ) from exc
        if not isinstance(record, Mapping):
            raise _blocked(
                "SOURCE_RECORD_INVALID", f"{source_name}:{line_number} must be an object"
            )
        records.append(record)
    return records


def _record_string(record: Mapping[str, Any], field: str, location: str) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise _blocked("SOURCE_RECORD_INVALID", f"{location}.{field} must be a string")
    return value


def _parse_documents(payload: bytes) -> dict[str, str]:
    documents: dict[str, str] = {}
    for line_number, record in enumerate(_jsonl_records(payload, "corpus.jsonl"), start=1):
        location = f"corpus.jsonl:{line_number}"
        document_id = _record_string(record, "_id", location)
        title = _record_string(record, "title", location)
        text = _record_string(record, "text", location)
        if not document_id:
            raise _blocked("SOURCE_RECORD_INVALID", f"{location} has an empty _id")
        if document_id in documents:
            raise _blocked("SOURCE_ID_DUPLICATE", f"duplicate document ID: {document_id}")
        documents[document_id] = f"{title}\n{text}" if title else text
    return documents


def _parse_queries(payload: bytes) -> dict[str, str]:
    queries: dict[str, str] = {}
    for line_number, record in enumerate(_jsonl_records(payload, "queries.jsonl"), start=1):
        location = f"queries.jsonl:{line_number}"
        query_id = _record_string(record, "_id", location)
        text = _record_string(record, "text", location)
        if not query_id:
            raise _blocked("SOURCE_RECORD_INVALID", f"{location} has an empty _id")
        if query_id in queries:
            raise _blocked("SOURCE_ID_DUPLICATE", f"duplicate query ID: {query_id}")
        queries[query_id] = text
    return queries


def _canonical_score(value: str, location: str) -> str:
    try:
        score = float(value)
    except ValueError as exc:
        raise _blocked("QREL_SCORE_INVALID", f"{location} score is not numeric") from exc
    if not math.isfinite(score) or score <= 0:
        raise _blocked("QREL_SCORE_INVALID", f"{location} score must be finite and positive")
    if score.is_integer():
        return str(int(score))
    return format(score, ".17g")


def _parse_qrels(payload: bytes, split: str) -> list[Qrel]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _blocked("SOURCE_ENCODING_INVALID", f"qrels/{split}.tsv is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if reader.fieldnames != ["query-id", "corpus-id", "score"]:
        raise _blocked("QREL_SCHEMA_INVALID", f"qrels/{split}.tsv has an invalid header")

    qrels: list[Qrel] = []
    identities: set[tuple[str, str]] = set()
    for line_number, row in enumerate(reader, start=2):
        location = f"qrels/{split}.tsv:{line_number}"
        query_id = row.get("query-id")
        document_id = row.get("corpus-id")
        score_raw = row.get("score")
        if not query_id or not document_id or score_raw is None:
            raise _blocked("QREL_RECORD_INVALID", f"{location} has a missing field")
        identity = (query_id, document_id)
        if identity in identities:
            raise _blocked("QREL_DUPLICATE", f"duplicate qrel at {location}: {identity}")
        identities.add(identity)
        qrels.append(Qrel(query_id, document_id, _canonical_score(score_raw, location)))
    return qrels


def _expected_count(counts: Mapping[str, Any], field: str) -> int:
    return _require_int(counts.get(field), f"source.expected_counts.{field}")


def _validate_source(
    payloads: Mapping[str, bytes], source_manifest: Mapping[str, Any]
) -> SourceData:
    documents = _parse_documents(payloads["corpus.jsonl"])
    queries = _parse_queries(payloads["queries.jsonl"])
    train_qrels = _parse_qrels(payloads["qrels/train.tsv"], "train")
    test_qrels = _parse_qrels(payloads["qrels/test.tsv"], "test")

    source = _require_mapping(source_manifest.get("source"), "source")
    counts = _require_mapping(source.get("expected_counts"), "source.expected_counts")
    observed_counts = {
        "corpus_documents": len(documents),
        "queries": len(queries),
        "train_qrel_rows": len(train_qrels),
        "train_queries_with_qrels": len({qrel.query_id for qrel in train_qrels}),
        "test_qrel_rows": len(test_qrels),
        "test_queries_with_qrels": len({qrel.query_id for qrel in test_qrels}),
    }
    for field, observed in observed_counts.items():
        expected = _expected_count(counts, field)
        if observed != expected:
            raise _blocked(
                "SOURCE_COUNT_MISMATCH",
                f"{field} mismatch: expected {expected}, got {observed}",
            )

    for split, split_qrels in (("train", train_qrels), ("test", test_qrels)):
        for qrel in split_qrels:
            if qrel.query_id not in queries:
                raise _blocked(
                    "QREL_QUERY_MISSING",
                    f"{split} qrel references missing query: {qrel.query_id}",
                )
            if qrel.document_id not in documents:
                raise _blocked(
                    "QREL_DOCUMENT_MISSING",
                    f"{split} qrel references missing document: {qrel.document_id}",
                )

    train_ids = {qrel.query_id for qrel in train_qrels}
    test_ids = {qrel.query_id for qrel in test_qrels}
    overlap = train_ids & test_ids
    if overlap:
        raise _blocked(
            "UPSTREAM_SPLIT_OVERLAP",
            f"train and test share query IDs: {sorted(overlap, key=str.encode)[:5]}",
        )
    unassigned = set(queries) - train_ids - test_ids
    if unassigned:
        raise _blocked(
            "QUERY_WITHOUT_QREL",
            f"queries without train/test qrels: {sorted(unassigned, key=str.encode)[:5]}",
        )

    return SourceData(
        documents=documents,
        queries=queries,
        qrels={"train": train_qrels, "test": test_qrels},
    )


def _partition_query_ids(
    source_data: SourceData, source_manifest: Mapping[str, Any]
) -> dict[str, list[str]]:
    partition = _require_mapping(source_manifest.get("partition"), "partition")
    domain_separator = _require_string(
        partition.get("domain_separator"), "partition.domain_separator"
    )
    upstream_ids = {
        split: {qrel.query_id for qrel in qrels} for split, qrels in source_data.qrels.items()
    }
    ordered_ids: dict[str, list[str]] = {}
    for split, query_ids in upstream_ids.items():
        ordered_ids[split] = sorted(
            query_ids,
            key=lambda query_id: (
                hashlib.sha256(f"{domain_separator}:{split}:{query_id}".encode()).digest(),
                query_id.encode("utf-8"),
            ),
        )

    role_membership: dict[str, list[str]] = {}
    for index, raw_role in enumerate(_require_sequence(partition.get("roles"), "partition.roles")):
        role = _require_mapping(raw_role, f"partition.roles[{index}]")
        name = _require_string(role.get("name"), f"partition.roles[{index}].name")
        if name in role_membership:
            raise _blocked("PARTITION_POLICY_INVALID", f"duplicate role: {name}")
        split = _require_string(
            role.get("upstream_split"), f"partition.roles[{index}].upstream_split"
        )
        if split not in ordered_ids:
            raise _blocked("PARTITION_POLICY_INVALID", f"unknown upstream split: {split}")
        start = _require_int(
            role.get("start_inclusive"), f"partition.roles[{index}].start_inclusive"
        )
        end = _require_int(role.get("end_exclusive"), f"partition.roles[{index}].end_exclusive")
        expected_count = _require_int(
            role.get("query_count"), f"partition.roles[{index}].query_count"
        )
        if start > end or end > len(ordered_ids[split]) or end - start != expected_count:
            raise _blocked("PARTITION_POLICY_INVALID", f"invalid interval for role: {name}")
        role_membership[name] = ordered_ids[split][start:end]

    if set(role_membership) != REQUIRED_ROLES:
        missing = sorted(REQUIRED_ROLES - set(role_membership))
        extra = sorted(set(role_membership) - REQUIRED_ROLES)
        raise _blocked(
            "PARTITION_POLICY_INVALID", f"role set mismatch; missing={missing}, extra={extra}"
        )

    all_members = [query_id for members in role_membership.values() for query_id in members]
    if len(all_members) != len(set(all_members)):
        raise _blocked("PARTITION_OVERLAP", "role membership is not disjoint")
    if set(all_members) != set(source_data.queries):
        raise _blocked("PARTITION_INCOMPLETE", "role membership is not exhaustive")
    return role_membership


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def _membership_sha256(role: str, query_ids: Sequence[str]) -> str:
    payload = f"{role}\n" + "".join(f"{query_id}\n" for query_id in query_ids)
    return _sha256_bytes(payload.encode("utf-8"))


def _artifact_entry(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _build_package(
    output_root: Path,
    source_manifest: Mapping[str, Any],
    source_manifest_bytes: bytes,
    archive_path: Path,
    source_data: SourceData,
    role_membership: Mapping[str, list[str]],
) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []

    corpus_path = output_root / "corpus.jsonl"
    corpus_records = [
        {"document_id": document_id, "text": source_data.documents[document_id]}
        for document_id in sorted(source_data.documents, key=str.encode)
    ]
    _write_bytes(corpus_path, _jsonl_bytes(corpus_records))
    artifacts.append(_artifact_entry(output_root, corpus_path))

    qrels_by_query = {
        split: {
            query_id: [qrel for qrel in qrels if qrel.query_id == query_id]
            for query_id in {qrel.query_id for qrel in qrels}
        }
        for split, qrels in source_data.qrels.items()
    }
    partition = _require_mapping(source_manifest.get("partition"), "partition")
    split_by_role = {
        _require_string(
            _require_mapping(role, "partition role").get("name"), "role.name"
        ): _require_string(
            _require_mapping(role, "partition role").get("upstream_split"),
            "role.upstream_split",
        )
        for role in _require_sequence(partition.get("roles"), "partition.roles")
    }

    role_entries: list[dict[str, Any]] = []
    for role in sorted(role_membership):
        query_ids = sorted(role_membership[role], key=str.encode)
        query_path = output_root / "roles" / f"{role}.queries.jsonl"
        query_records = [
            {"query_id": query_id, "text": source_data.queries[query_id]} for query_id in query_ids
        ]
        _write_bytes(query_path, _jsonl_bytes(query_records))
        artifacts.append(_artifact_entry(output_root, query_path))

        role_qrels = [
            qrel for query_id in query_ids for qrel in qrels_by_query[split_by_role[role]][query_id]
        ]
        role_qrels.sort(key=lambda qrel: (qrel.query_id.encode(), qrel.document_id.encode()))
        qrel_path = output_root / "roles" / f"{role}.qrels.tsv"
        qrel_lines = ["query-id\tcorpus-id\tscore\n"]
        qrel_lines.extend(
            f"{qrel.query_id}\t{qrel.document_id}\t{qrel.score}\n" for qrel in role_qrels
        )
        _write_bytes(qrel_path, "".join(qrel_lines).encode("utf-8"))
        artifacts.append(_artifact_entry(output_root, qrel_path))

        role_entries.append(
            {
                "name": role,
                "upstream_split": split_by_role[role],
                "query_count": len(query_ids),
                "qrel_count": len(role_qrels),
                "membership_sha256": _membership_sha256(role, query_ids),
                "queries_artifact": query_path.relative_to(output_root).as_posix(),
                "qrels_artifact": qrel_path.relative_to(output_root).as_posix(),
            }
        )

    source = _require_mapping(source_manifest.get("source"), "source")
    notice_payload = {
        "dataset_id": source_manifest.get("dataset_id"),
        "source": {
            "dataset_name": source.get("dataset_name"),
            "distribution": source.get("distribution"),
            "download_url": source.get("download_url"),
            "upstream_dataset_url": source.get("upstream_dataset_url"),
        },
        "licenses": source_manifest.get("licenses"),
    }
    notice_path = output_root / "source-notice.json"
    _write_bytes(notice_path, canonical_json_bytes(notice_payload) + b"\n")
    artifacts.append(_artifact_entry(output_root, notice_path))

    artifacts.sort(key=lambda artifact: artifact["path"])
    materialization_config = _require_mapping(
        source_manifest.get("materialization"), "materialization"
    )
    manifest = {
        "materialization_format_version": MATERIALIZATION_FORMAT_VERSION,
        "materializer": {
            "module": "neural_continuity.dataset_materialization",
            "version": MATERIALIZER_VERSION,
        },
        "dataset_id": source_manifest.get("dataset_id"),
        "qualification_state": "materialized_unqualified",
        "qualifying_m1_evidence": False,
        "source_identity": {
            "source_manifest_sha256": _sha256_bytes(source_manifest_bytes),
            "source_archive_sha256": sha256_file(archive_path),
            "materialization_policy_sha256": _sha256_bytes(
                canonical_json_bytes(dict(materialization_config))
            ),
            "partition_policy_sha256": _sha256_bytes(canonical_json_bytes(dict(partition))),
        },
        "counts": {
            "documents": len(source_data.documents),
            "queries": len(source_data.queries),
            "qrels": sum(len(qrels) for qrels in source_data.qrels.values()),
        },
        "roles": role_entries,
        "artifacts": artifacts,
        "integrity": {
            "source_hashes_verified": True,
            "source_schema_verified": True,
            "role_membership_disjoint": True,
            "role_membership_exhaustive": True,
            "artifact_hash_algorithm": "SHA-256",
            "missing_evidence_behavior": "BLOCKED",
        },
        "next_gate": "capture and replay the frozen real-teacher observation package",
    }
    manifest_path = output_root / "materialization-manifest.json"
    _write_bytes(manifest_path, canonical_json_bytes(manifest) + b"\n")
    return manifest


def materialize_dataset(
    source_manifest_path: str | Path,
    archive_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(source_manifest_path).resolve()
    source_archive = Path(archive_path).resolve()
    output_path = Path(output_directory).resolve()
    if output_path.exists():
        raise _blocked("OUTPUT_ALREADY_EXISTS", f"output already exists: {output_path}")

    source_manifest, source_manifest_bytes = _load_source_manifest(manifest_path)
    source_payloads = _read_source_files(source_archive, source_manifest)
    source_data = _validate_source(source_payloads, source_manifest)
    role_membership = _partition_query_ids(source_data, source_manifest)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.building-", dir=output_path.parent)
    )
    try:
        result = _build_package(
            temporary_path,
            source_manifest,
            source_manifest_bytes,
            source_archive,
            source_data,
            role_membership,
        )
        os.replace(temporary_path, output_path)
        return result
    except Exception:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize a frozen M1 dataset package without network access."
    )
    parser.add_argument("--manifest", required=True, help="Path to the frozen source manifest.")
    parser.add_argument("--archive", required=True, help="Path to the local source ZIP archive.")
    parser.add_argument(
        "--output", required=True, help="New output directory; existing paths fail closed."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = materialize_dataset(args.manifest, args.archive, args.output)
    except DatasetMaterializationError as exc:
        print(
            json.dumps(
                {"status": "BLOCKED", "error": {"code": exc.code, "message": exc.message}},
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": "MATERIALIZED_UNQUALIFIED",
                "dataset_id": result["dataset_id"],
                "query_count": result["counts"]["queries"],
                "role_count": len(result["roles"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
