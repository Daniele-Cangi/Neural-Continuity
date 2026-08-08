from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from neural_continuity.dataset_materialization import (
    DatasetMaterializationError,
    materialize_dataset,
)
from neural_continuity.evidence import canonical_json_bytes, sha256_file


def _jsonl(records):
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def _source_package(
    tmp_path: Path,
    *,
    orphan_document: bool = False,
    omit_test_qrels: bool = False,
) -> tuple[Path, Path]:
    documents = [
        {"_id": f"d{index}", "title": f"Title {index}", "text": f"Document {index}"}
        for index in range(6)
    ]
    queries = [{"_id": f"q{index}", "text": f"Query {index}", "metadata": {}} for index in range(6)]
    corpus = _jsonl(documents)
    query_payload = _jsonl(queries)
    train_qrels = (
        b"query-id\tcorpus-id\tscore\n" b"q0\td0\t1\n" b"q1\td1\t1\n" b"q2\td2\t1\n" b"q3\td3\t1\n"
    )
    test_document = "missing" if orphan_document else "d4"
    test_qrels = ("query-id\tcorpus-id\tscore\n" f"q4\t{test_document}\t1\n" "q5\td5\t1\n").encode()
    source_files = {
        "corpus.jsonl": corpus,
        "queries.jsonl": query_payload,
        "qrels/train.tsv": train_qrels,
        "qrels/test.tsv": test_qrels,
    }

    archive_path = tmp_path / "scifact.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative_path, payload in source_files.items():
            if omit_test_qrels and relative_path == "qrels/test.tsv":
                continue
            archive.writestr(f"scifact/{relative_path}", payload)

    manifest = {
        "dataset_id": "test-scifact",
        "source": {
            "dataset_name": "SciFact test double",
            "distribution": "test",
            "download_url": "https://example.invalid/scifact.zip",
            "upstream_dataset_url": "https://example.invalid/scifact",
            "archive": {"sha256": sha256_file(archive_path)},
            "files": [
                {
                    "path": path,
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                for path, payload in source_files.items()
            ],
            "expected_counts": {
                "corpus_documents": 6,
                "queries": 6,
                "train_qrel_rows": 4,
                "train_queries_with_qrels": 4,
                "test_qrel_rows": 2,
                "test_queries_with_qrels": 2,
            },
        },
        "licenses": [{"applies_to": "test", "license": "test"}],
        "materialization": {
            "encoding": "UTF-8",
            "query_text": "text",
            "document_text": "title + newline + text",
        },
        "partition": {
            "domain_separator": "test:m1:scifact:v1",
            "roles": [
                {
                    "name": "measurement_null",
                    "upstream_split": "train",
                    "start_inclusive": 0,
                    "end_exclusive": 1,
                    "query_count": 1,
                },
                {
                    "name": "quantization_calibration",
                    "upstream_split": "train",
                    "start_inclusive": 1,
                    "end_exclusive": 2,
                    "query_count": 1,
                },
                {
                    "name": "contract_development",
                    "upstream_split": "train",
                    "start_inclusive": 2,
                    "end_exclusive": 3,
                    "query_count": 1,
                },
                {
                    "name": "validation",
                    "upstream_split": "train",
                    "start_inclusive": 3,
                    "end_exclusive": 4,
                    "query_count": 1,
                },
                {
                    "name": "frozen_critical",
                    "upstream_split": "test",
                    "start_inclusive": 0,
                    "end_exclusive": 1,
                    "query_count": 1,
                },
                {
                    "name": "final_holdout",
                    "upstream_split": "test",
                    "start_inclusive": 1,
                    "end_exclusive": 2,
                    "query_count": 1,
                },
            ],
        },
    }
    manifest_path = tmp_path / "source-manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest_path, archive_path


def test_materialization_is_deterministic_and_role_complete(tmp_path: Path):
    manifest_path, archive_path = _source_package(tmp_path)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first = materialize_dataset(manifest_path, archive_path, first_output)
    second = materialize_dataset(manifest_path, archive_path, second_output)

    assert first == second
    assert first["qualification_state"] == "materialized_unqualified"
    assert first["qualifying_m1_evidence"] is False
    assert {role["name"] for role in first["roles"]} == {
        "measurement_null",
        "quantization_calibration",
        "contract_development",
        "validation",
        "frozen_critical",
        "final_holdout",
    }
    assert all(role["query_count"] == 1 for role in first["roles"])
    assert (first_output / "materialization-manifest.json").read_bytes() == (
        second_output / "materialization-manifest.json"
    ).read_bytes()
    assert len(first["artifacts"]) == 14


def test_materialization_rejects_archive_hash_mismatch(tmp_path: Path):
    manifest_path, archive_path = _source_package(tmp_path)
    archive_path.write_bytes(archive_path.read_bytes() + b"tampered")

    with pytest.raises(DatasetMaterializationError) as error:
        materialize_dataset(manifest_path, archive_path, tmp_path / "output")

    assert error.value.code == "SOURCE_ARCHIVE_HASH_MISMATCH"


def test_materialization_rejects_declared_missing_file(tmp_path: Path):
    manifest_path, archive_path = _source_package(tmp_path, omit_test_qrels=True)

    with pytest.raises(DatasetMaterializationError) as error:
        materialize_dataset(manifest_path, archive_path, tmp_path / "output")

    assert error.value.code == "SOURCE_FILE_MISSING"


def test_materialization_rejects_orphan_qrel(tmp_path: Path):
    manifest_path, archive_path = _source_package(tmp_path, orphan_document=True)

    with pytest.raises(DatasetMaterializationError) as error:
        materialize_dataset(manifest_path, archive_path, tmp_path / "output")

    assert error.value.code == "QREL_DOCUMENT_MISSING"


def test_materialization_does_not_overwrite_existing_output(tmp_path: Path):
    manifest_path, archive_path = _source_package(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(DatasetMaterializationError) as error:
        materialize_dataset(manifest_path, archive_path, output)

    assert error.value.code == "OUTPUT_ALREADY_EXISTS"


def test_materialization_uses_canonical_source_manifest_identity(tmp_path: Path):
    manifest_path, archive_path = _source_package(tmp_path)
    first = materialize_dataset(manifest_path, archive_path, tmp_path / "first")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    second = materialize_dataset(manifest_path, archive_path, tmp_path / "second")

    assert first == second
