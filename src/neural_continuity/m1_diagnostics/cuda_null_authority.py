"""Read-only, source-only authority gate for the proposed CUDA null study.

This module deliberately does not import ONNX or ONNX Runtime. Passing this
gate never authorizes graph loading, a session, or a measurement epoch.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.cuda_null_paths import (
    has_linked_ancestor,
    path_is_link_or_reparse,
    snapshot_file_inventory,
)

DATASET_MANIFEST_SHA256 = "0746d98f5e69c6a0ee48ca3f47b342de1d968a877c90df26ffe8f893437fd5de"
PARTITION_POLICY_SHA256 = "43eb7bd3a805792897de35cebd995d3d5b93931f08fca260f8a8d4aa1883457d"
MATERIALIZATION_POLICY_SHA256 = "445aa58c22faad40ee567d28c98115589cf5811acac9b30e7b7a48f383bf9037"
CORPUS_SHA256 = "58c378602a096373e00244657f12c693a5a02a333f893edb59d5349d699a524c"
MEASUREMENT_QUERIES_SHA256 = "95ecf07c102b7df46095aeec10feae19725c8b5cbab9f1ea432b9a4c3782fddb"
MEASUREMENT_QRELS_SHA256 = "f41b57315a122b7fee557b94d198c44827676c6c59fdd08169fd46f4070cb3fc"
DOCUMENT_IDS_SHA256 = "07590b0c35c31a15fda4883f8e9ebbcacf55ae74bd97d0bd95c743ee4191a2d7"
QUERY_IDS_SHA256 = "9f399d92c337bb03f6d9cc11b50b980eb282e92dcd464be6542942ce8ed22f4a"
QRELS_IDENTITY_SHA256 = "3acfb04a476865bbfb7c38b6ce18b431c61f813f5bccb7ac264e9082c04260c6"
SNAPSHOT_DECLARATION_SHA256 = "42d8d798e4f01e68d9bb10634b9c712de00f7f8495271636fd6311b2db58e506"
TRANSITION_A_MANIFEST_SHA256 = "12566ccbcc7f3f74a799abca2189a9b0906efd44a0f038ce0dc7c44b7b87fc3a"
TRANSITION_A_ONNX_SHA256 = "5c0d999bd6b5e64e36cad1f61a83ef8e7507d55be49086745780fabb7c648511"
TEACHER_MANIFEST_SHA256 = "c7a4548f10bcf8229c70d5fa7ef8676c7f9b7b03e5a1a515f2dad093f43ed724"
TRANSITION_B_CONTRACT_SHA256 = "0acd1b0218b513ebbb6f9ab480f9305c746591b49a84577d09cb2e29c881c795"
CPU_EXTENSION_MANIFEST_SHA256 = "9c23cee654d7ad4b0797aa052a9da1c3b1961d9e23788e85188046dd4e3fdd30"
HISTORICAL_CUDA_MANIFEST_SHA256 = "276ba5286ffb38ddab1aa2e9101142fdde5e8eb20db6d0a534d33c272274e073"
DATASET_ID = "nc-m1-beir-scifact-v1"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
CUDA_NULL_CONFIG_PATH = Path(__file__).resolve().parents[3] / "experiments" / "m1-cuda-null-v1.yaml"
CUDA_NULL_CONFIG_SHA256 = "df1a171d93e7fa07909e3f24b552baccd592a7239c5e6a5c4e3f972c475551d3"
ROLE_ORDER = (
    "contract_development",
    "final_holdout",
    "frozen_critical",
    "measurement_null",
    "quantization_calibration",
    "validation",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class CudaNullAuthorityBlocked(RuntimeError):
    """A declared frozen input is missing, changed, or not yet verified."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise CudaNullAuthorityBlocked(reason)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{label}: expected object")
    return value


def _digest(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256.fullmatch(value) is not None, f"{label}: invalid SHA-256"
    )
    return value


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CudaNullAuthorityBlocked(f"{label}: unreadable or invalid JSON") from exc


def _relative_file(root: Path, name: Any, label: str, *, allow_symlink: bool = False) -> Path:
    if not isinstance(name, str) or not name:
        raise CudaNullAuthorityBlocked(f"{label}: missing relative path")
    relative = PurePosixPath(name)
    _require(
        not relative.is_absolute()
        and not re.match(r"^[A-Za-z]:", name)
        and "\\" not in name
        and all(part not in (".", "..") for part in relative.parts),
        f"{label}: unsafe path",
    )
    path = root.joinpath(*relative.parts)
    _require(path.is_file(), f"{label}: declared file missing")
    if allow_symlink:
        _require(
            _snapshot_target_is_allowed(root, path),
            f"{label}: snapshot path escapes the frozen model cache",
        )
    else:
        _require(
            not any(part.is_symlink() for part in (path, *path.parents) if part != root.parent),
            f"{label}: symlink",
        )
        _require(path.resolve().is_relative_to(root.resolve()), f"{label}: path escapes package")
    return path


def _snapshot_target_is_allowed(snapshot_root: Path, declared_path: Path) -> bool:
    """Allow a snapshot file or its content-addressed blob, not arbitrary links."""
    model_root = snapshot_root.parent.parent
    expected_model_dir = "models--" + MODEL_ID.replace("/", "--")
    if (
        snapshot_root.name != MODEL_REVISION
        or snapshot_root.parent.name != "snapshots"
        or model_root.name != expected_model_dir
    ):
        return False
    if not snapshot_root.is_dir() or has_linked_ancestor(snapshot_root):
        return False
    target = declared_path.resolve()
    if target.is_relative_to(snapshot_root.resolve()):
        return True
    blobs = model_root / "blobs"
    if not blobs.is_dir() or path_is_link_or_reparse(blobs):
        return False
    blob_root = blobs.resolve()
    return blob_root.is_relative_to(model_root.resolve()) and target.parent == blob_root


def _verify_artifacts(
    root: Path, manifest: Mapping[str, Any], required: set[str]
) -> dict[str, str]:
    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise CudaNullAuthorityBlocked("artifact list missing")
    observed: dict[str, str] = {}
    for record in records:
        item = _mapping(record, "artifact record")
        name = item.get("path")
        if not isinstance(name, str):
            raise CudaNullAuthorityBlocked("artifact: missing path")
        path = _relative_file(root, name, "artifact")
        _require(name not in observed, "duplicate artifact declaration")
        expected = _digest(item.get("sha256"), f"artifact {name}")
        _require(
            type(item.get("size_bytes")) is int and item["size_bytes"] >= 0,
            f"artifact {name}: invalid size",
        )
        _require(path.stat().st_size == item["size_bytes"], f"artifact {name}: size mismatch")
        _require(sha256_file(path) == expected, f"artifact {name}: hash mismatch")
        observed[name] = expected
    _require(set(observed) == required, "artifact set mismatch")
    return observed


def _ids(path: Path, key: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                record = _mapping(json.loads(line), f"{path.name} record")
                value = record.get(key)
                if not isinstance(value, str) or not value:
                    raise CudaNullAuthorityBlocked(f"{path.name}: missing {key}")
                _require(value not in seen, f"{path.name}: duplicate {key}")
                _require(isinstance(record.get("text"), str), f"{path.name}: missing text")
                seen.add(value)
                values.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CudaNullAuthorityBlocked(f"{path.name}: invalid JSONL") from exc
    return values


def _measurement_qrels(
    path: Path, document_ids: set[str], query_ids: set[str]
) -> dict[str, list[str]]:
    qrels: dict[str, list[str]] = {}
    count = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            _require(
                reader.fieldnames == ["query-id", "corpus-id", "score"], "qrels header mismatch"
            )
            for row in reader:
                query_id, document_id = row["query-id"], row["corpus-id"]
                _require(
                    query_id in query_ids and document_id in document_ids,
                    "qrels reference unknown ID",
                )
                _require(row["score"] == "1", "qrels contain non-positive or non-canonical score")
                values = qrels.setdefault(query_id, [])
                _require(document_id not in values, "duplicate qrel")
                values.append(document_id)
                count += 1
    except (OSError, UnicodeError, csv.Error) as exc:
        raise CudaNullAuthorityBlocked("measurement qrels unreadable") from exc
    _require(
        count == 103 and set(qrels) == query_ids, "measurement qrels count or coverage mismatch"
    )
    return {key: sorted(value) for key, value in sorted(qrels.items())}


def _verify_dataset(root: Path) -> dict[str, Any]:
    manifest_path = root / "materialization-manifest.json"
    _require(
        sha256_file(manifest_path) == DATASET_MANIFEST_SHA256, "canonical dataset manifest mismatch"
    )
    manifest = _read_json(manifest_path, "dataset manifest")
    _require(manifest.get("dataset_id") == DATASET_ID, "dataset identity mismatch")
    _require(
        manifest.get("qualification_state") == "materialized_unqualified", "dataset state mismatch"
    )
    _require(
        manifest.get("qualifying_m1_evidence") is False, "dataset qualification claim mismatch"
    )
    counts = _mapping(manifest.get("counts"), "dataset counts")
    _require(
        counts == {"documents": 5183, "queries": 1109, "qrels": 1258}, "dataset counts mismatch"
    )
    source = _mapping(manifest.get("source_identity"), "dataset source identity")
    for key in (
        "source_archive_sha256",
        "source_manifest_sha256",
        "materialization_policy_sha256",
        "partition_policy_sha256",
    ):
        _digest(source.get(key), key)
    roles = manifest.get("roles")
    if not isinstance(roles, list):
        raise CudaNullAuthorityBlocked("dataset roles missing")
    _require(
        isinstance(roles, list)
        and [item.get("name") for item in roles if isinstance(item, dict)] == list(ROLE_ORDER),
        "dataset role order mismatch",
    )
    required = {"corpus.jsonl", "source-notice.json"}
    required.update(
        f"roles/{role}.{suffix}" for role in ROLE_ORDER for suffix in ("queries.jsonl", "qrels.tsv")
    )
    artifacts = _verify_artifacts(root, manifest, required)
    _require(
        source["partition_policy_sha256"] == PARTITION_POLICY_SHA256, "partition policy mismatch"
    )
    _require(
        source["materialization_policy_sha256"] == MATERIALIZATION_POLICY_SHA256,
        "materialization policy mismatch",
    )
    _require(artifacts["corpus.jsonl"] == CORPUS_SHA256, "corpus artifact mismatch")
    _require(
        artifacts["roles/measurement_null.queries.jsonl"] == MEASUREMENT_QUERIES_SHA256,
        "measurement queries mismatch",
    )
    _require(
        artifacts["roles/measurement_null.qrels.tsv"] == MEASUREMENT_QRELS_SHA256,
        "measurement qrels mismatch",
    )
    null_role = _mapping(roles[3], "measurement role")
    _require(
        null_role.get("query_count") == 81 and null_role.get("qrel_count") == 103,
        "measurement role counts mismatch",
    )
    _require(
        null_role.get("queries_artifact") == "roles/measurement_null.queries.jsonl",
        "measurement query path mismatch",
    )
    _require(
        null_role.get("qrels_artifact") == "roles/measurement_null.qrels.tsv",
        "measurement qrels path mismatch",
    )
    document_ids = _ids(root / "corpus.jsonl", "document_id")
    query_ids = _ids(root / "roles" / "measurement_null.queries.jsonl", "query_id")
    _require(len(document_ids) == 5183 and len(query_ids) == 81, "materialized ID count mismatch")
    qrels = _measurement_qrels(
        root / "roles" / "measurement_null.qrels.tsv", set(document_ids), set(query_ids)
    )
    return {
        "materialization_manifest_sha256": DATASET_MANIFEST_SHA256,
        "partition_policy_sha256": source["partition_policy_sha256"],
        "materialization_policy_sha256": source["materialization_policy_sha256"],
        "document_count": len(document_ids),
        "measurement_query_count": len(query_ids),
        "measurement_qrel_count": sum(len(values) for values in qrels.values()),
        "document_ids_sha256": hashlib.sha256(
            canonical_json_bytes(sorted(document_ids))
        ).hexdigest(),
        "query_ids_sha256": hashlib.sha256(canonical_json_bytes(sorted(query_ids))).hexdigest(),
        "qrels_sha256": hashlib.sha256(canonical_json_bytes(qrels)).hexdigest(),
        "corpus_sha256": artifacts["corpus.jsonl"],
        "measurement_queries_sha256": artifacts["roles/measurement_null.queries.jsonl"],
        "measurement_qrels_sha256": artifacts["roles/measurement_null.qrels.tsv"],
        "role_order": list(ROLE_ORDER),
    }


def _verify_source(bundle_path: Path, snapshot_root: Path) -> dict[str, Any]:
    _require(bundle_path.name == "replay-bundle.json", "Transition A bundle name mismatch")
    root = bundle_path.parent
    manifest_path = root / "evidence-manifest.json"
    _require(
        sha256_file(manifest_path) == TRANSITION_A_MANIFEST_SHA256, "Transition A manifest mismatch"
    )
    manifest = _read_json(manifest_path, "Transition A manifest")
    _require(
        manifest.get("transition_id") == "A" and manifest.get("qualifying_m1_evidence") is True,
        "Transition A evidence mismatch",
    )
    required = {
        "comparison-report.json",
        "decision.json",
        "onnx-manifest.json",
        "replay-bundle.json",
        "source-rankings.jsonl",
        "target-rankings.jsonl",
        "teacher-manifest.json",
        "teacher.onnx",
        "transition-observations.npz",
    }
    artifacts = _verify_artifacts(root, manifest, required)
    _require(
        artifacts["teacher.onnx"] == TRANSITION_A_ONNX_SHA256, "Transition A ONNX identity mismatch"
    )
    _require(
        artifacts["teacher-manifest.json"] == TEACHER_MANIFEST_SHA256,
        "teacher manifest identity mismatch",
    )
    bundle = _read_json(bundle_path, "Transition A replay bundle")
    dataset = _mapping(bundle.get("dataset"), "Transition A dataset")
    _require(
        bundle.get("transition_id") == "A" and dataset.get("dataset_id") == DATASET_ID,
        "Transition A bundle identity mismatch",
    )
    _require(
        dataset.get("materialization_manifest_sha256") == DATASET_MANIFEST_SHA256,
        "Transition A dataset mismatch",
    )
    observation = _mapping(bundle.get("observation"), "Transition A observation")
    _require(
        observation.get("embedding_dimension") == 384
        and observation.get("embedding_dtype") == "float32",
        "Transition A embedding identity mismatch",
    )
    decision = _read_json(root / "decision.json", "Transition A decision")
    _require(
        decision.get("transition_a_status") == "PASS"
        and decision.get("technical_state") == "VALID",
        "Transition A not PASS",
    )
    onnx_manifest = _read_json(root / "onnx-manifest.json", "Transition A ONNX manifest")
    _require(
        onnx_manifest.get("artifact_path") == "teacher.onnx"
        and onnx_manifest.get("artifact_sha256") == TRANSITION_A_ONNX_SHA256,
        "ONNX manifest mismatch",
    )
    teacher = _read_json(root / "teacher-manifest.json", "teacher manifest")
    _require(
        teacher.get("model_id") == MODEL_ID and teacher.get("revision") == MODEL_REVISION,
        "teacher identity mismatch",
    )
    _require(
        teacher.get("materialization_manifest_sha256") == DATASET_MANIFEST_SHA256,
        "teacher dataset mismatch",
    )
    _require(
        teacher.get("effective_output_normalization") == "l2_unit_after_encode",
        "normalization mismatch",
    )
    _require(
        teacher.get("encode_normalize_embeddings") is False
        and teacher.get("output_dtype") == "float32",
        "teacher preprocessing mismatch",
    )
    _require(teacher.get("cache_only") is True, "teacher cache policy mismatch")
    snapshot_files = teacher.get("snapshot_files")
    if not isinstance(snapshot_files, list):
        raise CudaNullAuthorityBlocked("teacher snapshot declaration missing")
    _require(
        isinstance(snapshot_files, list)
        and len(snapshot_files) == teacher.get("snapshot_file_count") == 11,
        "teacher snapshot declaration mismatch",
    )
    seen: set[str] = set()
    for item in snapshot_files:
        record = _mapping(item, "snapshot file")
        name = record.get("path")
        if not isinstance(name, str):
            raise CudaNullAuthorityBlocked("snapshot file path missing")
        path = _relative_file(snapshot_root, name, "snapshot file", allow_symlink=True)
        _require(name not in seen, "duplicate snapshot file")
        seen.add(name)
        _require(path.stat().st_size == record.get("size_bytes"), f"snapshot size mismatch: {name}")
        _require(
            sha256_file(path) == _digest(record.get("sha256"), f"snapshot {name}"),
            f"snapshot hash mismatch: {name}",
        )
    _require(snapshot_file_inventory(snapshot_root) == seen, "snapshot file inventory mismatch")
    return {
        "transition_a_manifest_sha256": TRANSITION_A_MANIFEST_SHA256,
        "onnx_sha256": TRANSITION_A_ONNX_SHA256,
        "teacher_manifest_sha256": TEACHER_MANIFEST_SHA256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "embedding_dimension": 384,
        "normalization": "l2_unit_after_encode",
        "snapshot_files_sha256": hashlib.sha256(canonical_json_bytes(snapshot_files)).hexdigest(),
    }


def _verify_config(config_path: Path, external_sha256: str) -> str:
    _require(
        config_path.resolve() == CUDA_NULL_CONFIG_PATH.resolve(),
        "CUDA null config path mismatch",
    )
    _require(
        _digest(external_sha256, "external config SHA-256") == CUDA_NULL_CONFIG_SHA256,
        "CUDA null external config hash is not frozen",
    )
    _require(
        sha256_file(config_path) == CUDA_NULL_CONFIG_SHA256,
        "CUDA null config hash mismatch",
    )
    try:
        config = _mapping(
            yaml.safe_load(config_path.read_text(encoding="utf-8")), "CUDA null config"
        )
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CudaNullAuthorityBlocked("CUDA null config unreadable") from exc
    _require(
        config.get("status") == "DRAFT_NOT_EXECUTABLE", "CUDA null config unexpectedly executable"
    )
    freeze = _mapping(config.get("freeze_gate"), "freeze gate")
    for key in (
        "independent_review_complete",
        "fail_closed_replay_verified",
        "fresh_source_only_preflight_verified",
    ):
        _require(freeze.get(key) is False, f"freeze gate {key} must remain false")
    _require(
        freeze.get("external_config_sha256") is None, "draft config contains self-asserted SHA-256"
    )
    return external_sha256


def build_static_authority(
    *,
    config_path: Path,
    external_config_sha256: str,
    dataset_root: Path,
    transition_a_bundle: Path,
    teacher_snapshot_root: Path,
    cpu_extension_bundle: Path,
    historical_cuda_bundle: Path,
) -> dict[str, Any]:
    """Verify frozen files only; return an explicitly non-executable record."""
    try:
        config_sha256 = _verify_config(config_path, external_config_sha256)
        contract = Path(__file__).resolve().parents[3] / "contracts" / "m1-transition-b-v1.json"
        _require(
            sha256_file(contract) == TRANSITION_B_CONTRACT_SHA256, "Transition B contract mismatch"
        )
        contract_data = _read_json(contract, "Transition B contract")
        _require(
            contract_data.get("contract_id") == "m1-transition-b-v1",
            "Transition B contract identity mismatch",
        )
        dataset = _verify_dataset(dataset_root)
        source = _verify_source(transition_a_bundle, teacher_snapshot_root)
        _require(
            dataset["materialization_manifest_sha256"] == DATASET_MANIFEST_SHA256,
            "source/dataset mismatch",
        )
        _require(
            cpu_extension_bundle.name == "replay-bundle.json", "CPU extension bundle name mismatch"
        )
        _require(
            historical_cuda_bundle.name == "replay-bundle.json",
            "historical CUDA bundle name mismatch",
        )
        from neural_continuity.m1_diagnostics.cuda_preflight_evidence import replay_cuda_preflight
        from neural_continuity.m1_diagnostics.measurement_null_extension_evidence import (
            replay_measurement_null_extension_plan,
        )

        cpu_replay = replay_measurement_null_extension_plan(
            cpu_extension_bundle, CPU_EXTENSION_MANIFEST_SHA256
        )
        _require(
            cpu_replay.get("replay_verified") is True
            and cpu_replay.get("execution_started") is False,
            "historical CPU plan replay failed",
        )
        cuda_replay = replay_cuda_preflight(historical_cuda_bundle, HISTORICAL_CUDA_MANIFEST_SHA256)
        _require(
            cuda_replay.get("replay_status") == "PASS", "historical CUDA preflight replay failed"
        )
        return {
            "kind": "m1_cuda_null_static_authority",
            "version": "1.0.0",
            "status": "STATIC_VERIFIED_EXECUTION_BLOCKED",
            "config_sha256": config_sha256,
            "contract_sha256": TRANSITION_B_CONTRACT_SHA256,
            "dataset": dataset,
            "source": source,
            "historical_cpu_extension_manifest_sha256": CPU_EXTENSION_MANIFEST_SHA256,
            "historical_cuda_preflight_manifest_sha256": HISTORICAL_CUDA_MANIFEST_SHA256,
            "historical_cuda_preflight_is_qualifying": False,
            "runtime_verified": False,
            "onnx_graph_loaded": False,
            "session_created": False,
            "activation_read": False,
            "execution_authorized": False,
        }
    except CudaNullAuthorityBlocked:
        raise
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        raise CudaNullAuthorityBlocked(f"static authority could not be verified: {exc}") from exc
