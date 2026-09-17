"""Tamper-evident package and model-free replay for the CUDA null static gate."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.cuda_null_authority import (
    CORPUS_SHA256,
    CPU_EXTENSION_MANIFEST_SHA256,
    DATASET_MANIFEST_SHA256,
    HISTORICAL_CUDA_MANIFEST_SHA256,
    MATERIALIZATION_POLICY_SHA256,
    MEASUREMENT_QRELS_SHA256,
    MEASUREMENT_QUERIES_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    PARTITION_POLICY_SHA256,
    ROLE_ORDER,
    TEACHER_MANIFEST_SHA256,
    TRANSITION_A_MANIFEST_SHA256,
    TRANSITION_A_ONNX_SHA256,
    TRANSITION_B_CONTRACT_SHA256,
    CudaNullAuthorityBlocked,
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ARTIFACTS = {"static-authority.json", "decision.json", "replay-bundle.json"}
_AUTHORITY_FIELDS = {
    "kind",
    "version",
    "status",
    "config_sha256",
    "contract_sha256",
    "dataset",
    "source",
    "historical_cpu_extension_manifest_sha256",
    "historical_cuda_preflight_manifest_sha256",
    "historical_cuda_preflight_is_qualifying",
    "runtime_verified",
    "onnx_graph_loaded",
    "session_created",
    "activation_read",
    "execution_authorized",
}
_DATASET_FIELDS = {
    "materialization_manifest_sha256",
    "partition_policy_sha256",
    "materialization_policy_sha256",
    "document_count",
    "measurement_query_count",
    "measurement_qrel_count",
    "document_ids_sha256",
    "query_ids_sha256",
    "qrels_sha256",
    "corpus_sha256",
    "measurement_queries_sha256",
    "measurement_qrels_sha256",
    "role_order",
}
_SOURCE_FIELDS = {
    "transition_a_manifest_sha256",
    "onnx_sha256",
    "teacher_manifest_sha256",
    "model_id",
    "model_revision",
    "embedding_dimension",
    "normalization",
    "snapshot_files_sha256",
}


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise CudaNullAuthorityBlocked(reason)


def _object(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{label}: expected object")
    return value


def _sha(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256.fullmatch(value) is not None, f"{label}: invalid SHA-256"
    )
    return value


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), path.name)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CudaNullAuthorityBlocked(f"{path.name}: unreadable") from exc


def static_decision(authority: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the limited static conclusion, never a scientific PASS."""
    _require(set(authority) == _AUTHORITY_FIELDS, "authority field set mismatch")
    _require(authority.get("kind") == "m1_cuda_null_static_authority", "authority kind mismatch")
    _require(authority.get("version") == "1.0.0", "authority version mismatch")
    _require(
        authority.get("status") == "STATIC_VERIFIED_EXECUTION_BLOCKED", "authority status mismatch"
    )
    _sha(authority.get("config_sha256"), "config")
    _require(authority.get("contract_sha256") == TRANSITION_B_CONTRACT_SHA256, "contract mismatch")
    dataset = _object(authority.get("dataset"), "dataset")
    source = _object(authority.get("source"), "source")
    _require(set(dataset) == _DATASET_FIELDS, "dataset field set mismatch")
    _require(set(source) == _SOURCE_FIELDS, "source field set mismatch")
    _require(
        dataset.get("materialization_manifest_sha256") == DATASET_MANIFEST_SHA256,
        "dataset mismatch",
    )
    _require(
        dataset.get("partition_policy_sha256") == PARTITION_POLICY_SHA256,
        "partition policy mismatch",
    )
    _require(
        dataset.get("materialization_policy_sha256") == MATERIALIZATION_POLICY_SHA256,
        "materialization policy mismatch",
    )
    _require(
        dataset.get("document_count") == 5183 and dataset.get("measurement_query_count") == 81,
        "dataset counts mismatch",
    )
    _require(dataset.get("measurement_qrel_count") == 103, "qrels count mismatch")
    _require(dataset.get("role_order") == list(ROLE_ORDER), "role order mismatch")
    for key in (
        "document_ids_sha256",
        "query_ids_sha256",
        "qrels_sha256",
        "corpus_sha256",
        "measurement_queries_sha256",
        "measurement_qrels_sha256",
    ):
        _sha(dataset.get(key), f"dataset {key}")
    _require(dataset["corpus_sha256"] == CORPUS_SHA256, "corpus artifact mismatch")
    _require(
        dataset["measurement_queries_sha256"] == MEASUREMENT_QUERIES_SHA256,
        "measurement queries mismatch",
    )
    _require(
        dataset["measurement_qrels_sha256"] == MEASUREMENT_QRELS_SHA256,
        "measurement qrels mismatch",
    )
    _require(
        source.get("transition_a_manifest_sha256") == TRANSITION_A_MANIFEST_SHA256,
        "source manifest mismatch",
    )
    _require(source.get("onnx_sha256") == TRANSITION_A_ONNX_SHA256, "ONNX identity mismatch")
    _require(
        source.get("teacher_manifest_sha256") == TEACHER_MANIFEST_SHA256,
        "teacher manifest mismatch",
    )
    _require(
        source.get("model_id") == MODEL_ID and source.get("model_revision") == MODEL_REVISION,
        "teacher model identity mismatch",
    )
    _require(
        source.get("embedding_dimension") == 384
        and source.get("normalization") == "l2_unit_after_encode",
        "source semantics mismatch",
    )
    _sha(source.get("snapshot_files_sha256"), "teacher snapshot")
    _require(
        authority.get("historical_cpu_extension_manifest_sha256") == CPU_EXTENSION_MANIFEST_SHA256,
        "CPU extension provenance mismatch",
    )
    _require(
        authority.get("historical_cuda_preflight_manifest_sha256")
        == HISTORICAL_CUDA_MANIFEST_SHA256,
        "preflight provenance mismatch",
    )
    _require(
        authority.get("historical_cuda_preflight_is_qualifying") is False,
        "historical preflight cannot qualify",
    )
    for key in (
        "runtime_verified",
        "onnx_graph_loaded",
        "session_created",
        "activation_read",
        "execution_authorized",
    ):
        _require(authority.get(key) is False, f"{key} cannot be asserted by static evidence")
    return {
        "kind": "m1_cuda_null_static_decision",
        "version": "1.0.0",
        "status": "STATIC_VERIFIED_EXECUTION_BLOCKED",
        "scientific_outcome": "NOT_EVALUATED",
        "execution_authorized": False,
        "source_only": True,
        "model_free_replay_required": True,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def write_static_package(authority: Mapping[str, Any], output_directory: Path) -> dict[str, str]:
    decision = static_decision(authority)
    _require(
        not output_directory.exists() and not output_directory.is_symlink(),
        "output directory already exists",
    )
    parent = output_directory.parent.resolve()
    _require(parent.is_dir(), "output parent missing")
    with tempfile.TemporaryDirectory(prefix=".cuda-null-static-", dir=parent) as temporary:
        staging = Path(temporary)
        _write_json(staging / "static-authority.json", authority)
        _write_json(staging / "decision.json", decision)
        bundle = {
            "kind": "m1_cuda_null_static_replay",
            "version": "1.0.0",
            "requires_model_execution": False,
            "authority_sha256": sha256_file(staging / "static-authority.json"),
            "decision_sha256": sha256_file(staging / "decision.json"),
        }
        _write_json(staging / "replay-bundle.json", bundle)
        records = [
            {
                "path": name,
                "sha256": sha256_file(staging / name),
                "size_bytes": (staging / name).stat().st_size,
            }
            for name in sorted(_ARTIFACTS)
        ]
        _write_json(
            staging / "artifact-manifest.json",
            {
                "kind": "m1_cuda_null_static_manifest",
                "version": "1.0.0",
                "artifacts": records,
            },
        )
        manifest_sha256 = sha256_file(staging / "artifact-manifest.json")
        staging.rename(output_directory)
    return {
        "status": "STATIC_VERIFIED_EXECUTION_BLOCKED",
        "manifest_sha256": manifest_sha256,
        "run_directory": str(output_directory.resolve()),
    }


def replay_static_package(bundle_path: Path, external_manifest_sha256: str) -> dict[str, Any]:
    """Replay solely from hashed JSON files; no source model or runtime import."""
    try:
        _require(
            bundle_path.name == "replay-bundle.json",
            "replay bundle path mismatch",
        )
        _require(
            not any(
                part.is_symlink() or getattr(part, "is_junction", lambda: False)()
                for part in (bundle_path, *bundle_path.parents)
            ),
            "replay package path contains a symlink or junction",
        )
        root = bundle_path.parent.resolve()
        expected_manifest_sha256 = _sha(external_manifest_sha256, "external manifest")
        manifest_path = root / "artifact-manifest.json"
        _require(not manifest_path.is_symlink(), "manifest symlink")
        _require(
            sha256_file(manifest_path) == expected_manifest_sha256,
            "artifact manifest hash mismatch",
        )
        manifest = _read_json(manifest_path)
        _require(
            manifest.get("kind") == "m1_cuda_null_static_manifest"
            and manifest.get("version") == "1.0.0",
            "manifest schema mismatch",
        )
        records = manifest.get("artifacts")
        if not isinstance(records, list):
            raise CudaNullAuthorityBlocked("manifest artifacts missing")
        _require(
            isinstance(records, list) and len(records) == len(_ARTIFACTS),
            "manifest artifact count mismatch",
        )
        names: set[str] = set()
        for record in records:
            item = _object(record, "artifact record")
            name = item.get("path")
            if not isinstance(name, str):
                raise CudaNullAuthorityBlocked("artifact path missing")
            _require(
                isinstance(name, str) and name in _ARTIFACTS and name not in names,
                "undeclared or duplicate artifact",
            )
            names.add(name)
            path = root / name
            _require(
                path.is_file() and not path.is_symlink(), f"artifact missing or linked: {name}"
            )
            _require(
                type(item.get("size_bytes")) is int and path.stat().st_size == item["size_bytes"],
                f"artifact size mismatch: {name}",
            )
            _require(
                sha256_file(path) == _sha(item.get("sha256"), name),
                f"artifact hash mismatch: {name}",
            )
        _require(names == _ARTIFACTS, "declared artifact missing")
        bundle = _read_json(bundle_path)
        _require(
            bundle.get("kind") == "m1_cuda_null_static_replay" and bundle.get("version") == "1.0.0",
            "replay bundle schema mismatch",
        )
        _require(bundle.get("requires_model_execution") is False, "replay requests model execution")
        _require(
            bundle.get("authority_sha256") == sha256_file(root / "static-authority.json"),
            "replay authority mismatch",
        )
        _require(
            bundle.get("decision_sha256") == sha256_file(root / "decision.json"),
            "replay decision mismatch",
        )
        authority = _read_json(root / "static-authority.json")
        decision = _read_json(root / "decision.json")
        _require(decision == static_decision(authority), "static decision mismatch")
        return {
            "integrity_replay_status": "PASS",
            "status_match": True,
            "decision_match": True,
            "scientific_outcome": "NOT_EVALUATED",
            "execution_authorized": False,
            "manifest_sha256": expected_manifest_sha256,
            "authority_sha256": hashlib.sha256(canonical_json_bytes(authority) + b"\n").hexdigest(),
        }
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise CudaNullAuthorityBlocked(f"static replay blocked: {exc}") from exc
