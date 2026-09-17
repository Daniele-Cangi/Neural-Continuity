"""Tamper-evident single-epoch CUDA sentinel package with model-free replay."""

from __future__ import annotations

import json
import shutil
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.cuda_null_paths import has_linked_ancestor
from neural_continuity.m1_diagnostics.cuda_null_sentinel_epoch_format import (
    DOCUMENT_COUNT,
    EMBEDDING_DIMENSION,
    FORMAT_VERSION,
    QUERY_COUNT,
    SHA256_PATTERN,
    CudaNullSentinelEpochBlocked,
    technical_summary,
)
from neural_continuity.m1_diagnostics.cuda_null_source_preflight_evidence import (
    _external_output_directory,
)

ARTIFACT_NAMES = (
    "epoch-plan.json",
    "runtime-inventory.json",
    "raw-observations.npz",
    "provider-profile.jsonl",
    "technical-summary.json",
    "replay-bundle.json",
)
PACKAGE_FILES = set(ARTIFACT_NAMES) | {"artifact-manifest.json"}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(dict(value)) + b"\n")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path.name}")
    return value


def _read_observations(path: Path) -> tuple[np.ndarray, np.ndarray]:
    expected = {
        "document_embeddings.npy": DOCUMENT_COUNT,
        "query_embeddings.npy": QUERY_COUNT,
    }
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) != len(expected) or {entry.filename for entry in entries} != set(expected):
            raise ValueError("sentinel observation archive member set differs")
        for entry in entries:
            maximum = 4 * expected[entry.filename] * EMBEDDING_DIMENSION * 4 + 4096
            if entry.file_size > maximum:
                raise ValueError("sentinel observation archive member exceeds frozen shape")
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ["document_embeddings", "query_embeddings"]:
            raise ValueError("sentinel observation array order differs")
        return archive["document_embeddings"], archive["query_embeddings"]


def replay_cuda_sentinel_epoch(bundle_path: Path, external_manifest_sha256: str) -> dict[str, Any]:
    """Check one package only; a later gate must verify chain and live authority."""
    try:
        if not isinstance(external_manifest_sha256, str) or not SHA256_PATTERN.fullmatch(
            external_manifest_sha256
        ):
            raise ValueError("external epoch manifest SHA-256 is invalid")
        bundle_file = Path(bundle_path).absolute()
        if bundle_file.name != "replay-bundle.json" or has_linked_ancestor(bundle_file):
            raise ValueError("sentinel replay bundle path is linked or invalid")
        package = bundle_file.parent
        if {entry.name for entry in package.iterdir()} != PACKAGE_FILES:
            raise ValueError("sentinel epoch artifact set is incomplete or expanded")
        manifest_path = package / "artifact-manifest.json"
        if has_linked_ancestor(manifest_path) or not manifest_path.is_file():
            raise ValueError("sentinel artifact manifest is linked or missing")
        if sha256_file(manifest_path) != external_manifest_sha256:
            raise ValueError("sentinel artifact manifest differs from external SHA-256")
        manifest = _read_json(manifest_path)
        entries = manifest.get("artifacts")
        if (
            set(manifest) != {"kind", "version", "artifacts"}
            or manifest["kind"] != "m1_cuda_null_sentinel_epoch_artifact_manifest"
            or manifest["version"] != FORMAT_VERSION
            or not isinstance(entries, list)
            or len(entries) != len(ARTIFACT_NAMES)
        ):
            raise ValueError("sentinel artifact manifest schema differs")
        for entry, name in zip(entries, ARTIFACT_NAMES, strict=True):
            artifact = package / name
            if (
                not isinstance(entry, dict)
                or set(entry) != {"path", "sha256"}
                or entry["path"] != name
                or not isinstance(entry["sha256"], str)
                or not SHA256_PATTERN.fullmatch(entry["sha256"])
                or has_linked_ancestor(artifact)
                or not artifact.is_file()
                or sha256_file(artifact) != entry["sha256"]
            ):
                raise ValueError(f"sentinel artifact integrity mismatch: {name}")
        if _read_json(bundle_file) != {
            "kind": "m1_cuda_null_sentinel_epoch_replay",
            "version": FORMAT_VERSION,
            "artifacts": list(ARTIFACT_NAMES[:-1]),
            "model_required_for_replay": False,
            "authority_verified": False,
            "checkpoint_chain_verified": False,
        }:
            raise ValueError("sentinel replay bundle scope differs")
        plan = _read_json(package / "epoch-plan.json")
        runtime = _read_json(package / "runtime-inventory.json")
        documents, queries = _read_observations(package / "raw-observations.npz")
        profiles = [
            json.loads(line, object_pairs_hook=_unique_pairs)
            for line in (package / "provider-profile.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        summary = technical_summary(plan, runtime, documents, queries, profiles)
        if _read_json(package / "technical-summary.json") != summary:
            raise ValueError("sentinel technical summary differs from model-free replay")
        return {
            "replay_status": "PASS",
            "technical_structure_status": "PASS",
            "summary_match": True,
            "artifact_manifest_sha256": external_manifest_sha256,
            "epoch_number": plan["epoch_number"],
            "authority_verified": False,
            "checkpoint_chain_verified": False,
            "qualifying_detection_evidence": False,
            "scientific_decision": "NOT_EVALUATED",
            "model_loaded": False,
            "onnx_graph_loaded": False,
        }
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        UnicodeError,
        zipfile.BadZipFile,
        EOFError,
        OverflowError,
    ) as exc:
        return {
            "replay_status": "BLOCKED",
            "technical_structure_status": "BLOCKED",
            "summary_match": False,
            "authority_verified": False,
            "checkpoint_chain_verified": False,
            "qualifying_detection_evidence": False,
            "scientific_decision": "NOT_EVALUATED",
            "model_loaded": False,
            "onnx_graph_loaded": False,
            "reason": str(exc),
        }


def write_cuda_sentinel_epoch(
    output_directory: Path,
    *,
    plan: Mapping[str, Any],
    runtime: Mapping[str, Any],
    document_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    profiles: Sequence[Mapping[str, Any]],
) -> tuple[Path, str]:
    """Package supplied observations; never run a model or grant authority."""
    summary = technical_summary(plan, runtime, document_embeddings, query_embeddings, profiles)
    output = _external_output_directory(output_directory)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        _write_json(temporary / "epoch-plan.json", plan)
        _write_json(temporary / "runtime-inventory.json", runtime)
        np.savez_compressed(
            temporary / "raw-observations.npz",
            document_embeddings=document_embeddings,
            query_embeddings=query_embeddings,
        )
        (temporary / "provider-profile.jsonl").write_bytes(
            b"".join(canonical_json_bytes(dict(profile)) + b"\n" for profile in profiles)
        )
        _write_json(temporary / "technical-summary.json", summary)
        _write_json(
            temporary / "replay-bundle.json",
            {
                "kind": "m1_cuda_null_sentinel_epoch_replay",
                "version": FORMAT_VERSION,
                "artifacts": list(ARTIFACT_NAMES[:-1]),
                "model_required_for_replay": False,
                "authority_verified": False,
                "checkpoint_chain_verified": False,
            },
        )
        _write_json(
            temporary / "artifact-manifest.json",
            {
                "kind": "m1_cuda_null_sentinel_epoch_artifact_manifest",
                "version": FORMAT_VERSION,
                "artifacts": [
                    {"path": name, "sha256": sha256_file(temporary / name)}
                    for name in ARTIFACT_NAMES
                ],
            },
        )
        manifest_sha256 = sha256_file(temporary / "artifact-manifest.json")
        replay = replay_cuda_sentinel_epoch(temporary / "replay-bundle.json", manifest_sha256)
        if replay["replay_status"] != "PASS":
            raise CudaNullSentinelEpochBlocked("new sentinel epoch package did not replay")
        temporary.replace(output)
    except Exception:
        if temporary.exists() and temporary.resolve().parent == output.parent.resolve():
            shutil.rmtree(temporary)
        raise
    return output, manifest_sha256
