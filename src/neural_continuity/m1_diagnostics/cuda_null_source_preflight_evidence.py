"""Tamper-evident source-only CUDA preflight package and model-free replay."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import uuid
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.cuda_null_paths import has_linked_ancestor
from neural_continuity.m1_diagnostics.cuda_null_source_preflight_authority import (
    READINESS_RECORD_SHA256,
    RUNTIME_IDENTITY_SHA256,
)
from neural_continuity.m1_diagnostics.cuda_null_source_preflight_inputs import (
    CORPUS_SHA256,
    DATASET_MANIFEST_SHA256,
    DOCUMENT_COUNT,
    EMBEDDING_DIMENSION,
    FROZEN_RUNS,
    QRELS_SHA256,
    QUERY_COUNT,
    QUERY_SHA256,
    SOURCE_ONNX_SHA256,
    TEACHER_REVISION,
)

ARTIFACT_NAMES = (
    "runtime-inventory.json",
    "input-identity.json",
    "source-observations.npz",
    "provider-profile.jsonl",
    "technical-decision.json",
    "replay-bundle.json",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_OBSERVATION_NAMES = tuple(f"{role}_batch_{batch}" for role, batch in FROZEN_RUNS)
_PROVIDERS = {"CUDAExecutionProvider", "CPUExecutionProvider"}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


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


def _sorted_unique_ids(value: object, count: int) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != count
        or value != sorted(value, key=lambda item: item.encode("utf-8"))
    ):
        raise ValueError("frozen input IDs are malformed, duplicated, or unordered")
    return value


def _verify_identity(identity: Mapping[str, Any]) -> None:
    expected = {
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
        "max_sequence_length": 256,
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("input identity differs from the frozen source-only scope")
    documents = _sorted_unique_ids(identity.get("all_document_ids"), 5183)
    queries = _sorted_unique_ids(identity.get("all_query_ids"), 81)
    if (
        identity.get("selected_document_ids") != documents[:DOCUMENT_COUNT]
        or identity.get("selected_query_ids") != queries[:QUERY_COUNT]
    ):
        raise ValueError("selected identities are not the first frozen UTF-8 IDs")
    authority = identity.get("source_preflight_authority")
    if not isinstance(authority, Mapping):
        raise ValueError("source-only preflight authority is missing")
    digest = authority.get("record_sha256")
    payload = {key: value for key, value in authority.items() if key != "record_sha256"}
    if (
        not isinstance(digest, str)
        or hashlib.sha256(canonical_json_bytes(payload) + b"\n").hexdigest() != digest
        or authority.get("status") != "SOURCE_ONLY_PREFLIGHT_AUTHORITY_VERIFIED"
        or authority.get("readiness_record_sha256") != READINESS_RECORD_SHA256
        or authority.get("runtime_identity_sha256") != RUNTIME_IDENTITY_SHA256
        or authority.get("source_only") is not True
        or authority.get("int8_allowed") is not False
        or authority.get("full_corpus_allowed") is not False
        or authority.get("holdout_allowed") is not False
        or authority.get("technical_preflight_permission") != "GRANTED_AFTER_REVIEW"
    ):
        raise ValueError("source-only preflight authority record is inconsistent")


def _verify_runtime(runtime: Mapping[str, Any]) -> None:
    digest = hashlib.sha256(canonical_json_bytes(runtime) + b"\n").hexdigest()
    if (
        digest != RUNTIME_IDENTITY_SHA256
        or runtime.get("status") != "RUNTIME_IDENTITY_VERIFIED_EXECUTION_BLOCKED"
        or runtime.get("onnx_graph_loaded") is not False
        or runtime.get("session_created") is not False
        or runtime.get("execution_authorized") is not False
    ):
        raise ValueError("runtime inventory differs from frozen pre-execution identity")


def _verify_profile(profile: Mapping[str, Any], role: str, batch: int) -> dict[str, Any]:
    name = f"{role}_batch_{batch}"
    count = DOCUMENT_COUNT if role == "documents" else QUERY_COUNT
    elapsed = profile.get("encode_seconds")
    startup = profile.get("session_seconds")
    rate = profile.get("items_per_second")
    if (
        profile.get("role") != role
        or type(profile.get("batch_size")) is not int
        or profile["batch_size"] != batch
        or profile.get("observation_name") != name
        or profile.get("item_count") != count
        or not isinstance(elapsed, int | float)
        or isinstance(elapsed, bool)
        or not math.isfinite(elapsed)
        or elapsed <= 0
        or not isinstance(startup, int | float)
        or isinstance(startup, bool)
        or not math.isfinite(startup)
        or startup < 0
        or not isinstance(rate, int | float)
        or isinstance(rate, bool)
        or not math.isfinite(rate)
        or not math.isclose(rate, count / elapsed, rel_tol=1e-9)
    ):
        raise ValueError(f"invalid run identity or timing: {name}")
    counts = profile.get("provider_event_counts")
    operators = profile.get("operator_event_counts")
    if (
        not isinstance(counts, Mapping)
        or not isinstance(operators, Mapping)
        or set(counts) != set(operators)
        or set(counts) - _PROVIDERS
        or type(counts.get("CUDAExecutionProvider")) is not int
        or counts["CUDAExecutionProvider"] < 1
    ):
        raise ValueError(f"CUDA provider activity is absent or unclassified: {name}")
    for provider, amount in counts.items():
        by_operator = operators[provider]
        if (
            type(amount) is not int
            or amount < 1
            or not isinstance(by_operator, Mapping)
            or not by_operator
            or any(
                not isinstance(operator, str)
                or not operator
                or operator == "UNCLASSIFIED"
                or type(events) is not int
                or events < 1
                for operator, events in by_operator.items()
            )
            or sum(by_operator.values()) != amount
        ):
            raise ValueError(f"invalid provider/operator classification: {name}")
    if (
        profile.get("cpu_fallback_operator_types")
        != sorted(operators.get("CPUExecutionProvider", {}))
        or profile.get("unclassified_cpu_events") != 0
        or profile.get("undeclared_providers") != []
    ):
        raise ValueError(f"CPU fallback or undeclared provider mismatch: {name}")
    return {
        "role": role,
        "batch_size": batch,
        "session_seconds": startup,
        "encode_seconds": elapsed,
        "items_per_second": rate,
        "cuda_operator_events": counts["CUDAExecutionProvider"],
        "cpu_operator_events": counts.get("CPUExecutionProvider", 0),
        "cpu_fallback_operator_types": profile["cpu_fallback_operator_types"],
    }


def _decision(
    runtime: Mapping[str, Any],
    identity: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    _verify_identity(identity)
    _verify_runtime(runtime)
    if set(arrays) != set(_OBSERVATION_NAMES) or len(profiles) != len(FROZEN_RUNS):
        raise ValueError("source observations or run profiles are missing")
    timings: list[dict[str, Any]] = []
    for profile, (role, batch) in zip(profiles, FROZEN_RUNS, strict=True):
        name = f"{role}_batch_{batch}"
        array = arrays[name]
        expected_count = DOCUMENT_COUNT if role == "documents" else QUERY_COUNT
        if (
            not isinstance(array, np.ndarray)
            or array.dtype != np.dtype("<f4")
            or array.shape != (expected_count, EMBEDDING_DIMENSION)
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"invalid float32 source observation: {name}")
        norms = np.linalg.norm(array.astype(np.float64), axis=1)
        if not np.isfinite(norms).all() or np.any(np.abs(norms - 1.0) > 1e-4):
            raise ValueError(f"source observation is not normalized: {name}")
        digest = hashlib.sha256(np.ascontiguousarray(array, dtype="<f4").tobytes()).hexdigest()
        if profile.get("observation_sha256") != digest:
            raise ValueError(f"source observation/profile hash mismatch: {name}")
        timings.append(_verify_profile(profile, role, batch))
    return {
        "kind": "m1_cuda_null_source_preflight_technical_decision",
        "technical_preflight_status": "PASS",
        "scientific_decision": "NOT_EVALUATED",
        "qualifying_m1_evidence": False,
        "source_only": True,
        "int8_executed": False,
        "holdout_accessed": False,
        "sentinel_started": False,
        "full_corpus_started": False,
        "run_timings": timings,
    }


def write_source_preflight_package(
    capture: Mapping[str, Any], output_directory: Path
) -> tuple[Path, str]:
    """Write six declared artifacts atomically, never into the repository."""
    runtime = capture["runtime_inventory"]
    identity = capture["input_identity"]
    arrays = capture["observations"]
    profiles = capture["provider_profiles"]
    decision = _decision(runtime, identity, arrays, profiles)
    output = Path(output_directory).absolute()
    repository = Path(__file__).resolve().parents[3]
    if output.is_relative_to(repository) or output.exists() or not output.parent.is_dir():
        raise ValueError("output must be a new directory outside the repository")
    if has_linked_ancestor(output.parent):
        raise ValueError("output parent contains a link or reparse point")
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        _write_json(temporary / "runtime-inventory.json", runtime)
        _write_json(temporary / "input-identity.json", identity)
        np.savez_compressed(
            temporary / "source-observations.npz",
            **{name: arrays[name] for name in _OBSERVATION_NAMES},
        )
        (temporary / "provider-profile.jsonl").write_bytes(
            b"".join(canonical_json_bytes(profile) + b"\n" for profile in profiles)
        )
        _write_json(temporary / "technical-decision.json", decision)
        bundle = {
            "kind": "m1_cuda_null_source_preflight_replay",
            "artifacts": list(ARTIFACT_NAMES[:-1]),
            "expected_status": "PASS",
            "model_required_for_replay": False,
            "onnx_graph_required_for_replay": False,
        }
        _write_json(temporary / "replay-bundle.json", bundle)
        manifest = {
            "kind": "m1_cuda_null_source_preflight_artifact_manifest",
            "artifacts": [
                {"path": name, "sha256": sha256_file(temporary / name)} for name in ARTIFACT_NAMES
            ],
        }
        _write_json(temporary / "artifact-manifest.json", manifest)
        temporary.replace(output)
    except Exception:
        if temporary.exists() and temporary.resolve().parent == output.parent.resolve():
            shutil.rmtree(temporary)
        raise
    return output, sha256_file(output / "artifact-manifest.json")


def replay_source_preflight(bundle_path: Path, external_manifest_sha256: str) -> dict[str, Any]:
    """Recompute the technical decision from hashes, profiles, and arrays only."""
    try:
        if not isinstance(external_manifest_sha256, str) or not _SHA256.fullmatch(
            external_manifest_sha256
        ):
            raise ValueError("external artifact-manifest SHA-256 is invalid")
        bundle_file = Path(bundle_path).absolute()
        if bundle_file.name != "replay-bundle.json" or has_linked_ancestor(bundle_file):
            raise ValueError("replay bundle path is not package-bound and unlinked")
        package = bundle_file.parent
        expected_files = set(ARTIFACT_NAMES) | {"artifact-manifest.json"}
        if {entry.name for entry in package.iterdir()} != expected_files:
            raise ValueError("declared preflight artifact set is incomplete or expanded")
        manifest_path = package / "artifact-manifest.json"
        if sha256_file(manifest_path) != external_manifest_sha256:
            raise ValueError("artifact manifest differs from external SHA-256")
        manifest = _read_json(manifest_path)
        entries = manifest.get("artifacts")
        if (
            manifest.get("kind") != "m1_cuda_null_source_preflight_artifact_manifest"
            or not isinstance(entries, list)
            or len(entries) != len(ARTIFACT_NAMES)
        ):
            raise ValueError("artifact manifest schema is invalid")
        for entry, name in zip(entries, ARTIFACT_NAMES, strict=True):
            artifact = package / name
            if (
                not isinstance(entry, dict)
                or entry.get("path") != name
                or not isinstance(entry.get("sha256"), str)
                or not _SHA256.fullmatch(entry["sha256"])
                or has_linked_ancestor(artifact)
                or not artifact.is_file()
                or sha256_file(artifact) != entry["sha256"]
            ):
                raise ValueError(f"artifact integrity mismatch: {name}")
        bundle = _read_json(bundle_file)
        if bundle != {
            "kind": "m1_cuda_null_source_preflight_replay",
            "artifacts": list(ARTIFACT_NAMES[:-1]),
            "expected_status": "PASS",
            "model_required_for_replay": False,
            "onnx_graph_required_for_replay": False,
        }:
            raise ValueError("replay bundle scope differs from frozen specification")
        runtime = _read_json(package / "runtime-inventory.json")
        identity = _read_json(package / "input-identity.json")
        with np.load(package / "source-observations.npz", allow_pickle=False) as archive:
            if archive.files != list(_OBSERVATION_NAMES):
                raise ValueError("source observation array set is incomplete or reordered")
            arrays = {name: archive[name] for name in archive.files}
        profiles = [
            json.loads(line, object_pairs_hook=_unique_pairs)
            for line in (package / "provider-profile.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        if any(not isinstance(profile, dict) for profile in profiles):
            raise ValueError("provider profile contains a malformed record")
        reproduced = _decision(runtime, identity, arrays, profiles)
        recorded = _read_json(package / "technical-decision.json")
        if reproduced != recorded:
            raise ValueError("technical decision differs from model-free replay")
        return {
            "replay_status": "PASS",
            "technical_preflight_status": "PASS",
            "decision_match": True,
            "artifact_manifest_sha256": external_manifest_sha256,
            "scientific_decision": "NOT_EVALUATED",
            "model_loaded": False,
            "onnx_graph_loaded": False,
            "run_timings": reproduced["run_timings"],
        }
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        UnicodeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
        EOFError,
    ) as exc:
        return {
            "replay_status": "BLOCKED",
            "technical_preflight_status": "BLOCKED",
            "decision_match": False,
            "scientific_decision": "NOT_EVALUATED",
            "model_loaded": False,
            "onnx_graph_loaded": False,
            "reason": str(exc),
        }
