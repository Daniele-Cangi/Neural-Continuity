"""Tamper-evident evidence and model-free replay for the CUDA preflight."""

from __future__ import annotations

import json
import math
import re
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes, sha256_file

ARTIFACT_NAMES = (
    "cuda-authority.json",
    "runtime-inventory.json",
    "provider-assignment.json",
    "benchmark-summary.json",
    "preflight-decision.json",
    "replay-bundle.json",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")
_FROZEN_LAYOUT = [
    {"label": "batch_1_primary", "batch_size": 1},
    {"label": "batch_16_primary", "batch_size": 16},
    {"label": "batch_16_repeat", "batch_size": 16},
    {"label": "batch_64_primary", "batch_size": 64},
]


def _frozen_layout_matches(value: object) -> bool:
    if not isinstance(value, list) or len(value) != len(_FROZEN_LAYOUT):
        return False
    return all(
        isinstance(run, Mapping)
        and set(run) == {"label", "batch_size"}
        and run.get("label") == frozen["label"]
        and type(run.get("batch_size")) is int
        and run["batch_size"] == frozen["batch_size"]
        for run, frozen in zip(value, _FROZEN_LAYOUT, strict=True)
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _read_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _valid_profile(model: object) -> bool:
    if not isinstance(model, Mapping):
        return False
    counts = model.get("provider_event_counts")
    operators = model.get("operator_event_counts")
    if not isinstance(counts, Mapping) or not isinstance(operators, Mapping):
        return False
    if not counts or set(counts) != set(operators) or set(counts) - set(_PROVIDERS):
        return False
    if type(counts.get("CUDAExecutionProvider")) is not int or counts["CUDAExecutionProvider"] < 1:
        return False
    for provider, count in counts.items():
        by_operator = operators[provider]
        if type(count) is not int or count < 1 or not isinstance(by_operator, Mapping):
            return False
        if not by_operator or any(
            not isinstance(name, str)
            or not name
            or name == "UNCLASSIFIED"
            or type(amount) is not int
            or amount < 1
            for name, amount in by_operator.items()
        ):
            return False
        if sum(by_operator.values()) != count:
            return False
    cpu_types = sorted(operators.get("CPUExecutionProvider", {}))
    return (
        model.get("cpu_fallback_operator_types") == cpu_types
        and type(model.get("unclassified_cpu_events")) is int
        and model["unclassified_cpu_events"] == 0
        and model.get("undeclared_providers") == []
    )


def _valid_embedding(record: object, count: int) -> int | None:
    if not isinstance(record, Mapping):
        return None
    shape = record.get("shape")
    digest = record.get("sha256")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or type(shape[0]) is not int
        or shape[0] != count
        or type(shape[1]) is not int
        or shape[1] < 1
        or record.get("dtype") != "float32"
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
    ):
        return None
    return shape[1]


def _valid_run(run: object, layout: object, document_count: int, query_count: int) -> int | None:
    if not isinstance(run, Mapping) or not isinstance(layout, Mapping):
        return None
    if type(run.get("batch_size")) is not int:
        return None
    if run.get("label") != layout.get("label") or run.get("batch_size") != layout.get("batch_size"):
        return None
    elapsed = run.get("elapsed_seconds")
    rate = run.get("items_per_second")
    item_count = document_count + query_count
    if (
        not isinstance(elapsed, int | float)
        or isinstance(elapsed, bool)
        or not isinstance(rate, int | float)
        or isinstance(rate, bool)
    ):
        return None
    if (
        type(run.get("item_count")) is not int
        or run["item_count"] != item_count
        or not math.isfinite(elapsed)
        or elapsed <= 0
        or not math.isfinite(rate)
        or not math.isclose(rate, item_count / elapsed, rel_tol=1e-9)
    ):
        return None
    document_dimension = _valid_embedding(run.get("documents"), document_count)
    query_dimension = _valid_embedding(run.get("queries"), query_count)
    if document_dimension is None or document_dimension != query_dimension:
        return None
    return document_dimension


def _decision(
    authority: Mapping[str, Any],
    runtime: Mapping[str, Any],
    assignment: Mapping[str, Any],
    benchmark: Mapping[str, Any],
) -> dict[str, Any]:
    expected_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    authority_ok = (
        authority.get("qualifying_m1_evidence") is False
        and authority.get("full_corpus_authorized") is False
        and authority.get("scientific_decision") == "NOT_EVALUATED"
        and authority.get("provider_order") == expected_providers
    )
    expected_gpu = authority.get("expected_gpu")
    observed_gpu = runtime.get("gpu")
    runtime_ok = (
        runtime.get("declared_provider_order") == expected_providers
        and runtime.get("onnxruntime_gpu_version") == authority.get("expected_onnxruntime_version")
        and runtime.get("onnxruntime_device") == "GPU"
        and runtime.get("cuda_dlls_preloaded") is True
        and isinstance(expected_gpu, Mapping)
        and isinstance(observed_gpu, Mapping)
        and all(
            isinstance(expected_gpu.get(key), str)
            and bool(expected_gpu[key])
            and observed_gpu.get(key) == expected_gpu[key]
            for key in ("name", "uuid", "compute_capability")
        )
        and isinstance(runtime.get("available_providers"), list)
        and set(expected_providers).issubset(set(runtime["available_providers"]))
    )

    expected_policy = {
        "provider_order": expected_providers,
        "classification_unit": "operator_type",
        "node_names_retained": False,
        "tensor_names_retained": False,
        "benchmark_specific_exceptions": False,
    }
    assignments_ok = assignment.get("policy") == expected_policy
    for model_label in ("source_fp32", "candidate_int8_qdq"):
        assignments_ok = assignments_ok and _valid_profile(assignment.get(model_label))

    expected_layout = authority.get("run_layout")
    document_count = authority.get("document_count")
    query_count = authority.get("query_count")
    document_count_value = document_count if isinstance(document_count, int) else 0
    query_count_value = query_count if isinstance(query_count, int) else 0
    benchmark_ok = (
        benchmark.get("document_count") == authority.get("document_count")
        and benchmark.get("query_count") == authority.get("query_count")
        and benchmark.get("raw_embeddings_retained") is False
        and benchmark.get("scientific_comparison_performed") is False
        and type(document_count) is int
        and document_count > 0
        and type(query_count) is int
        and query_count > 0
        and _frozen_layout_matches(expected_layout)
    )
    dimensions: set[int] = set()
    for model_label in ("source_fp32", "candidate_int8_qdq"):
        runs = benchmark.get(model_label)
        if not isinstance(runs, list) or not isinstance(expected_layout, list):
            benchmark_ok = False
            continue
        if len(runs) != len(expected_layout):
            benchmark_ok = False
            continue
        for run, layout in zip(runs, expected_layout, strict=True):
            dimension = _valid_run(run, layout, document_count_value, query_count_value)
            if dimension is None:
                benchmark_ok = False
            else:
                dimensions.add(dimension)
    benchmark_ok = benchmark_ok and len(dimensions) == 1

    passed = authority_ok and runtime_ok and assignments_ok and benchmark_ok
    return {
        "version": 1,
        "kind": "m1_cuda_hybrid_preflight_decision",
        "preflight_status": "PASS" if passed else "BLOCKED",
        "authority_match": authority_ok,
        "runtime_match": runtime_ok,
        "provider_assignment_match": assignments_ok,
        "benchmark_scope_match": benchmark_ok,
        "scientific_decision": "NOT_EVALUATED",
        "qualifying_m1_evidence": False,
        "full_corpus_authorized": False,
    }


def write_cuda_preflight_package(
    authority: Mapping[str, Any],
    runtime_result: Mapping[str, Mapping[str, Any]],
    output_directory: str | Path,
) -> tuple[Path, str]:
    """Write one immutable package, returning its path and manifest SHA-256."""

    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        runtime = runtime_result["runtime_inventory"]
        assignment = runtime_result["provider_assignment"]
        benchmark = runtime_result["benchmark"]
        decision = _decision(authority, runtime, assignment, benchmark)
        if decision["preflight_status"] != "PASS":
            raise ValueError("runtime result does not satisfy the preflight authority")

        replay_bundle = {
            "version": 1,
            "kind": "m1_cuda_hybrid_preflight_replay_bundle",
            "artifacts": {
                "authority": "cuda-authority.json",
                "runtime_inventory": "runtime-inventory.json",
                "provider_assignment": "provider-assignment.json",
                "benchmark": "benchmark-summary.json",
                "decision": "preflight-decision.json",
            },
            "expected_authority_sha256": authority["authority_sha256"],
            "expected_preflight_status": "PASS",
            "model_required_for_replay": False,
            "onnx_graph_required_for_replay": False,
        }
        payloads = {
            "cuda-authority.json": authority,
            "runtime-inventory.json": runtime,
            "provider-assignment.json": assignment,
            "benchmark-summary.json": benchmark,
            "preflight-decision.json": decision,
            "replay-bundle.json": replay_bundle,
        }
        for name, payload in payloads.items():
            _write_json(temporary / name, payload)

        manifest = {
            "version": 1,
            "kind": "m1_cuda_hybrid_preflight_artifact_manifest",
            "artifacts": [
                {"path": name, "sha256": sha256_file(temporary / name)} for name in ARTIFACT_NAMES
            ],
        }
        _write_json(temporary / "artifact-manifest.json", manifest)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output, sha256_file(output / "artifact-manifest.json")


def replay_cuda_preflight(bundle_path: str | Path, expected_manifest_sha256: str) -> dict[str, Any]:
    """Replay the package from JSON and hashes only; never import a model runtime."""

    bundle_file = Path(bundle_path).resolve()
    package = bundle_file.parent
    try:
        if bundle_file != package / "replay-bundle.json":
            raise ValueError("replay bundle is not the manifest-listed artifact")
        manifest_path = package / "artifact-manifest.json"
        if (
            not isinstance(expected_manifest_sha256, str)
            or _SHA256.fullmatch(expected_manifest_sha256) is None
            or sha256_file(manifest_path) != expected_manifest_sha256
        ):
            raise ValueError("artifact manifest SHA-256 differs from external authority")
        bundle = _read_mapping(bundle_file)
        if (
            type(bundle.get("version")) is not int
            or bundle["version"] != 1
            or bundle.get("kind") != "m1_cuda_hybrid_preflight_replay_bundle"
            or bundle.get("model_required_for_replay") is not False
            or bundle.get("onnx_graph_required_for_replay") is not False
        ):
            raise ValueError("replay bundle schema is invalid")
        manifest = _read_mapping(manifest_path)
        if manifest.get("kind") != "m1_cuda_hybrid_preflight_artifact_manifest":
            raise ValueError("artifact manifest kind is invalid")
        entries = manifest.get("artifacts")
        if not isinstance(entries, list):
            raise ValueError("artifact manifest entries are missing")
        expected_names = set(ARTIFACT_NAMES)
        observed_names: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("artifact manifest entry must be an object")
            name = entry.get("path")
            expected_hash = entry.get("sha256")
            if name not in expected_names or not isinstance(expected_hash, str):
                raise ValueError("artifact manifest contains an undeclared entry")
            if name in observed_names:
                raise ValueError(f"duplicate artifact manifest entry: {name}")
            artifact = package / name
            if not artifact.is_file() or sha256_file(artifact) != expected_hash:
                raise ValueError(f"artifact integrity mismatch: {name}")
            observed_names.add(name)
        if observed_names != expected_names:
            raise ValueError("artifact manifest is incomplete")

        artifact_names = bundle.get("artifacts")
        if artifact_names != {
            "authority": "cuda-authority.json",
            "runtime_inventory": "runtime-inventory.json",
            "provider_assignment": "provider-assignment.json",
            "benchmark": "benchmark-summary.json",
            "decision": "preflight-decision.json",
        }:
            raise ValueError("replay bundle artifact map is invalid")
        authority = _read_mapping(package / artifact_names["authority"])
        runtime = _read_mapping(package / artifact_names["runtime_inventory"])
        assignment = _read_mapping(package / artifact_names["provider_assignment"])
        benchmark = _read_mapping(package / artifact_names["benchmark"])
        recorded = _read_mapping(package / artifact_names["decision"])
        reproduced = _decision(authority, runtime, assignment, benchmark)
        authority_match = authority.get("authority_sha256") == bundle.get(
            "expected_authority_sha256"
        )
        status_match = reproduced == recorded and reproduced.get("preflight_status") == bundle.get(
            "expected_preflight_status"
        )
        replay_status = "PASS" if authority_match and status_match else "BLOCKED"
        return {
            "version": 1,
            "kind": "m1_cuda_hybrid_preflight_replay_result",
            "replay_status": replay_status,
            "artifact_integrity": True,
            "authority_match": authority_match,
            "status_match": status_match,
            "provider_assignment_match": reproduced["provider_assignment_match"],
            "benchmark_scope_match": reproduced["benchmark_scope_match"],
            "preflight_status": reproduced["preflight_status"],
            "scientific_decision": "NOT_EVALUATED",
            "model_loaded": False,
            "onnx_graph_loaded": False,
        }
    except (
        KeyError,
        OSError,
        OverflowError,
        TypeError,
        ValueError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        return {
            "version": 1,
            "kind": "m1_cuda_hybrid_preflight_replay_result",
            "replay_status": "BLOCKED",
            "artifact_integrity": False,
            "authority_match": False,
            "status_match": False,
            "provider_assignment_match": False,
            "benchmark_scope_match": False,
            "preflight_status": "BLOCKED",
            "scientific_decision": "NOT_EVALUATED",
            "model_loaded": False,
            "onnx_graph_loaded": False,
            "reason": str(exc),
        }
