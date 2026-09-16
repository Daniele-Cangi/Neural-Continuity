"""Tamper-evident evidence and model-free replay for the CUDA preflight."""

from __future__ import annotations

import json
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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _read_mapping(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


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
    runtime_ok = runtime.get("declared_provider_order") == expected_providers

    assignments_ok = True
    for model_label in ("source_fp32", "candidate_int8_qdq"):
        model = assignment.get(model_label)
        if not isinstance(model, dict):
            assignments_ok = False
            continue
        counts = model.get("provider_event_counts")
        assignments_ok = assignments_ok and isinstance(counts, dict)
        if isinstance(counts, dict):
            assignments_ok = assignments_ok and counts.get("CUDAExecutionProvider", 0) > 0
            assignments_ok = assignments_ok and not (set(counts) - set(expected_providers))
        assignments_ok = assignments_ok and model.get("unclassified_cpu_events") == 0
        assignments_ok = assignments_ok and model.get("undeclared_providers") == []

    expected_layout = authority.get("run_layout")
    benchmark_ok = (
        benchmark.get("document_count") == authority.get("document_count")
        and benchmark.get("query_count") == authority.get("query_count")
        and benchmark.get("raw_embeddings_retained") is False
        and benchmark.get("scientific_comparison_performed") is False
    )
    for model_label in ("source_fp32", "candidate_int8_qdq"):
        runs = benchmark.get(model_label)
        observed_layout = []
        if isinstance(runs, list):
            observed_layout = [
                {"label": run.get("label"), "batch_size": run.get("batch_size")}
                for run in runs
                if isinstance(run, dict)
            ]
        benchmark_ok = benchmark_ok and observed_layout == expected_layout

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


def replay_cuda_preflight(bundle_path: str | Path) -> dict[str, Any]:
    """Replay the package from JSON and hashes only; never import a model runtime."""

    bundle_file = Path(bundle_path).resolve()
    package = bundle_file.parent
    try:
        bundle = _read_mapping(bundle_file)
        manifest = _read_mapping(package / "artifact-manifest.json")
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
            artifact = package / name
            if not artifact.is_file() or sha256_file(artifact) != expected_hash:
                raise ValueError(f"artifact integrity mismatch: {name}")
            observed_names.add(name)
        if observed_names != expected_names:
            raise ValueError("artifact manifest is incomplete")

        artifact_names = bundle.get("artifacts")
        if not isinstance(artifact_names, dict):
            raise ValueError("replay bundle artifact map is missing")
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
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
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
