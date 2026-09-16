"""Fail-closed authority verification for the M1 hybrid CUDA preflight."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.measurement_null_sentinel_authority import (
    SentinelAuthority,
    verify_sentinel_authority,
)


class CudaPreflightBlocked(RuntimeError):
    """Raised when a declared preflight authority cannot be verified."""


@dataclass(frozen=True)
class CandidateAuthority:
    package_directory: Path
    manifest_path: Path
    manifest_sha256: str
    quantization_config_path: Path
    quantization_config_sha256: str
    artifact_path: Path
    artifact_sha256: str


@dataclass(frozen=True)
class CudaPreflightAuthority:
    config: Mapping[str, Any]
    config_sha256: str
    base: SentinelAuthority
    candidate: CandidateAuthority
    provider_order: tuple[str, ...]
    run_layout: tuple[tuple[str, int], ...]
    expected_gpu: Mapping[str, str]
    expected_onnxruntime_version: str
    authority_sha256: str


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CudaPreflightBlocked(f"{label} must be a mapping")
    return value


def _required_string(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise CudaPreflightBlocked(f"{label}.{key} must be a non-empty string")
    return value


def _assert_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise CudaPreflightBlocked(f"{label} is missing: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise CudaPreflightBlocked(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )


def _load_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CudaPreflightBlocked(f"{label} is not valid JSON: {path}") from exc
    return _mapping(value, label)


def _load_preflight_config(path: Path) -> tuple[Mapping[str, Any], str]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CudaPreflightBlocked(f"preflight config is invalid: {path}") from exc
    config = _mapping(value, "preflight config")
    config_sha256 = hashlib.sha256(canonical_json_bytes(config)).hexdigest()
    return config, config_sha256


def _parse_run_layout(config: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    benchmark = _mapping(config.get("benchmark"), "benchmark")
    raw_runs = benchmark.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise CudaPreflightBlocked("benchmark.runs must be a non-empty list")

    runs: list[tuple[str, int]] = []
    for index, raw_run in enumerate(raw_runs):
        run = _mapping(raw_run, f"benchmark.runs[{index}]")
        label = _required_string(run, "label", f"benchmark.runs[{index}]")
        batch_size = run.get("batch_size")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise CudaPreflightBlocked(
                f"benchmark.runs[{index}].batch_size must be a positive integer"
            )
        runs.append((label, batch_size))

    expected = (
        ("batch_1_primary", 1),
        ("batch_16_primary", 16),
        ("batch_16_repeat", 16),
        ("batch_64_primary", 64),
    )
    if tuple(runs) != expected:
        raise CudaPreflightBlocked(f"benchmark run layout must be exactly {expected}")
    return tuple(runs)


def _verify_candidate(
    config: Mapping[str, Any], candidate_package: str | Path
) -> CandidateAuthority:
    declared = _mapping(config.get("candidate"), "candidate")
    package = Path(candidate_package).resolve()
    if not package.is_dir():
        raise CudaPreflightBlocked(f"candidate package is missing: {package}")

    manifest_path = package / _required_string(declared, "manifest_file", "candidate")
    quantization_path = package / _required_string(
        declared, "quantization_config_file", "candidate"
    )
    artifact_path = package / _required_string(declared, "artifact_file", "candidate")
    manifest_sha256 = _required_string(declared, "manifest_sha256", "candidate")
    quantization_sha256 = _required_string(declared, "quantization_config_sha256", "candidate")
    artifact_sha256 = _required_string(declared, "artifact_sha256", "candidate")

    _assert_hash(manifest_path, manifest_sha256, "candidate manifest")
    _assert_hash(quantization_path, quantization_sha256, "candidate quantization config")
    _assert_hash(artifact_path, artifact_sha256, "candidate ONNX artifact")

    manifest = _load_json_mapping(manifest_path, "candidate manifest")
    quantization = _load_json_mapping(quantization_path, "candidate quantization config")
    manifest_material = canonical_json_bytes(manifest).decode("utf-8")
    if artifact_path.name not in manifest_material or artifact_sha256 not in manifest_material:
        raise CudaPreflightBlocked(
            "candidate manifest does not bind the declared ONNX artifact and SHA-256"
        )

    expected_quantization = _mapping(
        declared.get("expected_quantization"), "candidate.expected_quantization"
    )
    quantization_material = canonical_json_bytes(quantization).decode("utf-8")
    for key, expected_value in expected_quantization.items():
        if not isinstance(expected_value, str | bool | int | float):
            raise CudaPreflightBlocked(f"candidate.expected_quantization.{key} must be scalar")
        encoded = json.dumps(expected_value, ensure_ascii=True, separators=(",", ":"))
        if encoded not in quantization_material:
            raise CudaPreflightBlocked(
                f"quantization audit did not find declared value for {key}: " f"{expected_value!r}"
            )

    return CandidateAuthority(
        package_directory=package,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        quantization_config_path=quantization_path,
        quantization_config_sha256=quantization_sha256,
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
    )


def verify_cuda_preflight_authority(
    config_path: str | Path,
    dataset_directory: str | Path,
    transition_a_bundle: str | Path,
    extension_plan_bundle: str | Path,
    extension_plan_manifest_sha256: str,
    candidate_package: str | Path,
) -> CudaPreflightAuthority:
    """Verify every frozen authority before any ONNX graph may be loaded."""

    try:
        config, config_sha256 = _load_preflight_config(Path(config_path))
        base = verify_sentinel_authority(
            config_path=_required_string(
                _mapping(config.get("base_authority"), "base_authority"),
                "config_path",
                "base_authority",
            ),
            dataset_directory=dataset_directory,
            transition_a_bundle=transition_a_bundle,
            extension_plan_bundle=extension_plan_bundle,
            extension_plan_manifest_sha256=extension_plan_manifest_sha256,
        )
    except CudaPreflightBlocked:
        raise
    except Exception as exc:
        raise CudaPreflightBlocked(f"base authority verification failed: {exc}") from exc

    scope = _mapping(config.get("scope"), "scope")
    if scope.get("qualifying_m1_evidence") is not False:
        raise CudaPreflightBlocked("CUDA preflight must remain non-qualifying")
    if scope.get("full_corpus_authorized") is not False:
        raise CudaPreflightBlocked("CUDA preflight cannot authorize the full corpus")
    if scope.get("scientific_decision") != "NOT_EVALUATED":
        raise CudaPreflightBlocked("scientific decision must remain NOT_EVALUATED")

    base_declared = _mapping(config.get("base_authority"), "base_authority")
    expected_base = _required_string(base_declared, "authority_sha256", "base_authority")
    if base.authority_sha256 != expected_base:
        raise CudaPreflightBlocked(
            "base sentinel authority mismatch: "
            f"expected {expected_base}, observed {base.authority_sha256}"
        )
    expected_source = _required_string(base_declared, "source_artifact_sha256", "base_authority")
    if base.source.artifact_sha256 != expected_source:
        raise CudaPreflightBlocked("base source ONNX artifact identity mismatch")

    expected_document_count = config.get("document_count")
    expected_query_count = config.get("query_count")
    if len(base.selected_document_ids) != expected_document_count:
        raise CudaPreflightBlocked("selected document count mismatch")
    if len(base.query_ids) != expected_query_count:
        raise CudaPreflightBlocked("selected query count mismatch")

    provider_policy = _mapping(config.get("provider_policy"), "provider_policy")
    raw_order = provider_policy.get("ordered_providers")
    if raw_order != ["CUDAExecutionProvider", "CPUExecutionProvider"]:
        raise CudaPreflightBlocked(
            "provider order must be CUDAExecutionProvider then CPUExecutionProvider"
        )
    if provider_policy.get("require_cuda_activity") is not True:
        raise CudaPreflightBlocked("CUDA activity must be required")
    if provider_policy.get("classify_cpu_fallback_by_operator_type") is not True:
        raise CudaPreflightBlocked("CPU fallback must be classified by operator type")
    if provider_policy.get("reject_undeclared_providers") is not True:
        raise CudaPreflightBlocked("undeclared providers must be rejected")

    candidate = _verify_candidate(config, candidate_package)
    run_layout = _parse_run_layout(config)
    runtime = _mapping(config.get("runtime"), "runtime")
    expected_gpu = _mapping(runtime.get("expected_gpu"), "runtime.expected_gpu")
    for key in ("name", "uuid", "compute_capability"):
        _required_string(expected_gpu, key, "runtime.expected_gpu")

    authority_payload = {
        "base_authority_sha256": base.authority_sha256,
        "candidate_artifact_sha256": candidate.artifact_sha256,
        "candidate_manifest_sha256": candidate.manifest_sha256,
        "config_sha256": config_sha256,
        "document_ids": list(base.selected_document_ids),
        "expected_gpu": dict(expected_gpu),
        "provider_order": list(raw_order),
        "qrels": {key: list(value) for key, value in sorted(base.qrels.items())},
        "quantization_config_sha256": candidate.quantization_config_sha256,
        "query_ids": list(base.query_ids),
        "run_layout": [
            {"label": label, "batch_size": batch_size} for label, batch_size in run_layout
        ],
        "source_artifact_sha256": base.source.artifact_sha256,
    }
    authority_sha256 = hashlib.sha256(canonical_json_bytes(authority_payload)).hexdigest()

    return CudaPreflightAuthority(
        config=config,
        config_sha256=config_sha256,
        base=base,
        candidate=candidate,
        provider_order=tuple(raw_order),
        run_layout=run_layout,
        expected_gpu=dict(expected_gpu),
        expected_onnxruntime_version=_required_string(runtime, "onnxruntime_version", "runtime"),
        authority_sha256=authority_sha256,
    )


def authority_record(authority: CudaPreflightAuthority) -> dict[str, Any]:
    """Return the compact, replayable authority record without dataset text."""

    return {
        "version": 1,
        "kind": "m1_cuda_hybrid_preflight_authority",
        "authority_sha256": authority.authority_sha256,
        "config_sha256": authority.config_sha256,
        "base_authority_sha256": authority.base.authority_sha256,
        "source_artifact_sha256": authority.base.source.artifact_sha256,
        "candidate_manifest_sha256": authority.candidate.manifest_sha256,
        "candidate_artifact_sha256": authority.candidate.artifact_sha256,
        "quantization_config_sha256": authority.candidate.quantization_config_sha256,
        "document_count": len(authority.base.selected_document_ids),
        "query_count": len(authority.base.query_ids),
        "document_ids_sha256": hashlib.sha256(
            canonical_json_bytes(list(authority.base.selected_document_ids))
        ).hexdigest(),
        "query_ids_sha256": hashlib.sha256(
            canonical_json_bytes(list(authority.base.query_ids))
        ).hexdigest(),
        "qrels_sha256": hashlib.sha256(
            canonical_json_bytes(
                {key: list(value) for key, value in sorted(authority.base.qrels.items())}
            )
        ).hexdigest(),
        "provider_order": list(authority.provider_order),
        "run_layout": [
            {"label": label, "batch_size": batch_size} for label, batch_size in authority.run_layout
        ],
        "expected_gpu": dict(authority.expected_gpu),
        "expected_onnxruntime_version": authority.expected_onnxruntime_version,
        "qualifying_m1_evidence": False,
        "full_corpus_authorized": False,
        "scientific_decision": "NOT_EVALUATED",
    }
