import json
from pathlib import Path

from neural_continuity.evidence import sha256_file
from neural_continuity.m1_diagnostics.cuda_preflight_evidence import (
    replay_cuda_preflight,
    write_cuda_preflight_package,
)


def _authority() -> dict[str, object]:
    return {
        "authority_sha256": "authority",
        "provider_order": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "run_layout": [
            {"label": "batch_1_primary", "batch_size": 1},
            {"label": "batch_16_primary", "batch_size": 16},
            {"label": "batch_16_repeat", "batch_size": 16},
            {"label": "batch_64_primary", "batch_size": 64},
        ],
        "document_count": 256,
        "query_count": 81,
        "expected_gpu": {
            "name": "GPU",
            "uuid": "GPU-uuid",
            "compute_capability": "7.5",
        },
        "expected_onnxruntime_version": "1.28.0",
        "qualifying_m1_evidence": False,
        "full_corpus_authorized": False,
        "scientific_decision": "NOT_EVALUATED",
    }


def _runtime_result() -> dict[str, dict[str, object]]:
    layout = _authority()["run_layout"]
    runs = [
        {
            **run,
            "elapsed_seconds": 1.0,
            "item_count": 337,
            "items_per_second": 337.0,
            "documents": {
                "shape": [256, 384],
                "dtype": "float32",
                "sha256": "a" * 64,
            },
            "queries": {
                "shape": [81, 384],
                "dtype": "float32",
                "sha256": "b" * 64,
            },
        }
        for run in layout  # type: ignore[union-attr]
    ]
    profile = {
        "provider_event_counts": {
            "CUDAExecutionProvider": 10,
            "CPUExecutionProvider": 2,
        },
        "operator_event_counts": {
            "CUDAExecutionProvider": {"MatMul": 10},
            "CPUExecutionProvider": {"Shape": 2},
        },
        "cpu_fallback_operator_types": ["Shape"],
        "unclassified_cpu_events": 0,
        "undeclared_providers": [],
    }
    return {
        "runtime_inventory": {
            "gpu": {
                "name": "GPU",
                "uuid": "GPU-uuid",
                "compute_capability": "7.5",
            },
            "onnxruntime_gpu_version": "1.28.0",
            "onnxruntime_device": "GPU",
            "cuda_dlls_preloaded": True,
            "available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "declared_provider_order": [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        },
        "provider_assignment": {
            "policy": {
                "provider_order": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "classification_unit": "operator_type",
                "node_names_retained": False,
                "tensor_names_retained": False,
                "benchmark_specific_exceptions": False,
            },
            "source_fp32": profile,
            "candidate_int8_qdq": profile,
        },
        "benchmark": {
            "document_count": 256,
            "query_count": 81,
            "raw_embeddings_retained": False,
            "scientific_comparison_performed": False,
            "source_fp32": runs,
            "candidate_int8_qdq": runs,
        },
    }


def test_model_free_replay_matches_capture(tmp_path: Path) -> None:
    package, manifest_hash = write_cuda_preflight_package(
        _authority(), _runtime_result(), tmp_path / "package"
    )
    result = replay_cuda_preflight(package / "replay-bundle.json", manifest_hash)
    assert result["replay_status"] == "PASS"
    assert result["model_loaded"] is False
    assert result["onnx_graph_loaded"] is False


def test_replay_fails_closed_on_tampering(tmp_path: Path) -> None:
    package, manifest_hash = write_cuda_preflight_package(
        _authority(), _runtime_result(), tmp_path / "package"
    )
    benchmark = package / "benchmark-summary.json"
    payload = json.loads(benchmark.read_text(encoding="utf-8"))
    payload["document_count"] = 255
    benchmark.write_text(json.dumps(payload), encoding="utf-8")
    result = replay_cuda_preflight(package / "replay-bundle.json", manifest_hash)
    assert result["replay_status"] == "BLOCKED"
    assert result["artifact_integrity"] is False


def _rebind_artifact(package: Path, artifact_name: str) -> str:
    manifest_path = package / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["artifacts"]:
        if entry["path"] == artifact_name:
            entry["sha256"] = sha256_file(package / artifact_name)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return sha256_file(manifest_path)


def test_rewritten_manifest_cannot_replace_external_root(tmp_path: Path) -> None:
    package, trusted_hash = write_cuda_preflight_package(
        _authority(), _runtime_result(), tmp_path / "package"
    )
    benchmark = package / "benchmark-summary.json"
    benchmark.write_text('{"document_count":255}', encoding="utf-8")
    _rebind_artifact(package, benchmark.name)
    result = replay_cuda_preflight(package / "replay-bundle.json", trusted_hash)
    assert result["replay_status"] == "BLOCKED"
    assert result["artifact_integrity"] is False


def test_missing_benchmark_observation_fails_even_when_rebound(tmp_path: Path) -> None:
    package, _ = write_cuda_preflight_package(_authority(), _runtime_result(), tmp_path / "package")
    benchmark = package / "benchmark-summary.json"
    payload = json.loads(benchmark.read_text(encoding="utf-8"))
    del payload["source_fp32"][0]["documents"]
    benchmark.write_text(json.dumps(payload), encoding="utf-8")
    new_hash = _rebind_artifact(package, benchmark.name)
    result = replay_cuda_preflight(package / "replay-bundle.json", new_hash)
    assert result["replay_status"] == "BLOCKED"
    assert result["artifact_integrity"] is True
    assert result["benchmark_scope_match"] is False


def test_missing_operator_classification_fails_even_when_rebound(tmp_path: Path) -> None:
    package, _ = write_cuda_preflight_package(_authority(), _runtime_result(), tmp_path / "package")
    assignment = package / "provider-assignment.json"
    payload = json.loads(assignment.read_text(encoding="utf-8"))
    del payload["source_fp32"]["operator_event_counts"]
    assignment.write_text(json.dumps(payload), encoding="utf-8")
    new_hash = _rebind_artifact(package, assignment.name)
    result = replay_cuda_preflight(package / "replay-bundle.json", new_hash)
    assert result["replay_status"] == "BLOCKED"
    assert result["provider_assignment_match"] is False


def test_runtime_gpu_mismatch_fails_even_when_rebound(tmp_path: Path) -> None:
    package, _ = write_cuda_preflight_package(_authority(), _runtime_result(), tmp_path / "package")
    runtime = package / "runtime-inventory.json"
    payload = json.loads(runtime.read_text(encoding="utf-8"))
    payload["gpu"]["uuid"] = "different"
    runtime.write_text(json.dumps(payload), encoding="utf-8")
    new_hash = _rebind_artifact(package, runtime.name)
    result = replay_cuda_preflight(package / "replay-bundle.json", new_hash)
    assert result["replay_status"] == "BLOCKED"
    assert result["status_match"] is False


def test_invalid_utf8_replay_artifact_fails_closed(tmp_path: Path) -> None:
    package, _ = write_cuda_preflight_package(_authority(), _runtime_result(), tmp_path / "package")
    runtime = package / "runtime-inventory.json"
    runtime.write_bytes(b"\xff")
    new_hash = _rebind_artifact(package, runtime.name)
    result = replay_cuda_preflight(package / "replay-bundle.json", new_hash)
    assert result["replay_status"] == "BLOCKED"


def test_unlisted_sibling_bundle_fails_closed(tmp_path: Path) -> None:
    package, manifest_hash = write_cuda_preflight_package(
        _authority(), _runtime_result(), tmp_path / "package"
    )
    other = package / "unlisted-bundle.json"
    other.write_bytes((package / "replay-bundle.json").read_bytes())
    result = replay_cuda_preflight(other, manifest_hash)
    assert result["replay_status"] == "BLOCKED"
    assert result["artifact_integrity"] is False


def test_huge_json_number_fails_closed(tmp_path: Path) -> None:
    package, _ = write_cuda_preflight_package(_authority(), _runtime_result(), tmp_path / "package")
    benchmark = package / "benchmark-summary.json"
    payload = json.loads(benchmark.read_text(encoding="utf-8"))
    payload["source_fp32"][0]["elapsed_seconds"] = 10**400
    benchmark.write_text(json.dumps(payload), encoding="utf-8")
    manifest_hash = _rebind_artifact(package, benchmark.name)
    result = replay_cuda_preflight(package / "replay-bundle.json", manifest_hash)
    assert result["replay_status"] == "BLOCKED"


def test_boolean_batch_size_fails_closed(tmp_path: Path) -> None:
    package, _ = write_cuda_preflight_package(_authority(), _runtime_result(), tmp_path / "package")
    benchmark = package / "benchmark-summary.json"
    payload = json.loads(benchmark.read_text(encoding="utf-8"))
    payload["source_fp32"][0]["batch_size"] = True
    benchmark.write_text(json.dumps(payload), encoding="utf-8")
    manifest_hash = _rebind_artifact(package, benchmark.name)
    result = replay_cuda_preflight(package / "replay-bundle.json", manifest_hash)
    assert result["replay_status"] == "BLOCKED"
    assert result["benchmark_scope_match"] is False


def test_replay_rejects_nonfrozen_layout_even_when_records_match(tmp_path: Path) -> None:
    package, _ = write_cuda_preflight_package(_authority(), _runtime_result(), tmp_path / "package")
    authority_path = package / "cuda-authority.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["run_layout"][0]["label"] = "different"
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    _rebind_artifact(package, authority_path.name)
    benchmark_path = package / "benchmark-summary.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    for model in ("source_fp32", "candidate_int8_qdq"):
        benchmark[model][0]["label"] = "different"
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
    manifest_hash = _rebind_artifact(package, benchmark_path.name)
    result = replay_cuda_preflight(package / "replay-bundle.json", manifest_hash)
    assert result["replay_status"] == "BLOCKED"
    assert result["benchmark_scope_match"] is False


def test_boolean_authority_batch_size_fails_closed(tmp_path: Path) -> None:
    package, _ = write_cuda_preflight_package(_authority(), _runtime_result(), tmp_path / "package")
    authority_path = package / "cuda-authority.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["run_layout"][0]["batch_size"] = True
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    manifest_hash = _rebind_artifact(package, authority_path.name)
    result = replay_cuda_preflight(package / "replay-bundle.json", manifest_hash)
    assert result["replay_status"] == "BLOCKED"
    assert result["benchmark_scope_match"] is False


def test_replay_bundle_flags_are_required(tmp_path: Path) -> None:
    package, _ = write_cuda_preflight_package(_authority(), _runtime_result(), tmp_path / "package")
    bundle = package / "replay-bundle.json"
    payload = json.loads(bundle.read_text(encoding="utf-8"))
    payload["model_required_for_replay"] = True
    bundle.write_text(json.dumps(payload), encoding="utf-8")
    manifest_hash = _rebind_artifact(package, bundle.name)
    result = replay_cuda_preflight(bundle, manifest_hash)
    assert result["replay_status"] == "BLOCKED"


def test_boolean_replay_bundle_version_fails_closed(tmp_path: Path) -> None:
    package, _ = write_cuda_preflight_package(_authority(), _runtime_result(), tmp_path / "package")
    bundle = package / "replay-bundle.json"
    payload = json.loads(bundle.read_text(encoding="utf-8"))
    payload["version"] = True
    bundle.write_text(json.dumps(payload), encoding="utf-8")
    manifest_hash = _rebind_artifact(package, bundle.name)
    result = replay_cuda_preflight(bundle, manifest_hash)
    assert result["replay_status"] == "BLOCKED"
