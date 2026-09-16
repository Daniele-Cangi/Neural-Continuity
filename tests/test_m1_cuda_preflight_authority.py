import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from neural_continuity.evidence import sha256_file
from neural_continuity.m1_diagnostics import cuda_preflight_authority as authority_module
from neural_continuity.m1_diagnostics.cuda_preflight_authority import (
    CudaPreflightBlocked,
    _load_preflight_config,
    verify_cuda_preflight_authority,
)


def _write_config(path: Path, candidate: Path, source_hash: str) -> None:
    config = {
        "scope": {
            "qualifying_m1_evidence": False,
            "full_corpus_authorized": False,
            "scientific_decision": "NOT_EVALUATED",
        },
        "base_authority": {
            "config_path": "base.yaml",
            "authority_sha256": "base-authority",
            "source_artifact_sha256": source_hash,
        },
        "candidate": {
            "manifest_file": "candidate-manifest.json",
            "manifest_sha256": sha256_file(candidate / "candidate-manifest.json"),
            "quantization_config_file": "quantization-config.json",
            "quantization_config_sha256": sha256_file(candidate / "quantization-config.json"),
            "artifact_file": "teacher-int8-qdq.onnx",
            "artifact_sha256": sha256_file(candidate / "teacher-int8-qdq.onnx"),
            "expected_quantization": {
                "quantization_format": "QDQ",
                "per_channel": True,
            },
        },
        "provider_policy": {
            "ordered_providers": [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            "require_cuda_activity": True,
            "classify_cpu_fallback_by_operator_type": True,
            "reject_undeclared_providers": True,
            "node_name_exceptions": False,
            "tensor_name_exceptions": False,
            "benchmark_specific_exceptions": False,
        },
        "runtime": {
            "onnxruntime_version": "1.28.0",
            "expected_gpu": {
                "name": "GPU",
                "uuid": "GPU-uuid",
                "compute_capability": "7.5",
            },
        },
        "document_count": 2,
        "query_count": 1,
        "benchmark": {
            "runs": [
                {"label": "batch_1_primary", "batch_size": 1},
                {"label": "batch_16_primary", "batch_size": 16},
                {"label": "batch_16_repeat", "batch_size": 16},
                {"label": "batch_64_primary", "batch_size": 64},
            ]
        },
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    model = candidate / "teacher-int8-qdq.onnx"
    model.write_bytes(b"candidate")
    quantization = candidate / "quantization-config.json"
    source = tmp_path / "teacher.onnx"
    source.write_bytes(b"source")
    quantization_data = {"quantization_format": "QDQ", "per_channel": True}
    quantization.write_text(json.dumps(quantization_data), encoding="utf-8")
    manifest = candidate / "candidate-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "package_kind": "m1_onnx_int8_static_qdq_candidate",
                "candidate_status": "CAPTURED_PENDING_OBSERVATION",
                "source_identity": {"onnx_fp32_artifact_sha256": sha256_file(source)},
                "quantization_configuration": quantization_data,
                "artifacts": [
                    {"path": model.name, "sha256": sha256_file(model)},
                    {"path": quantization.name, "sha256": sha256_file(quantization)},
                ],
            }
        ),
        encoding="utf-8",
    )
    base = SimpleNamespace(
        authority_sha256="base-authority",
        source=SimpleNamespace(
            artifact_path=source,
            artifact_sha256=sha256_file(source),
        ),
        selected_document_ids=("d1", "d2"),
        selected_document_texts=("one", "two"),
        query_ids=("q1",),
        query_texts=("query",),
        qrels={"q1": ("d1",)},
    )
    monkeypatch.setattr(authority_module, "verify_sentinel_authority", lambda **_kwargs: base)
    config = tmp_path / "cuda.yaml"
    _write_config(config, candidate, sha256_file(source))
    return config, candidate


def _verify(
    config: Path, candidate: Path, tmp_path: Path, expected_sha256: str | None = None
) -> object:
    return verify_cuda_preflight_authority(
        config,
        tmp_path,
        tmp_path / "transition-a.json",
        tmp_path / "extension.json",
        "extension-hash",
        candidate,
        expected_sha256 or _load_preflight_config(config)[1],
    )


def test_authority_verifies_before_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, candidate = _fixture(tmp_path, monkeypatch)
    authority = _verify(config, candidate, tmp_path)
    assert authority.provider_order == (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    assert authority.candidate.artifact_sha256 == sha256_file(candidate / "teacher-int8-qdq.onnx")


def test_candidate_tampering_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, candidate = _fixture(tmp_path, monkeypatch)
    (candidate / "teacher-int8-qdq.onnx").write_bytes(b"tampered")
    with pytest.raises(CudaPreflightBlocked, match="SHA-256 mismatch"):
        _verify(config, candidate, tmp_path)


def test_config_hash_is_external_authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, candidate = _fixture(tmp_path, monkeypatch)
    with pytest.raises(CudaPreflightBlocked, match="config SHA-256 mismatch"):
        _verify(config, candidate, tmp_path, "0" * 64)


def test_candidate_path_cannot_escape_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, candidate = _fixture(tmp_path, monkeypatch)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["candidate"]["artifact_file"] = "../teacher.onnx"
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(CudaPreflightBlocked, match="package-relative"):
        _verify(config, candidate, tmp_path)


def test_manifest_artifact_record_is_structural(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, candidate = _fixture(tmp_path, monkeypatch)
    manifest_path = candidate / "candidate-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["path"] = "unrelated.onnx"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _write_config(config, candidate, sha256_file(tmp_path / "teacher.onnx"))
    with pytest.raises(CudaPreflightBlocked, match="does not bind"):
        _verify(config, candidate, tmp_path)


def test_quantization_value_is_checked_by_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, candidate = _fixture(tmp_path, monkeypatch)
    quantization_path = candidate / "quantization-config.json"
    quantization = json.loads(quantization_path.read_text(encoding="utf-8"))
    quantization["per_channel"] = False
    quantization_path.write_text(json.dumps(quantization), encoding="utf-8")
    manifest_path = candidate / "candidate-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["quantization_configuration"] = quantization
    manifest["artifacts"][1]["sha256"] = sha256_file(quantization_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _write_config(config, candidate, sha256_file(tmp_path / "teacher.onnx"))
    with pytest.raises(CudaPreflightBlocked, match="quantization per_channel mismatch"):
        _verify(config, candidate, tmp_path)


def test_invalid_utf8_manifest_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, candidate = _fixture(tmp_path, monkeypatch)
    (candidate / "candidate-manifest.json").write_bytes(b"\xff")
    _write_config(config, candidate, sha256_file(tmp_path / "teacher.onnx"))
    with pytest.raises(CudaPreflightBlocked, match="not valid JSON"):
        _verify(config, candidate, tmp_path)


def test_provider_exceptions_are_forbidden(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, candidate = _fixture(tmp_path, monkeypatch)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["provider_policy"]["node_name_exceptions"] = True
    config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(CudaPreflightBlocked, match="node_name_exceptions"):
        _verify(config, candidate, tmp_path)
