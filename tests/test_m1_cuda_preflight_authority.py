from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from neural_continuity.evidence import sha256_file
from neural_continuity.m1_diagnostics import cuda_preflight_authority as authority_module
from neural_continuity.m1_diagnostics.cuda_preflight_authority import (
    CudaPreflightBlocked,
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
            "expected_quantization": {"format": "QDQ", "per_channel": True},
        },
        "provider_policy": {
            "ordered_providers": [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            "require_cuda_activity": True,
            "classify_cpu_fallback_by_operator_type": True,
            "reject_undeclared_providers": True,
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
    quantization.write_text('{"format":"QDQ","per_channel":true}', encoding="utf-8")
    manifest = candidate / "candidate-manifest.json"
    manifest.write_text(
        '{"artifact":"teacher-int8-qdq.onnx","sha256":"' + sha256_file(model) + '"}',
        encoding="utf-8",
    )
    source = tmp_path / "teacher.onnx"
    source.write_bytes(b"source")
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


def test_authority_verifies_before_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, candidate = _fixture(tmp_path, monkeypatch)
    authority = verify_cuda_preflight_authority(
        config,
        tmp_path,
        tmp_path / "transition-a.json",
        tmp_path / "extension.json",
        "extension-hash",
        candidate,
    )
    assert authority.provider_order == (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    assert authority.candidate.artifact_sha256 == sha256_file(candidate / "teacher-int8-qdq.onnx")


def test_candidate_tampering_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, candidate = _fixture(tmp_path, monkeypatch)
    (candidate / "teacher-int8-qdq.onnx").write_bytes(b"tampered")
    with pytest.raises(CudaPreflightBlocked, match="SHA-256 mismatch"):
        verify_cuda_preflight_authority(
            config,
            tmp_path,
            tmp_path / "transition-a.json",
            tmp_path / "extension.json",
            "extension-hash",
            candidate,
        )
