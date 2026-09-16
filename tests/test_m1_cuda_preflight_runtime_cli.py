import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from neural_continuity.m1_diagnostics import cuda_preflight_cli, cuda_preflight_runtime
from neural_continuity.m1_diagnostics.cuda_preflight_authority import CudaPreflightBlocked
from neural_continuity.m1_diagnostics.cuda_preflight_runtime import _profile_summary


def test_malformed_profile_node_fails_closed(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps([{"cat": "Node", "args": {}}]), encoding="utf-8")
    with pytest.raises(CudaPreflightBlocked, match="lacks a provider"):
        _profile_summary(profile, ("CUDAExecutionProvider", "CPUExecutionProvider"))


def test_runtime_exception_becomes_blocked(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cuda_preflight_cli, "verify_cuda_preflight_authority", lambda **_kwargs: object()
    )

    def failing_runtime(*_args: object) -> None:
        raise RuntimeError("session creation failed")

    monkeypatch.setattr(cuda_preflight_cli, "run_cuda_preflight", failing_runtime)
    result = cuda_preflight_cli.main(
        [
            "capture",
            "--config",
            "config.yaml",
            "--config-sha256",
            "0" * 64,
            "--dataset-directory",
            "dataset",
            "--transition-a-bundle",
            "transition-a.json",
            "--extension-plan-bundle",
            "extension.json",
            "--extension-plan-manifest-sha256",
            "1" * 64,
            "--candidate-package",
            "candidate",
            "--output",
            "output",
        ]
    )
    assert result == 2
    response = json.loads(capsys.readouterr().out)
    assert response["preflight_status"] == "BLOCKED"
    assert response["scientific_decision"] == "NOT_EVALUATED"


def _install_fake_ort(
    monkeypatch: pytest.MonkeyPatch,
    module_version: str = "1.28.0",
    reverse_session_providers: bool = False,
) -> list[list[str]]:
    created_providers: list[list[str]] = []

    class FakeSession:
        def __init__(self, _path: str, sess_options: object, providers: list[str]) -> None:
            self.profile_path = Path(sess_options.profile_file_prefix + ".json")  # type: ignore[attr-defined]
            self.providers = list(providers)
            created_providers.append(list(providers))

        def get_providers(self) -> list[str]:
            if reverse_session_providers:
                return list(reversed(self.providers))
            return self.providers

        def end_profiling(self) -> str:
            self.profile_path.write_text(
                json.dumps(
                    [
                        {
                            "cat": "Node",
                            "args": {
                                "provider": "CUDAExecutionProvider",
                                "op_name": "MatMul",
                            },
                        },
                        {
                            "cat": "Node",
                            "args": {
                                "provider": "CPUExecutionProvider",
                                "op_name": "Shape",
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )
            return str(self.profile_path)

    fake_ort = SimpleNamespace(
        __version__=module_version,
        preload_dlls=lambda: None,
        get_available_providers=lambda: [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
        get_device=lambda: "GPU",
        SessionOptions=lambda: SimpleNamespace(),
        InferenceSession=FakeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(
        cuda_preflight_runtime,
        "_gpu_inventory",
        lambda: {
            "name": "GPU",
            "uuid": "GPU-uuid",
            "compute_capability": "7.5",
        },
    )
    monkeypatch.setattr(
        cuda_preflight_runtime,
        "_distribution_version",
        lambda name: "1.28.0" if name == "onnxruntime-gpu" else "test-version",
    )
    return created_providers


def _runtime_authority(tmp_path: Path) -> SimpleNamespace:
    source = tmp_path / "source.onnx"
    candidate = tmp_path / "candidate.onnx"
    source.write_bytes(b"source")
    candidate.write_bytes(b"candidate")
    return SimpleNamespace(
        base=SimpleNamespace(
            config={},
            source=SimpleNamespace(artifact_path=source),
            selected_document_texts=("doc one", "doc two"),
            selected_document_ids=("d1", "d2"),
            query_texts=("query",),
            query_ids=("q1",),
        ),
        candidate=SimpleNamespace(artifact_path=candidate),
        provider_order=("CUDAExecutionProvider", "CPUExecutionProvider"),
        expected_gpu={
            "name": "GPU",
            "uuid": "GPU-uuid",
            "compute_capability": "7.5",
        },
        expected_onnxruntime_version="1.28.0",
        run_layout=(
            ("batch_1_primary", 1),
            ("batch_16_primary", 16),
            ("batch_16_repeat", 16),
            ("batch_64_primary", 64),
        ),
    )


def test_bounded_runtime_capture_with_mocked_ort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created_providers = _install_fake_ort(monkeypatch)
    authority = _runtime_authority(tmp_path)
    calls: list[tuple[str, int, int]] = []
    monkeypatch.setattr(cuda_preflight_runtime, "_load_teacher", lambda _config: (object(), {}))

    def fake_encode(
        _teacher: object,
        session: object,
        texts: tuple[str, ...],
        batch_size: int,
        label: str,
    ) -> np.ndarray:
        calls.append((label, len(texts), batch_size))
        return np.full((len(texts), 384), 0.05, dtype=np.float32)

    monkeypatch.setattr(cuda_preflight_runtime, "encode_onnx_source", fake_encode)
    result = cuda_preflight_runtime.run_cuda_preflight(authority, tmp_path / "working")
    assert created_providers == [list(authority.provider_order)] * 2
    assert result["runtime_inventory"]["onnxruntime_gpu_version"] == "1.28.0"
    for model in ("source_fp32", "candidate_int8_qdq"):
        assignment = result["provider_assignment"][model]
        assert assignment["provider_event_counts"] == {
            "CUDAExecutionProvider": 1,
            "CPUExecutionProvider": 1,
        }
        assert assignment["cpu_fallback_operator_types"] == ["Shape"]
        runs = result["benchmark"][model]
        assert len(runs) == 4
        assert all(len(run["documents"]["sha256"]) == 64 for run in runs)
        assert all(len(run["queries"]["sha256"]) == 64 for run in runs)
    assert len(calls) == 16
    assert result["benchmark"]["raw_embeddings_retained"] is False


def test_imported_ort_module_version_must_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_ort(monkeypatch, module_version="0.0.0")
    with pytest.raises(CudaPreflightBlocked, match="imported module 0.0.0"):
        cuda_preflight_runtime.capture_runtime_inventory(_runtime_authority(tmp_path))


def test_session_provider_order_must_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_ort(monkeypatch, reverse_session_providers=True)
    with pytest.raises(CudaPreflightBlocked, match="provider order differs"):
        cuda_preflight_runtime._profiled_session(
            tmp_path / "source.onnx",
            tmp_path / "profile",
            ("CUDAExecutionProvider", "CPUExecutionProvider"),
        )


@pytest.mark.parametrize("content", [b"\xff", b"null"])
def test_invalid_profile_payload_fails_closed(tmp_path: Path, content: bytes) -> None:
    profile = tmp_path / "profile.json"
    profile.write_bytes(content)
    with pytest.raises(CudaPreflightBlocked):
        _profile_summary(profile, ("CUDAExecutionProvider", "CPUExecutionProvider"))


def test_temporary_directory_error_becomes_blocked(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cuda_preflight_cli, "verify_cuda_preflight_authority", lambda **_kwargs: object()
    )

    def failing_temporary_directory(**_kwargs: object) -> None:
        raise PermissionError("temporary storage unavailable")

    monkeypatch.setattr(
        cuda_preflight_cli.tempfile, "TemporaryDirectory", failing_temporary_directory
    )
    result = cuda_preflight_cli.main(
        [
            "capture",
            "--config",
            "config.yaml",
            "--config-sha256",
            "0" * 64,
            "--dataset-directory",
            "dataset",
            "--transition-a-bundle",
            "transition-a.json",
            "--extension-plan-bundle",
            "extension.json",
            "--extension-plan-manifest-sha256",
            "1" * 64,
            "--candidate-package",
            "candidate",
            "--output",
            "output",
        ]
    )
    assert result == 2
    assert json.loads(capsys.readouterr().out)["preflight_status"] == "BLOCKED"
