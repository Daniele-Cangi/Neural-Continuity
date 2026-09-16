import json
from pathlib import Path

import pytest

from neural_continuity.m1_diagnostics import cuda_preflight_cli
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
