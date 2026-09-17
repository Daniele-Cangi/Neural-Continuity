"""Public source-preflight CLI preserves technical status and exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_continuity.m1_diagnostics import cuda_null_source_preflight_cli as cli
from neural_continuity.m1_diagnostics import cuda_null_source_preflight_runtime as runtime
from neural_continuity.m1_diagnostics.cuda_null_runtime_authority import CudaNullRuntimeBlocked
from neural_continuity.m1_diagnostics.cuda_null_source_preflight_authority import (
    CudaNullSourcePreflightBlocked,
)
from neural_continuity.m1_diagnostics.cuda_preflight_authority import CudaPreflightBlocked


def _capture_args(root: Path) -> list[str]:
    arguments = ["capture"]
    for name in (
        "authorization-spec",
        "preflight-spec",
        "static-bundle",
        "config",
        "dataset",
        "transition-a-bundle",
        "teacher-snapshot",
        "cpu-extension-bundle",
        "historical-cuda-bundle",
        "output",
    ):
        arguments.extend((f"--{name}", str(root / name)))
    for name in (
        "reviewed-spec-sha256",
        "preflight-spec-sha256",
        "static-manifest-sha256",
        "config-sha256",
    ):
        arguments.extend((f"--{name}", "a" * 64))
    return arguments


@pytest.mark.parametrize(("status", "exit_code"), [("PASS", 0), ("BLOCKED", 2)])
def test_replay_cli_status_and_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    status: str,
    exit_code: int,
) -> None:
    monkeypatch.setattr(cli, "replay_source_preflight", lambda *_: {"replay_status": status})
    result = cli.main(
        [
            "replay",
            "--bundle",
            str(tmp_path / "replay-bundle.json"),
            "--artifact-manifest-sha256",
            "a" * 64,
        ]
    )
    assert result == exit_code
    assert json.loads(capsys.readouterr().out)["replay_status"] == status


@pytest.mark.parametrize(
    "error_type",
    [CudaNullSourcePreflightBlocked, CudaNullRuntimeBlocked, CudaPreflightBlocked],
)
def test_capture_prerequisites_report_blocked(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    error_type: type[Exception],
) -> None:
    def blocked(**_kwargs: object) -> None:
        raise error_type("frozen prerequisite missing")

    monkeypatch.setattr(runtime, "capture_source_preflight", blocked)
    assert cli.main(_capture_args(tmp_path)) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "BLOCKED"


def test_capture_unexpected_failure_remains_execution_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    def failed(**_kwargs: object) -> None:
        raise RuntimeError("synthetic GPU execution error")

    monkeypatch.setattr(runtime, "capture_source_preflight", failed)
    assert cli.main(_capture_args(tmp_path)) == 3
    assert json.loads(capsys.readouterr().out)["status"] == "EXECUTION_ERROR"


def test_capture_success_reports_model_free_replay(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr(runtime, "capture_source_preflight", lambda **_: {"stub": True})
    monkeypatch.setattr(
        cli,
        "write_source_preflight_package",
        lambda *_: (tmp_path / "run", "a" * 64),
    )
    monkeypatch.setattr(
        cli,
        "replay_source_preflight",
        lambda *_: {"replay_status": "PASS", "run_timings": []},
    )
    assert cli.main(_capture_args(tmp_path)) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "TECHNICAL_PREFLIGHT_PASS"
    assert result["replay_status"] == "PASS"
    assert result["scientific_decision"] == "NOT_EVALUATED"
