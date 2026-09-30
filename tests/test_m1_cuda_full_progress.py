from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_continuity.m1_diagnostics import cuda_null_full_progress as progress
from neural_continuity.m1_diagnostics import cuda_null_full_runner as runner


@pytest.fixture(autouse=True)
def isolated_progress():
    progress.configure_progress()
    yield
    progress.configure_progress()


def test_phase_percentage_eta_and_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(progress, "monotonic", lambda: clock[0])
    path = tmp_path / "progress.json"
    progress.configure_progress(path)
    progress.report_progress("corpus_replay", "started")
    initial = json.loads(path.read_text())
    assert initial["percent"] == 0
    assert initial["phase_eta_seconds"] is None
    clock[0] = 100
    progress.report_progress("corpus_replay", "epoch_completed", epoch=1)
    first = json.loads(path.read_text())
    assert first["completed_epochs"] == 1
    assert first["phase_eta_seconds"] == 11900
    assert first["process_id"] > 0
    clock[0] = 200
    progress.report_progress("corpus_replay", "epoch_completed", epoch=2)
    progress.report_progress("corpus_replay", "epoch_completed", epoch=1)
    latest = json.loads(path.read_text())
    assert latest["completed_epochs"] == 2
    assert latest["percent"] == 1.67
    assert latest["phase_elapsed_seconds"] == 200
    assert latest["phase_eta_seconds"] == 11800
    assert "2/120" in latest["message"]
    assert not list(tmp_path.glob("*.tmp"))
    progress.report_progress("final_anchor", "completed")
    assert json.loads(path.read_text())["state"] == "completed"


def test_new_phase_and_restarted_phase_have_separate_counters(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    progress.configure_progress(path)
    progress.report_progress("corpus_aggregation", "epoch_completed", epoch=120)
    progress.report_progress("corpus_replay", "started")
    assert json.loads(path.read_text())["completed_epochs"] == 0
    progress.report_progress("corpus_replay", "epoch_completed", epoch=1)
    progress.report_progress("corpus_replay", "started")
    assert json.loads(path.read_text())["completed_epochs"] == 0


def test_unrelated_existing_file_is_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    path.write_text('{"kind":"research_evidence"}')
    original = path.read_bytes()
    progress.configure_progress(path)
    progress.report_progress("corpus_replay", "started")
    assert path.read_bytes() == original


def test_snapshot_failure_preserves_previous_status_without_stopping_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "progress.json"
    progress.configure_progress(path)
    progress.report_progress("corpus_replay", "started")
    original = path.read_bytes()

    def fail(*args: object) -> None:
        raise OSError("test unavailable")

    monkeypatch.setattr(progress.os, "replace", fail)
    progress.report_progress("corpus_replay", "epoch_completed", epoch=1)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_journal_reports_coverage_without_extrapolating_phase_eta(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    progress.configure_progress(path)
    progress.report_progress("journal_epoch_replay", "epoch_completed", epoch=1)
    assert json.loads(path.read_text())["phase_eta_seconds"] is None
    progress.report_progress("controller", "interrupted")
    assert json.loads(path.read_text())["state"] == "interrupted"


def test_closed_terminal_does_not_stop_snapshot_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ClosedTerminal:
        def write(self, text: str) -> None:
            raise OSError("terminal closed")

        def flush(self) -> None:
            raise OSError("terminal closed")

    path = tmp_path / "progress.json"
    progress.configure_progress(path)
    monkeypatch.setattr(progress.sys, "stderr", ClosedTerminal())
    progress.report_progress("corpus_replay", "epoch_completed", epoch=1)
    assert json.loads(path.read_text())["completed_epochs"] == 1


@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt])
def test_cli_records_failure_or_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[BaseException]
) -> None:
    path = tmp_path / "progress.json"
    progress.configure_progress(path)

    def fail(*args: object, **kwargs: object) -> None:
        raise error("test failure")

    monkeypatch.setattr(runner, "run_full_corpus", fail)
    monkeypatch.setattr(runner.sys, "argv", ["runner", "resume", "--authority-sha256", "a" * 64])
    with pytest.raises(error):
        runner.main()
    expected = "interrupted" if error is KeyboardInterrupt else "error"
    assert json.loads(path.read_text())["state"] == expected
