from __future__ import annotations

from pathlib import Path

import pytest

from neural_continuity.m1_diagnostics import cuda_null_full_budget_replay as budget
from neural_continuity.m1_diagnostics import cuda_null_full_segment_capture as capture
from neural_continuity.m1_diagnostics import cuda_null_sentinel_postgate_package as gate


def test_budget_replay_rejects_linked_package_before_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(budget, "has_linked_ancestor", lambda _path: True)
    result = budget.replay_budget_package(tmp_path / "replay-bundle.json", "a" * 64)
    assert result["replay_status"] == "BLOCKED"


def test_gate_replay_rejects_linked_package_before_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(gate, "has_linked_ancestor", lambda _path: True)
    result = gate.replay_sentinel_postgate(tmp_path / "replay-bundle.json", "a" * 64)
    assert result["replay_status"] == "BLOCKED"


def test_segment_capture_rejects_linked_child_target_before_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging = tmp_path / "staging"
    scratch = tmp_path / "scratch"
    staging.mkdir()
    scratch.mkdir()
    monkeypatch.setattr(
        capture,
        "has_linked_ancestor",
        lambda path: path not in {staging, scratch},
    )
    monkeypatch.setattr(
        capture,
        "_profiled_session",
        lambda *_args: pytest.fail("session must not be created for a linked target"),
    )
    with pytest.raises(capture.FullSegmentCaptureBlocked):
        capture._capture_profiled_role(
            object(),
            tmp_path / "unused.onnx",
            ids=["d1"],
            texts=["document"],
            run_label="batch_1_primary",
            role="documents",
            batch_size=1,
            staging=staging,
            scratch=scratch,
        )
