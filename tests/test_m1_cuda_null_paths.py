"""Deterministic path-boundary checks without symlink creation privileges."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from neural_continuity.m1_diagnostics import cuda_null_paths as paths_module
from neural_continuity.m1_diagnostics.cuda_null_authority import _snapshot_target_is_allowed


def test_windows_reparse_attribute_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        paths_module,
        "lstat",
        lambda _: SimpleNamespace(st_file_attributes=paths_module._REPARSE_POINT),
    )
    assert paths_module._windows_reparse(tmp_path)


def test_snapshot_root_reparse_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_root = tmp_path / "models--sentence-transformers--all-MiniLM-L6-v2"
    snapshot = model_root / "snapshots" / "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    snapshot.mkdir(parents=True)
    monkeypatch.setattr(paths_module, "_WINDOWS", True)
    monkeypatch.setattr(paths_module, "_windows_reparse", lambda path: path == model_root)
    assert not _snapshot_target_is_allowed(snapshot, snapshot / "tokenizer.json")


def test_snapshot_inventory_reports_extra_files(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert paths_module.snapshot_file_inventory(snapshot) == {"tokenizer.json"}
    (snapshot / "unexpected.json").write_text("{}", encoding="utf-8")
    assert paths_module.snapshot_file_inventory(snapshot) == {
        "tokenizer.json",
        "unexpected.json",
    }
