from __future__ import annotations

from pathlib import Path

import pytest

from neural_continuity.m1_diagnostics.authority import DiagnosticPreflightError
from neural_continuity.m1_diagnostics.static_package import (
    verify_static_preflight_package,
)


def test_static_manifest_mismatch_blocks_before_artifact_loading(tmp_path: Path) -> None:
    (tmp_path / "artifact-manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(DiagnosticPreflightError) as captured:
        verify_static_preflight_package(tmp_path)

    assert captured.value.status == "BLOCKED"
    assert captured.value.code == "STATIC_MANIFEST_HASH_MISMATCH"
