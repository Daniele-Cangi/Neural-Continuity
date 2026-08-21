from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neural_continuity.m1_diagnostics.fidelity_authority import FidelityGateError
from neural_continuity.m1_diagnostics.fidelity_evidence import (
    replay_fidelity,
    write_fidelity_package,
)


def _outputs() -> dict[str, np.ndarray]:
    source = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    target = np.asarray([[0.5, 0.5], [0.25, 0.75]], dtype=np.float32)
    return {
        "source_original": source,
        "source_instrumented": source.copy(),
        "target_original": target,
        "target_instrumented": target.copy(),
    }


def _capture(root: Path, outputs: dict[str, np.ndarray]) -> dict[str, object]:
    return write_fidelity_package(
        root,
        ["q-001", "q-002"],
        outputs,
        {"authority_test": True},
        {"execution_test": True},
    )


def test_fidelity_package_is_deterministic_and_replays_without_model(tmp_path: Path) -> None:
    first = _capture(tmp_path / "first", _outputs())
    second = _capture(tmp_path / "second", _outputs())
    assert first["status"] == "COMPLETE"
    assert first["artifact_manifest_sha256"] == second["artifact_manifest_sha256"]

    replay = replay_fidelity(
        tmp_path / "first" / "replay-bundle.json",
        str(first["artifact_manifest_sha256"]),
    )
    assert replay["status"] == "COMPLETE"
    assert replay["fidelity_status"] == "PASS"
    assert replay["replay_verified"] is True
    assert replay["model_execution_used"] is False


def test_fidelity_mismatch_is_blocked_and_replay_verified(tmp_path: Path) -> None:
    outputs = _outputs()
    outputs["target_instrumented"][0, 0] = np.float32(0.5001)
    captured = _capture(tmp_path / "blocked", outputs)
    assert captured["status"] == "BLOCKED"
    assert captured["fidelity_status"] == "BLOCKED"

    replay = replay_fidelity(
        tmp_path / "blocked" / "replay-bundle.json",
        str(captured["artifact_manifest_sha256"]),
    )
    assert replay["status"] == "BLOCKED"
    assert replay["comparison_match"] is True


def test_fidelity_replay_fails_closed_when_artifact_is_missing(tmp_path: Path) -> None:
    captured = _capture(tmp_path / "missing", _outputs())
    (tmp_path / "missing" / "final-output-observations.npz").unlink()
    with pytest.raises(FidelityGateError) as error:
        replay_fidelity(
            tmp_path / "missing" / "replay-bundle.json",
            str(captured["artifact_manifest_sha256"]),
        )
    assert error.value.code == "ARTIFACT_MISSING"
    assert error.value.status == "BLOCKED"
