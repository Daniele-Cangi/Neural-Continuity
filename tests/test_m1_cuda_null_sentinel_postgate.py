from __future__ import annotations

from pathlib import Path

import pytest

from neural_continuity.m1_diagnostics import cuda_null_sentinel_postgate as gate
from neural_continuity.m1_diagnostics import cuda_null_sentinel_postgate_package as package


def test_postgate_requires_external_tip_before_evidence_access() -> None:
    with pytest.raises(gate.SentinelPostgateBlocked, match="tip"):
        gate._verify_roots(Path("unused"), "bad", "a" * 64)


def test_postgate_package_replays_and_rejects_tampering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = {"status": "TECHNICAL_GATE_PASS_NO_SCIENTIFIC_RELEASE", "epoch_count": 120}
    monkeypatch.setattr(package, "build_sentinel_postgate_report", lambda **_kwargs: expected)
    monkeypatch.setattr(package, "_external_output_directory", lambda path: path)
    output, digest = package.write_sentinel_postgate(
        output=tmp_path / "gate",
        root=tmp_path / "source",
        external_tip_sha256="a" * 64,
        external_authority_sha256="b" * 64,
    )
    assert (
        package.replay_sentinel_postgate(output / "replay-bundle.json", digest)["replay_status"]
        == "PASS"
    )
    (output / "technical-gate-report.json").write_text("{}", encoding="utf-8")
    assert (
        package.replay_sentinel_postgate(output / "replay-bundle.json", digest)["replay_status"]
        == "BLOCKED"
    )
