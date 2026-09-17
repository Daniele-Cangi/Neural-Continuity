"""CUDA sentinel planning cannot grant execution or trust a broken preflight."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_continuity.m1_diagnostics import cuda_null_sentinel_readiness as readiness
from neural_continuity.m1_diagnostics import cuda_null_sentinel_readiness_cli as cli


def _valid_replay() -> dict[str, object]:
    return {
        "replay_status": "PASS",
        "technical_preflight_status": "PASS",
        "decision_match": True,
        "artifact_manifest_sha256": readiness.SOURCE_PREFLIGHT_MANIFEST_SHA256,
        "scientific_decision": "NOT_EVALUATED",
        "model_loaded": False,
        "onnx_graph_loaded": False,
        "run_timings": [
            {"role": role, "batch_size": batch}
            for role in ("documents", "measurement_null_queries")
            for batch in (1, 16, 64)
        ],
    }


def _arguments(bundle: Path) -> dict[str, object]:
    return {
        "config_path": readiness.CONFIG_PATH,
        "external_config_sha256": readiness.CONFIG_SHA256,
        "source_preflight_bundle": bundle,
        "external_source_manifest_sha256": readiness.SOURCE_PREFLIGHT_MANIFEST_SHA256,
    }


def test_valid_replay_still_requires_review_and_never_authorizes_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(readiness, "replay_source_preflight", lambda *_: _valid_replay())
    result = readiness.verify_cuda_sentinel_readiness(**_arguments(tmp_path / "bundle.json"))
    assert result["status"] == "SENTINEL_DESIGN_REVIEW_REQUIRED"
    assert result["source_preflight_replay_status"] == "PASS"
    assert result["raw_observation_bytes_before_overhead"] == 248463360
    assert result["execution_authorized"] is False
    assert result["sentinel_started"] is False
    assert result["scientific_decision"] == "NOT_EVALUATED"


def test_missing_source_bundle_blocks(tmp_path: Path) -> None:
    with pytest.raises(readiness.CudaNullSentinelReadinessBlocked, match="replay did not verify"):
        readiness.verify_cuda_sentinel_readiness(**_arguments(tmp_path / "missing.json"))


def test_external_root_mismatch_blocks_before_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        readiness,
        "replay_source_preflight",
        lambda *_: pytest.fail("replay ran after external root mismatch"),
    )
    arguments = _arguments(tmp_path / "bundle.json")
    arguments["external_source_manifest_sha256"] = "0" * 64
    with pytest.raises(readiness.CudaNullSentinelReadinessBlocked, match="manifest SHA-256"):
        readiness.verify_cuda_sentinel_readiness(**arguments)


def test_incomplete_run_coverage_blocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    replay = _valid_replay()
    runs = replay["run_timings"]
    assert isinstance(runs, list)
    replay["run_timings"] = runs[:-1]
    monkeypatch.setattr(readiness, "replay_source_preflight", lambda *_: replay)
    with pytest.raises(readiness.CudaNullSentinelReadinessBlocked, match="run coverage"):
        readiness.verify_cuda_sentinel_readiness(**_arguments(tmp_path / "bundle.json"))


def test_cli_reports_blocked_without_execution(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def blocked(**_kwargs: object) -> dict[str, object]:
        raise readiness.CudaNullSentinelReadinessBlocked("synthetic missing artifact")

    monkeypatch.setattr(cli, "verify_cuda_sentinel_readiness", blocked)
    assert (
        cli.main(
            [
                "--config",
                str(readiness.CONFIG_PATH),
                "--config-sha256",
                readiness.CONFIG_SHA256,
                "--source-preflight-bundle",
                "missing.json",
                "--source-manifest-sha256",
                readiness.SOURCE_PREFLIGHT_MANIFEST_SHA256,
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out)["status"] == "BLOCKED"
