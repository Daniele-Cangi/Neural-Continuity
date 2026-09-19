from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from neural_continuity.m1_diagnostics import cuda_null_full_budget_authority as authority
from neural_continuity.m1_diagnostics import cuda_null_full_budget_capture as capture
from neural_continuity.m1_diagnostics import cuda_null_full_budget_replay as replay


def test_budget_authority_requires_external_hash() -> None:
    with pytest.raises(authority.FullBudgetBlocked, match="SHA-256"):
        authority.load_budget_spec("not-a-hash")


def test_budget_replay_fails_closed_on_missing_package(tmp_path: Path) -> None:
    result = replay.replay_budget_package(tmp_path / "replay-bundle.json", "a" * 64)
    assert result["replay_status"] == "BLOCKED"
    assert result["model_loaded"] is False


def test_budget_record_rejects_missing_profile() -> None:
    with pytest.raises(ValueError, match="scope"):
        replay.validate_budget_record({"profiles": []})


def test_failed_v1_spec_cannot_authorize_capture() -> None:
    with pytest.raises(ValueError):
        authority.load_budget_spec(authority.FAILED_V1_SPEC_SHA256)


def test_full_input_timing_profiles_only_bounded_sample(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[int] = []

    class Session:
        def get_providers(self) -> list[str]:
            return list(capture.PROVIDERS)

        def end_profiling(self) -> str:
            return str(tmp_path / "profile.json")

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(InferenceSession=lambda *_args, **_kwargs: Session()),
    )
    monkeypatch.setattr(capture, "_profiled_session", lambda *_args: Session())
    monkeypatch.setattr(
        capture,
        "encode_onnx_source",
        lambda _teacher, _session, texts, _batch_size, _label: (
            seen.append(len(texts)) or np.empty((len(texts), 384), dtype=np.float32)
        ),
    )
    monkeypatch.setattr(capture, "_profile_summary", lambda *_args: {})
    result = capture._timed_profile(
        object(),
        tmp_path / "source.onnx",
        ["x"] * 5183,
        "batch_1_primary.documents",
        1,
        tmp_path,
    )
    assert seen == [5183, 64]
    assert result["profiled_item_count"] == 64
    assert result["full_input_profiled"] is False
