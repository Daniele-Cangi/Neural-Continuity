from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_continuity.m1_diagnostics import (
    activation_analysis_authority,
)
from neural_continuity.m1_diagnostics.activation_analysis_authority import (
    ActivationAnalysisError,
    verify_activation_analysis_input,
)


def test_analysis_authority_rejects_model_requiring_source_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "replay-bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "replay_requires_model_execution": True,
                "capture_plan_path": "capture-plan.json",
                "batch_index_path": "batch-index.json",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        activation_analysis_authority,
        "replay_activation_capture",
        lambda *_args, **_kwargs: {
            "status": "COMPLETE",
            "activation_capture_status": "CAPTURED",
            "replay_verified": True,
            "status_match": True,
            "summary_match": True,
            "model_execution_used": False,
            "query_count": 364,
            "batch_count": 23,
            "floating_probe_count": 283,
            "integer_probe_count": 248,
        },
    )

    with pytest.raises(ActivationAnalysisError) as error:
        verify_activation_analysis_input(bundle, "a" * 64)
    assert error.value.code == "MODEL_EXECUTION_REQUIRED"
