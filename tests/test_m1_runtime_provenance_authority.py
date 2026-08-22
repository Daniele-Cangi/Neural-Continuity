from __future__ import annotations

import pytest

from neural_continuity.m1_diagnostics.runtime_provenance_authority import (
    RuntimeProvenanceError,
    _validate_stage0_replay,
)


def test_stage0_authority_requires_model_free_blocked_replay() -> None:
    with pytest.raises(RuntimeProvenanceError) as error:
        _validate_stage0_replay(
            {
                "replay_verified": True,
                "status": "PASS",
                "stage_1_execution_started": False,
                "model_execution_used": False,
            }
        )
    assert error.value.code == "RUNTIME_PROVENANCE_STAGE0_STATUS_INVALID"


def test_stage0_authority_rejects_started_stage1() -> None:
    with pytest.raises(RuntimeProvenanceError) as error:
        _validate_stage0_replay(
            {
                "replay_verified": True,
                "status": "BLOCKED",
                "stage_1_execution_started": True,
                "model_execution_used": False,
            }
        )
    assert error.value.code == "RUNTIME_PROVENANCE_STAGE1_ALREADY_STARTED"
