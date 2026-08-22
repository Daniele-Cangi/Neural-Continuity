from __future__ import annotations

import pytest

from neural_continuity.m1_diagnostics.measurement_null_extension_authority import (
    MeasurementNullPlanError,
    _validate_provenance_replay,
)


def _valid_replay() -> dict[str, object]:
    return {
        "replay_verified": True,
        "status": "BLOCKED",
        "attribution_status": "INCONCLUSIVE",
        "classification": "FROZEN_BATCH_ENVELOPE_DOES_NOT_COVER_CANONICAL_BASELINE",
        "model_execution_used": False,
        "onnx_graph_loaded": False,
        "activation_read": False,
        "stage_1_execution_started": False,
    }


def test_runtime_provenance_authority_accepts_exact_frozen_result() -> None:
    _validate_provenance_replay(_valid_replay())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("classification", "UNKNOWN"),
        ("model_execution_used", True),
        ("onnx_graph_loaded", True),
        ("activation_read", True),
        ("stage_1_execution_started", True),
    ],
)
def test_runtime_provenance_authority_fails_closed(field: str, value: object) -> None:
    replay = _valid_replay()
    replay[field] = value

    with pytest.raises(MeasurementNullPlanError):
        _validate_provenance_replay(replay)
