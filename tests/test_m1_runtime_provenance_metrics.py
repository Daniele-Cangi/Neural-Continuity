from __future__ import annotations

from neural_continuity.m1_diagnostics.runtime_provenance_metrics import _attribution


def _intra(outcome: str) -> dict[str, str]:
    return {"outcome": outcome}


def test_attribution_is_inconclusive_for_cross_epoch_drift_without_runtime_authority() -> None:
    result = _attribution(
        {
            "baseline_fp32": _intra("PASS"),
            "baseline_int8": _intra("PASS"),
            "fresh_fp32": _intra("PASS"),
            "fresh_int8": _intra("PASS"),
        },
        {"controls": {"fp32": {"outcome": "BLOCKED"}}},
        {"complete": False},
    )
    assert result["status"] == "INCONCLUSIVE"
    assert result["classification"] == "CROSS_EPOCH_DRIFT_WITH_INCOMPLETE_RUNTIME_AUTHORITY"
    assert result["causal_runtime_claim_made"] is False
    assert result["scientific_regression_recorded"] is False


def test_attribution_reports_process_local_variation_first() -> None:
    result = _attribution(
        {
            "baseline_fp32": _intra("PASS"),
            "baseline_int8": _intra("PASS"),
            "fresh_fp32": _intra("PASS"),
            "fresh_int8": _intra("BLOCKED"),
        },
        {"controls": {"fp32": {"outcome": "BLOCKED"}}},
        {"complete": False},
    )
    assert result["classification"] == "PROCESS_LOCAL_VARIATION_EXCEEDS_FROZEN_ENVELOPE"


def test_attribution_identifies_canonical_envelope_coverage_failure() -> None:
    result = _attribution(
        {
            "baseline_fp32": _intra("BLOCKED"),
            "baseline_int8": _intra("BLOCKED"),
            "fresh_fp32": _intra("BLOCKED"),
            "fresh_int8": _intra("BLOCKED"),
        },
        {"controls": {"fp32": {"outcome": "BLOCKED"}}},
        {"complete": False},
    )
    assert result["classification"] == "FROZEN_BATCH_ENVELOPE_DOES_NOT_COVER_CANONICAL_BASELINE"
    assert result["status"] == "INCONCLUSIVE"
