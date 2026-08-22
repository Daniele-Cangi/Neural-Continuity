from __future__ import annotations

from neural_continuity.m1_diagnostics.runtime_provenance_environment import (
    HISTORICAL_RUNTIME_FIELDS,
    historical_runtime_coverage,
)


def test_historical_runtime_coverage_fails_closed_when_ort_is_missing() -> None:
    report = historical_runtime_coverage(
        [
            {
                "teacher_tokenizer_identity": {
                    "torch_version": "2.10.0+cpu",
                    "sentence_transformers_version": "5.6.1",
                },
                "source_identity": {"execution_provider": "CPUExecutionProvider"},
            }
        ]
    )
    assert report["complete"] is False
    assert "onnxruntime_version" in report["missing_fields"]
    assert "onnxruntime_binary_sha256" in report["missing_fields"]


def test_historical_runtime_coverage_accepts_complete_identity() -> None:
    report = historical_runtime_coverage([{field: "value" for field in HISTORICAL_RUNTIME_FIELDS}])
    assert report["complete"] is True
    assert report["missing_fields"] == []
