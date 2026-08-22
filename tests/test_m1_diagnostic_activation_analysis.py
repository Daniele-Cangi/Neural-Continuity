from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neural_continuity.m1_diagnostics import (
    activation_analysis_evidence,
)
from neural_continuity.m1_diagnostics.activation_analysis_authority import (
    ActivationAnalysisError,
    VerifiedActivationAnalysisInput,
)
from neural_continuity.m1_diagnostics.activation_analysis_evidence import (
    create_activation_analysis_package,
    replay_activation_analysis,
)
from neural_continuity.m1_diagnostics.activation_analysis_metrics import (
    analyze_activation_batches,
)


def _capture_plan() -> dict[str, object]:
    return {
        "batch_size": 1,
        "query_count": 1,
        "probe_count": 2,
        "integer_probe_count": 1,
        "query_order": "query_id_utf8_byte_order",
        "probe_mappings": [
            {
                "probe_id": "probe-0001",
                "target_tensor_basis": ("post_quantize_dequantize_output"),
                "structural_families": ["QUANTIZED_COMPUTE"],
            },
            {
                "probe_id": "probe-0002",
                "target_tensor_basis": "direct_compute_output",
                "structural_families": ["DIRECT_COMPUTE"],
            },
        ],
        "integer_mappings": [{"probe_id": "probe-0001"}],
    }


def _batch_index() -> dict[str, object]:
    return {
        "batch_count": 1,
        "query_count": 1,
        "batches": [
            {
                "batch_id": "batch-0001",
                "floating_path": "batch-0001-floating.npz",
                "integer_path": "batch-0001-integer.npz",
                "query_ids": ["q-1"],
            }
        ],
    }


def _write_batches(root: Path, *, omit_integer: bool = False) -> None:
    np.savez_compressed(
        root / "batch-0001-floating.npz",
        query_ids=np.asarray(["q-1"]),
        source__probe_0001=np.asarray([[1.0, 2.0]], dtype=np.float32),
        target__probe_0001=np.asarray([[1.0, 2.0]], dtype=np.float32),
        source__probe_0002=np.asarray([[1, 1]], dtype=np.int64),
        target__probe_0002=np.asarray([[2, 1]], dtype=np.int64),
    )
    integer_payload = {"query_ids": np.asarray(["q-1"])}
    if not omit_integer:
        integer_payload["target_integer__probe_0001"] = np.asarray([[0, 127]], dtype=np.int8)
    np.savez_compressed(root / "batch-0001-integer.npz", **integer_payload)


def _authority(root: Path) -> VerifiedActivationAnalysisInput:
    capture_plan = _capture_plan()
    bundle = root / "replay-bundle.json"
    bundle.write_text("{}\n", encoding="utf-8")
    (root / "capture-plan.json").write_text(
        json.dumps(capture_plan, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return VerifiedActivationAnalysisInput(
        root=root,
        bundle_path=bundle,
        manifest_sha256="a" * 64,
        capture_plan=capture_plan,
        batch_index=_batch_index(),
        replay_result={"replay_verified": True},
    )


def test_analysis_ranks_and_localizes_without_threshold(
    tmp_path: Path,
) -> None:
    _write_batches(tmp_path)
    analysis = analyze_activation_batches(tmp_path, _capture_plan(), _batch_index())

    assert analysis["summary"]["first_bitwise_divergence"]["probe_id"] == "probe-0002"
    assert analysis["summary"]["ranked_probe_ids"][0] == "probe-0002"
    records = {record["probe_id"]: record for record in analysis["probes"]}
    assert records["probe-0001"]["floating"]["classification"] == "BITWISE_EQUAL"
    assert records["probe-0001"]["integer_dtype_extremes"]["dtype_extreme_rate"] == 0.5
    assert records["probe-0002"]["floating"]["differing_value_count"] == 1
    assert records["probe-0002"]["floating"]["metric_domain"] == "INTEGER_NUMERIC"


def test_analysis_fails_closed_for_missing_integer_probe(
    tmp_path: Path,
) -> None:
    _write_batches(tmp_path, omit_integer=True)
    with pytest.raises(ActivationAnalysisError) as error:
        analyze_activation_batches(tmp_path, _capture_plan(), _batch_index())
    assert error.value.code == "INTEGER_BATCH_SCHEMA_MISMATCH"


def test_analysis_package_replays_model_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_batches(source)
    authority = _authority(source)
    monkeypatch.setattr(
        activation_analysis_evidence,
        "verify_activation_analysis_input",
        lambda *_args, **_kwargs: authority,
    )
    output = tmp_path / "analysis"
    result = create_activation_analysis_package(
        authority.bundle_path,
        authority.manifest_sha256,
        output,
    )
    replay = replay_activation_analysis(
        output / "replay-bundle.json",
        result["artifact_manifest_sha256"],
    )

    assert replay["replay_verified"] is True
    assert replay["plan_match"] is True
    assert replay["probe_diagnostics_match"] is True
    assert replay["report_match"] is True
    assert replay["model_execution_used"] is False


def test_analysis_replay_rejects_tampered_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_batches(source)
    authority = _authority(source)
    monkeypatch.setattr(
        activation_analysis_evidence,
        "verify_activation_analysis_input",
        lambda *_args, **_kwargs: authority,
    )
    output = tmp_path / "analysis"
    result = create_activation_analysis_package(
        authority.bundle_path,
        authority.manifest_sha256,
        output,
    )
    report = output / "diagnostic-report.json"
    report.write_text(
        report.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )

    with pytest.raises(ActivationAnalysisError) as error:
        replay_activation_analysis(
            output / "replay-bundle.json",
            result["artifact_manifest_sha256"],
        )
    assert error.value.code == "ANALYSIS_ARTIFACT_HASH_MISMATCH"
