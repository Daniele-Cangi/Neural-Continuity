from __future__ import annotations

import hashlib
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from neural_continuity.m1_diagnostics import authority
from neural_continuity.m1_diagnostics.authority import (
    FrozenAuthorityPaths,
    verify_frozen_authority_set,
)
from neural_continuity.m1_diagnostics.instrumentation import derive_instrumented_graphs
from neural_continuity.m1_diagnostics.static_package import VerifiedStaticPreflight


def _write_model(path: Path) -> None:
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["input"], ["hidden"]),
            helper.make_node("Identity", ["hidden"], ["embeddings"]),
        ],
        "instrumentation-test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("embeddings", TensorProto.FLOAT, [1, 2])],
    )
    onnx.save_model(helper.make_model(graph), str(path))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_instrumentation_creates_derived_copies_without_overwriting_frozen_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.onnx"
    target = tmp_path / "target.onnx"
    _write_model(source)
    _write_model(target)
    other_paths = {
        role: tmp_path / f"{role}.json"
        for role in authority.FROZEN_AUTHORITY_SHA256
        if role not in {"onnx_fp32_source", "onnx_int8_candidate"}
    }
    for role, path in other_paths.items():
        path.write_text(f"{role}\n", encoding="utf-8")
    paths = FrozenAuthorityPaths(
        onnx_fp32_source=source,
        onnx_int8_candidate=target,
        calibration_manifest=other_paths["calibration_manifest"],
        paired_fp32_evidence=other_paths["paired_fp32_evidence"],
        int8_target_evidence=other_paths["int8_target_evidence"],
        transition_b_decision=other_paths["transition_b_decision"],
        transition_a_contract=other_paths["transition_a_contract"],
        transition_b_v1_contract=other_paths["transition_b_v1_contract"],
    )
    expected = {
        role: (
            authority._sha256_lf_normalized_file(paths.path_for(role))
            if role in {"transition_a_contract", "transition_b_v1_contract"}
            else authority._sha256_file(paths.path_for(role))
        )
        for role in authority.FROZEN_AUTHORITY_SHA256
    }
    monkeypatch.setattr(authority, "FROZEN_AUTHORITY_SHA256", expected)
    verified = VerifiedStaticPreflight(
        package_directory=tmp_path,
        manifest_sha256="a" * 64,
        probe_plan_sha256="b" * 64,
        authorities=verify_frozen_authority_set(paths),
        probe_plan={
            "probes": [
                {
                    "probe_id": "probe-0001",
                    "source_tensor": "hidden",
                    "target_tensor": "hidden",
                }
            ]
        },
    )
    source_before = _sha256(source)
    target_before = _sha256(target)

    result = derive_instrumented_graphs(verified, tmp_path / "derived")

    assert _sha256(source) == source_before
    assert _sha256(target) == target_before
    assert result.source.probe_outputs == ("hidden",)
    assert result.target.probe_outputs == ("hidden",)
    assert [value.name for value in onnx.load(result.source.path).graph.output] == [
        "embeddings",
        "hidden",
    ]
    assert result.to_dict()["onnx_runtime_session_created"] is False
