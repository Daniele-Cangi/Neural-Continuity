"""Fail-closed executable authority for the source-only CUDA full corpus."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from neural_continuity.m1_diagnostics.cuda_null_full_budget_replay import (
    replay_budget_package,
)
from neural_continuity.m1_diagnostics.cuda_null_full_profile_compression_package import (
    replay_compressed_profile_package,
)
from neural_continuity.m1_diagnostics.cuda_null_paths import has_linked_ancestor
from neural_continuity.m1_diagnostics.cuda_null_preflight_readiness import (
    CONFIG_PATH,
    CONFIG_SHA256,
    PROTOCOL_PATH,
    PROTOCOL_SHA256,
    _pinned_file,
)
from neural_continuity.m1_diagnostics.cuda_null_sentinel_execution_authority import (
    SentinelExecutionAuthority,
    verify_sentinel_execution_authority,
)
from neural_continuity.m1_diagnostics.cuda_null_sentinel_postgate_package import (
    replay_sentinel_postgate,
)

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "experiments" / "m1-cuda-null-full-corpus-authority-v1.json"
PROPOSAL_PATH = ROOT / "experiments" / "m1-cuda-null-full-corpus-proposal-v1.json"
PLAN_PATH = ROOT / "experiments" / "m1-cuda-null-full-corpus-execution-plan-v1.json"
REVIEW_PATH = ROOT / "docs" / "M1_CUDA_FULL_CORPUS_PRE_EXECUTION_REVIEW.md"
COMPRESSION_RESULT_PATH = ROOT / "experiments" / "m1-cuda-full-profile-compression-result-v1.json"

PROPOSAL_SHA256 = "4e78cd0497becbde1c974369c6fbac82cda190cffae006c2e1892790a161859c"
PLAN_SHA256 = "20cc3ec5f7bcd086681b97197c2f1ae8d730f4e4d32f1ee32c2324d789bc5fac"
REVIEW_SHA256 = "956f47b0e0c9c7ab9213e42c0bdfe3f650fe6f0b13805248e59813b03d372cd3"
SENTINEL_AUTHORITY_SHA256 = "987e038164916549e9bda3f1ce362c7f8b6490ab45875b73c4228863621259eb"
SENTINEL_TIP_SHA256 = "ded69e251d15dd6360deeeaaf970435ddb6b1e004086ea30274b53766fbc48ca"
GATE_MANIFEST_SHA256 = "331411a9523690afaf0e56678141e4b2358151fa817e91b6366d811edd12a1b3"
BUDGET_SPEC_SHA256 = "5ecd713388e1947c0aa88c45ae67e62c0792f4783f21b0e9b75ad0b3736dc60c"
BUDGET_MANIFEST_SHA256 = "93aa797cebad75892bc4771ef24d5c4893978d8fc1f4cab45c40fddbb897ef6a"
COMPRESSION_RESULT_SHA256 = "15b2fdf65f0ca9a71c0698b971eed044f955f35287fcffc40d61fc867deea0f1"
COMPRESSION_MANIFEST_SHA256 = "b48f2bf025e514807df066493c3dfef998149df4878cdc49cc1031afa18b8fd5"
DATASET_MANIFEST_SHA256 = "0746d98f5e69c6a0ee48ca3f47b342de1d968a877c90df26ffe8f893437fd5de"
SOURCE_ONNX_SHA256 = "5c0d999bd6b5e64e36cad1f61a83ef8e7507d55be49086745780fabb7c648511"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class FullCorpusExecutionBlocked(ValueError):
    """A frozen prerequisite for qualifying execution did not verify."""


@dataclass(frozen=True)
class FullCorpusExecutionAuthority:
    spec_sha256: str
    sentinel: SentinelExecutionAuthority
    paths: dict[str, Path]
    minimum_free_bytes: int
    observed_free_bytes: int


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise FullCorpusExecutionBlocked(reason)


def _read_spec(external_sha256: str) -> dict[str, Any]:
    _require(SHA256.fullmatch(external_sha256) is not None, "authority SHA-256 is invalid")
    try:
        _pinned_file(SPEC_PATH, SPEC_PATH, external_sha256, "full-corpus execution authority")
    except ValueError as exc:
        raise FullCorpusExecutionBlocked(str(exc)) from exc
    try:
        value = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FullCorpusExecutionBlocked("execution authority is unreadable") from exc
    _require(isinstance(value, dict), "execution authority is malformed")
    expected: dict[str, Any] = {
        "kind": "m1_cuda_null_full_corpus_execution_authority",
        "version": "1.0.0",
        "status": "IMPLEMENTATION_REVIEW_REQUIRED_EXECUTION_BLOCKED",
        "design_pre_execution_review_completed": True,
        "implementation_independent_review_completed": False,
        "proposal_sha256": PROPOSAL_SHA256,
        "execution_plan_sha256": PLAN_SHA256,
        "review_sha256": REVIEW_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "config_sha256": CONFIG_SHA256,
        "sentinel_authority_sha256": SENTINEL_AUTHORITY_SHA256,
        "sentinel_final_checkpoint_sha256": SENTINEL_TIP_SHA256,
        "technical_gate_manifest_sha256": GATE_MANIFEST_SHA256,
        "budget_preflight_spec_sha256": BUDGET_SPEC_SHA256,
        "budget_preflight_manifest_sha256": BUDGET_MANIFEST_SHA256,
        "profile_compression_result_sha256": COMPRESSION_RESULT_SHA256,
        "profile_compression_manifest_sha256": COMPRESSION_MANIFEST_SHA256,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "source_onnx_sha256": SOURCE_ONNX_SHA256,
        "phase_id": "full_corpus_qualification",
        "epoch_count": 120,
        "document_count": 5183,
        "measurement_null_query_count": 81,
        "ordered_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "profile_storage": "deterministic_gzip_level_9",
        "independent_process_per_epoch": True,
        "early_stopping_allowed": False,
        "adaptive_sample_size_allowed": False,
        "candidate_or_int8_execution_allowed": False,
        "holdout_access_allowed": False,
        "operational_tolerance_change_allowed": False,
        "execution_authorized": False,
        "scientific_decision": "NOT_EVALUATED",
    }
    _require(
        set(value) == set(expected) | {"authorization_basis", "paths"}
        and all(value.get(key) == item for key, item in expected.items()),
        "execution authority scope differs",
    )
    _require(
        isinstance(value.get("authorization_basis"), str) and bool(value["authorization_basis"]),
        "authorization basis is missing",
    )
    return value


def _paths(spec: dict[str, Any]) -> dict[str, Path]:
    names = {
        "technical_gate_bundle",
        "budget_preflight_bundle",
        "profile_compression_bundle",
        "output_root",
        "scratch_root",
        "external_checkpoint_tip",
    }
    raw = spec.get("paths")
    _require(isinstance(raw, dict), "execution authority path set differs")
    path_values = cast(dict[str, Any], raw)
    _require(
        set(path_values) == names
        and all(isinstance(path_values[name], str) and path_values[name] for name in names),
        "execution authority path set differs",
    )
    return {name: Path(path_values[name]) for name in names}


def verify_full_corpus_execution_authority(
    external_sha256: str,
) -> FullCorpusExecutionAuthority:
    """Reverify every frozen prerequisite before any ONNX graph may be loaded."""
    spec = _read_spec(external_sha256)
    paths = _paths(spec)
    for path, digest, label in (
        (PROPOSAL_PATH, PROPOSAL_SHA256, "full-corpus proposal"),
        (PLAN_PATH, PLAN_SHA256, "full-corpus execution plan"),
        (REVIEW_PATH, REVIEW_SHA256, "independent pre-execution review"),
        (COMPRESSION_RESULT_PATH, COMPRESSION_RESULT_SHA256, "profile compression result"),
        (CONFIG_PATH, CONFIG_SHA256, "frozen configuration"),
        (PROTOCOL_PATH, PROTOCOL_SHA256, "frozen protocol"),
    ):
        _pinned_file(path, path, digest, label)

    _require(
        not any(has_linked_ancestor(path.absolute()) for path in paths.values()),
        "declared execution path contains a link or reparse point",
    )
    _require(
        spec["implementation_independent_review_completed"] is True
        and spec["execution_authorized"] is True,
        "independent implementation review is required before execution",
    )

    sentinel = verify_sentinel_execution_authority(SENTINEL_AUTHORITY_SHA256)
    gate = replay_sentinel_postgate(paths["technical_gate_bundle"], GATE_MANIFEST_SHA256)
    _require(
        gate.get("replay_status") == "PASS"
        and gate.get("status") == "TECHNICAL_GATE_PASS_NO_SCIENTIFIC_RELEASE"
        and gate.get("epoch_count") == 120
        and gate.get("model_loaded") is False
        and gate.get("full_corpus_execution_authorized") is False,
        "declared sentinel technical gate replay blocked",
    )
    budget = replay_budget_package(paths["budget_preflight_bundle"], BUDGET_MANIFEST_SHA256)
    _require(
        budget.get("replay_status") == "PASS"
        and budget.get("model_loaded") is False
        and budget.get("full_corpus_qualification_execution_authorized") is False,
        "budget preflight replay blocked",
    )
    compression = replay_compressed_profile_package(
        paths["profile_compression_bundle"], COMPRESSION_MANIFEST_SHA256
    )
    _require(
        compression.get("replay_status") == "PASS"
        and compression.get("model_loaded") is False
        and compression.get("storage_budget_satisfied") is True,
        "compressed profile storage replay blocked",
    )

    output = paths["output_root"]
    scratch = paths["scratch_root"]
    checkpoint = paths["external_checkpoint_tip"]
    evidence_parent = sentinel.paths["output_root"].parent
    _require(
        output.drive.upper() == "D:"
        and scratch.drive.upper() == "D:"
        and checkpoint.drive.upper() == "D:"
        and output.parent == evidence_parent
        and scratch.parent == evidence_parent
        and checkpoint.parent == evidence_parent,
        "full-corpus storage paths differ from the frozen D-drive evidence root",
    )
    _require(not output.exists(), "fresh full-corpus output root already exists")
    _require(not checkpoint.exists(), "external checkpoint tip already exists")
    _require(scratch.is_dir() and not scratch.is_symlink(), "declared scratch root is unavailable")
    raw_minimum_free = compression.get("minimum_free_bytes_before_execution")
    _require(
        type(raw_minimum_free) is int and raw_minimum_free > 0,
        "storage minimum is invalid",
    )
    minimum_free = cast(int, raw_minimum_free)
    free = shutil.disk_usage(evidence_parent).free
    _require(free >= minimum_free, "live free storage is below the frozen minimum")
    return FullCorpusExecutionAuthority(external_sha256, sentinel, paths, minimum_free, free)
