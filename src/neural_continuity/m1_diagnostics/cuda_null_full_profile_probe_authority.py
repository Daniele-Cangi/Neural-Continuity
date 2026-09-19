"""Authority for one nonqualifying raw-profile storage probe."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural_continuity.m1_diagnostics.cuda_null_full_budget_replay import replay_budget_package
from neural_continuity.m1_diagnostics.cuda_null_preflight_readiness import _pinned_file
from neural_continuity.m1_diagnostics.cuda_null_sentinel_execution_authority import (
    SentinelExecutionAuthority,
    verify_sentinel_execution_authority,
)
from neural_continuity.m1_diagnostics.cuda_null_sentinel_postgate_package import (
    replay_sentinel_postgate,
)
from neural_continuity.m1_diagnostics.cuda_null_sentinel_readiness import EPOCH_LAYOUT

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "experiments" / "m1-cuda-full-profile-storage-probe-v1.json"
REVIEW_PATH = ROOT / "docs" / "M1_CUDA_FULL_CORPUS_PRE_EXECUTION_REVIEW.md"
PLAN_PATH = ROOT / "experiments" / "m1-cuda-null-full-corpus-execution-plan-v1.json"
REVIEW_SHA256 = "956f47b0e0c9c7ab9213e42c0bdfe3f650fe6f0b13805248e59813b03d372cd3"
PLAN_SHA256 = "20cc3ec5f7bcd086681b97197c2f1ae8d730f4e4d32f1ee32c2324d789bc5fac"
SENTINEL_AUTHORITY_SHA256 = "987e038164916549e9bda3f1ce362c7f8b6490ab45875b73c4228863621259eb"
GATE_MANIFEST_SHA256 = "331411a9523690afaf0e56678141e4b2358151fa817e91b6366d811edd12a1b3"
BUDGET_MANIFEST_SHA256 = "93aa797cebad75892bc4771ef24d5c4893978d8fc1f4cab45c40fddbb897ef6a"
DATASET_SHA256 = "0746d98f5e69c6a0ee48ca3f47b342de1d968a877c90df26ffe8f893437fd5de"
SOURCE_SHA256 = "5c0d999bd6b5e64e36cad1f61a83ef8e7507d55be49086745780fabb7c648511"
_SPEC_FIELDS = {
    "approval_basis",
    "technical_gate_bundle",
    "budget_preflight_bundle",
    "output_directory",
}


class FullProfileProbeBlocked(ValueError):
    """The bounded storage probe authority did not verify."""


@dataclass(frozen=True)
class FullProfileProbeAuthority:
    spec_sha256: str
    sentinel: SentinelExecutionAuthority
    output_directory: Path


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise FullProfileProbeBlocked(reason)


def _load_spec(external_sha256: str) -> dict[str, Any]:
    _pinned_file(SPEC_PATH, SPEC_PATH, external_sha256, "profile storage probe authority")
    try:
        spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FullProfileProbeBlocked("profile storage probe authority unreadable") from exc
    expected: dict[str, Any] = {
        "kind": "m1_cuda_full_profile_storage_probe",
        "version": "1.0.0",
        "status": "OWNER_AUTHORIZED_NONQUALIFYING_STORAGE_ONLY",
        "review_sha256": REVIEW_SHA256,
        "execution_plan_sha256": PLAN_SHA256,
        "sentinel_authority_sha256": SENTINEL_AUTHORITY_SHA256,
        "technical_gate_manifest_sha256": GATE_MANIFEST_SHA256,
        "budget_preflight_manifest_sha256": BUDGET_MANIFEST_SHA256,
        "dataset_manifest_sha256": DATASET_SHA256,
        "source_onnx_sha256": SOURCE_SHA256,
        "document_profile_item_count": 256,
        "query_profile_item_count": 81,
        "maximum_profile_segment_items": 256,
        "run_layout": [{"label": label, "batch_size": size} for label, size in EPOCH_LAYOUT],
        "ordered_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "retain_raw_profiles": True,
        "retain_embeddings": False,
        "retain_rankings_or_metrics": False,
        "timings_are_descriptive_only": True,
        "qualifying_detection_evidence": False,
        "full_corpus_qualification_started": False,
        "full_corpus_execution_authorized": False,
        "candidate_or_int8_execution_allowed": False,
        "holdout_access_allowed": False,
        "scientific_decision": "NOT_EVALUATED",
    }
    _require(isinstance(spec, dict), "profile storage probe authority malformed")
    _require(
        set(spec) == set(expected) | _SPEC_FIELDS
        and all(spec.get(key) == value for key, value in expected.items()),
        "profile storage probe scope differs",
    )
    _require(
        isinstance(spec.get("approval_basis"), str) and bool(spec["approval_basis"]),
        "approval basis missing",
    )
    return spec


def verify_full_profile_probe_authority(external_sha256: str) -> FullProfileProbeAuthority:
    """Verify every frozen root and live runtime before graph loading is permitted."""
    spec = _load_spec(external_sha256)
    _pinned_file(REVIEW_PATH, REVIEW_PATH, REVIEW_SHA256, "independent pre-execution review")
    _pinned_file(PLAN_PATH, PLAN_PATH, PLAN_SHA256, "full-corpus execution plan")
    gate = replay_sentinel_postgate(Path(spec["technical_gate_bundle"]), GATE_MANIFEST_SHA256)
    _require(
        gate.get("replay_status") == "PASS"
        and gate.get("status") == "TECHNICAL_GATE_PASS_NO_SCIENTIFIC_RELEASE"
        and gate.get("epoch_count") == 120,
        "sentinel technical gate replay blocked",
    )
    budget = replay_budget_package(Path(spec["budget_preflight_bundle"]), BUDGET_MANIFEST_SHA256)
    _require(
        budget.get("replay_status") == "PASS"
        and budget.get("status") == "TECHNICAL_TIMING_CAPTURED_NOT_QUALIFYING"
        and budget.get("full_corpus_qualification_execution_authorized") is False,
        "full-input budget replay blocked",
    )
    sentinel = verify_sentinel_execution_authority(SENTINEL_AUTHORITY_SHA256)
    output = Path(spec["output_directory"])
    _require(
        output.drive.upper() == "D:"
        and not output.exists()
        and output.parent == sentinel.paths["output_root"].parent,
        "profile storage probe output is not a fresh D evidence path",
    )
    return FullProfileProbeAuthority(external_sha256, sentinel, output)
