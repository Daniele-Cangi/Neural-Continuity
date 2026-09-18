"""Bounded timing permission; never authorizes a qualifying full-corpus epoch."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
SPEC_PATH = ROOT / "experiments" / "m1-cuda-full-corpus-budget-preflight-v2.json"
FAILED_V1_SPEC_SHA256 = "929b707ce64a39a74559f1325d4d679d2b61a55e276cb24b706914d1f5251379"
SENTINEL_AUTHORITY_SHA256 = "987e038164916549e9bda3f1ce362c7f8b6490ab45875b73c4228863621259eb"
SENTINEL_TIP_SHA256 = "ded69e251d15dd6360deeeaaf970435ddb6b1e004086ea30274b53766fbc48ca"
GATE_MANIFEST_SHA256 = "331411a9523690afaf0e56678141e4b2358151fa817e91b6366d811edd12a1b3"
DATASET_MANIFEST_SHA256 = "0746d98f5e69c6a0ee48ca3f47b342de1d968a877c90df26ffe8f893437fd5de"
SOURCE_ONNX_SHA256 = "5c0d999bd6b5e64e36cad1f61a83ef8e7507d55be49086745780fabb7c648511"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class FullBudgetBlocked(ValueError):
    """The frozen timing-only authority did not verify."""


@dataclass(frozen=True)
class FullBudgetAuthority:
    spec_sha256: str
    sentinel: SentinelExecutionAuthority
    gate_bundle: Path
    output_directory: Path


def load_budget_spec(external_sha256: str) -> dict[str, Any]:
    if SHA256.fullmatch(external_sha256) is None:
        raise FullBudgetBlocked("external budget-preflight SHA-256 is invalid")
    _pinned_file(SPEC_PATH, SPEC_PATH, external_sha256, "budget-preflight authority")
    try:
        spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FullBudgetBlocked("budget-preflight authority is unreadable") from exc
    if not isinstance(spec, dict):
        raise FullBudgetBlocked("budget-preflight authority is malformed")
    expected: dict[str, Any] = {
        "kind": "m1_cuda_full_corpus_budget_preflight",
        "version": "2.0.0",
        "status": "OWNER_AUTHORIZED_NONQUALIFYING_TIMING_ONLY",
        "supersedes_failed_spec_sha256": FAILED_V1_SPEC_SHA256,
        "failed_v1_attempt": (
            "EXECUTION_ERROR: ONNX Runtime profiler reached its event limit on the "
            "full-input batch-1 pass; process terminated without a budget evidence package."
        ),
        "sentinel_authority_sha256": SENTINEL_AUTHORITY_SHA256,
        "sentinel_final_checkpoint_sha256": SENTINEL_TIP_SHA256,
        "technical_gate_manifest_sha256": GATE_MANIFEST_SHA256,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "source_onnx_sha256": SOURCE_ONNX_SHA256,
        "document_count": 5183,
        "measurement_null_query_count": 81,
        "one_process": True,
        "bounded_full_input_timing_authorized": True,
        "full_input_profiling_allowed": False,
        "provider_profile_scope": "canonical_first_64_ids_per_role",
        "provider_profile_item_count_per_role": 64,
        "run_layout": [{"label": label, "batch_size": size} for label, size in EPOCH_LAYOUT],
        "ordered_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "retain_embeddings": False,
        "retain_rankings_or_metrics": False,
        "timings_are_descriptive_only": True,
        "qualifying_detection_evidence": False,
        "full_corpus_qualification_started": False,
        "full_corpus_qualification_execution_authorized": False,
        "candidate_or_int8_execution_allowed": False,
        "holdout_access_allowed": False,
        "operational_tolerance_change_allowed": False,
        "scientific_decision": "NOT_EVALUATED",
    }
    if set(spec) != set(expected) | {
        "approval_basis",
        "technical_gate_bundle",
        "output_directory",
    } or any(spec.get(key) != value for key, value in expected.items()):
        raise FullBudgetBlocked("budget-preflight scope differs from frozen design")
    if not isinstance(spec.get("approval_basis"), str) or not spec["approval_basis"]:
        raise FullBudgetBlocked("owner approval basis is missing")
    if not all(
        isinstance(spec.get(key), str) and spec[key]
        for key in ("technical_gate_bundle", "output_directory")
    ):
        raise FullBudgetBlocked("budget-preflight paths are missing")
    return spec


def verify_full_budget_authority(external_sha256: str) -> FullBudgetAuthority:
    """Verify all frozen evidence and runtime before caller may create a session."""
    spec = load_budget_spec(external_sha256)
    gate_bundle = Path(spec["technical_gate_bundle"])
    gate = replay_sentinel_postgate(gate_bundle, GATE_MANIFEST_SHA256)
    if not (
        gate.get("replay_status") == "PASS"
        and gate.get("status") == "TECHNICAL_GATE_PASS_NO_SCIENTIFIC_RELEASE"
        and gate.get("epoch_count") == 120
        and gate.get("full_corpus_execution_authorized") is False
        and gate.get("model_loaded") is False
    ):
        raise FullBudgetBlocked("sentinel technical gate replay blocked")
    sentinel = verify_sentinel_execution_authority(SENTINEL_AUTHORITY_SHA256)
    output = Path(spec["output_directory"])
    if output.drive.upper() != "D:" or output.exists():
        raise FullBudgetBlocked("budget output is not a fresh directory on D")
    if output.parent != sentinel.paths["output_root"].parent:
        raise FullBudgetBlocked("budget output parent differs from declared evidence root")
    return FullBudgetAuthority(external_sha256, sentinel, gate_bundle, output)
