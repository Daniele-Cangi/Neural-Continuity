"""Model-free technical aggregation of the completed CUDA sentinel."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from neural_continuity.evidence import sha256_file
from neural_continuity.m1_diagnostics.cuda_null_evidence import replay_static_package
from neural_continuity.m1_diagnostics.cuda_null_paths import has_linked_ancestor
from neural_continuity.m1_diagnostics.cuda_null_sentinel_chain_replay import (
    replay_cuda_sentinel_chain_packages,
)
from neural_continuity.m1_diagnostics.cuda_null_sentinel_execution_authority import (
    EXPECTED_SOURCE_MANIFEST,
    EXPECTED_STATIC_AUTHORITY,
    EXPECTED_STATIC_MANIFEST,
    _load_spec,
)
from neural_continuity.m1_diagnostics.cuda_null_sentinel_readiness import EPOCH_LAYOUT
from neural_continuity.m1_diagnostics.cuda_null_sentinel_runner import _replay_bound_epochs
from neural_continuity.m1_diagnostics.cuda_null_source_preflight_evidence import (
    replay_source_preflight,
)
from neural_continuity.m1_diagnostics.cuda_null_source_preflight_inputs import QRELS_SHA256

SHA256 = re.compile(r"[0-9a-f]{64}\Z")
ROLES = ("documents", "measurement_null_queries")


class SentinelPostgateBlocked(ValueError):
    """A required sentinel, provenance, or provider assertion did not verify."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise SentinelPostgateBlocked(reason)


def _verify_roots(root: Path, tip: str, authority_sha256: str) -> dict[str, Path]:
    _require(SHA256.fullmatch(tip) is not None, "external checkpoint tip is invalid")
    _require(SHA256.fullmatch(authority_sha256) is not None, "external authority hash is invalid")
    spec = _load_spec(authority_sha256)
    paths = {key: Path(value) for key, value in spec["paths"].items()}
    _require(root.absolute() == paths["output_root"].absolute(), "sentinel root differs")
    _require(not has_linked_ancestor(root), "sentinel root contains a link")
    _require(
        sha256_file(paths["dataset_root"] / "roles" / "measurement_null.qrels.tsv") == QRELS_SHA256,
        "measurement-null qrels changed",
    )
    static = replay_static_package(paths["fresh_static_bundle"], EXPECTED_STATIC_MANIFEST)
    _require(
        static.get("integrity_replay_status") == "PASS"
        and static.get("authority_sha256") == EXPECTED_STATIC_AUTHORITY
        and static.get("decision_match") is True
        and static.get("status_match") is True,
        "frozen static authority replay blocked",
    )
    source = replay_source_preflight(paths["source_preflight_bundle"], EXPECTED_SOURCE_MANIFEST)
    _require(
        source.get("replay_status") == "PASS"
        and source.get("technical_preflight_status") == "PASS"
        and source.get("decision_match") is True,
        "source preflight replay blocked",
    )
    return paths


def _profile_rows(
    root: Path,
) -> tuple[list[dict[str, Any]], int, int, int, dict[str, int]]:
    keys = [(label, role) for label, _batch in EPOCH_LAYOUT for role in ROLES]
    totals = {
        key: {
            "cuda_total": 0,
            "cpu_total": 0,
            "cuda_min": 2**63,
            "cuda_max": 0,
            "cpu_min": 2**63,
            "cpu_max": 0,
        }
        for key in keys
    }
    fallback: Counter[str] = Counter()
    processes: set[str] = set()
    runtimes: set[str] = set()
    profile_count = 0
    artifact_bytes = 0
    for number in range(1, 121):
        epoch = root / f"epoch-{number:04d}"
        runtime = json.loads((epoch / "runtime-inventory.json").read_text(encoding="utf-8"))
        processes.add(runtime["process_instance_id"])
        runtimes.add(runtime["runtime_identity_sha256"])
        lines = (epoch / "provider-profile.jsonl").read_text(encoding="utf-8").splitlines()
        _require(len(lines) == len(keys), f"epoch {number}: profile count differs")
        for line, key in zip(lines, keys, strict=True):
            profile = json.loads(line)
            _require(
                (profile["run_label"], profile["role"]) == key,
                f"epoch {number}: profile order differs",
            )
            cuda = profile["provider_event_counts"].get("CUDAExecutionProvider", 0)
            cpu = profile["provider_event_counts"].get("CPUExecutionProvider", 0)
            _require(type(cuda) is int and cuda > 0, f"epoch {number}: CUDA inactive")
            _require(type(cpu) is int and cpu >= 0, f"epoch {number}: CPU count invalid")
            row = totals[key]
            row["cuda_total"] += cuda
            row["cpu_total"] += cpu
            row["cuda_min"] = min(row["cuda_min"], cuda)
            row["cuda_max"] = max(row["cuda_max"], cuda)
            row["cpu_min"] = min(row["cpu_min"], cpu)
            row["cpu_max"] = max(row["cpu_max"], cpu)
            fallback.update(profile["cpu_fallback_operator_types"])
            profile_count += 1
        artifact_bytes += sum(path.stat().st_size for path in epoch.iterdir())
    _require(len(processes) == 120, "process isolation did not verify")
    _require(len(runtimes) == 1, "runtime identity varied between epochs")
    _require(profile_count == 960, "sentinel profile coverage is incomplete")
    rows = [{"run_label": label, "role": role, **totals[(label, role)]} for label, role in keys]
    return rows, profile_count, artifact_bytes, len(runtimes), dict(sorted(fallback.items()))


def build_sentinel_postgate_report(
    *, root: Path, external_tip_sha256: str, external_authority_sha256: str
) -> dict[str, Any]:
    """Replay all raw epochs; aggregate technical structure, never numerical limits."""
    paths = _verify_roots(root, external_tip_sha256, external_authority_sha256)
    chain = replay_cuda_sentinel_chain_packages(
        root, expected_count=120, external_tip_sha256=external_tip_sha256
    )
    _require(chain.get("status") == "TECHNICAL_REPLAY_PASS_NOT_QUALIFYING", "chain replay blocked")
    _replay_bound_epochs(
        root, 120, external_tip_sha256, external_authority_sha256, paths["dataset_root"]
    )
    rows, profile_count, artifact_bytes, runtime_count, fallback = _profile_rows(root)
    # Recheck after aggregation so a concurrent mutation cannot pass unnoticed.
    _replay_bound_epochs(
        root, 120, external_tip_sha256, external_authority_sha256, paths["dataset_root"]
    )
    _verify_roots(root, external_tip_sha256, external_authority_sha256)
    return {
        "kind": "m1_cuda_null_sentinel_postgate_report",
        "version": "1.0.0",
        "status": "TECHNICAL_GATE_PASS_NO_SCIENTIFIC_RELEASE",
        "sentinel_authority_sha256": external_authority_sha256,
        "sentinel_final_checkpoint_sha256": external_tip_sha256,
        "epoch_count": 120,
        "profile_count": profile_count,
        "unique_process_count": 120,
        "unique_runtime_identity_count": runtime_count,
        "provider_event_counts": rows,
        "classified_cpu_fallback_profile_occurrences": fallback,
        "epoch_artifact_bytes": artifact_bytes,
        "run_timing_status": "NOT_CAPTURED",
        "gpu_utilization_status": "NOT_CAPTURED",
        "qualifying_detection_evidence": False,
        "independent_review_completed": False,
        "full_corpus_execution_authorized": False,
        "int8_executed": False,
        "scientific_decision": "NOT_EVALUATED",
        "model_required_for_replay": False,
    }
