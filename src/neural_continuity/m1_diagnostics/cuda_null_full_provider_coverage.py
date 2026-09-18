"""Model-free replay of full-input CUDA provider coverage, one role and run.

Raw ORT profiles are retained. This module does not create a session, verify
embeddings, or authorize a qualifying epoch.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.cuda_null_paths import (
    has_linked_ancestor,
    snapshot_file_inventory,
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_LAYOUT = {
    "batch_1_primary": 1,
    "batch_16_primary": 16,
    "batch_16_repeat": 16,
    "batch_64_primary": 64,
}
_ROLE_COUNTS = {"documents": 5183, "measurement_null_queries": 81}
_PROVIDERS = {"CUDAExecutionProvider", "CPUExecutionProvider"}
_SEGMENT_LIMIT = 64
_MAX_PROFILE_BYTES = 256 * 1024 * 1024
_FIELDS = {
    "run_label",
    "role",
    "start",
    "end",
    "batch_size",
    "ordered_ids_sha256",
    "embeddings_sha256",
    "profile_path",
    "profile_sha256",
    "inference_call_count",
    "provider_event_counts",
    "operator_event_counts",
}


class FullProviderCoverageBlocked(ValueError):
    """An input segment or its retained provider evidence did not verify."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise FullProviderCoverageBlocked(reason)


def _digest(value: Any, label: str) -> None:
    _require(isinstance(value, str) and _SHA256.fullmatch(value) is not None, f"{label} invalid")


def _counts(profile: Path, expected_calls: int) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    try:
        _require(profile.stat().st_size <= _MAX_PROFILE_BYTES, "profile exceeds memory bound")
        events = json.loads(profile.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FullProviderCoverageBlocked("raw provider profile cannot be read") from exc
    _require(isinstance(events, list), "raw provider profile is not an event list")
    calls = 0
    providers: Counter[str] = Counter()
    operators: dict[str, Counter[str]] = {}
    for event in events:
        _require(isinstance(event, dict), "profile event malformed")
        if event.get("cat") == "Session" and event.get("name") == "model_run":
            calls += 1
        if event.get("cat") != "Node":
            continue
        args = event.get("args")
        _require(isinstance(args, dict), "Node event has no arguments")
        provider = args.get("provider")
        operator = args.get("op_name")
        _require(provider in _PROVIDERS, "Node event has an undeclared provider")
        _require(isinstance(operator, str) and bool(operator), "Node operator is unclassified")
        providers[provider] += 1
        operators.setdefault(provider, Counter())[operator] += 1
    _require(calls == expected_calls, "profile inference-call coverage differs")
    _require(sum(providers.values()) > 0, "profile contains no Node events")
    return dict(sorted(providers.items())), {
        provider: dict(sorted(counts.items())) for provider, counts in sorted(operators.items())
    }


def replay_full_provider_coverage(
    root: Path,
    *,
    run_label: str,
    role: str,
    ordered_ids: Sequence[str],
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recompute exact ID coverage and provider counts from retained raw profiles."""
    _require(run_label in _LAYOUT, "run label is not frozen")
    _require(role in _ROLE_COUNTS, "query role is not frozen")
    _require(
        len(ordered_ids) == _ROLE_COUNTS[role]
        and all(isinstance(item, str) and bool(item) for item in ordered_ids)
        and len(set(ordered_ids)) == len(ordered_ids),
        "ordered input IDs are invalid",
    )
    _require(isinstance(segments, list) and bool(segments), "provider segments missing")
    _require(not has_linked_ancestor(root), "profile package path contains a link")
    profile_dir = root / "profiles" / run_label / role
    _require(profile_dir.is_dir(), "profile segment directory missing")
    _require(not has_linked_ancestor(profile_dir), "profile segment directory linked")
    batch_size = _LAYOUT[run_label]
    next_start = 0
    expected_files: set[str] = set()
    provider_totals: Counter[str] = Counter()
    operator_totals: dict[str, Counter[str]] = {}
    call_total = 0
    for index, segment in enumerate(segments, start=1):
        _require(isinstance(segment, dict) and set(segment) == _FIELDS, "segment schema differs")
        _require(
            segment["run_label"] == run_label
            and segment["role"] == role
            and type(segment["batch_size"]) is int
            and segment["batch_size"] == batch_size,
            "segment run identity differs",
        )
        start, end = segment["start"], segment["end"]
        _require(
            type(start) is int
            and type(end) is int
            and start == next_start
            and start < end <= len(ordered_ids)
            and end - start <= _SEGMENT_LIMIT,
            "segment skips, repeats, or exceeds the bounded input range",
        )
        expected_id_hash = hashlib.sha256(
            canonical_json_bytes(list(ordered_ids[start:end]))
        ).hexdigest()
        _require(segment["ordered_ids_sha256"] == expected_id_hash, "segment ID hash differs")
        _digest(segment["embeddings_sha256"], "segment embedding hash")
        _digest(segment["profile_sha256"], "segment profile hash")
        relative = f"profiles/{run_label}/{role}/segment-{index:04d}.json"
        _require(segment["profile_path"] == relative, "segment profile path differs")
        profile = root / relative
        _require(
            profile.is_file() and not has_linked_ancestor(profile), "profile missing or linked"
        )
        _require(sha256_file(profile) == segment["profile_sha256"], "raw profile hash differs")
        calls = math.ceil((end - start) / batch_size)
        _require(
            type(segment["inference_call_count"]) is int
            and segment["inference_call_count"] == calls,
            "declared inference-call count differs",
        )
        providers, operators = _counts(profile, calls)
        _require(
            segment["provider_event_counts"] == providers
            and segment["operator_event_counts"] == operators,
            "provider/operator summary differs from raw events",
        )
        provider_totals.update(providers)
        for provider, counts in operators.items():
            operator_totals.setdefault(provider, Counter()).update(counts)
        call_total += calls
        expected_files.add(profile.name)
        next_start = end
    _require(next_start == len(ordered_ids), "full-input provider coverage incomplete")
    _require(
        snapshot_file_inventory(profile_dir) == expected_files,
        "profile directory has missing or undeclared artifacts",
    )
    _require(provider_totals["CUDAExecutionProvider"] > 0, "CUDA inactive for full run")
    return {
        "status": "FULL_INPUT_PROVIDER_COVERAGE_REPLAY_PASS",
        "run_label": run_label,
        "role": role,
        "profiled_item_count": len(ordered_ids),
        "segment_count": len(segments),
        "inference_call_count": call_total,
        "provider_event_counts": dict(sorted(provider_totals.items())),
        "operator_event_counts": {
            provider: dict(sorted(counts.items()))
            for provider, counts in sorted(operator_totals.items())
        },
        "cpu_fallback_operator_types": sorted(operator_totals.get("CPUExecutionProvider", {})),
        "raw_profiles_retained": True,
        "embeddings_replayed": False,
        "model_loaded": False,
        "execution_authorized": False,
    }
