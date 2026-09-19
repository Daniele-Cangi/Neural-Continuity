"""Tamper-evident package and model-free replay for the profile storage probe."""

from __future__ import annotations

import json
import math
import shutil
import uuid
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.cuda_null_full_profile_probe_authority import (
    BUDGET_MANIFEST_SHA256,
    DATASET_SHA256,
    GATE_MANIFEST_SHA256,
    PLAN_SHA256,
    REVIEW_SHA256,
    SOURCE_SHA256,
)
from neural_continuity.m1_diagnostics.cuda_null_full_provider_coverage import _counts
from neural_continuity.m1_diagnostics.cuda_null_paths import (
    has_linked_ancestor,
    snapshot_file_inventory,
)
from neural_continuity.m1_diagnostics.cuda_null_sentinel_readiness import EPOCH_LAYOUT

_VERSION = "1.0.0"
_ROLES = {"documents": 256, "measurement_null_queries": 81}
_RAW_OBSERVATION_BYTES = 3_881_041_920
_METADATA_RESERVE_BYTES = 2 * 1024**3
_OPERATIONAL_FREE_RESERVE_BYTES = 20 * 1024**3
_FILES = {"probe-record.json", "replay-bundle.json", "artifact-manifest.json"}
_MAX_JSON_BYTES = 16 * 1024 * 1024
_DYNAMIC_RECORD_FIELDS = {
    "profiles",
    "projected_profile_bytes_per_epoch",
    "projected_profile_bytes_120_epochs",
    "required_evidence_storage_bytes",
    "minimum_free_bytes_before_execution",
    "free_bytes_before_probe",
    "storage_budget_satisfied",
}


class FullProfileProbeReplayBlocked(ValueError):
    """The profile storage probe package did not replay exactly."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise FullProfileProbeReplayBlocked(reason)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        _require(path.stat().st_size <= _MAX_JSON_BYTES, f"{path.name} exceeds size limit")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FullProfileProbeReplayBlocked(f"cannot decode {path.name}") from exc
    _require(isinstance(value, dict), f"{path.name} is not an object")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _profile_paths(record: dict[str, Any]) -> set[str]:
    profiles = record.get("profiles")
    if not isinstance(profiles, list):
        raise FullProfileProbeReplayBlocked("probe profile set differs")
    _require(len(profiles) == 8, "probe profile set differs")
    expected_keys = [(label, role) for label, _batch in EPOCH_LAYOUT for role in _ROLES]
    paths: set[str] = set()
    for profile, (label, role) in zip(profiles, expected_keys, strict=True):
        _require(
            isinstance(profile, dict)
            and profile.get("run_label") == label
            and profile.get("role") == role,
            "probe profile order differs",
        )
        relative = profile.get("profile_path")
        _require(
            isinstance(relative, str)
            and relative == f"profiles/{label}.{role}.json"
            and relative not in paths,
            "probe profile path differs",
        )
        _require(
            _is_sha256(profile.get("embedding_sha256"))
            and _is_sha256(profile.get("profile_sha256")),
            "probe profile digest malformed",
        )
        paths.add(relative)
    return paths


def _recompute(record: dict[str, Any], root: Path) -> dict[str, Any]:
    expected = {
        "kind": "m1_cuda_full_profile_storage_probe_observation",
        "version": _VERSION,
        "status": "TECHNICAL_PROFILE_STORAGE_CAPTURED_NOT_QUALIFYING",
        "review_sha256": REVIEW_SHA256,
        "execution_plan_sha256": PLAN_SHA256,
        "technical_gate_manifest_sha256": GATE_MANIFEST_SHA256,
        "budget_preflight_manifest_sha256": BUDGET_MANIFEST_SHA256,
        "dataset_manifest_sha256": DATASET_SHA256,
        "source_onnx_sha256": SOURCE_SHA256,
        "document_profile_item_count": 256,
        "query_profile_item_count": 81,
        "maximum_profile_segment_items": 256,
        "run_layout": [{"label": label, "batch_size": size} for label, size in EPOCH_LAYOUT],
        "ordered_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "raw_observation_bytes_120_epochs": _RAW_OBSERVATION_BYTES,
        "profile_projection_safety_multiplier": 2,
        "metadata_reserve_bytes": _METADATA_RESERVE_BYTES,
        "operational_free_reserve_bytes": _OPERATIONAL_FREE_RESERVE_BYTES,
        "projection_is_guaranteed_upper_bound": False,
        "raw_profiles_retained": True,
        "embeddings_retained": False,
        "rankings_or_metrics_retained": False,
        "qualifying_detection_evidence": False,
        "full_corpus_qualification_started": False,
        "full_corpus_execution_authorized": False,
        "candidate_or_int8_executed": False,
        "holdout_accessed": False,
        "scientific_decision": "NOT_EVALUATED",
    }
    _require(
        set(record) == set(expected) | _DYNAMIC_RECORD_FIELDS,
        "probe record schema differs",
    )
    _require(
        all(record.get(key) == value for key, value in expected.items()), "probe scope differs"
    )
    paths = _profile_paths(record)
    summaries: list[dict[str, Any]] = []
    projected_per_epoch = 0
    for profile, (label, batch_size) in zip(
        record["profiles"],
        [(label, size) for label, size in EPOCH_LAYOUT for _role in _ROLES],
        strict=True,
    ):
        role = profile["role"]
        count = _ROLES[role]
        path = root / profile["profile_path"]
        _require(
            path.is_file() and not has_linked_ancestor(path), "probe raw profile missing or linked"
        )
        _require(
            path.stat().st_size == profile["size_bytes"]
            and sha256_file(path) == profile["profile_sha256"],
            "probe raw profile integrity differs",
        )
        calls = math.ceil(count / batch_size)
        providers, operators = _counts(path, calls)
        _require(
            profile["batch_size"] == batch_size
            and profile["item_count"] == count
            and profile["inference_call_count"] == calls
            and profile["provider_event_counts"] == providers
            and profile["operator_event_counts"] == operators,
            "probe profile summary differs",
        )
        multiplier = math.ceil(5183 / 256) if role == "documents" else 1
        projected_per_epoch += profile["size_bytes"] * multiplier
        summaries.append({"run_label": label, "role": role, "size_bytes": profile["size_bytes"]})
    projected_120 = projected_per_epoch * 120
    required = _RAW_OBSERVATION_BYTES + 2 * projected_120 + _METADATA_RESERVE_BYTES
    _require(
        record["projected_profile_bytes_per_epoch"] == projected_per_epoch
        and record["projected_profile_bytes_120_epochs"] == projected_120
        and record["required_evidence_storage_bytes"] == required
        and record["minimum_free_bytes_before_execution"]
        == required + _OPERATIONAL_FREE_RESERVE_BYTES,
        "probe storage projection differs",
    )
    free_bytes = record.get("free_bytes_before_probe")
    if not isinstance(free_bytes, int) or isinstance(free_bytes, bool):
        raise FullProfileProbeReplayBlocked("probe free-space observation invalid")
    _require(free_bytes >= 0, "probe free-space observation invalid")
    _require(
        record.get("storage_budget_satisfied")
        is (free_bytes >= required + _OPERATIONAL_FREE_RESERVE_BYTES),
        "probe storage budget outcome differs",
    )
    return {
        "profiles": summaries,
        "projected_profile_bytes_per_epoch": projected_per_epoch,
        "projected_profile_bytes_120_epochs": projected_120,
        "required_evidence_storage_bytes": required,
        "minimum_free_bytes_before_execution": required + _OPERATIONAL_FREE_RESERVE_BYTES,
        "free_bytes_before_probe": free_bytes,
        "storage_budget_satisfied": record["storage_budget_satisfied"],
        "profile_paths": paths,
    }


def replay_full_profile_probe(bundle: Path, external_manifest_sha256: str) -> dict[str, Any]:
    try:
        _require(_is_sha256(external_manifest_sha256), "probe manifest external hash malformed")
        bundle = Path(bundle).absolute()
        _require(
            bundle.name == "replay-bundle.json" and not has_linked_ancestor(bundle),
            "probe bundle path invalid",
        )
        root = bundle.parent
        record = _read_json(root / "probe-record.json")
        computed = _recompute(record, root)
        expected_files = _FILES | computed["profile_paths"]
        _require(
            snapshot_file_inventory(root) == expected_files, "probe artifact inventory differs"
        )
        manifest_path = root / "artifact-manifest.json"
        _require(
            sha256_file(manifest_path) == external_manifest_sha256,
            "probe manifest external hash differs",
        )
        manifest = _read_json(manifest_path)
        entries = manifest.get("artifacts")
        if not isinstance(entries, list):
            raise FullProfileProbeReplayBlocked("probe manifest schema differs")
        names = sorted(expected_files - {"artifact-manifest.json"})
        _require(
            manifest.get("kind") == "m1_cuda_full_profile_storage_probe_manifest"
            and manifest.get("version") == _VERSION
            and len(entries) == len(names),
            "probe manifest schema differs",
        )
        for entry, name in zip(entries, names, strict=True):
            path = root / name
            _require(
                isinstance(entry, dict)
                and entry
                == {
                    "path": name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                },
                f"probe manifest entry differs: {name}",
            )
        _require(
            _read_json(bundle)
            == {
                "kind": "m1_cuda_full_profile_storage_probe_replay",
                "version": _VERSION,
                "model_required_for_replay": False,
                "qualifying_detection_evidence": False,
                "full_corpus_execution_authorized": False,
            },
            "probe replay bundle scope differs",
        )
        return {
            "replay_status": "PASS",
            "status": record["status"],
            "artifact_manifest_sha256": external_manifest_sha256,
            **{key: value for key, value in computed.items() if key != "profile_paths"},
            "model_loaded": False,
            "full_corpus_execution_authorized": False,
            "scientific_decision": "NOT_EVALUATED",
        }
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, OverflowError) as exc:
        return {
            "replay_status": "BLOCKED",
            "model_loaded": False,
            "full_corpus_execution_authorized": False,
            "scientific_decision": "NOT_EVALUATED",
            "reason": str(exc),
        }


def write_full_profile_probe(
    output: Path,
    record: dict[str, Any],
    raw_profiles: dict[str, Path],
) -> tuple[Path, str]:
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        for relative, source in raw_profiles.items():
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        (temporary / "probe-record.json").write_bytes(canonical_json_bytes(record) + b"\n")
        (temporary / "replay-bundle.json").write_bytes(
            canonical_json_bytes(
                {
                    "kind": "m1_cuda_full_profile_storage_probe_replay",
                    "version": _VERSION,
                    "model_required_for_replay": False,
                    "qualifying_detection_evidence": False,
                    "full_corpus_execution_authorized": False,
                }
            )
            + b"\n"
        )
        names = sorted(snapshot_file_inventory(temporary))
        manifest = {
            "kind": "m1_cuda_full_profile_storage_probe_manifest",
            "version": _VERSION,
            "artifacts": [
                {
                    "path": name,
                    "size_bytes": (temporary / name).stat().st_size,
                    "sha256": sha256_file(temporary / name),
                }
                for name in names
            ],
        }
        (temporary / "artifact-manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
        digest = sha256_file(temporary / "artifact-manifest.json")
        _require(
            replay_full_profile_probe(temporary / "replay-bundle.json", digest)["replay_status"]
            == "PASS",
            "new probe package did not replay",
        )
        temporary.replace(output)
        return output, digest
    except Exception:
        if temporary.exists() and temporary.parent == output.parent:
            shutil.rmtree(temporary)
        raise
