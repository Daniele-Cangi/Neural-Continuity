"""Tamper-evident compressed derivative of the bounded profile probe."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.cuda_null_full_profile_archive import (
    ProfileArchiveBlocked,
    compress_profile,
    materialize_profile,
)
from neural_continuity.m1_diagnostics.cuda_null_full_profile_probe_replay import (
    replay_full_profile_probe,
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
_MAX_JSON_BYTES = 16 * 1024 * 1024
_PACKAGE_FILES = {
    "compression-record.json",
    "replay-bundle.json",
    "artifact-manifest.json",
}


class CompressedProfilePackageBlocked(ValueError):
    """The compressed profile derivative did not verify exactly."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise CompressedProfilePackageBlocked(reason)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        _require(path.stat().st_size <= _MAX_JSON_BYTES, f"{path.name} exceeds size limit")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CompressedProfilePackageBlocked(f"cannot decode {path.name}") from exc
    _require(isinstance(value, dict), f"{path.name} is not an object")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _expected_profile_keys() -> list[tuple[str, int, str]]:
    return [(label, batch_size, role) for label, batch_size in EPOCH_LAYOUT for role in _ROLES]


def _projection(profiles: list[dict[str, Any]]) -> dict[str, int]:
    per_epoch = 0
    for profile in profiles:
        multiplier = 21 if profile["role"] == "documents" else 1
        per_epoch += profile["archive"]["archive_size_bytes"] * multiplier
    projected = per_epoch * 120
    required = _RAW_OBSERVATION_BYTES + 2 * projected + _METADATA_RESERVE_BYTES
    return {
        "projected_compressed_profile_bytes_per_epoch": per_epoch,
        "projected_compressed_profile_bytes_120_epochs": projected,
        "required_evidence_storage_bytes": required,
        "minimum_free_bytes_before_execution": required + _OPERATIONAL_FREE_RESERVE_BYTES,
    }


def _verify_profiles(record: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    profiles = record.get("profiles")
    if not isinstance(profiles, list):
        raise CompressedProfilePackageBlocked("compressed profile set differs")
    expected = _expected_profile_keys()
    _require(len(profiles) == len(expected), "compressed profile count differs")
    summaries: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="compressed-profile-replay-") as temporary:
        scratch = Path(temporary)
        for profile, (label, batch_size, role) in zip(profiles, expected, strict=True):
            _require(
                isinstance(profile, dict)
                and profile.get("run_label") == label
                and profile.get("batch_size") == batch_size
                and profile.get("role") == role
                and profile.get("item_count") == _ROLES[role]
                and profile.get("inference_call_count") == math.ceil(_ROLES[role] / batch_size)
                and _is_sha256(profile.get("embedding_sha256")),
                "compressed profile identity differs",
            )
            archive_path = profile.get("archive_path")
            archive = profile.get("archive")
            _require(
                isinstance(archive_path, str)
                and archive_path == f"profiles/{label}.{role}.json.gz"
                and isinstance(archive, dict),
                "compressed profile archive declaration differs",
            )
            path = root / archive_path
            try:
                with materialize_profile(path, archive, scratch) as raw_profile:
                    providers, operators = _counts(
                        raw_profile,
                        math.ceil(_ROLES[role] / batch_size),
                    )
            except ProfileArchiveBlocked as exc:
                raise CompressedProfilePackageBlocked(str(exc)) from exc
            _require(
                profile.get("provider_event_counts") == providers
                and profile.get("operator_event_counts") == operators,
                "compressed profile provider summary differs",
            )
            summaries.append(
                {
                    "run_label": label,
                    "role": role,
                    "raw_size_bytes": archive["raw_size_bytes"],
                    "archive_size_bytes": archive["archive_size_bytes"],
                }
            )
    return summaries


def replay_compressed_profile_package(
    bundle: Path,
    external_manifest_sha256: str,
) -> dict[str, Any]:
    """Replay the compressed derivative without a model or source package."""
    try:
        _require(_is_sha256(external_manifest_sha256), "external manifest hash malformed")
        bundle = Path(bundle).absolute()
        _require(
            bundle.name == "replay-bundle.json" and not has_linked_ancestor(bundle),
            "compressed replay bundle path invalid",
        )
        root = bundle.parent
        record = _read_json(root / "compression-record.json")
        expected_record = {
            "kind": "m1_cuda_full_profile_compression_observation",
            "version": _VERSION,
            "status": "TECHNICAL_PROFILE_COMPRESSION_VERIFIED_NOT_QUALIFYING",
            "compression": "deterministic_gzip_level_9",
            "source_profile_manifest_sha256": record.get("source_profile_manifest_sha256"),
            "raw_observation_bytes_120_epochs": _RAW_OBSERVATION_BYTES,
            "profile_projection_safety_multiplier": 2,
            "metadata_reserve_bytes": _METADATA_RESERVE_BYTES,
            "operational_free_reserve_bytes": _OPERATIONAL_FREE_RESERVE_BYTES,
            "projection_is_guaranteed_upper_bound": False,
            "raw_profiles_losslessly_archived": True,
            "model_required_for_replay": False,
            "qualifying_detection_evidence": False,
            "full_corpus_qualification_started": False,
            "full_corpus_execution_authorized": False,
            "candidate_or_int8_executed": False,
            "holdout_accessed": False,
            "scientific_decision": "NOT_EVALUATED",
        }
        dynamic_fields = {
            "profiles",
            "projected_compressed_profile_bytes_per_epoch",
            "projected_compressed_profile_bytes_120_epochs",
            "required_evidence_storage_bytes",
            "minimum_free_bytes_before_execution",
            "free_bytes_before_compression",
            "storage_budget_satisfied",
        }
        _require(
            set(record) == set(expected_record) | dynamic_fields
            and all(record.get(key) == value for key, value in expected_record.items())
            and _is_sha256(record.get("source_profile_manifest_sha256")),
            "compressed record scope differs",
        )
        summaries = _verify_profiles(record, root)
        projection = _projection(record["profiles"])
        _require(
            all(record.get(key) == value for key, value in projection.items()),
            "compressed storage projection differs",
        )
        free_bytes = record.get("free_bytes_before_compression")
        if not isinstance(free_bytes, int) or isinstance(free_bytes, bool):
            raise CompressedProfilePackageBlocked("compressed free-space value invalid")
        _require(
            record.get("storage_budget_satisfied")
            is (free_bytes >= projection["minimum_free_bytes_before_execution"]),
            "compressed storage budget outcome differs",
        )
        archive_paths = {profile["archive_path"] for profile in record["profiles"]}
        expected_files = _PACKAGE_FILES | archive_paths
        _require(
            snapshot_file_inventory(root) == expected_files,
            "compressed package inventory differs",
        )
        manifest_path = root / "artifact-manifest.json"
        _require(
            sha256_file(manifest_path) == external_manifest_sha256,
            "compressed manifest external hash differs",
        )
        manifest = _read_json(manifest_path)
        entries = manifest.get("artifacts")
        if not isinstance(entries, list):
            raise CompressedProfilePackageBlocked("compressed manifest schema differs")
        names = sorted(expected_files - {"artifact-manifest.json"})
        _require(
            manifest.get("kind") == "m1_cuda_full_profile_compression_manifest"
            and manifest.get("version") == _VERSION
            and len(entries) == len(names),
            "compressed manifest schema differs",
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
                f"compressed manifest entry differs: {name}",
            )
        _require(
            _read_json(bundle)
            == {
                "kind": "m1_cuda_full_profile_compression_replay",
                "version": _VERSION,
                "model_required_for_replay": False,
                "full_corpus_execution_authorized": False,
            },
            "compressed replay declaration differs",
        )
        return {
            "replay_status": "PASS",
            "status": record["status"],
            "artifact_manifest_sha256": external_manifest_sha256,
            "source_profile_manifest_sha256": record["source_profile_manifest_sha256"],
            "profiles": summaries,
            **projection,
            "free_bytes_before_compression": free_bytes,
            "storage_budget_satisfied": record["storage_budget_satisfied"],
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


def build_compressed_profile_package(
    source_bundle: Path,
    source_manifest_sha256: str,
    output: Path,
) -> dict[str, Any]:
    """Create a nonqualifying compressed derivative of a verified probe."""
    source_bundle = Path(source_bundle).absolute()
    output = Path(output).absolute()
    source_replay = replay_full_profile_probe(source_bundle, source_manifest_sha256)
    _require(source_replay.get("replay_status") == "PASS", "source probe replay blocked")
    _require(
        output.parent == source_bundle.parent.parent
        and not output.exists()
        and not has_linked_ancestor(output.parent),
        "compressed package output must be a fresh sibling evidence path",
    )
    source_record = _read_json(source_bundle.parent / "probe-record.json")
    source_profiles = source_record.get("profiles")
    if not isinstance(source_profiles, list):
        raise CompressedProfilePackageBlocked("source profile set differs")
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        profiles: list[dict[str, Any]] = []
        for source_profile, (label, batch_size, role) in zip(
            source_profiles,
            _expected_profile_keys(),
            strict=True,
        ):
            _require(
                isinstance(source_profile, dict)
                and source_profile.get("run_label") == label
                and source_profile.get("batch_size") == batch_size
                and source_profile.get("role") == role,
                "source profile order differs",
            )
            source_path = source_bundle.parent / str(source_profile["profile_path"])
            relative = f"profiles/{label}.{role}.json.gz"
            descriptor = compress_profile(source_path, temporary / relative)
            _require(
                descriptor["raw_size_bytes"] == source_profile["size_bytes"]
                and descriptor["raw_sha256"] == source_profile["profile_sha256"],
                "compressed source profile integrity differs",
            )
            profiles.append(
                {
                    "run_label": label,
                    "batch_size": batch_size,
                    "role": role,
                    "item_count": source_profile["item_count"],
                    "inference_call_count": source_profile["inference_call_count"],
                    "embedding_sha256": source_profile["embedding_sha256"],
                    "provider_event_counts": source_profile["provider_event_counts"],
                    "operator_event_counts": source_profile["operator_event_counts"],
                    "archive_path": relative,
                    "archive": descriptor,
                }
            )
        projection = _projection(profiles)
        free_bytes = shutil.disk_usage(output.parent).free
        record = {
            "kind": "m1_cuda_full_profile_compression_observation",
            "version": _VERSION,
            "status": "TECHNICAL_PROFILE_COMPRESSION_VERIFIED_NOT_QUALIFYING",
            "compression": "deterministic_gzip_level_9",
            "source_profile_manifest_sha256": source_manifest_sha256,
            "profiles": profiles,
            "raw_observation_bytes_120_epochs": _RAW_OBSERVATION_BYTES,
            **projection,
            "profile_projection_safety_multiplier": 2,
            "metadata_reserve_bytes": _METADATA_RESERVE_BYTES,
            "operational_free_reserve_bytes": _OPERATIONAL_FREE_RESERVE_BYTES,
            "free_bytes_before_compression": free_bytes,
            "storage_budget_satisfied": free_bytes
            >= projection["minimum_free_bytes_before_execution"],
            "projection_is_guaranteed_upper_bound": False,
            "raw_profiles_losslessly_archived": True,
            "model_required_for_replay": False,
            "qualifying_detection_evidence": False,
            "full_corpus_qualification_started": False,
            "full_corpus_execution_authorized": False,
            "candidate_or_int8_executed": False,
            "holdout_accessed": False,
            "scientific_decision": "NOT_EVALUATED",
        }
        (temporary / "compression-record.json").write_bytes(canonical_json_bytes(record) + b"\n")
        (temporary / "replay-bundle.json").write_bytes(
            canonical_json_bytes(
                {
                    "kind": "m1_cuda_full_profile_compression_replay",
                    "version": _VERSION,
                    "model_required_for_replay": False,
                    "full_corpus_execution_authorized": False,
                }
            )
            + b"\n"
        )
        names = sorted(snapshot_file_inventory(temporary))
        manifest = {
            "kind": "m1_cuda_full_profile_compression_manifest",
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
        replay = replay_compressed_profile_package(
            temporary / "replay-bundle.json",
            digest,
        )
        _require(replay.get("replay_status") == "PASS", "new compressed package blocked")
        temporary.replace(output)
        return {"output": str(output), **replay}
    except Exception:
        if temporary.exists() and temporary.parent == output.parent:
            shutil.rmtree(temporary)
        raise
