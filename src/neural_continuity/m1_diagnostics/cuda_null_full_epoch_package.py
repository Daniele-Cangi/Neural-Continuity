"""Tamper-evident, model-free replay for one CUDA full-corpus epoch package."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.cuda_null_full_comparison_replay import (
    replay_full_comparison,
)
from neural_continuity.m1_diagnostics.cuda_null_full_epoch_format import (
    FORMAT_VERSION,
    ROLES,
    RUN_LAYOUT,
    SHA256,
    validate_full_epoch_plan,
    validate_full_epoch_runtime,
)
from neural_continuity.m1_diagnostics.cuda_null_full_extrema import (
    batch_epoch_unit,
    single_comparison_unit,
)
from neural_continuity.m1_diagnostics.cuda_null_paths import (
    has_linked_ancestor,
    snapshot_file_inventory,
)

_METADATA_LIMIT = 16 * 1024 * 1024
_TOP_LEVEL = {
    "epoch-plan.json",
    "runtime-inventory.json",
    "run-records.json",
    "rankings.json",
    "metrics.json",
    "within-epoch-comparisons.json",
    "replay-bundle.json",
}
COMPARISON_PAIRS = (
    ("repeated_inference", "batch_16_primary", "batch_16_repeat"),
    ("batch_size_variation", "batch_1_primary", "batch_16_primary"),
    ("batch_size_variation", "batch_1_primary", "batch_64_primary"),
    ("batch_size_variation", "batch_16_primary", "batch_64_primary"),
)


class FullEpochPackageBlocked(ValueError):
    """A full-corpus epoch package is incomplete, altered, or inconsistent."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise FullEpochPackageBlocked(reason)


def _validate_retrieval_run_maps(rankings: Any, metrics: Any) -> None:
    labels = {label for label, _batch in RUN_LAYOUT}
    _require(
        isinstance(rankings, dict)
        and isinstance(metrics, dict)
        and set(rankings) == labels
        and set(metrics) == labels,
        "retrieval run set differs",
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    try:
        _require(path.stat().st_size <= _METADATA_LIMIT, "metadata exceeds memory bound")
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FullEpochPackageBlocked(f"metadata cannot be decoded: {path.name}") from exc


def _safe_relative(value: Any) -> str:
    _require(isinstance(value, str) and bool(value), "artifact path missing")
    pure = PurePosixPath(value)
    windows = Path(value)
    _require(
        not pure.is_absolute()
        and not windows.is_absolute()
        and not windows.drive
        and "\\" not in value
        and all(part not in ("", ".", "..") for part in pure.parts),
        "artifact path escapes package",
    )
    return value


def _records(value: Any) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    _require(
        isinstance(value, list) and len(value) == len(RUN_LAYOUT) * len(ROLES), "run set differs"
    )
    expected = [(label, role) for label, _batch in RUN_LAYOUT for role in ROLES]
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for record, key in zip(value, expected, strict=True):
        _require(isinstance(record, dict), "run record malformed")
        _require((record.get("run_label"), record.get("role")) == key, "run record order differs")
        _require(key not in by_key, "duplicate run record")
        by_key[key] = record
    return value, by_key


def _expected_artifacts(records: list[dict[str, Any]]) -> set[str]:
    expected = set(_TOP_LEVEL)
    segment_count = 0
    for record in records:
        expected.add(_safe_relative(record.get("array_path")))
        segments = record.get("segments")
        if not isinstance(segments, list) or not segments:
            raise FullEpochPackageBlocked("provider segments missing")
        segment_count += len(segments)
        for segment in segments:
            _require(isinstance(segment, dict), "provider segment malformed")
            expected.add(_safe_relative(segment.get("profile_path")))
    _require(
        len(expected) == len(_TOP_LEVEL) + len(records) + segment_count,
        "artifact paths overlap",
    )
    return expected


def _verify_manifest(package: Path, external_sha256: str, expected: set[str]) -> None:
    _require(
        isinstance(external_sha256, str) and SHA256.fullmatch(external_sha256) is not None,
        "external manifest hash invalid",
    )
    manifest_path = package / "artifact-manifest.json"
    _require(
        manifest_path.is_file() and not has_linked_ancestor(manifest_path),
        "manifest missing or linked",
    )
    _require(sha256_file(manifest_path) == external_sha256, "manifest external hash differs")
    manifest = _read_json(manifest_path)
    _require(
        isinstance(manifest, dict)
        and set(manifest) == {"kind", "version", "artifacts"}
        and manifest["kind"] == "m1_cuda_null_full_corpus_epoch_artifact_manifest"
        and manifest["version"] == FORMAT_VERSION
        and isinstance(manifest["artifacts"], list),
        "manifest schema differs",
    )
    entries = manifest["artifacts"]
    _require(len(entries) == len(expected), "manifest artifact count differs")
    for entry, relative in zip(entries, sorted(expected), strict=True):
        artifact = package / relative
        _require(
            isinstance(entry, dict)
            and set(entry) == {"path", "size_bytes", "sha256"}
            and entry["path"] == relative
            and type(entry["size_bytes"]) is int
            and entry["size_bytes"] >= 0
            and isinstance(entry["sha256"], str)
            and SHA256.fullmatch(entry["sha256"]) is not None
            and artifact.is_file()
            and not has_linked_ancestor(artifact)
            and artifact.stat().st_size == entry["size_bytes"]
            and sha256_file(artifact) == entry["sha256"],
            f"artifact integrity differs: {relative}",
        )
    _require(
        snapshot_file_inventory(package) == expected | {"artifact-manifest.json"},
        "package artifact inventory differs",
    )


def replay_full_epoch_package(
    bundle_path: Path,
    external_manifest_sha256: str,
    external_authority_sha256: str,
) -> dict[str, Any]:
    """Replay one complete epoch without a model; never authenticate the journal."""
    try:
        bundle = Path(bundle_path).absolute()
        _require(bundle.name == "replay-bundle.json", "replay bundle name differs")
        _require(
            bundle.is_file() and not has_linked_ancestor(bundle), "replay bundle missing or linked"
        )
        package = bundle.parent
        plan = _read_json(package / "epoch-plan.json")
        runtime = _read_json(package / "runtime-inventory.json")
        records, by_key = _records(_read_json(package / "run-records.json"))
        expected = _expected_artifacts(records)
        _verify_manifest(package, external_manifest_sha256, expected)
        _require(
            _read_json(bundle)
            == {
                "kind": "m1_cuda_null_full_corpus_epoch_replay",
                "version": FORMAT_VERSION,
                "model_required_for_replay": False,
                "journal_verified": False,
                "full_corpus_complete": False,
                "scientific_decision": "NOT_EVALUATED",
            },
            "replay bundle scope differs",
        )
        _require(isinstance(plan, dict) and isinstance(runtime, dict), "epoch metadata malformed")
        validate_full_epoch_plan(plan, external_authority_sha256)
        validate_full_epoch_runtime(runtime)
        rankings = _read_json(package / "rankings.json")
        metrics = _read_json(package / "metrics.json")
        _validate_retrieval_run_maps(rankings, metrics)
        comparisons = _read_json(package / "within-epoch-comparisons.json")
        _require(
            isinstance(comparisons, list) and len(comparisons) == len(COMPARISON_PAIRS),
            "comparison set differs",
        )
        replayed: list[dict[str, Any]] = []
        for item, (family, left, right) in zip(comparisons, COMPARISON_PAIRS, strict=True):
            _require(
                isinstance(item, dict)
                and set(item) == {"family", "comparison"}
                and item["family"] == family
                and isinstance(item["comparison"], dict)
                and item["comparison"].get("left_run_label") == left
                and item["comparison"].get("right_run_label") == right,
                "comparison identity differs",
            )
            result = replay_full_comparison(
                package,
                package,
                recorded_comparison=item["comparison"],
                left_run_label=left,
                right_run_label=right,
                document_ids=plan["document_ids"],
                query_ids=plan["query_ids"],
                qrels=plan["qrels"],
                left_document_record=by_key[(left, "documents")],
                left_query_record=by_key[(left, "measurement_null_queries")],
                right_document_record=by_key[(right, "documents")],
                right_query_record=by_key[(right, "measurement_null_queries")],
                left_rankings=rankings[left],
                left_metrics=metrics[left],
                right_rankings=rankings[right],
                right_metrics=metrics[right],
            )
            _require(result["status"] == "FULL_COMPARISON_REPLAY_PASS", "comparison replay blocked")
            replayed.append(result["comparison"])
        repeated = single_comparison_unit("repeated_inference", plan["epoch_number"], replayed[0])
        batch = batch_epoch_unit(plan["epoch_number"], replayed[1:])
        return {
            "replay_status": "PASS",
            "technical_structure_status": "PASS",
            "artifact_manifest_sha256": external_manifest_sha256,
            "execution_authority_sha256": external_authority_sha256,
            "epoch_number": plan["epoch_number"],
            "attempt_number": plan["attempt_number"],
            "process_instance_id": runtime["process_instance_id"],
            "previous_completed_epoch_manifest_sha256": plan[
                "previous_completed_epoch_manifest_sha256"
            ],
            "within_epoch_units": [repeated, batch],
            "journal_verified": False,
            "full_corpus_complete": False,
            "qualifying_detection_evidence": True,
            "capture_status": "CAPTURED_NOT_DECIDED",
            "scientific_decision": "NOT_EVALUATED",
            "model_loaded": False,
            "onnx_graph_loaded": False,
            "execution_authorized": False,
        }
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, OverflowError) as exc:
        return {
            "replay_status": "BLOCKED",
            "technical_structure_status": "BLOCKED",
            "journal_verified": False,
            "full_corpus_complete": False,
            "qualifying_detection_evidence": False,
            "scientific_decision": "NOT_EVALUATED",
            "model_loaded": False,
            "onnx_graph_loaded": False,
            "execution_authorized": False,
            "reason": str(exc),
        }


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def finalize_full_epoch_package(
    staging_directory: Path,
    output_directory: Path,
    *,
    external_authority_sha256: str,
) -> tuple[Path, str]:
    """Seal and replay a populated staging package before atomic publication.

    Failures deliberately leave staging intact so the attempt can be retained.
    """
    staging = Path(staging_directory).absolute()
    output = Path(output_directory).absolute()
    _require(staging.is_dir() and not has_linked_ancestor(staging), "staging missing or linked")
    _require(
        staging.parent == output.parent
        and not output.exists()
        and not output.is_symlink()
        and output.name.startswith("epoch-")
        and output.name[6:].isdigit()
        and len(output.name) == 10,
        "final epoch output path is not fresh or canonical",
    )
    _require(
        "artifact-manifest.json" not in snapshot_file_inventory(staging)
        and "replay-bundle.json" not in snapshot_file_inventory(staging),
        "staging is already sealed",
    )
    _write_json(
        staging / "replay-bundle.json",
        {
            "kind": "m1_cuda_null_full_corpus_epoch_replay",
            "version": FORMAT_VERSION,
            "model_required_for_replay": False,
            "journal_verified": False,
            "full_corpus_complete": False,
            "scientific_decision": "NOT_EVALUATED",
        },
    )
    records, _by_key = _records(_read_json(staging / "run-records.json"))
    expected = _expected_artifacts(records)
    _require(snapshot_file_inventory(staging) == expected, "staging artifact inventory differs")
    _write_json(
        staging / "artifact-manifest.json",
        {
            "kind": "m1_cuda_null_full_corpus_epoch_artifact_manifest",
            "version": FORMAT_VERSION,
            "artifacts": [
                {
                    "path": relative,
                    "size_bytes": (staging / relative).stat().st_size,
                    "sha256": sha256_file(staging / relative),
                }
                for relative in sorted(expected)
            ],
        },
    )
    manifest_sha256 = sha256_file(staging / "artifact-manifest.json")
    replay = replay_full_epoch_package(
        staging / "replay-bundle.json",
        manifest_sha256,
        external_authority_sha256,
    )
    _require(replay.get("replay_status") == "PASS", "new full epoch package did not replay")
    _require(output.name == f"epoch-{replay['epoch_number']:04d}", "output epoch number differs")
    staging.replace(output)
    return output, manifest_sha256
