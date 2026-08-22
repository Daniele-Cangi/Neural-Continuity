from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.fidelity_authority import (
    FidelityGateError,
    verify_artifact_manifest,
)
from neural_continuity.m1_diagnostics.fidelity_evidence import _write_deterministic_npz

ACTIVATION_FORMAT_VERSION = "1.0.0"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _load_json(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FidelityGateError(code, f"cannot load JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FidelityGateError(code, f"JSON artifact is not an object: {path}")
    return value


def _artifact_entry(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _probe_key(prefix: str, probe_id: str) -> str:
    normalized = probe_id.replace("-", "_")
    if not normalized or not normalized.replace("_", "").isalnum():
        raise FidelityGateError("PROBE_ID_INVALID", f"probe ID is not archive-safe: {probe_id}")
    return f"{prefix}__{normalized}"


def _safe_batch_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise FidelityGateError("BATCH_INDEX_INVALID", "batch artifact path is invalid")
    candidate = (root / value).resolve()
    if candidate.parent != root or not candidate.is_file():
        raise FidelityGateError("BATCH_ARTIFACT_MISSING", f"batch artifact missing: {value}")
    return candidate


def prepare_capture_package(
    build_root: Path,
    capture_plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> str:
    plan_path = build_root / "capture-plan.json"
    preflight_path = build_root / "capture-preflight.json"
    _write_json(plan_path, capture_plan)
    plan_sha256 = sha256_file(plan_path)
    preflight_record = {**dict(preflight), "capture_plan_sha256": plan_sha256}
    _write_json(preflight_path, preflight_record)
    if sha256_file(plan_path) != plan_sha256:
        raise FidelityGateError("CAPTURE_PLAN_MUTATED", "capture plan changed before execution")
    return plan_sha256


def write_activation_batch(
    build_root: Path,
    batch_id: str,
    query_ids: Sequence[str],
    probe_mappings: Sequence[Mapping[str, Any]],
    integer_mappings: Sequence[Mapping[str, Any]],
    source_values: Sequence[np.ndarray],
    target_values: Sequence[np.ndarray],
    integer_values: Sequence[np.ndarray],
) -> dict[str, Any]:
    if (
        len(probe_mappings) != len(source_values)
        or len(probe_mappings) != len(target_values)
        or len(integer_mappings) != len(integer_values)
    ):
        raise FidelityGateError("ACTIVATION_COUNT_MISMATCH", "captured output count mismatch")
    floating_arrays: dict[str, np.ndarray] = {"query_ids": np.asarray(query_ids, dtype=np.str_)}
    integer_arrays: dict[str, np.ndarray] = {"query_ids": np.asarray(query_ids, dtype=np.str_)}
    for mapping, source_value, target_value in zip(
        probe_mappings, source_values, target_values, strict=True
    ):
        probe_id = mapping.get("probe_id")
        if not isinstance(probe_id, str):
            raise FidelityGateError("PROBE_ID_INVALID", "floating probe ID is missing")
        source = np.ascontiguousarray(source_value)
        target = np.ascontiguousarray(target_value)
        if source.shape != target.shape:
            raise FidelityGateError(
                "PAIRED_ACTIVATION_SHAPE_MISMATCH",
                f"source and target shape mismatch: {probe_id}",
            )
        if mapping.get("target_tensor_basis") == "post_quantize_dequantize_output" and (
            source.dtype != np.float32 or target.dtype != np.float32
        ):
            raise FidelityGateError(
                "PAIRED_ACTIVATION_DTYPE_INVALID",
                f"post-QDQ pair is not float32: {probe_id}",
            )
        floating_arrays[_probe_key("source", probe_id)] = source
        floating_arrays[_probe_key("target", probe_id)] = target

    for mapping, integer_value in zip(integer_mappings, integer_values, strict=True):
        probe_id = mapping.get("probe_id")
        if not isinstance(probe_id, str):
            raise FidelityGateError("PROBE_ID_INVALID", "integer probe ID is missing")
        value = np.ascontiguousarray(integer_value)
        if value.dtype.itemsize != 1 or not np.issubdtype(value.dtype, np.integer):
            raise FidelityGateError(
                "INTEGER_ACTIVATION_INVALID",
                f"integer activation is not 8-bit integral: {probe_id}",
            )
        integer_arrays[_probe_key("target_integer", probe_id)] = value

    floating_path = build_root / f"{batch_id}-floating.npz"
    integer_path = build_root / f"{batch_id}-integer.npz"
    _write_deterministic_npz(floating_path, floating_arrays)
    _write_deterministic_npz(integer_path, integer_arrays)
    return {
        "batch_id": batch_id,
        "query_ids": list(query_ids),
        "floating_path": floating_path.name,
        "integer_path": integer_path.name,
        "floating_sha256": sha256_file(floating_path),
        "integer_sha256": sha256_file(integer_path),
        "floating_size_bytes": floating_path.stat().st_size,
        "integer_size_bytes": integer_path.stat().st_size,
    }


def _nonfinite_count(value: np.ndarray) -> int:
    if np.issubdtype(value.dtype, np.floating) or np.issubdtype(value.dtype, np.complexfloating):
        return int(np.count_nonzero(~np.isfinite(value)))
    return 0


def _load_archive(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    except (OSError, KeyError, ValueError) as exc:
        raise FidelityGateError(
            "ACTIVATION_ARCHIVE_INVALID", f"cannot load activation archive {path}: {exc}"
        ) from exc


def summarize_activation_package(
    root: Path,
    capture_plan: Mapping[str, Any],
    batch_index: Mapping[str, Any],
) -> dict[str, Any]:
    probe_mappings = capture_plan.get("probe_mappings")
    integer_mappings = capture_plan.get("integer_mappings")
    batches = batch_index.get("batches")
    if (
        not isinstance(probe_mappings, list)
        or not isinstance(integer_mappings, list)
        or not isinstance(batches, list)
        or not batches
    ):
        raise FidelityGateError("CAPTURE_METADATA_INVALID", "capture plan or batch index invalid")

    probe_summary: dict[str, dict[str, Any]] = {}
    integer_summary: dict[str, dict[str, Any]] = {}
    all_query_ids: list[str] = []
    total_floating_bytes = 0
    total_integer_bytes = 0

    for mapping in probe_mappings:
        probe_id = mapping.get("probe_id")
        if not isinstance(probe_id, str) or probe_id in probe_summary:
            raise FidelityGateError("CAPTURE_METADATA_INVALID", "floating probe IDs invalid")
        probe_summary[probe_id] = {
            "probe_id": probe_id,
            "target_tensor_basis": mapping.get("target_tensor_basis"),
            "source_dtypes": set(),
            "target_dtypes": set(),
            "source_nonfinite_count": 0,
            "target_nonfinite_count": 0,
            "batch_shapes": [],
        }
    for mapping in integer_mappings:
        probe_id = mapping.get("probe_id")
        if not isinstance(probe_id, str) or probe_id in integer_summary:
            raise FidelityGateError("CAPTURE_METADATA_INVALID", "integer probe IDs invalid")
        integer_summary[probe_id] = {
            "probe_id": probe_id,
            "dtype": None,
            "value_count": 0,
            "qmin": None,
            "qmax": None,
            "qmin_count": 0,
            "qmax_count": 0,
            "observed_min": None,
            "observed_max": None,
        }

    for batch in batches:
        if not isinstance(batch, Mapping):
            raise FidelityGateError("BATCH_INDEX_INVALID", "batch record is not an object")
        batch_id = batch.get("batch_id")
        query_ids = batch.get("query_ids")
        if not isinstance(batch_id, str) or not isinstance(query_ids, list):
            raise FidelityGateError("BATCH_INDEX_INVALID", "batch identity is invalid")
        if any(not isinstance(query_id, str) for query_id in query_ids):
            raise FidelityGateError("BATCH_INDEX_INVALID", "batch query IDs are invalid")
        floating_path = _safe_batch_path(root, batch.get("floating_path"))
        integer_path = _safe_batch_path(root, batch.get("integer_path"))
        if sha256_file(floating_path) != batch.get("floating_sha256") or sha256_file(
            integer_path
        ) != batch.get("integer_sha256"):
            raise FidelityGateError("BATCH_HASH_MISMATCH", f"batch hash mismatch: {batch_id}")
        total_floating_bytes += floating_path.stat().st_size
        total_integer_bytes += integer_path.stat().st_size
        floating = _load_archive(floating_path)
        integer = _load_archive(integer_path)
        archived_query_ids = [str(value) for value in floating.pop("query_ids").tolist()]
        integer_query_ids = [str(value) for value in integer.pop("query_ids").tolist()]
        if archived_query_ids != query_ids or integer_query_ids != query_ids:
            raise FidelityGateError("BATCH_QUERY_MISMATCH", f"batch query mismatch: {batch_id}")
        all_query_ids.extend(query_ids)

        expected_floating_keys: set[str] = set()
        for mapping in probe_mappings:
            probe_id = mapping["probe_id"]
            source_key = _probe_key("source", probe_id)
            target_key = _probe_key("target", probe_id)
            expected_floating_keys.update((source_key, target_key))
            source = floating.get(source_key)
            target = floating.get(target_key)
            if source is None or target is None or source.shape != target.shape:
                raise FidelityGateError(
                    "PAIRED_ACTIVATION_SHAPE_MISMATCH",
                    f"replay pair mismatch: {probe_id}/{batch_id}",
                )
            if mapping.get("target_tensor_basis") == "post_quantize_dequantize_output" and (
                source.dtype != np.float32 or target.dtype != np.float32
            ):
                raise FidelityGateError(
                    "PAIRED_ACTIVATION_DTYPE_INVALID",
                    f"replay post-QDQ pair is not float32: {probe_id}",
                )
            summary = probe_summary[probe_id]
            summary["source_dtypes"].add(str(source.dtype))
            summary["target_dtypes"].add(str(target.dtype))
            summary["source_nonfinite_count"] += _nonfinite_count(source)
            summary["target_nonfinite_count"] += _nonfinite_count(target)
            summary["batch_shapes"].append(
                {
                    "batch_id": batch_id,
                    "source": list(source.shape),
                    "target": list(target.shape),
                }
            )
        if set(floating) != expected_floating_keys:
            raise FidelityGateError(
                "ACTIVATION_ARCHIVE_INVALID", f"unexpected floating keys: {batch_id}"
            )

        expected_integer_keys: set[str] = set()
        for mapping in integer_mappings:
            probe_id = mapping["probe_id"]
            key = _probe_key("target_integer", probe_id)
            expected_integer_keys.add(key)
            value = integer.get(key)
            if (
                value is None
                or value.dtype.itemsize != 1
                or not np.issubdtype(value.dtype, np.integer)
            ):
                raise FidelityGateError(
                    "INTEGER_ACTIVATION_INVALID",
                    f"replay integer activation invalid: {probe_id}/{batch_id}",
                )
            limits = np.iinfo(value.dtype)
            summary = integer_summary[probe_id]
            dtype_name = str(value.dtype)
            if summary["dtype"] not in (None, dtype_name):
                raise FidelityGateError(
                    "INTEGER_ACTIVATION_INVALID", f"integer dtype changed: {probe_id}"
                )
            observed_min = int(np.min(value))
            observed_max = int(np.max(value))
            summary["dtype"] = dtype_name
            summary["value_count"] += int(value.size)
            summary["qmin"] = int(limits.min)
            summary["qmax"] = int(limits.max)
            summary["qmin_count"] += int(np.count_nonzero(value == limits.min))
            summary["qmax_count"] += int(np.count_nonzero(value == limits.max))
            if summary["observed_min"] is None:
                summary["observed_min"] = observed_min
                summary["observed_max"] = observed_max
            else:
                summary["observed_min"] = min(summary["observed_min"], observed_min)
                summary["observed_max"] = max(summary["observed_max"], observed_max)
        if set(integer) != expected_integer_keys:
            raise FidelityGateError(
                "ACTIVATION_ARCHIVE_INVALID", f"unexpected integer keys: {batch_id}"
            )

    if (
        len(all_query_ids) != capture_plan.get("query_count")
        or len(all_query_ids) != len(set(all_query_ids))
        or all_query_ids != sorted(all_query_ids, key=str.encode)
    ):
        raise FidelityGateError("CAPTURE_QUERY_IDENTITY_INVALID", "captured query set is invalid")

    probes = []
    for probe_id in [mapping["probe_id"] for mapping in probe_mappings]:
        summary = probe_summary[probe_id]
        summary["source_dtypes"] = sorted(summary["source_dtypes"], key=str.encode)
        summary["target_dtypes"] = sorted(summary["target_dtypes"], key=str.encode)
        probes.append(summary)
    saturation = [integer_summary[mapping["probe_id"]] for mapping in integer_mappings]
    return {
        "batch_count": len(batches),
        "query_count": len(all_query_ids),
        "floating_probe_count": len(probe_mappings),
        "integer_probe_count": len(integer_mappings),
        "paired_shape_match": True,
        "total_floating_bytes": total_floating_bytes,
        "total_integer_bytes": total_integer_bytes,
        "probes": probes,
        "integer_saturation": saturation,
    }


def finalize_activation_package(
    build_root: Path,
    output_directory: str | Path,
    batch_index: Mapping[str, Any],
) -> dict[str, Any]:
    capture_plan = _load_json(build_root / "capture-plan.json", "CAPTURE_PLAN_INVALID")
    preflight = _load_json(build_root / "capture-preflight.json", "CAPTURE_PREFLIGHT_INVALID")
    plan_sha256 = sha256_file(build_root / "capture-plan.json")
    if (
        preflight.get("status") != "PASS"
        or preflight.get("capture_plan_sha256") != plan_sha256
        or preflight.get("derivative_final_output_fidelity") != "PASS"
        or preflight.get("activations_read_before_preflight") is not False
    ):
        raise FidelityGateError("CAPTURE_PREFLIGHT_INVALID", "capture preflight did not pass")

    batch_index_path = build_root / "batch-index.json"
    _write_json(batch_index_path, batch_index)
    summary = summarize_activation_package(build_root, capture_plan, batch_index)

    report_path = build_root / "activation-report.json"
    report = {
        "kind": "m1_transition_b_v2_activation_capture_report",
        "status": "COMPLETE",
        "activation_capture_status": "CAPTURED",
        "summary": summary,
        "capture_plan_sha256": plan_sha256,
        "activations_read": True,
        "scientific_decision_recomputed": False,
        "frozen_transition_b_v1_scientific_decision": "FAIL",
    }
    _write_json(report_path, report)

    replay_path = build_root / "replay-bundle.json"
    replay = {
        "replay_format_version": ACTIVATION_FORMAT_VERSION,
        "capture_plan_path": "capture-plan.json",
        "capture_preflight_path": "capture-preflight.json",
        "batch_index_path": batch_index_path.name,
        "report_path": report_path.name,
        "expected_status": "COMPLETE",
        "expected_activation_capture_status": "CAPTURED",
        "replay_requires_model_execution": False,
    }
    _write_json(replay_path, replay)

    artifact_paths = sorted(
        (path for path in build_root.iterdir() if path.is_file()),
        key=lambda path: path.name.encode(),
    )
    artifacts = [_artifact_entry(build_root, path) for path in artifact_paths]
    manifest_path = build_root / "artifact-manifest.json"
    manifest = {
        "kind": "m1_transition_b_v2_activation_capture_manifest",
        "status": "COMPLETE",
        "activation_capture_status": "CAPTURED",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "tamper_evident": True,
        "model_execution_used": True,
        "replay_requires_model_execution": False,
    }
    _write_json(manifest_path, manifest)

    output_path = Path(output_directory).resolve()
    if output_path.exists():
        raise FidelityGateError("OUTPUT_ALREADY_EXISTS", f"output exists: {output_path}")
    os.replace(build_root, output_path)
    return {
        "output_directory": str(output_path),
        "artifact_manifest_sha256": sha256_file(output_path / manifest_path.name),
        "status": "COMPLETE",
        "activation_capture_status": "CAPTURED",
        "batch_count": summary["batch_count"],
        "query_count": summary["query_count"],
        "floating_probe_count": summary["floating_probe_count"],
        "integer_probe_count": summary["integer_probe_count"],
        "total_floating_bytes": summary["total_floating_bytes"],
        "total_integer_bytes": summary["total_integer_bytes"],
    }


def replay_activation_capture(
    bundle_path: str | Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    bundle_file = Path(bundle_path).resolve()
    root = bundle_file.parent
    manifest = verify_artifact_manifest(root, expected_manifest_sha256)
    bundle = _load_json(bundle_file, "REPLAY_BUNDLE_INVALID")
    if (
        bundle.get("replay_format_version") != ACTIVATION_FORMAT_VERSION
        or bundle.get("replay_requires_model_execution") is not False
    ):
        raise FidelityGateError("REPLAY_POLICY_INVALID", "activation replay policy invalid")
    capture_plan = _load_json(root / "capture-plan.json", "CAPTURE_PLAN_INVALID")
    batch_index = _load_json(root / "batch-index.json", "BATCH_INDEX_INVALID")
    report = _load_json(root / "activation-report.json", "ACTIVATION_REPORT_INVALID")
    summary = summarize_activation_package(root, capture_plan, batch_index)
    if (
        report.get("summary") != summary
        or report.get("status") != "COMPLETE"
        or report.get("activation_capture_status") != "CAPTURED"
        or bundle.get("expected_status") != "COMPLETE"
        or bundle.get("expected_activation_capture_status") != "CAPTURED"
        or manifest.get("status") != "COMPLETE"
        or manifest.get("activation_capture_status") != "CAPTURED"
    ):
        raise FidelityGateError("REPLAY_STATUS_MISMATCH", "activation replay mismatch")
    return {
        "status": "COMPLETE",
        "activation_capture_status": "CAPTURED",
        "replay_verified": True,
        "status_match": True,
        "summary_match": True,
        "model_execution_used": False,
        "batch_count": summary["batch_count"],
        "query_count": summary["query_count"],
        "floating_probe_count": summary["floating_probe_count"],
        "integer_probe_count": summary["integer_probe_count"],
    }
