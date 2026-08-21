from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.fidelity_authority import (
    FidelityGateError,
    verify_artifact_manifest,
)

FIDELITY_FORMAT_VERSION = "1.0.0"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _artifact_entry(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.ascontiguousarray(array), allow_pickle=False)
    return stream.getvalue()


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name in sorted(arrays, key=str.encode):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(arrays[name]))


def _validate_outputs(query_ids: Sequence[str], outputs: Mapping[str, np.ndarray]) -> None:
    if not query_ids or len(query_ids) != len(set(query_ids)):
        raise FidelityGateError("FIDELITY_INPUT_INVALID", "query IDs are empty or duplicated")
    if list(query_ids) != sorted(query_ids, key=str.encode):
        raise FidelityGateError("FIDELITY_INPUT_INVALID", "query IDs are not in canonical order")
    expected_shape: tuple[int, int] | None = None
    for name in (
        "source_original",
        "source_instrumented",
        "target_original",
        "target_instrumented",
    ):
        value = outputs.get(name)
        if not isinstance(value, np.ndarray):
            raise FidelityGateError("FIDELITY_INPUT_INVALID", f"missing output matrix: {name}")
        if value.dtype != np.float32 or value.ndim != 2 or value.shape[0] != len(query_ids):
            raise FidelityGateError("FIDELITY_INPUT_INVALID", f"invalid output matrix: {name}")
        if not np.isfinite(value).all():
            raise FidelityGateError("FIDELITY_INPUT_INVALID", f"non-finite output matrix: {name}")
        if expected_shape is None:
            expected_shape = value.shape
        elif value.shape != expected_shape:
            raise FidelityGateError("FIDELITY_INPUT_INVALID", "output matrix shapes differ")


def exact_fidelity_comparison(
    query_ids: Sequence[str], outputs: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    _validate_outputs(query_ids, outputs)
    comparisons: dict[str, Any] = {}
    all_exact = True
    for role in ("source", "target"):
        original = outputs[f"{role}_original"]
        instrumented = outputs[f"{role}_instrumented"]
        exact = bool(np.array_equal(original, instrumented))
        all_exact = all_exact and exact
        delta = np.abs(original - instrumented)
        comparisons[role] = {
            "bitwise_equal": exact,
            "differing_value_count": int(np.count_nonzero(original != instrumented)),
            "max_abs_delta": float(np.max(delta)),
            "output_dtype": "float32",
            "output_shape": list(original.shape),
        }
    return {
        "status": "COMPLETE" if all_exact else "BLOCKED",
        "fidelity_status": "PASS" if all_exact else "BLOCKED",
        "comparison_semantics": "bitwise_exact_float32",
        "comparisons": comparisons,
    }


def write_fidelity_package(
    output_directory: str | Path,
    query_ids: Sequence[str],
    outputs: Mapping[str, np.ndarray],
    authority: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    output_path = Path(output_directory).resolve()
    if output_path.exists():
        raise FidelityGateError("OUTPUT_ALREADY_EXISTS", f"output exists: {output_path}")
    comparison = exact_fidelity_comparison(query_ids, outputs)
    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.building-", dir=output_path.parent)
    )
    try:
        observations_path = temporary_path / "final-output-observations.npz"
        arrays = {name: np.ascontiguousarray(value) for name, value in outputs.items()}
        arrays["query_ids"] = np.asarray(query_ids, dtype=np.str_)
        _write_deterministic_npz(observations_path, arrays)

        authority_path = temporary_path / "fidelity-authority.json"
        authority_record = {
            "kind": "m1_transition_b_v2_fidelity_authority",
            "status": "PASS",
            **dict(authority),
        }
        _write_json(authority_path, authority_record)

        report_path = temporary_path / "fidelity-report.json"
        report = {
            "kind": "m1_transition_b_v2_final_output_fidelity_report",
            **comparison,
            "query_count": len(query_ids),
            "batch_size": 16,
            "execution_provider": "CPUExecutionProvider",
            "final_outputs_only": True,
            "activations_read": False,
            "scientific_decision_recomputed": False,
            "frozen_transition_b_v1_scientific_decision": "FAIL",
            "execution": dict(execution),
        }
        _write_json(report_path, report)

        bundle_path = temporary_path / "replay-bundle.json"
        bundle = {
            "replay_format_version": FIDELITY_FORMAT_VERSION,
            "observation_path": observations_path.name,
            "authority_path": authority_path.name,
            "report_path": report_path.name,
            "query_count": len(query_ids),
            "expected_status": comparison["status"],
            "expected_fidelity_status": comparison["fidelity_status"],
            "comparison_semantics": "bitwise_exact_float32",
            "capture_model_execution_used": True,
            "replay_requires_model_execution": False,
        }
        _write_json(bundle_path, bundle)

        artifact_paths = [observations_path, authority_path, report_path, bundle_path]
        artifacts = sorted(
            (_artifact_entry(temporary_path, path) for path in artifact_paths),
            key=lambda item: str(item["path"]).encode(),
        )
        manifest_path = temporary_path / "artifact-manifest.json"
        manifest = {
            "kind": "m1_transition_b_v2_final_output_fidelity_manifest",
            "status": comparison["status"],
            "fidelity_status": comparison["fidelity_status"],
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "tamper_evident": True,
            "replay_requires_model_execution": False,
        }
        _write_json(manifest_path, manifest)
        os.replace(temporary_path, output_path)
        return {
            "output_directory": str(output_path),
            "artifact_manifest_sha256": sha256_file(output_path / manifest_path.name),
            "status": comparison["status"],
            "fidelity_status": comparison["fidelity_status"],
            "comparisons": comparison["comparisons"],
            "query_count": len(query_ids),
        }
    except Exception:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise


def _load_observations(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            query_ids = [str(value) for value in archive["query_ids"].tolist()]
            outputs = {
                name: np.ascontiguousarray(archive[name], dtype=np.float32)
                for name in (
                    "source_original",
                    "source_instrumented",
                    "target_original",
                    "target_instrumented",
                )
            }
    except (OSError, KeyError, ValueError) as exc:
        raise FidelityGateError(
            "REPLAY_OBSERVATION_INVALID", f"cannot load fidelity observations: {exc}"
        ) from exc
    return query_ids, outputs


def replay_fidelity(bundle_path: str | Path, expected_manifest_sha256: str) -> dict[str, Any]:
    bundle_file = Path(bundle_path).resolve()
    root = bundle_file.parent
    manifest = verify_artifact_manifest(root, expected_manifest_sha256)
    try:
        bundle = json.loads(bundle_file.read_text(encoding="utf-8"))
        report = json.loads((root / bundle["report_path"]).read_text(encoding="utf-8"))
        authority = json.loads((root / bundle["authority_path"]).read_text(encoding="utf-8"))
    except (OSError, KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FidelityGateError("REPLAY_BUNDLE_INVALID", f"invalid replay metadata: {exc}") from exc
    if (
        not isinstance(bundle, dict)
        or bundle.get("replay_format_version") != FIDELITY_FORMAT_VERSION
        or bundle.get("replay_requires_model_execution") is not False
        or bundle.get("comparison_semantics") != "bitwise_exact_float32"
    ):
        raise FidelityGateError("REPLAY_POLICY_INVALID", "replay policy is not authoritative")
    if not isinstance(authority, dict) or authority.get("status") != "PASS":
        raise FidelityGateError("REPLAY_AUTHORITY_INVALID", "captured authority did not pass")
    observation_name = bundle.get("observation_path")
    if not isinstance(observation_name, str):
        raise FidelityGateError("REPLAY_BUNDLE_INVALID", "observation path is missing")
    query_ids, outputs = _load_observations(root / observation_name)
    comparison = exact_fidelity_comparison(query_ids, outputs)
    if not isinstance(report, dict) or report.get("comparisons") != comparison["comparisons"]:
        raise FidelityGateError("REPLAY_COMPARISON_MISMATCH", "replayed comparisons differ")
    if (
        report.get("status") != comparison["status"]
        or report.get("fidelity_status") != comparison["fidelity_status"]
        or bundle.get("expected_status") != comparison["status"]
        or bundle.get("expected_fidelity_status") != comparison["fidelity_status"]
        or manifest.get("status") != comparison["status"]
        or manifest.get("fidelity_status") != comparison["fidelity_status"]
        or bundle.get("query_count") != len(query_ids)
    ):
        raise FidelityGateError("REPLAY_STATUS_MISMATCH", "replayed fidelity status differs")
    return {
        "status": comparison["status"],
        "fidelity_status": comparison["fidelity_status"],
        "replay_verified": True,
        "status_match": True,
        "comparison_match": True,
        "model_execution_used": False,
        "query_count": len(query_ids),
        "comparisons": comparison["comparisons"],
    }
