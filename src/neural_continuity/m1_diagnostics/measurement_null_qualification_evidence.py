from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.measurement_null_qualification_authority import (
    QUALIFICATION_FORMAT_VERSION,
    QualificationPreflightAuthority,
    QualificationPreflightError,
    build_qualification_authority_document,
    verify_qualification_preflight_authority,
)

PACKAGE_ARTIFACTS = frozenset({"qualification-authority.json", "replay-bundle.json"})


def _blocked(code: str, message: str) -> QualificationPreflightError:
    return QualificationPreflightError(code, message)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(dict(value)) + b"\n")


def _load_json(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise _blocked(code, f"cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise _blocked(code, f"JSON artifact is not an object: {path}")
    return value


def _safe_path(root: Path, relative: str, code: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise _blocked(code, f"artifact path is not relative: {relative}")
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise _blocked(code, f"artifact escapes package root: {relative}")
    return path


def _replay_bundle(authority: QualificationPreflightAuthority) -> dict[str, Any]:
    return {
        "replay_format_version": QUALIFICATION_FORMAT_VERSION,
        "config_path": str(authority.config_path),
        "dataset_directory": str(authority.dataset_directory),
        "transition_a_bundle": str(authority.transition_a_bundle),
        "extension_plan_bundle": str(authority.extension_plan_bundle),
        "extension_plan_manifest_sha256": authority.extension_plan_manifest_sha256,
        "sentinel_run_directory": str(authority.sentinel_run_directory),
        "sentinel_root_manifest_sha256": authority.sentinel_root_manifest_sha256,
        "sentinel_checkpoint_sha256": authority.sentinel_checkpoint_sha256,
        "authority_path": "qualification-authority.json",
        "expected_authority_sha256": authority.authority_sha256,
        "expected_status": "QUALIFICATION_PREFLIGHT_VERIFIED",
        "replay_requires_model_execution": False,
        "replay_deserializes_observation_archives": False,
    }


def _write_manifest(root: Path) -> str:
    artifacts = [
        {
            "path": relative,
            "sha256": sha256_file(root / relative),
            "size_bytes": (root / relative).stat().st_size,
        }
        for relative in sorted(PACKAGE_ARTIFACTS)
    ]
    manifest = {
        "kind": "m1-measurement-null-qualification-preflight-manifest",
        "version": QUALIFICATION_FORMAT_VERSION,
        "status": "QUALIFICATION_PREFLIGHT_VERIFIED",
        "tamper_evident": True,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "model_execution_used": False,
        "onnx_graph_loaded": False,
        "runtime_session_created": False,
        "activation_read": False,
        "numeric_observation_read": False,
        "full_corpus_execution_authorized": False,
        "full_corpus_execution_started": False,
        "replay_requires_model_execution": False,
    }
    path = root / "artifact-manifest.json"
    _write_json(path, manifest)
    return sha256_file(path)


def _verify_manifest(root: Path, expected_manifest_sha256: str) -> None:
    path = root / "artifact-manifest.json"
    if not path.is_file() or sha256_file(path) != expected_manifest_sha256:
        raise _blocked(
            "QUALIFICATION_PACKAGE_MANIFEST_HASH_MISMATCH",
            "qualification package manifest is missing or differs",
        )
    manifest = _load_json(path, "QUALIFICATION_PACKAGE_MANIFEST_INVALID")
    entries = manifest.get("artifacts")
    if (
        manifest.get("kind") != "m1-measurement-null-qualification-preflight-manifest"
        or manifest.get("version") != QUALIFICATION_FORMAT_VERSION
        or manifest.get("status") != "QUALIFICATION_PREFLIGHT_VERIFIED"
        or manifest.get("tamper_evident") is not True
        or not isinstance(entries, list)
    ):
        raise _blocked(
            "QUALIFICATION_PACKAGE_MANIFEST_INVALID",
            "qualification package manifest identity differs",
        )
    observed: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise _blocked(
                "QUALIFICATION_PACKAGE_MANIFEST_INVALID",
                "qualification artifact entry is invalid",
            )
        relative = entry.get("path")
        if not isinstance(relative, str) or relative in observed:
            raise _blocked(
                "QUALIFICATION_PACKAGE_MANIFEST_INVALID",
                "qualification artifact path is invalid or duplicated",
            )
        observed.add(relative)
        artifact = _safe_path(
            root,
            relative,
            "QUALIFICATION_PACKAGE_ARTIFACT_PATH_INVALID",
        )
        if (
            not artifact.is_file()
            or entry.get("sha256") != sha256_file(artifact)
            or entry.get("size_bytes") != artifact.stat().st_size
        ):
            raise _blocked(
                "QUALIFICATION_PACKAGE_ARTIFACT_INTEGRITY_FAILED",
                f"qualification artifact integrity failed: {relative}",
            )
    if observed != set(PACKAGE_ARTIFACTS):
        raise _blocked(
            "QUALIFICATION_PACKAGE_ARTIFACT_SET_MISMATCH",
            "qualification package artifact set is incomplete or unexpected",
        )


def capture_qualification_preflight_package(
    config_path: str | Path,
    dataset_directory: str | Path,
    transition_a_bundle: str | Path,
    extension_plan_bundle: str | Path,
    extension_plan_manifest_sha256: str,
    sentinel_run_directory: str | Path,
    sentinel_root_manifest_sha256: str,
    sentinel_checkpoint_sha256: str,
    output_directory: str | Path,
) -> dict[str, Any]:
    authority = verify_qualification_preflight_authority(
        config_path,
        dataset_directory,
        transition_a_bundle,
        extension_plan_bundle,
        extension_plan_manifest_sha256,
        sentinel_run_directory,
        sentinel_root_manifest_sha256,
        sentinel_checkpoint_sha256,
    )
    output = Path(output_directory).resolve()
    if output.exists():
        raise _blocked(
            "QUALIFICATION_PACKAGE_OUTPUT_EXISTS",
            f"qualification output directory already exists: {output}",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        document = build_qualification_authority_document(authority)
        replay = _replay_bundle(authority)
        _write_json(temporary / "qualification-authority.json", document)
        _write_json(temporary / "replay-bundle.json", replay)
        manifest_sha256 = _write_manifest(temporary)
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "status": "QUALIFICATION_PREFLIGHT_VERIFIED",
        "output_directory": str(output),
        "artifact_manifest_sha256": manifest_sha256,
        "authority_sha256": authority.authority_sha256,
        "sentinel_complete": True,
        "completed_epoch_count": 120,
        "full_corpus_execution_authorized": False,
        "full_corpus_execution_started": False,
        "qualifying_detection_evidence": False,
        "scientific_decision": "NOT_EVALUATED",
        "model_execution_used": False,
        "onnx_graph_loaded": False,
        "runtime_session_created": False,
        "activation_read": False,
        "numeric_observation_read": False,
    }


def replay_qualification_preflight_package(
    bundle_path: str | Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    bundle = Path(bundle_path).resolve()
    root = bundle.parent.resolve()
    _verify_manifest(root, expected_manifest_sha256)
    replay = _load_json(bundle, "QUALIFICATION_PACKAGE_REPLAY_BUNDLE_INVALID")
    if (
        replay.get("replay_format_version") != QUALIFICATION_FORMAT_VERSION
        or replay.get("replay_requires_model_execution") is not False
        or replay.get("replay_deserializes_observation_archives") is not False
    ):
        raise _blocked(
            "QUALIFICATION_PACKAGE_REPLAY_BUNDLE_INVALID",
            "qualification replay policy differs",
        )
    authority = verify_qualification_preflight_authority(
        str(replay.get("config_path", "")),
        str(replay.get("dataset_directory", "")),
        str(replay.get("transition_a_bundle", "")),
        str(replay.get("extension_plan_bundle", "")),
        str(replay.get("extension_plan_manifest_sha256", "")),
        str(replay.get("sentinel_run_directory", "")),
        str(replay.get("sentinel_root_manifest_sha256", "")),
        str(replay.get("sentinel_checkpoint_sha256", "")),
    )
    relative_authority = replay.get("authority_path")
    if not isinstance(relative_authority, str):
        raise _blocked(
            "QUALIFICATION_PACKAGE_REPLAY_BUNDLE_INVALID",
            "qualification authority path is missing",
        )
    captured = _load_json(
        _safe_path(
            root,
            relative_authority,
            "QUALIFICATION_PACKAGE_AUTHORITY_PATH_INVALID",
        ),
        "QUALIFICATION_PACKAGE_AUTHORITY_INVALID",
    )
    expected_document = build_qualification_authority_document(authority)
    authority_match = canonical_json_bytes(captured) == canonical_json_bytes(expected_document)
    authority_sha256_match = replay.get("expected_authority_sha256") == authority.authority_sha256
    status_match = (
        replay.get("expected_status") == "QUALIFICATION_PREFLIGHT_VERIFIED"
        and captured.get("status") == "QUALIFICATION_PREFLIGHT_VERIFIED"
    )
    if not authority_match or not authority_sha256_match or not status_match:
        raise _blocked(
            "QUALIFICATION_PACKAGE_REPLAY_MISMATCH",
            "qualification replay differs from captured authority",
        )
    return {
        "status": "QUALIFICATION_PREFLIGHT_VERIFIED",
        "replay_verified": True,
        "authority_match": True,
        "authority_sha256_match": True,
        "status_match": True,
        "sentinel_complete": True,
        "completed_epoch_count": 120,
        "full_corpus_execution_authorized": False,
        "full_corpus_execution_started": False,
        "qualifying_detection_evidence": False,
        "scientific_decision": "NOT_EVALUATED",
        "model_execution_used": False,
        "onnx_graph_loaded": False,
        "runtime_session_created": False,
        "activation_read": False,
        "numeric_observation_read": False,
    }
