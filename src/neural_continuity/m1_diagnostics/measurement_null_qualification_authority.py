from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.measurement_null_extension_evidence import (
    replay_measurement_null_extension_plan,
)
from neural_continuity.m1_diagnostics.measurement_null_sentinel_authority import (
    verify_sentinel_authority,
)

QUALIFICATION_FORMAT_VERSION = "1.0.0"
QUALIFICATION_EPOCH_COUNT = 120
EPOCH_DIRECTORY_PATTERN = re.compile(r"epoch-(\d{4})")
ROOT_FILES = frozenset({"artifact-manifest.json", "sentinel-run-plan.json"})
EPOCH_ARTIFACTS = frozenset(
    {
        "epoch-plan.json",
        "epoch-summary.json",
        "raw-observations.npz",
        "runtime-inventory.json",
    }
)
EPOCH_FILES = EPOCH_ARTIFACTS | {"epoch-manifest.json"}


class QualificationPreflightError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.status = "BLOCKED"


@dataclass(frozen=True)
class QualificationPreflightAuthority:
    config_path: Path
    dataset_directory: Path
    transition_a_bundle: Path
    extension_plan_bundle: Path
    extension_plan_manifest_sha256: str
    extension_plan_sha256: str
    qualification_phase: Mapping[str, Any]
    qualification_phase_sha256: str
    sentinel_run_directory: Path
    sentinel_root_manifest_sha256: str
    sentinel_checkpoint_sha256: str
    sentinel_run_plan_sha256: str
    sentinel_authority_sha256: str
    epoch_manifest_sha256s: tuple[str, ...]
    authority_sha256: str


def _blocked(code: str, message: str) -> QualificationPreflightError:
    return QualificationPreflightError(code, message)


def _load_json(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise _blocked(code, f"cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise _blocked(code, f"JSON artifact is not an object: {path}")
    return value


def _safe_artifact_path(root: Path, relative: str, code: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise _blocked(code, f"artifact path is not relative: {relative}")
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise _blocked(code, f"artifact escapes evidence root: {relative}")
    return path


def _sha256_payload(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def _require_sha256(value: str, field: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise _blocked("QUALIFICATION_HASH_INVALID", f"{field} is not a lowercase SHA-256")


def _verify_artifacts(
    root: Path,
    entries: object,
    expected_paths: frozenset[str],
    code_prefix: str,
) -> None:
    if not isinstance(entries, list):
        raise _blocked(f"{code_prefix}_MANIFEST_INVALID", "artifact entries are missing")
    observed: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise _blocked(f"{code_prefix}_MANIFEST_INVALID", "artifact entry is invalid")
        relative = entry.get("path")
        if not isinstance(relative, str) or relative in observed:
            raise _blocked(
                f"{code_prefix}_MANIFEST_INVALID",
                "artifact path is missing or duplicated",
            )
        observed.add(relative)
        artifact = _safe_artifact_path(root, relative, f"{code_prefix}_ARTIFACT_PATH_INVALID")
        if not artifact.is_file():
            raise _blocked(
                f"{code_prefix}_ARTIFACT_MISSING",
                f"declared artifact is missing: {relative}",
            )
        if entry.get("sha256") != sha256_file(artifact):
            raise _blocked(
                f"{code_prefix}_ARTIFACT_INTEGRITY_FAILED",
                f"artifact SHA-256 differs: {relative}",
            )
        if entry.get("size_bytes") != artifact.stat().st_size:
            raise _blocked(
                f"{code_prefix}_ARTIFACT_INTEGRITY_FAILED",
                f"artifact size differs: {relative}",
            )
    if observed != set(expected_paths):
        raise _blocked(
            f"{code_prefix}_ARTIFACT_SET_MISMATCH",
            "artifact set is incomplete or unexpected",
        )


def _verify_extension_phase(
    extension_plan_bundle: Path,
    extension_plan_manifest_sha256: str,
    expected_plan_sha256: str,
) -> tuple[dict[str, Any], str]:
    replay = replay_measurement_null_extension_plan(
        extension_plan_bundle,
        extension_plan_manifest_sha256,
    )
    expected_replay = {
        "status": "PREREGISTERED_NOT_EXECUTED",
        "replay_verified": True,
        "plan_match": True,
        "status_match": True,
        "invariants_match": True,
        "execution_started": False,
        "model_execution_used": False,
        "stage_1_execution_started": False,
    }
    for field, expected in expected_replay.items():
        if replay.get(field) != expected:
            raise _blocked(
                "QUALIFICATION_EXTENSION_PLAN_REPLAY_INVALID",
                f"extension-plan replay field differs: {field}",
            )
    replay_bundle = _load_json(
        extension_plan_bundle,
        "QUALIFICATION_EXTENSION_REPLAY_BUNDLE_INVALID",
    )
    relative_plan = replay_bundle.get("plan_path")
    if not isinstance(relative_plan, str):
        raise _blocked(
            "QUALIFICATION_EXTENSION_REPLAY_BUNDLE_INVALID",
            "extension-plan path is missing",
        )
    plan_path = _safe_artifact_path(
        extension_plan_bundle.parent.resolve(),
        relative_plan,
        "QUALIFICATION_EXTENSION_PLAN_PATH_INVALID",
    )
    if sha256_file(plan_path) != expected_plan_sha256:
        raise _blocked(
            "QUALIFICATION_EXTENSION_PLAN_HASH_MISMATCH",
            "extension-plan SHA-256 differs from sentinel authority",
        )
    plan = _load_json(plan_path, "QUALIFICATION_EXTENSION_PLAN_INVALID")
    phases = plan.get("phases")
    if not isinstance(phases, list):
        raise _blocked("QUALIFICATION_EXTENSION_PLAN_INVALID", "plan phases are missing")
    matches = [
        phase
        for phase in phases
        if isinstance(phase, Mapping) and phase.get("phase_id") == "full_corpus_qualification"
    ]
    if len(matches) != 1:
        raise _blocked(
            "QUALIFICATION_PHASE_INVALID",
            "exactly one full-corpus qualification phase is required",
        )
    phase = dict(matches[0])
    required_phase = {
        "phase_id": "full_corpus_qualification",
        "qualifying_detection_evidence": True,
        "process_epoch_count": QUALIFICATION_EPOCH_COUNT,
        "documents": "all frozen document IDs",
        "queries": "all and only measurement_null query IDs",
    }
    for field, expected in required_phase.items():
        if phase.get(field) != expected:
            raise _blocked(
                "QUALIFICATION_PHASE_INVALID",
                f"frozen qualification phase field differs: {field}",
            )
    start_condition = phase.get("start_condition")
    if not isinstance(start_condition, str) or not start_condition.strip():
        raise _blocked(
            "QUALIFICATION_PHASE_INVALID",
            "qualification phase start condition is missing",
        )
    return phase, _sha256_payload(phase)


def _verify_run_plan(
    run_plan: Mapping[str, Any],
    sentinel_authority: Any,
    extension_plan_bundle: Path,
    extension_plan_manifest_sha256: str,
) -> None:
    if (
        run_plan.get("kind") != "m1-measurement-null-sentinel-run-plan"
        or run_plan.get("version") != QUALIFICATION_FORMAT_VERSION
        or run_plan.get("phase_id") != "tensor_sentinel_preflight"
        or run_plan.get("process_epoch_count") != QUALIFICATION_EPOCH_COUNT
    ):
        raise _blocked(
            "QUALIFICATION_SENTINEL_RUN_PLAN_INVALID",
            "sentinel run plan identity or epoch count differs",
        )
    authority = run_plan.get("authority")
    if not isinstance(authority, Mapping):
        raise _blocked(
            "QUALIFICATION_SENTINEL_RUN_PLAN_INVALID",
            "sentinel authority record is missing",
        )
    expected_authority = {
        "authority_sha256": sentinel_authority.authority_sha256,
        "config_sha256": sentinel_authority.config_sha256,
        "extension_plan_manifest_sha256": extension_plan_manifest_sha256,
        "extension_plan_sha256": sentinel_authority.extension_plan_sha256,
    }
    for field, expected in expected_authority.items():
        if authority.get(field) != expected:
            raise _blocked(
                "QUALIFICATION_SENTINEL_AUTHORITY_MISMATCH",
                f"sentinel authority field differs: {field}",
            )
    recorded_bundle = authority.get("extension_plan_bundle")
    if (
        not isinstance(recorded_bundle, str)
        or Path(recorded_bundle).resolve() != extension_plan_bundle
    ):
        raise _blocked(
            "QUALIFICATION_SENTINEL_AUTHORITY_MISMATCH",
            "sentinel extension-plan bundle identity differs",
        )
    comparisons: Sequence[tuple[str, object, object]] = (
        (
            "document_ids",
            run_plan.get("document_ids"),
            list(sentinel_authority.selected_document_ids),
        ),
        ("query_ids", run_plan.get("query_ids"), list(sentinel_authority.query_ids)),
        ("qrels", run_plan.get("qrels"), sentinel_authority.qrels),
    )
    for field, observed, expected in comparisons:
        if canonical_json_bytes({"value": observed}) != canonical_json_bytes({"value": expected}):
            raise _blocked(
                "QUALIFICATION_SENTINEL_SCOPE_MISMATCH",
                f"sentinel run-plan scope differs: {field}",
            )
    query_roles = run_plan.get("query_roles")
    if not isinstance(query_roles, list) or query_roles != ["measurement_null"] * len(
        sentinel_authority.query_ids
    ):
        raise _blocked(
            "QUALIFICATION_SENTINEL_SCOPE_MISMATCH",
            "sentinel query-role ordering differs",
        )
    evidence_policy = run_plan.get("evidence_policy")
    execution_policy = run_plan.get("execution_policy")
    required_evidence_policy = {
        "operational_tolerance_selection_allowed": False,
        "qualifying_detection_evidence": False,
        "scientific_decision": "NOT_EVALUATED",
        "technical_preflight_only": True,
    }
    required_execution_policy = {
        "adaptive_sample_size_allowed": False,
        "candidate_or_int8_execution_allowed": False,
        "early_stopping_allowed": False,
        "full_corpus_execution_allowed": False,
        "holdout_query_access_allowed": False,
        "one_epoch_per_process": True,
        "resume_requires_checkpoint_hash": True,
    }
    if evidence_policy != required_evidence_policy or execution_policy != required_execution_policy:
        raise _blocked(
            "QUALIFICATION_SENTINEL_POLICY_MISMATCH",
            "sentinel evidence or execution policy differs",
        )


def _verify_sentinel_chain(
    sentinel_run_directory: Path,
    sentinel_root_manifest_sha256: str,
    sentinel_checkpoint_sha256: str,
    sentinel_authority: Any,
    extension_plan_bundle: Path,
    extension_plan_manifest_sha256: str,
) -> tuple[str, tuple[str, ...]]:
    root = sentinel_run_directory.resolve()
    root_manifest_path = root / "artifact-manifest.json"
    if not root_manifest_path.is_file() or sha256_file(root_manifest_path) != (
        sentinel_root_manifest_sha256
    ):
        raise _blocked(
            "QUALIFICATION_SENTINEL_ROOT_MISMATCH",
            "sentinel root manifest is missing or differs",
        )
    root_manifest = _load_json(
        root_manifest_path,
        "QUALIFICATION_SENTINEL_ROOT_INVALID",
    )
    if (
        root_manifest.get("kind") != "m1-measurement-null-sentinel-root-manifest"
        or root_manifest.get("version") != QUALIFICATION_FORMAT_VERSION
    ):
        raise _blocked(
            "QUALIFICATION_SENTINEL_ROOT_INVALID",
            "sentinel root manifest identity differs",
        )
    if {path.name for path in root.iterdir() if path.is_file()} != set(ROOT_FILES):
        raise _blocked(
            "QUALIFICATION_SENTINEL_ROOT_FILE_SET_MISMATCH",
            "sentinel root file set is incomplete or unexpected",
        )
    _verify_artifacts(
        root,
        root_manifest.get("artifacts"),
        frozenset({"sentinel-run-plan.json"}),
        "QUALIFICATION_SENTINEL_ROOT",
    )
    run_plan_path = root / "sentinel-run-plan.json"
    run_plan = _load_json(run_plan_path, "QUALIFICATION_SENTINEL_RUN_PLAN_INVALID")
    _verify_run_plan(
        run_plan,
        sentinel_authority,
        extension_plan_bundle,
        extension_plan_manifest_sha256,
    )
    directory_names = {path.name for path in root.iterdir() if path.is_dir()}
    expected_directories = {f"epoch-{number:04d}" for number in range(1, 121)}
    if directory_names != expected_directories:
        raise _blocked(
            "QUALIFICATION_SENTINEL_CHECKPOINT_SET_MISMATCH",
            "sentinel checkpoint directory set is incomplete or unexpected",
        )
    previous_checkpoint_sha256 = sentinel_root_manifest_sha256
    epoch_manifest_sha256s: list[str] = []
    for epoch_number in range(1, QUALIFICATION_EPOCH_COUNT + 1):
        epoch_root = root / f"epoch-{epoch_number:04d}"
        if {path.name for path in epoch_root.iterdir() if path.is_file()} != set(EPOCH_FILES):
            raise _blocked(
                "QUALIFICATION_SENTINEL_EPOCH_FILE_SET_MISMATCH",
                f"epoch {epoch_number} file set is incomplete or unexpected",
            )
        if any(path.is_dir() for path in epoch_root.iterdir()):
            raise _blocked(
                "QUALIFICATION_SENTINEL_EPOCH_FILE_SET_MISMATCH",
                f"epoch {epoch_number} contains an undeclared directory",
            )
        manifest_path = epoch_root / "epoch-manifest.json"
        manifest = _load_json(
            manifest_path,
            "QUALIFICATION_SENTINEL_EPOCH_MANIFEST_INVALID",
        )
        if (
            manifest.get("kind") != "m1-measurement-null-sentinel-epoch-manifest"
            or manifest.get("version") != QUALIFICATION_FORMAT_VERSION
            or manifest.get("epoch_number") != epoch_number
            or manifest.get("previous_checkpoint_sha256") != previous_checkpoint_sha256
            or manifest.get("qualifying_detection_evidence") is not False
            or manifest.get("full_corpus_execution") is not False
        ):
            raise _blocked(
                "QUALIFICATION_SENTINEL_EPOCH_MANIFEST_INVALID",
                f"epoch {epoch_number} manifest identity or chain differs",
            )
        integrity = manifest.get("integrity")
        if integrity != {
            "hash_algorithm": "SHA-256",
            "missing_or_tampered_artifact_behavior": "BLOCKED",
        }:
            raise _blocked(
                "QUALIFICATION_SENTINEL_EPOCH_MANIFEST_INVALID",
                f"epoch {epoch_number} integrity policy differs",
            )
        _verify_artifacts(
            epoch_root,
            manifest.get("artifacts"),
            EPOCH_ARTIFACTS,
            "QUALIFICATION_SENTINEL_EPOCH",
        )
        current_sha256 = sha256_file(manifest_path)
        epoch_manifest_sha256s.append(current_sha256)
        previous_checkpoint_sha256 = current_sha256
    if previous_checkpoint_sha256 != sentinel_checkpoint_sha256:
        raise _blocked(
            "QUALIFICATION_SENTINEL_CHECKPOINT_HASH_MISMATCH",
            "final sentinel checkpoint SHA-256 differs",
        )
    return sha256_file(run_plan_path), tuple(epoch_manifest_sha256s)


def _authority_fingerprint_payload(
    *,
    sentinel_authority_sha256: str,
    sentinel_root_manifest_sha256: str,
    sentinel_checkpoint_sha256: str,
    sentinel_run_plan_sha256: str,
    extension_plan_manifest_sha256: str,
    extension_plan_sha256: str,
    qualification_phase_sha256: str,
    epoch_manifest_sha256s: Sequence[str],
) -> dict[str, Any]:
    return {
        "kind": "m1-measurement-null-qualification-preflight-fingerprint",
        "version": QUALIFICATION_FORMAT_VERSION,
        "sentinel_authority_sha256": sentinel_authority_sha256,
        "sentinel_root_manifest_sha256": sentinel_root_manifest_sha256,
        "sentinel_checkpoint_sha256": sentinel_checkpoint_sha256,
        "sentinel_run_plan_sha256": sentinel_run_plan_sha256,
        "extension_plan_manifest_sha256": extension_plan_manifest_sha256,
        "extension_plan_sha256": extension_plan_sha256,
        "qualification_phase_sha256": qualification_phase_sha256,
        "epoch_manifest_sha256s": list(epoch_manifest_sha256s),
    }


def verify_qualification_preflight_authority(
    config_path: str | Path,
    dataset_directory: str | Path,
    transition_a_bundle: str | Path,
    extension_plan_bundle: str | Path,
    extension_plan_manifest_sha256: str,
    sentinel_run_directory: str | Path,
    sentinel_root_manifest_sha256: str,
    sentinel_checkpoint_sha256: str,
) -> QualificationPreflightAuthority:
    for value, field in (
        (extension_plan_manifest_sha256, "extension_plan_manifest_sha256"),
        (sentinel_root_manifest_sha256, "sentinel_root_manifest_sha256"),
        (sentinel_checkpoint_sha256, "sentinel_checkpoint_sha256"),
    ):
        _require_sha256(value, field)
    config = Path(config_path).resolve()
    dataset = Path(dataset_directory).resolve()
    transition = Path(transition_a_bundle).resolve()
    extension = Path(extension_plan_bundle).resolve()
    sentinel = Path(sentinel_run_directory).resolve()
    try:
        sentinel_authority = verify_sentinel_authority(
            config,
            dataset,
            transition,
            extension,
            extension_plan_manifest_sha256,
        )
        phase, phase_sha256 = _verify_extension_phase(
            extension,
            extension_plan_manifest_sha256,
            sentinel_authority.extension_plan_sha256,
        )
        run_plan_sha256, epoch_manifest_sha256s = _verify_sentinel_chain(
            sentinel,
            sentinel_root_manifest_sha256,
            sentinel_checkpoint_sha256,
            sentinel_authority,
            extension,
            extension_plan_manifest_sha256,
        )
    except QualificationPreflightError:
        raise
    except Exception as exc:
        code = str(getattr(exc, "code", "QUALIFICATION_AUTHORITY_INVALID"))
        raise _blocked(code, f"qualification authority verification failed: {exc}") from exc
    fingerprint = _authority_fingerprint_payload(
        sentinel_authority_sha256=sentinel_authority.authority_sha256,
        sentinel_root_manifest_sha256=sentinel_root_manifest_sha256,
        sentinel_checkpoint_sha256=sentinel_checkpoint_sha256,
        sentinel_run_plan_sha256=run_plan_sha256,
        extension_plan_manifest_sha256=extension_plan_manifest_sha256,
        extension_plan_sha256=sentinel_authority.extension_plan_sha256,
        qualification_phase_sha256=phase_sha256,
        epoch_manifest_sha256s=epoch_manifest_sha256s,
    )
    return QualificationPreflightAuthority(
        config_path=config,
        dataset_directory=dataset,
        transition_a_bundle=transition,
        extension_plan_bundle=extension,
        extension_plan_manifest_sha256=extension_plan_manifest_sha256,
        extension_plan_sha256=sentinel_authority.extension_plan_sha256,
        qualification_phase=phase,
        qualification_phase_sha256=phase_sha256,
        sentinel_run_directory=sentinel,
        sentinel_root_manifest_sha256=sentinel_root_manifest_sha256,
        sentinel_checkpoint_sha256=sentinel_checkpoint_sha256,
        sentinel_run_plan_sha256=run_plan_sha256,
        sentinel_authority_sha256=sentinel_authority.authority_sha256,
        epoch_manifest_sha256s=epoch_manifest_sha256s,
        authority_sha256=_sha256_payload(fingerprint),
    )


def build_qualification_authority_document(
    authority: QualificationPreflightAuthority,
) -> dict[str, Any]:
    return {
        "kind": "m1-measurement-null-qualification-preflight-authority",
        "version": QUALIFICATION_FORMAT_VERSION,
        "status": "QUALIFICATION_PREFLIGHT_VERIFIED",
        "authority_sha256": authority.authority_sha256,
        "frozen_authority": {
            "sentinel_authority_sha256": authority.sentinel_authority_sha256,
            "extension_plan_manifest_sha256": authority.extension_plan_manifest_sha256,
            "extension_plan_sha256": authority.extension_plan_sha256,
            "qualification_phase_sha256": authority.qualification_phase_sha256,
        },
        "sentinel": {
            "status": "SENTINEL_PREFLIGHT_COMPLETE",
            "completed_epoch_count": QUALIFICATION_EPOCH_COUNT,
            "next_epoch_number": None,
            "root_manifest_sha256": authority.sentinel_root_manifest_sha256,
            "checkpoint_manifest_sha256": authority.sentinel_checkpoint_sha256,
            "run_plan_sha256": authority.sentinel_run_plan_sha256,
            "epoch_manifest_sha256s": list(authority.epoch_manifest_sha256s),
            "manifest_chain_verified": True,
            "observation_archives_hash_verified": True,
            "observation_archives_deserialized": False,
        },
        "qualification_phase": dict(authority.qualification_phase),
        "gate": {
            "qualification_start_condition_satisfied": True,
            "independent_review_required_before_executor_activation": True,
            "full_corpus_execution_authorized": False,
            "candidate_or_int8_execution_allowed": False,
            "holdout_query_access_allowed": False,
            "operational_tolerance_change_allowed": False,
            "scientific_decision": "NOT_EVALUATED",
            "qualifying_detection_evidence": False,
        },
        "execution": {
            "model_execution_used": False,
            "onnx_graph_loaded": False,
            "runtime_session_created": False,
            "activation_read": False,
            "numeric_observation_read": False,
            "full_corpus_execution_started": False,
        },
        "failure_policy": {
            "missing_or_tampered_authority": "BLOCKED",
            "missing_or_tampered_sentinel_artifact": "BLOCKED",
            "authority_or_scope_drift": "BLOCKED",
            "runtime_identity_mismatch": "BLOCKED",
        },
    }
