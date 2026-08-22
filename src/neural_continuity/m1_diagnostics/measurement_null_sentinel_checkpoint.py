from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_teacher_evidence import (
    _fail,
    _load_json,
    _verify_artifacts,
)

SENTINEL_RUN_FORMAT_VERSION = "1.0.0"
REQUIRED_EPOCH_ARTIFACTS = (
    "epoch-plan.json",
    "runtime-inventory.json",
    "raw-observations.npz",
    "epoch-summary.json",
)
EPOCH_DIRECTORY_PATTERN = re.compile(r"epoch-(\d{4})")


@dataclass(frozen=True)
class SentinelCheckpointState:
    run_directory: Path
    run_plan: Mapping[str, Any]
    root_manifest_sha256: str
    completed_epoch_count: int
    next_epoch_number: int | None
    latest_checkpoint_sha256: str
    complete: bool


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(dict(value)) + b"\n")


def _artifact_entry(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def create_sentinel_run(
    output_directory: str | Path,
    run_plan: Mapping[str, Any],
) -> dict[str, Any]:
    output = Path(output_directory).resolve()
    if output.exists():
        raise _fail(
            "OUTPUT_ALREADY_EXISTS",
            f"sentinel run output already exists: {output}",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        plan_path = temporary / "sentinel-run-plan.json"
        _write_json(plan_path, run_plan)
        manifest = {
            "kind": "m1-measurement-null-sentinel-root-manifest",
            "version": SENTINEL_RUN_FORMAT_VERSION,
            "artifacts": [_artifact_entry(temporary, plan_path)],
            "checkpoint_policy": {
                "append_only_epoch_directories": True,
                "hash_chained_epoch_manifests": True,
                "missing_or_tampered_checkpoint_behavior": "BLOCKED",
                "one_epoch_per_process_required": True,
            },
        }
        manifest_path = temporary / "artifact-manifest.json"
        _write_json(manifest_path, manifest)
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    root_sha256 = sha256_file(output / "artifact-manifest.json")
    return {
        "status": "SENTINEL_PREPARED_NOT_EXECUTED",
        "output_directory": str(output),
        "root_manifest_sha256": root_sha256,
        "latest_checkpoint_sha256": root_sha256,
        "completed_epoch_count": 0,
        "execution_started": False,
        "full_corpus_execution_allowed": False,
    }


def _expected_epoch_plan(
    run_plan: Mapping[str, Any],
    epoch_number: int,
    previous_checkpoint_sha256: str,
) -> dict[str, Any]:
    return {
        "kind": "m1-measurement-null-sentinel-epoch-plan",
        "version": SENTINEL_RUN_FORMAT_VERSION,
        "phase_id": "tensor_sentinel_preflight",
        "epoch_number": epoch_number,
        "previous_checkpoint_sha256": previous_checkpoint_sha256,
        "runs": list(run_plan["runs"]),
        "document_ids": list(run_plan["document_ids"]),
        "query_ids": list(run_plan["query_ids"]),
        "query_role": "measurement_null",
        "qualifying_detection_evidence": False,
        "full_corpus_execution": False,
        "scientific_decision": "NOT_EVALUATED",
    }


def _validate_observation_archive(
    path: Path,
    run_plan: Mapping[str, Any],
) -> None:
    expected_runs = list(run_plan["runs"])
    try:
        with np.load(path, allow_pickle=False) as archive:
            run_ids = archive["run_ids"].astype(str).tolist()
            batch_sizes = archive["batch_sizes"].astype(int).tolist()
            document_ids = archive["document_ids"].astype(str).tolist()
            query_ids = archive["query_ids"].astype(str).tolist()
            document_embeddings = np.ascontiguousarray(
                archive["document_embeddings"],
                dtype=np.float32,
            )
            query_embeddings = np.ascontiguousarray(
                archive["query_embeddings"],
                dtype=np.float32,
            )
    except (KeyError, OSError, ValueError) as exc:
        raise _fail(
            "SENTINEL_CHECKPOINT_OBSERVATION_INVALID",
            f"cannot load sentinel observations: {exc}",
        ) from exc
    actual_runs = [
        {"run_id": run_id, "batch_size": batch_size}
        for run_id, batch_size in zip(run_ids, batch_sizes, strict=True)
    ]
    if (
        actual_runs != expected_runs
        or document_ids != list(run_plan["document_ids"])
        or query_ids != list(run_plan["query_ids"])
        or document_embeddings.ndim != 3
        or query_embeddings.ndim != 3
        or document_embeddings.shape[0] != len(expected_runs)
        or query_embeddings.shape[0] != len(expected_runs)
        or document_embeddings.shape[1] != len(document_ids)
        or query_embeddings.shape[1] != len(query_ids)
        or document_embeddings.shape[2] != query_embeddings.shape[2]
        or document_embeddings.shape[2] == 0
        or not np.isfinite(document_embeddings).all()
        or not np.isfinite(query_embeddings).all()
    ):
        raise _fail(
            "SENTINEL_CHECKPOINT_OBSERVATION_INVALID",
            "sentinel observation identities or shapes differ",
        )
    document_norms = np.linalg.norm(document_embeddings, axis=2)
    query_norms = np.linalg.norm(query_embeddings, axis=2)
    if not np.allclose(document_norms, 1.0, rtol=1e-5, atol=1e-6) or not np.allclose(
        query_norms,
        1.0,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise _fail(
            "SENTINEL_CHECKPOINT_NORMALIZATION_INVALID",
            "sentinel observations are not L2 normalized",
        )


def _verify_epoch(
    epoch_directory: Path,
    run_plan: Mapping[str, Any],
    epoch_number: int,
    previous_checkpoint_sha256: str,
) -> tuple[str, str]:
    manifest_path = epoch_directory / "epoch-manifest.json"
    manifest = _load_json(
        manifest_path,
        "SENTINEL_EPOCH_MANIFEST_INVALID",
    )
    if (
        manifest.get("kind") != "m1-measurement-null-sentinel-epoch-manifest"
        or manifest.get("version") != SENTINEL_RUN_FORMAT_VERSION
        or manifest.get("epoch_number") != epoch_number
        or manifest.get("previous_checkpoint_sha256") != previous_checkpoint_sha256
        or manifest.get("qualifying_detection_evidence") is not False
        or manifest.get("full_corpus_execution") is not False
    ):
        raise _fail(
            "SENTINEL_EPOCH_MANIFEST_INVALID",
            f"epoch {epoch_number} manifest differs from checkpoint chain",
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or {
        artifact.get("path") for artifact in artifacts if isinstance(artifact, Mapping)
    } != set(REQUIRED_EPOCH_ARTIFACTS):
        raise _fail(
            "SENTINEL_EPOCH_MANIFEST_INVALID",
            f"epoch {epoch_number} does not declare the exact artifact set",
        )
    _verify_artifacts(epoch_directory, manifest, "artifacts")
    epoch_plan = _load_json(
        epoch_directory / "epoch-plan.json",
        "SENTINEL_EPOCH_PLAN_INVALID",
    )
    if epoch_plan != _expected_epoch_plan(
        run_plan,
        epoch_number,
        previous_checkpoint_sha256,
    ):
        raise _fail(
            "SENTINEL_EPOCH_PLAN_INVALID",
            f"epoch {epoch_number} plan differs from frozen run plan",
        )
    runtime = _load_json(
        epoch_directory / "runtime-inventory.json",
        "SENTINEL_RUNTIME_INVENTORY_INVALID",
    )
    if (
        runtime.get("model_execution_used") is not True
        or runtime.get("onnx_graph_loaded") is not True
        or runtime.get("activation_read") is not False
        or runtime.get("execution_provider") != "CPUExecutionProvider"
    ):
        raise _fail(
            "SENTINEL_RUNTIME_INVENTORY_INVALID",
            f"epoch {epoch_number} runtime inventory is incomplete",
        )
    process_instance_id = runtime.get("process_instance_id")
    if not isinstance(process_instance_id, str) or not process_instance_id:
        raise _fail(
            "SENTINEL_RUNTIME_INVENTORY_INVALID",
            f"epoch {epoch_number} lacks process identity",
        )
    summary = _load_json(
        epoch_directory / "epoch-summary.json",
        "SENTINEL_EPOCH_SUMMARY_INVALID",
    )
    if (
        summary.get("epoch_number") != epoch_number
        or summary.get("scientific_decision") != "NOT_EVALUATED"
        or summary.get("qualifying_detection_evidence") is not False
        or summary.get("full_corpus_execution") is not False
    ):
        raise _fail(
            "SENTINEL_EPOCH_SUMMARY_INVALID",
            f"epoch {epoch_number} summary exceeds technical preflight scope",
        )
    _validate_observation_archive(
        epoch_directory / "raw-observations.npz",
        run_plan,
    )
    return sha256_file(manifest_path), process_instance_id


def inspect_sentinel_checkpoint(
    run_directory: str | Path,
    root_manifest_sha256: str,
    expected_checkpoint_sha256: str,
) -> SentinelCheckpointState:
    root = Path(run_directory).resolve()
    manifest_path = root / "artifact-manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != root_manifest_sha256:
        raise _fail(
            "SENTINEL_ROOT_MANIFEST_MISMATCH",
            "sentinel root manifest is missing or differs",
        )
    manifest = _load_json(
        manifest_path,
        "SENTINEL_ROOT_MANIFEST_INVALID",
    )
    if (
        manifest.get("kind") != "m1-measurement-null-sentinel-root-manifest"
        or manifest.get("version") != SENTINEL_RUN_FORMAT_VERSION
    ):
        raise _fail(
            "SENTINEL_ROOT_MANIFEST_INVALID",
            "sentinel root manifest is not authoritative",
        )
    _verify_artifacts(root, manifest, "artifacts")
    run_plan = _load_json(
        root / "sentinel-run-plan.json",
        "SENTINEL_RUN_PLAN_INVALID",
    )
    epoch_count = run_plan.get("process_epoch_count")
    if not isinstance(epoch_count, int) or isinstance(epoch_count, bool) or epoch_count <= 0:
        raise _fail(
            "SENTINEL_RUN_PLAN_INVALID",
            "sentinel process epoch count is invalid",
        )
    epoch_directories: list[tuple[int, Path]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = EPOCH_DIRECTORY_PATTERN.fullmatch(path.name)
        if match is None:
            raise _fail(
                "SENTINEL_UNDECLARED_CHECKPOINT_DIRECTORY",
                f"undeclared directory exists in sentinel run: {path.name}",
            )
        epoch_directories.append((int(match.group(1)), path))
    epoch_directories.sort()
    expected_numbers = list(range(1, len(epoch_directories) + 1))
    if [number for number, _ in epoch_directories] != expected_numbers:
        raise _fail(
            "SENTINEL_CHECKPOINT_SEQUENCE_INVALID",
            "sentinel epoch checkpoints are not contiguous",
        )
    if len(epoch_directories) > epoch_count:
        raise _fail(
            "SENTINEL_CHECKPOINT_SEQUENCE_INVALID",
            "sentinel checkpoint count exceeds preregistration",
        )
    latest_sha256 = root_manifest_sha256
    process_ids: set[str] = set()
    for epoch_number, epoch_directory in epoch_directories:
        latest_sha256, process_instance_id = _verify_epoch(
            epoch_directory,
            run_plan,
            epoch_number,
            latest_sha256,
        )
        if process_instance_id in process_ids:
            raise _fail(
                "SENTINEL_PROCESS_RESTART_REQUIRED",
                "each sentinel epoch must use an independent process",
            )
        process_ids.add(process_instance_id)
    if latest_sha256 != expected_checkpoint_sha256:
        raise _fail(
            "SENTINEL_CHECKPOINT_HASH_MISMATCH",
            "latest sentinel checkpoint SHA-256 differs",
        )
    completed = len(epoch_directories)
    return SentinelCheckpointState(
        run_directory=root,
        run_plan=run_plan,
        root_manifest_sha256=root_manifest_sha256,
        completed_epoch_count=completed,
        next_epoch_number=None if completed == epoch_count else completed + 1,
        latest_checkpoint_sha256=latest_sha256,
        complete=completed == epoch_count,
    )


def write_epoch_checkpoint(
    state: SentinelCheckpointState,
    *,
    document_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    runtime_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    epoch_number = state.next_epoch_number
    if epoch_number is None:
        raise _fail(
            "SENTINEL_PREFLIGHT_COMPLETE",
            "all preregistered sentinel epochs are already complete",
        )
    final_directory = state.run_directory / f"epoch-{epoch_number:04d}"
    if final_directory.exists():
        raise _fail(
            "SENTINEL_CHECKPOINT_ALREADY_EXISTS",
            f"sentinel epoch already exists: {epoch_number}",
        )
    expected_runs = list(state.run_plan["runs"])
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{state.run_directory.name}.epoch-{epoch_number:04d}.building-",
            dir=state.run_directory.parent,
        )
    )
    try:
        epoch_plan = _expected_epoch_plan(
            state.run_plan,
            epoch_number,
            state.latest_checkpoint_sha256,
        )
        _write_json(temporary / "epoch-plan.json", epoch_plan)
        _write_json(temporary / "runtime-inventory.json", runtime_inventory)
        np.savez_compressed(
            temporary / "raw-observations.npz",
            run_ids=np.asarray([run["run_id"] for run in expected_runs]),
            batch_sizes=np.asarray(
                [run["batch_size"] for run in expected_runs],
                dtype=np.int64,
            ),
            document_ids=np.asarray(state.run_plan["document_ids"]),
            query_ids=np.asarray(state.run_plan["query_ids"]),
            document_embeddings=np.ascontiguousarray(
                document_embeddings,
                dtype=np.float32,
            ),
            query_embeddings=np.ascontiguousarray(
                query_embeddings,
                dtype=np.float32,
            ),
        )
        summary = {
            "kind": "m1-measurement-null-sentinel-epoch-summary",
            "version": SENTINEL_RUN_FORMAT_VERSION,
            "epoch_number": epoch_number,
            "run_count": len(expected_runs),
            "document_count": len(state.run_plan["document_ids"]),
            "query_count": len(state.run_plan["query_ids"]),
            "embedding_dimension": int(document_embeddings.shape[2]),
            "scientific_decision": "NOT_EVALUATED",
            "qualifying_detection_evidence": False,
            "full_corpus_execution": False,
        }
        _write_json(temporary / "epoch-summary.json", summary)
        artifacts = [
            _artifact_entry(temporary, temporary / artifact_name)
            for artifact_name in REQUIRED_EPOCH_ARTIFACTS
        ]
        artifacts.sort(key=lambda artifact: str(artifact["path"]))
        epoch_manifest = {
            "kind": "m1-measurement-null-sentinel-epoch-manifest",
            "version": SENTINEL_RUN_FORMAT_VERSION,
            "epoch_number": epoch_number,
            "previous_checkpoint_sha256": state.latest_checkpoint_sha256,
            "artifacts": artifacts,
            "qualifying_detection_evidence": False,
            "full_corpus_execution": False,
            "integrity": {
                "hash_algorithm": "SHA-256",
                "missing_or_tampered_artifact_behavior": "BLOCKED",
            },
        }
        _write_json(temporary / "epoch-manifest.json", epoch_manifest)
        _verify_epoch(
            temporary,
            state.run_plan,
            epoch_number,
            state.latest_checkpoint_sha256,
        )
        os.replace(temporary, final_directory)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    checkpoint_sha256 = sha256_file(final_directory / "epoch-manifest.json")
    return {
        "status": "SENTINEL_EPOCH_CAPTURED",
        "epoch_number": epoch_number,
        "completed_epoch_count": epoch_number,
        "next_epoch_number": (
            None if epoch_number == state.run_plan["process_epoch_count"] else epoch_number + 1
        ),
        "checkpoint_manifest_sha256": checkpoint_sha256,
        "qualifying_detection_evidence": False,
        "scientific_decision": "NOT_EVALUATED",
        "full_corpus_execution": False,
    }
