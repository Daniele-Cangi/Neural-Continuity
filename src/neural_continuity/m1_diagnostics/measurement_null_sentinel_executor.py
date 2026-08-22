from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from neural_continuity.evidence import canonical_json_bytes
from neural_continuity.m1_diagnostics.measurement_null_sentinel_authority import (
    EPOCH_LAYOUT,
    PROCESS_EPOCH_COUNT,
    SentinelAuthority,
    verify_sentinel_authority,
)
from neural_continuity.m1_diagnostics.measurement_null_sentinel_checkpoint import (
    create_sentinel_run,
    inspect_sentinel_checkpoint,
    write_epoch_checkpoint,
)
from neural_continuity.m1_teacher_evidence import TeacherEvidenceError, _fail


class SentinelBackend(Protocol):
    def encode(
        self,
        texts: Sequence[str],
        batch_size: int,
        label: str,
    ) -> np.ndarray: ...

    def runtime_inventory(self) -> Mapping[str, Any]: ...


SentinelBackendFactory = Callable[[SentinelAuthority], SentinelBackend]


def _build_run_plan(authority: SentinelAuthority) -> dict[str, Any]:
    return {
        "kind": "m1-measurement-null-sentinel-run-plan",
        "version": "1.0.0",
        "phase_id": "tensor_sentinel_preflight",
        "authority": {
            "authority_sha256": authority.authority_sha256,
            "config_sha256": authority.config_sha256,
            "extension_plan_bundle": str(authority.extension_plan_bundle),
            "extension_plan_manifest_sha256": (authority.extension_plan_manifest_sha256),
            "extension_plan_sha256": authority.extension_plan_sha256,
            "transition_a_manifest_sha256": (authority.source.transition_a_manifest_sha256),
            "onnx_fp32_artifact_sha256": authority.source.artifact_sha256,
            "dataset_id": authority.dataset.dataset_id,
            "materialization_manifest_sha256": authority.dataset.manifest_sha256,
            "materialization_policy_sha256": (authority.dataset.materialization_policy_sha256),
            "partition_policy_sha256": authority.dataset.partition_policy_sha256,
            "execution_provider": authority.source.execution_provider,
        },
        "process_epoch_count": PROCESS_EPOCH_COUNT,
        "runs": [
            {"run_id": run_id, "batch_size": batch_size} for run_id, batch_size in EPOCH_LAYOUT
        ],
        "document_ids": list(authority.selected_document_ids),
        "query_ids": list(authority.query_ids),
        "query_roles": ["measurement_null"] * len(authority.query_ids),
        "qrels": {query_id: list(authority.qrels[query_id]) for query_id in authority.query_ids},
        "execution_policy": {
            "one_epoch_per_process": True,
            "resume_requires_checkpoint_hash": True,
            "early_stopping_allowed": False,
            "adaptive_sample_size_allowed": False,
            "full_corpus_execution_allowed": False,
            "candidate_or_int8_execution_allowed": False,
            "holdout_query_access_allowed": False,
        },
        "evidence_policy": {
            "qualifying_detection_evidence": False,
            "technical_preflight_only": True,
            "scientific_decision": "NOT_EVALUATED",
            "operational_tolerance_selection_allowed": False,
        },
    }


def prepare_sentinel_run(
    *,
    config_path: str | Path,
    dataset_directory: str | Path,
    transition_a_bundle: str | Path,
    extension_plan_bundle: str | Path,
    extension_plan_manifest_sha256: str,
    output_directory: str | Path,
) -> dict[str, Any]:
    authority = verify_sentinel_authority(
        config_path,
        dataset_directory,
        transition_a_bundle,
        extension_plan_bundle,
        extension_plan_manifest_sha256,
    )
    result = create_sentinel_run(
        output_directory,
        _build_run_plan(authority),
    )
    return {
        **result,
        "authority_sha256": authority.authority_sha256,
        "document_count": len(authority.selected_document_ids),
        "query_count": len(authority.query_ids),
        "process_epoch_count": PROCESS_EPOCH_COUNT,
    }


def _validated_embeddings(
    values: np.ndarray,
    *,
    expected_count: int,
    expected_dimension: int | None,
    label: str,
) -> np.ndarray:
    embeddings = np.ascontiguousarray(values, dtype=np.float32)
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != expected_count
        or embeddings.shape[1] == 0
        or (expected_dimension is not None and embeddings.shape[1] != expected_dimension)
        or not np.isfinite(embeddings).all()
    ):
        raise _fail(
            "SENTINEL_EXECUTION_OUTPUT_INVALID",
            f"invalid sentinel embedding output for {label}",
            "EXECUTION_ERROR",
        )
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
        raise _fail(
            "SENTINEL_EXECUTION_NORMALIZATION_INVALID",
            f"sentinel embeddings are not L2 normalized for {label}",
            "EXECUTION_ERROR",
        )
    return embeddings


def run_next_sentinel_epoch(
    *,
    config_path: str | Path,
    dataset_directory: str | Path,
    transition_a_bundle: str | Path,
    extension_plan_bundle: str | Path,
    extension_plan_manifest_sha256: str,
    run_directory: str | Path,
    root_manifest_sha256: str,
    expected_checkpoint_sha256: str,
    backend_factory: SentinelBackendFactory | None = None,
) -> dict[str, Any]:
    authority = verify_sentinel_authority(
        config_path,
        dataset_directory,
        transition_a_bundle,
        extension_plan_bundle,
        extension_plan_manifest_sha256,
    )
    expected_plan = _build_run_plan(authority)
    state = inspect_sentinel_checkpoint(
        run_directory,
        root_manifest_sha256,
        expected_checkpoint_sha256,
    )
    if canonical_json_bytes(dict(state.run_plan)) != canonical_json_bytes(expected_plan):
        raise _fail(
            "SENTINEL_RUN_AUTHORITY_MISMATCH",
            "sentinel run plan differs from the currently verified authority",
        )
    if state.complete:
        return {
            "status": "SENTINEL_PREFLIGHT_COMPLETE",
            "completed_epoch_count": state.completed_epoch_count,
            "latest_checkpoint_sha256": state.latest_checkpoint_sha256,
            "execution_started": False,
            "full_corpus_execution_allowed": False,
        }
    if backend_factory is None:
        from neural_continuity.m1_diagnostics.measurement_null_sentinel_runtime import (
            create_onnx_sentinel_backend,
        )

        backend_factory = create_onnx_sentinel_backend

    try:
        backend = backend_factory(authority)
        document_runs: list[np.ndarray] = []
        query_runs: list[np.ndarray] = []
        embedding_dimension: int | None = None
        for run_id, batch_size in EPOCH_LAYOUT:
            documents = _validated_embeddings(
                backend.encode(
                    authority.selected_document_texts,
                    batch_size,
                    f"{run_id} sentinel documents",
                ),
                expected_count=len(authority.selected_document_ids),
                expected_dimension=embedding_dimension,
                label=f"{run_id} sentinel documents",
            )
            if embedding_dimension is None:
                embedding_dimension = int(documents.shape[1])
            queries = _validated_embeddings(
                backend.encode(
                    authority.query_texts,
                    batch_size,
                    f"{run_id} measurement-null queries",
                ),
                expected_count=len(authority.query_ids),
                expected_dimension=embedding_dimension,
                label=f"{run_id} measurement-null queries",
            )
            document_runs.append(documents)
            query_runs.append(queries)
        runtime_inventory = backend.runtime_inventory()
    except TeacherEvidenceError:
        raise
    except Exception as exc:
        raise _fail(
            "SENTINEL_EXECUTION_FAILED",
            f"sentinel epoch execution failed: {exc}",
            "EXECUTION_ERROR",
        ) from exc

    result = write_epoch_checkpoint(
        state,
        document_embeddings=np.stack(document_runs),
        query_embeddings=np.stack(query_runs),
        runtime_inventory=runtime_inventory,
    )
    return {
        **result,
        "root_manifest_sha256": root_manifest_sha256,
        "authority_sha256": authority.authority_sha256,
        "execution_started": True,
        "model_execution_used": True,
    }


def sentinel_status(
    *,
    run_directory: str | Path,
    root_manifest_sha256: str,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    state = inspect_sentinel_checkpoint(
        run_directory,
        root_manifest_sha256,
        expected_checkpoint_sha256,
    )
    return {
        "status": (
            "SENTINEL_PREFLIGHT_COMPLETE" if state.complete else "SENTINEL_PREFLIGHT_INCOMPLETE"
        ),
        "completed_epoch_count": state.completed_epoch_count,
        "next_epoch_number": state.next_epoch_number,
        "latest_checkpoint_sha256": state.latest_checkpoint_sha256,
        "model_execution_used": False,
        "replay_verified": True,
        "qualifying_detection_evidence": False,
        "full_corpus_execution_allowed": False,
    }
