from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_b.onnx_source import (
    VerifiedOnnxSource,
    load_verified_onnx_source,
)
from neural_continuity.m1_diagnostics.measurement_null_extension_authority import (
    MeasurementNullPlanError,
)
from neural_continuity.m1_diagnostics.measurement_null_extension_evidence import (
    replay_measurement_null_extension_plan,
)
from neural_continuity.m1_teacher_evidence import (
    MaterializedDataset,
    _fail,
    _load_config,
    _load_json,
    _require_mapping,
    _require_string,
    load_materialized_dataset,
)

SENTINEL_DOCUMENT_COUNT = 256
SENTINEL_SELECTION_DOMAIN = "neural-continuity:m1:null-extension:v1:document"
PROCESS_EPOCH_COUNT = 120
EPOCH_LAYOUT = (
    ("batch_1_primary", 1),
    ("batch_16_primary", 16),
    ("batch_16_repeat", 16),
    ("batch_64_primary", 64),
)


@dataclass(frozen=True)
class SentinelAuthority:
    config: Mapping[str, Any]
    config_sha256: str
    contract: Mapping[str, Any]
    dataset: MaterializedDataset
    source: VerifiedOnnxSource
    extension_plan_bundle: Path
    extension_plan_manifest_sha256: str
    extension_plan_sha256: str
    selected_document_ids: tuple[str, ...]
    selected_document_texts: tuple[str, ...]
    query_ids: tuple[str, ...]
    query_texts: tuple[str, ...]
    qrels: Mapping[str, tuple[str, ...]]
    authority_sha256: str


def _select_sentinel_ids(
    document_ids: list[str] | tuple[str, ...],
    count: int = SENTINEL_DOCUMENT_COUNT,
) -> tuple[str, ...]:
    if len(document_ids) < count or len(document_ids) != len(set(document_ids)):
        raise _fail(
            "SENTINEL_DOCUMENT_POPULATION_INVALID",
            f"sentinel selection requires at least {count} unique document IDs",
        )

    def selection_key(document_id: str) -> tuple[bytes, bytes]:
        digest = hashlib.sha256(f"{SENTINEL_SELECTION_DOMAIN}\0{document_id}".encode()).digest()
        return digest, document_id.encode()

    return tuple(sorted(document_ids, key=selection_key)[:count])


def _validate_extension_plan(plan: Mapping[str, Any]) -> None:
    if (
        plan.get("kind") != "m1-measurement-null-extension-plan"
        or plan.get("status") != "PREREGISTERED_NOT_EXECUTED"
        or plan.get("execution_started") is not False
        or plan.get("model_execution_used_for_preregistration") is not False
        or plan.get("candidate_or_holdout_result_selected_design") is not False
    ):
        raise _fail(
            "SENTINEL_EXTENSION_PLAN_INVALID",
            "measurement-null extension plan does not preserve preregistered scope",
        )
    scope = _require_mapping(plan.get("scope"), "scope")
    required_scope = {
        "query_role": "measurement_null",
        "execution_provider": "CPUExecutionProvider",
        "candidate_or_int8_execution_allowed": False,
        "holdout_query_access_allowed": False,
        "stage_1_execution_allowed": False,
        "existing_evidence_mutation_allowed": False,
        "operational_tolerance_change_allowed": False,
    }
    if any(scope.get(key) != value for key, value in required_scope.items()):
        raise _fail(
            "SENTINEL_EXTENSION_PLAN_INVALID",
            "measurement-null extension scope differs from frozen authority",
        )
    frozen = _require_mapping(plan.get("frozen_design"), "frozen_design")
    if (
        frozen.get("process_epoch_count") != PROCESS_EPOCH_COUNT
        or frozen.get("passes_per_epoch") != len(EPOCH_LAYOUT)
        or frozen.get("passes_per_phase") != PROCESS_EPOCH_COUNT * len(EPOCH_LAYOUT)
        or frozen.get("total_planned_passes") != 2 * PROCESS_EPOCH_COUNT * len(EPOCH_LAYOUT)
        or frozen.get("batch_sizes") != [1, 16, 64]
        or frozen.get("repeat_batch_size") != 16
        or frozen.get("early_stopping_allowed") is not False
        or frozen.get("adaptive_sample_size_allowed") is not False
        or frozen.get("independent_process_required_per_epoch") is not True
    ):
        raise _fail(
            "SENTINEL_EXTENSION_PLAN_INVALID",
            "measurement-null sampling design differs from preregistration",
        )
    expected_layout = [
        {"run": run_id, "batch_size": batch_size} for run_id, batch_size in EPOCH_LAYOUT
    ]
    if plan.get("epoch_layout") != expected_layout:
        raise _fail(
            "SENTINEL_EXTENSION_PLAN_INVALID",
            "measurement-null epoch layout differs from preregistration",
        )
    phases = plan.get("phases")
    if not isinstance(phases, list) or len(phases) != 2:
        raise _fail(
            "SENTINEL_EXTENSION_PLAN_INVALID",
            "measurement-null phases are missing",
        )
    sentinel = _require_mapping(phases[0], "phases[0]")
    sentinel_documents = _require_mapping(sentinel.get("documents"), "phases[0].documents")
    if (
        sentinel.get("phase_id") != "tensor_sentinel_preflight"
        or sentinel.get("process_epoch_count") != PROCESS_EPOCH_COUNT
        or sentinel.get("qualifying_detection_evidence") is not False
        or sentinel_documents.get("count") != SENTINEL_DOCUMENT_COUNT
        or sentinel_documents.get("selection")
        != "first IDs after domain-separated SHA-256 ordering"
        or sentinel_documents.get("selection_domain") != SENTINEL_SELECTION_DOMAIN
        or sentinel_documents.get("text_or_qrel_dependent_selection") is not False
    ):
        raise _fail(
            "SENTINEL_EXTENSION_PLAN_INVALID",
            "sentinel phase differs from preregistration",
        )


def _authority_fingerprint(
    *,
    config_sha256: str,
    dataset: MaterializedDataset,
    source: VerifiedOnnxSource,
    extension_plan_manifest_sha256: str,
    extension_plan_sha256: str,
    document_ids: tuple[str, ...],
    query_ids: tuple[str, ...],
    qrels: Mapping[str, tuple[str, ...]],
) -> str:
    payload = {
        "config_sha256": config_sha256,
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "materialization_manifest_sha256": dataset.manifest_sha256,
            "materialization_policy_sha256": dataset.materialization_policy_sha256,
            "partition_policy_sha256": dataset.partition_policy_sha256,
        },
        "source": {
            "artifact_sha256": source.artifact_sha256,
            "transition_a_manifest_sha256": source.transition_a_manifest_sha256,
            "execution_provider": source.execution_provider,
        },
        "extension_plan_manifest_sha256": extension_plan_manifest_sha256,
        "extension_plan_sha256": extension_plan_sha256,
        "document_ids": list(document_ids),
        "query_ids": list(query_ids),
        "qrels": {query_id: list(qrels[query_id]) for query_id in query_ids},
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def verify_sentinel_authority(
    config_path: str | Path,
    dataset_directory: str | Path,
    transition_a_bundle: str | Path,
    extension_plan_bundle: str | Path,
    extension_plan_manifest_sha256: str,
) -> SentinelAuthority:
    try:
        replay = replay_measurement_null_extension_plan(
            extension_plan_bundle,
            extension_plan_manifest_sha256,
        )
    except MeasurementNullPlanError as exc:
        raise _fail(
            "SENTINEL_EXTENSION_AUTHORITY_INVALID",
            f"measurement-null extension replay failed: {exc}",
        ) from exc
    required_replay = {
        "replay_verified": True,
        "plan_match": True,
        "invariants_match": True,
        "status_match": True,
        "status": "PREREGISTERED_NOT_EXECUTED",
        "execution_started": False,
        "model_execution_used": False,
        "stage_1_execution_started": False,
    }
    if any(replay.get(key) != value for key, value in required_replay.items()):
        raise _fail(
            "SENTINEL_EXTENSION_AUTHORITY_INVALID",
            "measurement-null extension replay does not match frozen authority",
        )
    extension_bundle_path = Path(extension_plan_bundle).resolve()
    extension_plan_path = extension_bundle_path.parent / "measurement-null-extension-plan.json"
    extension_plan = _load_json(
        extension_plan_path,
        "SENTINEL_EXTENSION_PLAN_INVALID",
    )
    _validate_extension_plan(extension_plan)

    config, config_sha256 = _load_config(config_path)
    contract_path = Path(_require_string(config.get("contract_path"), "contract_path"))
    contract = _load_json(contract_path, "TRANSITION_B_CONTRACT_INVALID")
    if contract.get("contract_id") != "m1-transition-b-v1":
        raise _fail(
            "TRANSITION_B_CONTRACT_INVALID",
            "contract_id must remain m1-transition-b-v1",
        )
    dataset = load_materialized_dataset(dataset_directory)
    expected_dataset_id = _require_string(
        _require_mapping(config.get("dataset"), "dataset").get("dataset_id"),
        "dataset.dataset_id",
    )
    if dataset.dataset_id != expected_dataset_id:
        raise _fail(
            "DATASET_ID_MISMATCH",
            "materialized dataset identity differs",
        )
    source = load_verified_onnx_source(transition_a_bundle, contract)
    if source.execution_provider != "CPUExecutionProvider":
        raise _fail(
            "ONNX_PROVIDER_UNVERIFIED",
            "sentinel execution requires CPUExecutionProvider",
        )

    if len(dataset.document_ids) != len(dataset.document_texts):
        raise _fail(
            "SENTINEL_DOCUMENT_POPULATION_INVALID",
            "document IDs and texts differ in length",
        )
    document_text_by_id = dict(zip(dataset.document_ids, dataset.document_texts, strict=True))
    selected_document_ids = _select_sentinel_ids(tuple(dataset.document_ids))
    selected_document_texts = tuple(
        document_text_by_id[document_id] for document_id in selected_document_ids
    )

    role = dataset.roles.get("measurement_null")
    if role is None:
        raise _fail(
            "MEASUREMENT_NULL_ROLE_MISSING",
            "materialized dataset lacks measurement_null",
        )
    query_pairs = sorted(
        zip(role.query_ids, role.query_texts, strict=True),
        key=lambda item: item[0].encode(),
    )
    query_ids = tuple(query_id for query_id, _ in query_pairs)
    query_texts = tuple(text for _, text in query_pairs)
    if not query_ids or len(query_ids) != len(set(query_ids)):
        raise _fail(
            "SENTINEL_QUERY_POPULATION_INVALID",
            "measurement-null query identities are empty or duplicated",
        )
    document_id_set = set(dataset.document_ids)
    qrels: dict[str, tuple[str, ...]] = {}
    for query_id in query_ids:
        values = tuple(role.relevant_document_ids[query_id])
        if not values or any(value not in document_id_set for value in values):
            raise _fail(
                "SENTINEL_QRELS_INVALID",
                f"measurement-null qrels are invalid for query: {query_id}",
            )
        qrels[query_id] = values

    extension_plan_sha256 = sha256_file(extension_plan_path)
    authority_sha256 = _authority_fingerprint(
        config_sha256=config_sha256,
        dataset=dataset,
        source=source,
        extension_plan_manifest_sha256=extension_plan_manifest_sha256,
        extension_plan_sha256=extension_plan_sha256,
        document_ids=selected_document_ids,
        query_ids=query_ids,
        qrels=qrels,
    )
    return SentinelAuthority(
        config=config,
        config_sha256=config_sha256,
        contract=contract,
        dataset=dataset,
        source=source,
        extension_plan_bundle=extension_bundle_path,
        extension_plan_manifest_sha256=extension_plan_manifest_sha256,
        extension_plan_sha256=extension_plan_sha256,
        selected_document_ids=selected_document_ids,
        selected_document_texts=selected_document_texts,
        query_ids=query_ids,
        query_texts=query_texts,
        qrels=qrels,
        authority_sha256=authority_sha256,
    )
