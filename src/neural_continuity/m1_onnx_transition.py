from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_teacher_evidence import (
    EVIDENCE_FORMAT_VERSION,
    REPLAY_FORMAT_VERSION,
    ROLE_ORDER,
    MaterializedDataset,
    TeacherEvidenceError,
    TeacherObservation,
    _artifact_entry,
    _encode,
    _fail,
    _jsonl_bytes,
    _load_config,
    _load_json,
    _load_teacher,
    _ordered_observation,
    _rank_and_measure,
    _require_int,
    _require_mapping,
    _require_string,
    _safe_artifact_path,
    _validate_observation,
    _verify_artifacts,
    _write_bytes,
    load_materialized_dataset,
)

TRANSITION_FORMAT_VERSION = "1.0.0"
TRANSITION_ID = "A"
ONNX_FILENAME = "teacher.onnx"


@dataclass(frozen=True)
class PairedRun:
    run_id: str
    batch_size: int
    source: TeacherObservation
    target: TeacherObservation


def _positive_int(value: Any, field: str) -> int:
    parsed = _require_int(value, field)
    if parsed == 0:
        raise _fail("TRANSITION_CONFIG_INVALID", f"{field} must be positive")
    return parsed


def _load_contract(path: str | Path) -> tuple[dict[str, Any], str]:
    contract_path = Path(path)
    contract = _load_json(contract_path, "TRANSITION_CONTRACT_INVALID")
    if contract.get("contract_id") != "m1-transition-a-v1":
        raise _fail("TRANSITION_CONTRACT_INVALID", "contract_id must be m1-transition-a-v1")
    transition = _require_mapping(contract.get("transition"), "transition")
    if transition.get("id") != TRANSITION_ID:
        raise _fail("TRANSITION_CONTRACT_INVALID", "contract does not authorize transition A")
    return contract, sha256_file(contract_path)


def _transition_config(
    config: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[int, list[int]]:
    evaluation = _require_mapping(config.get("evaluation"), "evaluation")
    top_k = _positive_int(evaluation.get("top_k"), "evaluation.top_k")
    transition = _require_mapping(config.get("transition_a"), "transition_a")
    raw_batch_sizes = transition.get("batch_sizes")
    if not isinstance(raw_batch_sizes, list) or not raw_batch_sizes:
        raise _fail(
            "TRANSITION_CONFIG_INVALID", "transition_a.batch_sizes must be a non-empty array"
        )
    batch_sizes = [
        _positive_int(value, f"transition_a.batch_sizes[{index}]")
        for index, value in enumerate(raw_batch_sizes)
    ]
    if len(set(batch_sizes)) != len(batch_sizes):
        raise _fail("TRANSITION_CONFIG_INVALID", "transition_a.batch_sizes must be unique")
    required = _require_mapping(
        _require_mapping(contract.get("preconditions"), "preconditions").get("target_capture"),
        "preconditions.target_capture",
    ).get("required_batch_sizes")
    if not isinstance(required, list) or set(batch_sizes) != set(required):
        raise _fail(
            "TRANSITION_CONFIG_INVALID",
            "transition_a.batch_sizes must equal the frozen contract batch sizes",
        )
    return top_k, batch_sizes


def _ordered_query_texts(dataset: MaterializedDataset) -> list[str]:
    texts: list[str] = []
    for role in ROLE_ORDER:
        role_data = dataset.roles[role]
        texts.extend(
            text
            for _, text in sorted(
                zip(role_data.query_ids, role_data.query_texts, strict=True),
                key=lambda item: item[0].encode(),
            )
        )
    return texts


def _load_onnx_dependencies() -> tuple[Any, Any, Any]:
    try:
        import onnx
        import onnxruntime
        import torch
    except ModuleNotFoundError as exc:
        raise _fail("ONNX_DEPENDENCY_MISSING", f"missing dependency: {exc.name}") from exc
    return onnx, onnxruntime, torch


def _export_teacher_onnx(
    teacher: Any, output_path: Path, opset_version: int, requested_provider: str
) -> dict[str, Any]:
    onnx, onnxruntime, torch = _load_onnx_dependencies()
    if requested_provider != "CPUExecutionProvider":
        raise _fail("ONNX_PROVIDER_UNVERIFIED", "only CPUExecutionProvider is authorized")
    if requested_provider not in onnxruntime.get_available_providers():
        raise _fail("ONNX_PROVIDER_MISSING", f"provider unavailable: {requested_provider}")
    modules = list(teacher._modules.values())
    if (
        len(modules) != 3
        or not hasattr(modules[0], "auto_model")
        or type(modules[2]).__name__ != "Normalize"
    ):
        raise _fail(
            "ONNX_TEACHER_UNSUPPORTED",
            "frozen teacher must contain Transformer, mean Pooling, and Normalize only",
        )
    pooling = modules[1]
    if (
        not hasattr(pooling, "get_config_dict")
        or pooling.get_config_dict().get("pooling_mode") != "mean"
    ):
        raise _fail(
            "ONNX_POOLING_UNSUPPORTED",
            "frozen teacher pooling is not exactly mean-token pooling",
        )

    class MeanPoolingEncoder(torch.nn.Module):  # type: ignore[name-defined]
        def __init__(self, encoder: Any) -> None:
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids: Any, attention_mask: Any, token_type_ids: Any) -> Any:
            token_embeddings = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=False,
            )[0]
            expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return (token_embeddings * expanded_mask).sum(1) / expanded_mask.sum(1).clamp(min=1e-9)

    tokens = teacher.tokenize(["Neural Continuity ONNX export qualification."])
    input_ids = tokens.get("input_ids")
    attention_mask = tokens.get("attention_mask")
    if input_ids is None or attention_mask is None:
        raise _fail(
            "ONNX_TOKENIZATION_INVALID", "teacher tokenizer lacks input_ids or attention_mask"
        )
    token_type_ids = tokens.get("token_type_ids")
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids)
    wrapper = MeanPoolingEncoder(modules[0].auto_model).eval()
    try:
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask, token_type_ids),
            output_path,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["embeddings"],
            opset_version=opset_version,
            dynamo=False,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "token_type_ids": {0: "batch", 1: "sequence"},
                "embeddings": {0: "batch"},
            },
        )
    except Exception as exc:
        raise _fail(
            "ONNX_EXPORT_FAILED", f"cannot export frozen teacher: {exc}", "EXECUTION_ERROR"
        ) from exc
    try:
        model = onnx.load(str(output_path), load_external_data=False)
        onnx.checker.check_model(model)
    except Exception as exc:
        raise _fail(
            "ONNX_GRAPH_INVALID", f"exported graph is invalid: {exc}", "EXECUTION_ERROR"
        ) from exc
    try:
        session = onnxruntime.InferenceSession(str(output_path), providers=[requested_provider])
    except Exception as exc:
        raise _fail(
            "ONNX_SESSION_FAILED", f"cannot load exported graph: {exc}", "EXECUTION_ERROR"
        ) from exc
    if requested_provider not in session.get_providers():
        raise _fail("ONNX_PROVIDER_MISSING", f"requested provider not active: {requested_provider}")
    return {
        "onnx_version": onnx.__version__,
        "onnxruntime_version": onnxruntime.__version__,
        "opset_version": opset_version,
        "requested_execution_provider": requested_provider,
        "active_execution_providers": session.get_providers(),
        "graph_node_count": len(model.graph.node),
        "graph_input_names": [value.name for value in model.graph.input],
        "graph_output_names": [value.name for value in model.graph.output],
    }


def _onnx_session(onnx_path: Path, requested_provider: str) -> Any:
    _, onnxruntime, _ = _load_onnx_dependencies()
    try:
        session = onnxruntime.InferenceSession(str(onnx_path), providers=[requested_provider])
    except Exception as exc:
        raise _fail(
            "ONNX_SESSION_FAILED", f"cannot load exported graph: {exc}", "EXECUTION_ERROR"
        ) from exc
    if requested_provider not in session.get_providers():
        raise _fail("ONNX_PROVIDER_MISSING", f"requested provider not active: {requested_provider}")
    return session


def _encode_onnx(
    teacher: Any, session: Any, texts: Sequence[str], batch_size: int, label: str
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    session_inputs = {value.name for value in session.get_inputs()}
    if session_inputs != {"input_ids", "attention_mask", "token_type_ids"}:
        raise _fail("ONNX_IO_INVALID", "exported graph input names are not authoritative")
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        tokens = teacher.tokenize(batch)
        input_ids = tokens.get("input_ids")
        attention_mask = tokens.get("attention_mask")
        if input_ids is None or attention_mask is None:
            raise _fail("ONNX_TOKENIZATION_INVALID", f"tokenizer lacks inputs for {label}")
        token_type_ids = tokens.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = np.zeros_like(input_ids.detach().cpu().numpy())
        else:
            token_type_ids = token_type_ids.detach().cpu().numpy()
        inputs = {
            "input_ids": input_ids.detach().cpu().numpy().astype(np.int64, copy=False),
            "attention_mask": attention_mask.detach().cpu().numpy().astype(np.int64, copy=False),
            "token_type_ids": np.asarray(token_type_ids, dtype=np.int64),
        }
        try:
            values = session.run(["embeddings"], inputs)[0]
        except Exception as exc:
            raise _fail(
                "ONNX_INFERENCE_FAILED", f"cannot encode {label}: {exc}", "EXECUTION_ERROR"
            ) from exc
        outputs.append(np.asarray(values, dtype=np.float32))
    embeddings = np.ascontiguousarray(np.concatenate(outputs, axis=0), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts) or embeddings.shape[1] == 0:
        raise _fail(
            "ONNX_OUTPUT_INVALID", f"invalid embedding shape for {label}", "EXECUTION_ERROR"
        )
    if not np.isfinite(embeddings).all():
        raise _fail(
            "ONNX_OUTPUT_INVALID", f"non-finite embedding values for {label}", "EXECUTION_ERROR"
        )
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise _fail(
            "ONNX_OUTPUT_INVALID", f"non-positive embedding norm for {label}", "EXECUTION_ERROR"
        )
    return np.ascontiguousarray(embeddings / norms[:, np.newaxis], dtype=np.float32)


def _metric_values(metrics: Mapping[str, Any], role: str) -> Mapping[str, float]:
    try:
        values = metrics["roles"][role]["metrics"]
    except (KeyError, TypeError) as exc:
        raise _fail("TRANSITION_METRICS_INVALID", f"metrics missing for role: {role}") from exc
    if not isinstance(values, Mapping):
        raise _fail("TRANSITION_METRICS_INVALID", f"invalid metrics for role: {role}")
    return {str(name): float(value) for name, value in values.items()}


def _rankings_by_role(
    rankings: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    result: dict[str, dict[str, Mapping[str, Any]]] = {role: {} for role in ROLE_ORDER}
    for ranking in rankings:
        role = ranking.get("role")
        query_id = ranking.get("query_id")
        if role not in result or not isinstance(query_id, str) or query_id in result[role]:
            raise _fail(
                "TRANSITION_RANKING_INVALID", "rankings do not preserve canonical role/query IDs"
            )
        result[role][query_id] = ranking
    return result


def _comparison(
    run: PairedRun,
    source_rankings: Sequence[Mapping[str, Any]],
    target_rankings: Sequence[Mapping[str, Any]],
    source_metrics: Mapping[str, Any],
    target_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    source = run.source
    target = run.target
    _validate_observation(source)
    _validate_observation(target)
    if (
        source.document_ids != target.document_ids
        or source.query_ids != target.query_ids
        or source.query_roles != target.query_roles
        or source.relevant_document_ids != target.relevant_document_ids
    ):
        raise _fail("TRANSITION_OBSERVATION_INVALID", "source and target identities do not match")
    document_delta = np.abs(source.document_embeddings - target.document_embeddings)
    query_delta = np.abs(source.query_embeddings - target.query_embeddings)
    source_by_role = _rankings_by_role(source_rankings)
    target_by_role = _rankings_by_role(target_rankings)
    roles: dict[str, Any] = {}
    for role in ROLE_ORDER:
        source_role = source_by_role[role]
        target_role = target_by_role[role]
        if set(source_role) != set(target_role) or not source_role:
            raise _fail("TRANSITION_RANKING_INVALID", f"role rankings differ structurally: {role}")
        ranking_changes = sum(
            source_role[query_id]["ranked_document_ids"]
            != target_role[query_id]["ranked_document_ids"]
            for query_id in source_role
        )
        source_values = _metric_values(source_metrics, role)
        target_values = _metric_values(target_metrics, role)
        if set(source_values) != set(target_values):
            raise _fail("TRANSITION_METRICS_INVALID", f"metric names differ for role: {role}")
        roles[role] = {
            "query_count": len(source_role),
            "ranking_change_count": ranking_changes,
            "ranking_change_fraction": ranking_changes / len(source_role),
            "metric_decrease": {
                name: max(0.0, source_values[name] - target_values[name]) for name in source_values
            },
        }
    return {
        "run_id": run.run_id,
        "batch_size": run.batch_size,
        "functional": {
            "document_max_abs_delta": float(np.max(document_delta)),
            "query_max_abs_delta": float(np.max(query_delta)),
            "document_min_cosine_similarity": float(
                np.min(np.sum(source.document_embeddings * target.document_embeddings, axis=1))
            ),
            "query_min_cosine_similarity": float(
                np.min(np.sum(source.query_embeddings * target.query_embeddings, axis=1))
            ),
        },
        "roles": roles,
    }


def _decision_for_comparison(
    comparison: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    tolerances = _require_mapping(contract.get("operational_tolerances"), "operational_tolerances")
    functional_limit = _require_mapping(
        tolerances.get("functional_all_decision_roles"),
        "operational_tolerances.functional_all_decision_roles",
    )
    functional = _require_mapping(comparison.get("functional"), "comparison.functional")
    reasons: list[str] = []
    if max(
        float(functional["document_max_abs_delta"]), float(functional["query_max_abs_delta"])
    ) > float(functional_limit["embedding_max_abs_delta_lte"]):
        reasons.append("embedding_max_abs_delta_exceeds_operational_tolerance")
    if min(
        float(functional["document_min_cosine_similarity"]),
        float(functional["query_min_cosine_similarity"]),
    ) < float(functional_limit["embedding_min_cosine_similarity_gte"]):
        reasons.append("embedding_min_cosine_similarity_below_operational_tolerance")
    roles = _require_mapping(comparison.get("roles"), "comparison.roles")
    outcomes: dict[str, Any] = {}
    for role in ROLE_ORDER:
        observed = _require_mapping(roles.get(role), f"comparison.roles.{role}")
        if role == "frozen_critical":
            limits = _require_mapping(
                tolerances.get("topology_frozen_critical"),
                "operational_tolerances.topology_frozen_critical",
            )
        elif role in {"validation", "final_holdout"}:
            limits = _require_mapping(
                tolerances.get("topology_validation_and_final_holdout"),
                "operational_tolerances.topology_validation_and_final_holdout",
            )
        else:
            outcomes[role] = {"outcome": "OBSERVED_ONLY", "reasons": []}
            continue
        role_reasons: list[str] = []
        if float(observed["ranking_change_fraction"]) > float(
            limits.get("ranking_change_fraction_lte", 0.0)
        ) or int(observed["ranking_change_count"]) > int(limits.get("ranking_change_count_lte", 0)):
            role_reasons.append("ranking_change_exceeds_operational_tolerance")
        metric_decrease = _require_mapping(observed.get("metric_decrease"), "metric_decrease")
        for metric in ("recall_at_k", "mrr_at_k", "ndcg_at_k"):
            if float(metric_decrease[metric]) > float(limits[f"{metric}_decrease_lte"]):
                role_reasons.append(f"{metric}_decrease_exceeds_operational_tolerance")
        outcomes[role] = {"outcome": "FAIL" if role_reasons else "PASS", "reasons": role_reasons}
        reasons.extend(f"{role}:{reason}" for reason in role_reasons)
    return {
        "run_id": comparison["run_id"],
        "batch_size": comparison["batch_size"],
        "status": "FAIL" if reasons else "PASS",
        "reasons": reasons,
        "role_outcomes": outcomes,
    }


def _report_and_decision(
    runs: Sequence[PairedRun], contract: Mapping[str, Any], top_k: int
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not runs:
        raise _fail("TRANSITION_OBSERVATION_INVALID", "at least one paired run is required")
    comparisons: list[dict[str, Any]] = []
    per_run_metrics: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    target_records: list[dict[str, Any]] = []
    for run in runs:
        source_rankings, source_metrics = _rank_and_measure(run.source, top_k)
        target_rankings, target_metrics = _rank_and_measure(run.target, top_k)
        comparisons.append(
            _comparison(run, source_rankings, target_rankings, source_metrics, target_metrics)
        )
        per_run_metrics.append(
            {
                "run_id": run.run_id,
                "source": source_metrics,
                "target": target_metrics,
            }
        )
        source_records.extend({"run_id": run.run_id, **record} for record in source_rankings)
        target_records.extend({"run_id": run.run_id, **record} for record in target_rankings)
    run_decisions = [_decision_for_comparison(comparison, contract) for comparison in comparisons]
    statuses = [str(decision["status"]) for decision in run_decisions]
    decision = {
        "transition_id": TRANSITION_ID,
        "transition_a_status": "FAIL" if "FAIL" in statuses else "PASS",
        "technical_state": "VALID",
        "run_decisions": run_decisions,
        "transition_b_authorized": False,
    }
    decision["transition_b_authorized"] = decision["transition_a_status"] == "PASS"
    report = {
        "transition_format_version": TRANSITION_FORMAT_VERSION,
        "transition_id": TRANSITION_ID,
        "source_target_pair_count": len(runs),
        "source_runs": [{"run_id": run.run_id, "batch_size": run.batch_size} for run in runs],
        "top_k": top_k,
        "comparisons": comparisons,
        "per_run_metrics": per_run_metrics,
        "decision_status": decision["transition_a_status"],
    }
    return report, decision, source_records, target_records


def _write_observations(path: Path, runs: Sequence[PairedRun]) -> None:
    reference = runs[0].source
    for run in runs:
        for observation in (run.source, run.target):
            _validate_observation(observation)
            if (
                observation.document_ids != reference.document_ids
                or observation.query_ids != reference.query_ids
                or observation.query_roles != reference.query_roles
                or observation.relevant_document_ids != reference.relevant_document_ids
            ):
                raise _fail(
                    "TRANSITION_OBSERVATION_INVALID", "paired observations lack canonical IDs"
                )
    np.savez_compressed(
        path,
        run_ids=np.asarray([run.run_id for run in runs]),
        batch_sizes=np.asarray([run.batch_size for run in runs], dtype=np.int64),
        document_ids=np.asarray(reference.document_ids),
        query_ids=np.asarray(reference.query_ids),
        query_roles=np.asarray(reference.query_roles),
        source_document_embeddings=np.stack([run.source.document_embeddings for run in runs]),
        source_query_embeddings=np.stack([run.source.query_embeddings for run in runs]),
        target_document_embeddings=np.stack([run.target.document_embeddings for run in runs]),
        target_query_embeddings=np.stack([run.target.query_embeddings for run in runs]),
    )


def write_transition_a_package(
    output_directory: str | Path,
    dataset: MaterializedDataset,
    runs: Sequence[PairedRun],
    contract: Mapping[str, Any],
    contract_sha256: str,
    config_sha256: str,
    teacher_manifest: Mapping[str, Any],
    onnx_source_path: str | Path,
    onnx_manifest: Mapping[str, Any],
    evidence_scope: Mapping[str, Any],
    top_k: int,
) -> dict[str, Any]:
    output_path = Path(output_directory).resolve()
    source_path = Path(onnx_source_path)
    if output_path.exists():
        raise _fail("OUTPUT_ALREADY_EXISTS", f"output already exists: {output_path}")
    if not source_path.is_file():
        raise _fail("ONNX_ARTIFACT_MISSING", f"missing ONNX source artifact: {source_path}")
    if len({run.run_id for run in runs}) != len(runs):
        raise _fail("TRANSITION_OBSERVATION_INVALID", "paired run IDs are duplicated")
    report, decision, source_rankings, target_rankings = _report_and_decision(runs, contract, top_k)
    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.building-", dir=output_path.parent)
    )
    try:
        onnx_path = temporary_path / ONNX_FILENAME
        shutil.copyfile(source_path, onnx_path)
        observations_path = temporary_path / "transition-observations.npz"
        _write_observations(observations_path, runs)
        source_rankings_path = temporary_path / "source-rankings.jsonl"
        _write_bytes(source_rankings_path, _jsonl_bytes(source_rankings))
        target_rankings_path = temporary_path / "target-rankings.jsonl"
        _write_bytes(target_rankings_path, _jsonl_bytes(target_rankings))
        report_path = temporary_path / "comparison-report.json"
        _write_bytes(report_path, canonical_json_bytes(report) + b"\n")
        decision_path = temporary_path / "decision.json"
        _write_bytes(decision_path, canonical_json_bytes(decision) + b"\n")
        teacher_manifest_path = temporary_path / "teacher-manifest.json"
        _write_bytes(teacher_manifest_path, canonical_json_bytes(dict(teacher_manifest)) + b"\n")
        complete_onnx_manifest = {
            **onnx_manifest,
            "artifact_path": ONNX_FILENAME,
            "artifact_sha256": sha256_file(onnx_path),
            "artifact_size_bytes": onnx_path.stat().st_size,
        }
        onnx_manifest_path = temporary_path / "onnx-manifest.json"
        _write_bytes(onnx_manifest_path, canonical_json_bytes(complete_onnx_manifest) + b"\n")
        reference = runs[0].source
        replay_bundle = {
            "replay_format_version": REPLAY_FORMAT_VERSION,
            "transition_format_version": TRANSITION_FORMAT_VERSION,
            "transition_id": TRANSITION_ID,
            "contract": {
                "contract_id": _require_string(contract.get("contract_id"), "contract.contract_id"),
                "sha256": contract_sha256,
            },
            "configuration_sha256": config_sha256,
            "dataset": {
                "dataset_id": dataset.dataset_id,
                "materialization_manifest_sha256": dataset.manifest_sha256,
                "materialization_policy_sha256": dataset.materialization_policy_sha256,
                "partition_policy_sha256": dataset.partition_policy_sha256,
            },
            "observation": {
                "path": "transition-observations.npz",
                "embedding_dtype": "float32",
                "embedding_dimension": int(reference.document_embeddings.shape[1]),
                "document_count": len(reference.document_ids),
                "query_count": len(reference.query_ids),
            },
            "required_pairs": [
                {"run_id": run.run_id, "batch_size": run.batch_size} for run in runs
            ],
            "evaluation": {
                "top_k": top_k,
                "source_ranking_path": "source-rankings.jsonl",
                "target_ranking_path": "target-rankings.jsonl",
                "comparison_report_path": "comparison-report.json",
                "decision_path": "decision.json",
            },
            "qrels": {
                query_id: reference.relevant_document_ids[query_id]
                for query_id in sorted(reference.query_ids, key=str.encode)
            },
            "replay_requires_model_execution": False,
        }
        bundle_path = temporary_path / "replay-bundle.json"
        _write_bytes(bundle_path, canonical_json_bytes(replay_bundle) + b"\n")
        artifacts = [
            _artifact_entry(temporary_path, path)
            for path in (
                onnx_path,
                observations_path,
                source_rankings_path,
                target_rankings_path,
                report_path,
                decision_path,
                teacher_manifest_path,
                onnx_manifest_path,
                bundle_path,
            )
        ]
        artifacts.sort(key=lambda artifact: artifact["path"])
        evidence_manifest = {
            "evidence_format_version": EVIDENCE_FORMAT_VERSION,
            "evidence_status": "CAPTURED_PENDING_REPLAY",
            "qualifying_m1_evidence": bool(evidence_scope.get("qualifying_m1_evidence")),
            "dataset_id": dataset.dataset_id,
            "transition_id": TRANSITION_ID,
            "artifacts": artifacts,
            "integrity": {
                "artifact_hash_algorithm": "SHA-256",
                "missing_evidence_behavior": "BLOCKED",
                "replay_without_model_execution_required": True,
            },
        }
        _write_bytes(
            temporary_path / "evidence-manifest.json",
            canonical_json_bytes(evidence_manifest) + b"\n",
        )
        os.replace(temporary_path, output_path)
        return {
            "output_directory": str(output_path),
            "evidence_manifest_sha256": sha256_file(output_path / "evidence-manifest.json"),
            "transition_a_status": decision["transition_a_status"],
            "source_target_pair_count": len(runs),
            "document_count": len(reference.document_ids),
            "query_count": len(reference.query_ids),
            "evidence_status": "CAPTURED_PENDING_REPLAY",
        }
    except Exception:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise


def capture_transition_a(
    config_path: str | Path, dataset_directory: str | Path, output_directory: str | Path
) -> dict[str, Any]:
    config, config_sha256 = _load_config(config_path)
    contract_path = _require_string(config.get("contract_path"), "contract_path")
    contract, contract_sha256 = _load_contract(contract_path)
    top_k, batch_sizes = _transition_config(config, contract)
    dataset = load_materialized_dataset(dataset_directory)
    expected_dataset_id = _require_string(
        _require_mapping(config.get("dataset"), "dataset").get("dataset_id"), "dataset.dataset_id"
    )
    if dataset.dataset_id != expected_dataset_id:
        raise _fail(
            "DATASET_ID_MISMATCH",
            f"expected dataset {expected_dataset_id}, got {dataset.dataset_id}",
        )
    model, teacher_manifest = _load_teacher(config)
    onnx_config = _require_mapping(config.get("onnx"), "onnx")
    opset_version = _positive_int(onnx_config.get("opset_version"), "onnx.opset_version")
    provider = _require_string(onnx_config.get("execution_provider"), "onnx.execution_provider")
    temporary_descriptor, temporary_name = tempfile.mkstemp(
        prefix="nc-m1-transition-a-", suffix=".onnx"
    )
    os.close(temporary_descriptor)
    temporary_onnx = Path(temporary_name)
    try:
        onnx_manifest = _export_teacher_onnx(model, temporary_onnx, opset_version, provider)
        session = _onnx_session(temporary_onnx, provider)
        query_texts = _ordered_query_texts(dataset)
        runs: list[PairedRun] = []
        for batch_size in batch_sizes:
            source_document_embeddings = _encode(
                model, dataset.document_texts, batch_size, f"source batch {batch_size} documents"
            )
            source_query_embeddings = _encode(
                model, query_texts, batch_size, f"source batch {batch_size} queries"
            )
            target_document_embeddings = _encode_onnx(
                model,
                session,
                dataset.document_texts,
                batch_size,
                f"target batch {batch_size} documents",
            )
            target_query_embeddings = _encode_onnx(
                model, session, query_texts, batch_size, f"target batch {batch_size} queries"
            )
            runs.append(
                PairedRun(
                    run_id=f"batch-size-{batch_size:04d}",
                    batch_size=batch_size,
                    source=_ordered_observation(
                        dataset, source_document_embeddings, source_query_embeddings
                    ),
                    target=_ordered_observation(
                        dataset, target_document_embeddings, target_query_embeddings
                    ),
                )
            )
        teacher_manifest = {
            **teacher_manifest,
            "configuration_sha256": config_sha256,
            "contract_sha256": contract_sha256,
            "dataset_id": dataset.dataset_id,
            "materialization_manifest_sha256": dataset.manifest_sha256,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        }
        return write_transition_a_package(
            output_directory=output_directory,
            dataset=dataset,
            runs=runs,
            contract=contract,
            contract_sha256=contract_sha256,
            config_sha256=config_sha256,
            teacher_manifest=teacher_manifest,
            onnx_source_path=temporary_onnx,
            onnx_manifest=onnx_manifest,
            evidence_scope=_require_mapping(config.get("evidence_scope"), "evidence_scope"),
            top_k=top_k,
        )
    finally:
        temporary_onnx.unlink(missing_ok=True)


def _load_replayed_runs(bundle: Mapping[str, Any], root: Path) -> list[PairedRun]:
    observation = _require_mapping(bundle.get("observation"), "observation")
    observation_path = _safe_artifact_path(
        root, _require_string(observation.get("path"), "observation.path")
    )
    raw_pairs = bundle.get("required_pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise _fail(
            "MISSING_DECLARED_SOURCE_OBSERVATION",
            "replay bundle does not declare source-target pairs",
        )
    pairs: list[dict[str, Any]] = []
    for index, raw_pair in enumerate(raw_pairs):
        pair = _require_mapping(raw_pair, f"required_pairs[{index}]")
        pairs.append(
            {
                "run_id": _require_string(pair.get("run_id"), f"required_pairs[{index}].run_id"),
                "batch_size": _positive_int(
                    pair.get("batch_size"), f"required_pairs[{index}].batch_size"
                ),
            }
        )
    if len({pair["run_id"] for pair in pairs}) != len(pairs):
        raise _fail("REPLAY_OBSERVATION_INVALID", "replay bundle duplicates paired run IDs")
    try:
        with np.load(observation_path, allow_pickle=False) as archive:
            run_ids = [str(value) for value in archive["run_ids"].tolist()]
            batch_sizes = [int(value) for value in archive["batch_sizes"].tolist()]
            document_ids = [str(value) for value in archive["document_ids"].tolist()]
            query_ids = [str(value) for value in archive["query_ids"].tolist()]
            query_roles = [str(value) for value in archive["query_roles"].tolist()]
            source_document_embeddings = np.ascontiguousarray(
                archive["source_document_embeddings"], dtype=np.float32
            )
            source_query_embeddings = np.ascontiguousarray(
                archive["source_query_embeddings"], dtype=np.float32
            )
            target_document_embeddings = np.ascontiguousarray(
                archive["target_document_embeddings"], dtype=np.float32
            )
            target_query_embeddings = np.ascontiguousarray(
                archive["target_query_embeddings"], dtype=np.float32
            )
    except (OSError, KeyError, ValueError) as exc:
        raise _fail(
            "REPLAY_OBSERVATION_INVALID", f"cannot load transition observations: {exc}"
        ) from exc
    recorded_pairs = [
        {"run_id": run_id, "batch_size": batch_size}
        for run_id, batch_size in zip(run_ids, batch_sizes, strict=True)
    ]
    if pairs != recorded_pairs:
        raise _fail(
            "MISSING_DECLARED_SOURCE_OBSERVATION",
            "declared source-target pairs are missing or have mismatched metadata",
        )
    run_count = len(pairs)
    arrays = (
        source_document_embeddings,
        source_query_embeddings,
        target_document_embeddings,
        target_query_embeddings,
    )
    if (
        any(array.ndim != 3 or array.shape[0] != run_count for array in arrays)
        or source_document_embeddings.shape != target_document_embeddings.shape
        or source_query_embeddings.shape != target_query_embeddings.shape
        or source_document_embeddings.shape[1] != len(document_ids)
        or source_query_embeddings.shape[1] != len(query_ids)
        or source_document_embeddings.shape[2] != source_query_embeddings.shape[2]
    ):
        raise _fail("REPLAY_OBSERVATION_INVALID", "transition observation shapes are invalid")
    if any(array.dtype != np.float32 for array in arrays):
        raise _fail("REPLAY_OBSERVATION_INVALID", "transition observations must be float32")
    if source_document_embeddings.shape[2] != _require_int(
        observation.get("embedding_dimension"), "observation.embedding_dimension"
    ):
        raise _fail("REPLAY_OBSERVATION_INVALID", "embedding dimension mismatch")
    if len(document_ids) != _require_int(
        observation.get("document_count"), "observation.document_count"
    ) or len(query_ids) != _require_int(observation.get("query_count"), "observation.query_count"):
        raise _fail("REPLAY_OBSERVATION_INVALID", "transition observation counts are invalid")
    qrels = _require_mapping(bundle.get("qrels"), "qrels")
    if set(qrels) != set(query_ids) or len(query_roles) != len(query_ids):
        raise _fail("REPLAY_QRELS_INVALID", "replay qrels or query roles are invalid")
    relevant_document_ids: dict[str, list[str]] = {}
    for query_id in query_ids:
        values = qrels[query_id]
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise _fail("REPLAY_QRELS_INVALID", f"invalid qrels for query: {query_id}")
        relevant_document_ids[query_id] = values
    return [
        PairedRun(
            run_id=pair["run_id"],
            batch_size=pair["batch_size"],
            source=TeacherObservation(
                document_ids=document_ids,
                document_embeddings=source_document_embeddings[index],
                query_ids=query_ids,
                query_embeddings=source_query_embeddings[index],
                query_roles=query_roles,
                relevant_document_ids=relevant_document_ids,
            ),
            target=TeacherObservation(
                document_ids=document_ids,
                document_embeddings=target_document_embeddings[index],
                query_ids=query_ids,
                query_embeddings=target_query_embeddings[index],
                query_roles=query_roles,
                relevant_document_ids=relevant_document_ids,
            ),
        )
        for index, pair in enumerate(pairs)
    ]


def replay_transition_a(bundle_path: str | Path) -> dict[str, Any]:
    bundle_file = Path(bundle_path).resolve()
    root = bundle_file.parent
    bundle = _load_json(bundle_file, "REPLAY_BUNDLE_INVALID")
    if bundle.get("replay_format_version") != REPLAY_FORMAT_VERSION:
        raise _fail("REPLAY_FORMAT_INVALID", "replay bundle format is not authoritative")
    if bundle.get("transition_format_version") != TRANSITION_FORMAT_VERSION:
        raise _fail("REPLAY_FORMAT_INVALID", "transition format is not authoritative")
    if bundle.get("transition_id") != TRANSITION_ID:
        raise _fail("REPLAY_FORMAT_INVALID", "replay bundle is not transition A")
    evidence_manifest = _load_json(root / "evidence-manifest.json", "EVIDENCE_MANIFEST_INVALID")
    _verify_artifacts(root, evidence_manifest, "artifacts")
    contract_info = _require_mapping(bundle.get("contract"), "contract")
    if contract_info.get("contract_id") != "m1-transition-a-v1":
        raise _fail("REPLAY_CONTRACT_INVALID", "replay bundle does not bind transition A contract")
    runs = _load_replayed_runs(bundle, root)
    evaluation = _require_mapping(bundle.get("evaluation"), "evaluation")
    top_k = _positive_int(evaluation.get("top_k"), "evaluation.top_k")
    contract_path = Path("contracts/m1-transition-a-v1.json")
    contract, contract_sha256 = _load_contract(contract_path)
    if contract_info.get("sha256") != contract_sha256:
        raise _fail("REPLAY_CONTRACT_MISMATCH", "local authoritative contract SHA-256 differs")
    report, decision, source_rankings, target_rankings = _report_and_decision(runs, contract, top_k)
    artifact_paths = {
        "source": _safe_artifact_path(
            root, _require_string(evaluation.get("source_ranking_path"), "source_ranking_path")
        ),
        "target": _safe_artifact_path(
            root, _require_string(evaluation.get("target_ranking_path"), "target_ranking_path")
        ),
        "report": _safe_artifact_path(
            root,
            _require_string(evaluation.get("comparison_report_path"), "comparison_report_path"),
        ),
        "decision": _safe_artifact_path(
            root, _require_string(evaluation.get("decision_path"), "decision_path")
        ),
    }
    if artifact_paths["source"].read_bytes() != _jsonl_bytes(source_rankings):
        raise _fail("REPLAY_RANKING_MISMATCH", "source rankings do not match replay")
    if artifact_paths["target"].read_bytes() != _jsonl_bytes(target_rankings):
        raise _fail("REPLAY_RANKING_MISMATCH", "target rankings do not match replay")
    if artifact_paths["report"].read_bytes() != canonical_json_bytes(report) + b"\n":
        raise _fail("REPLAY_COMPARISON_MISMATCH", "comparison report does not match replay")
    if artifact_paths["decision"].read_bytes() != canonical_json_bytes(decision) + b"\n":
        raise _fail("REPLAY_DECISION_MISMATCH", "decision does not match replay")
    if bundle.get("replay_requires_model_execution") is not False:
        raise _fail("REPLAY_POLICY_INVALID", "replay must not execute source or target models")
    return {
        "status": "PASS",
        "replay_verified": True,
        "model_execution_used": False,
        "transition_a_status": decision["transition_a_status"],
        "source_target_pair_count": len(runs),
        "dataset_id": _require_string(
            _require_mapping(bundle.get("dataset"), "dataset").get("dataset_id"),
            "dataset.dataset_id",
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture and replay M1 Transition A PyTorch FP32 to ONNX FP32 evidence."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture", help="Export and capture Transition A evidence.")
    capture.add_argument("--config", required=True)
    capture.add_argument("--dataset", required=True)
    capture.add_argument("--output", required=True)
    replay = subparsers.add_parser("replay", help="Replay Transition A evidence without models.")
    replay.add_argument("--bundle", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "capture":
            result = capture_transition_a(args.config, args.dataset, args.output)
        else:
            result = replay_transition_a(args.bundle)
    except TeacherEvidenceError as exc:
        print(
            json.dumps(
                {"status": exc.status, "error": {"code": exc.code, "message": exc.message}},
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
