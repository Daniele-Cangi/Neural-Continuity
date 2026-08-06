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
    _rank_and_measure,
    _require_int,
    _require_mapping,
    _require_string,
    _safe_artifact_path,
    _verify_artifacts,
    _write_bytes,
    load_materialized_dataset,
)

MEASUREMENT_NULL_FORMAT_VERSION = "1.0.0"
MEASUREMENT_NULL_ROLE = "measurement_null"
REPEATED_INFERENCE_FAMILY = "repeated_inference"
BATCH_SIZE_VARIATION_FAMILY = "batch_size_variation"


@dataclass(frozen=True)
class SourceRun:
    run_id: str
    family: str
    batch_size: int
    observation: TeacherObservation


def _positive_int(value: Any, field: str) -> int:
    parsed = _require_int(value, field)
    if parsed == 0:
        raise _fail("MEASUREMENT_NULL_CONFIG_INVALID", f"{field} must be positive")
    return parsed


def _measurement_config(config: Mapping[str, Any]) -> tuple[int, int, list[int]]:
    measurement_null = _require_mapping(config.get("measurement_null"), "measurement_null")
    repeated = _require_mapping(
        measurement_null.get(REPEATED_INFERENCE_FAMILY),
        f"measurement_null.{REPEATED_INFERENCE_FAMILY}",
    )
    repeat_count = _positive_int(
        repeated.get("count"), f"measurement_null.{REPEATED_INFERENCE_FAMILY}.count"
    )
    if repeat_count < 3:
        raise _fail(
            "MEASUREMENT_NULL_CONFIG_INVALID",
            "measurement_null.repeated_inference.count must be at least 3",
        )
    repeated_batch_size = _positive_int(
        repeated.get("batch_size"),
        f"measurement_null.{REPEATED_INFERENCE_FAMILY}.batch_size",
    )
    variation = _require_mapping(
        measurement_null.get(BATCH_SIZE_VARIATION_FAMILY),
        f"measurement_null.{BATCH_SIZE_VARIATION_FAMILY}",
    )
    raw_batch_sizes = variation.get("batch_sizes")
    if not isinstance(raw_batch_sizes, list) or len(raw_batch_sizes) < 2:
        raise _fail(
            "MEASUREMENT_NULL_CONFIG_INVALID",
            "measurement_null.batch_size_variation.batch_sizes must contain at least two values",
        )
    batch_sizes = [
        _positive_int(value, f"measurement_null.batch_size_variation.batch_sizes[{index}]")
        for index, value in enumerate(raw_batch_sizes)
    ]
    if len(set(batch_sizes)) != len(batch_sizes):
        raise _fail(
            "MEASUREMENT_NULL_CONFIG_INVALID",
            "measurement_null.batch_size_variation.batch_sizes must be unique",
        )
    return repeat_count, repeated_batch_size, batch_sizes


def _measurement_observation(
    dataset: MaterializedDataset,
    document_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
) -> TeacherObservation:
    role_data = dataset.roles.get(MEASUREMENT_NULL_ROLE)
    if role_data is None:
        raise _fail("MEASUREMENT_NULL_ROLE_MISSING", "materialized dataset lacks measurement_null")
    pairs = sorted(
        zip(role_data.query_ids, role_data.query_texts, strict=True),
        key=lambda item: item[0].encode(),
    )
    query_ids = [query_id for query_id, _ in pairs]
    if query_embeddings.shape[0] != len(query_ids):
        raise _fail(
            "MEASUREMENT_NULL_OBSERVATION_INVALID",
            "measurement-null query embedding count does not match role membership",
            "EXECUTION_ERROR",
        )
    return TeacherObservation(
        document_ids=list(dataset.document_ids),
        document_embeddings=document_embeddings,
        query_ids=query_ids,
        query_embeddings=query_embeddings,
        query_roles=[MEASUREMENT_NULL_ROLE] * len(query_ids),
        relevant_document_ids={
            query_id: role_data.relevant_document_ids[query_id] for query_id in query_ids
        },
    )


def _run_metadata(runs: Sequence[SourceRun]) -> list[dict[str, Any]]:
    return [
        {"run_id": run.run_id, "family": run.family, "batch_size": run.batch_size} for run in runs
    ]


def _metric_values(metrics: Mapping[str, Any]) -> Mapping[str, float]:
    try:
        values = metrics["roles"][MEASUREMENT_NULL_ROLE]["metrics"]
    except (KeyError, TypeError) as exc:
        raise _fail(
            "MEASUREMENT_NULL_METRICS_INVALID",
            "measurement-null metrics are missing",
            "EXECUTION_ERROR",
        ) from exc
    if not isinstance(values, Mapping) or any(
        not isinstance(value, float | int) for value in values.values()
    ):
        raise _fail(
            "MEASUREMENT_NULL_METRICS_INVALID",
            "measurement-null metrics are invalid",
            "EXECUTION_ERROR",
        )
    return {str(key): float(value) for key, value in values.items()}


def _ranking_change_count(
    reference_rankings: Sequence[Mapping[str, Any]], observed_rankings: Sequence[Mapping[str, Any]]
) -> int:
    reference_by_query = {str(record["query_id"]): record for record in reference_rankings}
    observed_by_query = {str(record["query_id"]): record for record in observed_rankings}
    if set(reference_by_query) != set(observed_by_query):
        raise _fail(
            "MEASUREMENT_NULL_RANKING_INVALID",
            "source runs do not contain the same measurement-null queries",
            "EXECUTION_ERROR",
        )
    return sum(
        reference_by_query[query_id].get("ranked_document_ids")
        != observed_by_query[query_id].get("ranked_document_ids")
        for query_id in reference_by_query
    )


def _comparison(
    reference: SourceRun,
    observed: SourceRun,
    reference_rankings: Sequence[Mapping[str, Any]],
    observed_rankings: Sequence[Mapping[str, Any]],
    reference_metrics: Mapping[str, Any],
    observed_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    reference_observation = reference.observation
    observed_observation = observed.observation
    if (
        reference_observation.document_ids != observed_observation.document_ids
        or reference_observation.query_ids != observed_observation.query_ids
    ):
        raise _fail(
            "MEASUREMENT_NULL_OBSERVATION_INVALID",
            "source run identities are not stable across executions",
            "EXECUTION_ERROR",
        )
    document_delta = np.abs(
        reference_observation.document_embeddings - observed_observation.document_embeddings
    )
    query_delta = np.abs(
        reference_observation.query_embeddings - observed_observation.query_embeddings
    )
    document_cosine = np.sum(
        reference_observation.document_embeddings * observed_observation.document_embeddings, axis=1
    )
    query_cosine = np.sum(
        reference_observation.query_embeddings * observed_observation.query_embeddings, axis=1
    )
    reference_values = _metric_values(reference_metrics)
    observed_values = _metric_values(observed_metrics)
    if set(reference_values) != set(observed_values):
        raise _fail(
            "MEASUREMENT_NULL_METRICS_INVALID",
            "source run metric names are not stable",
            "EXECUTION_ERROR",
        )
    metric_delta = {
        name: abs(observed_values[name] - reference_values[name]) for name in reference_values
    }
    changed_rankings = _ranking_change_count(reference_rankings, observed_rankings)
    return {
        "reference_run_id": reference.run_id,
        "observed_run_id": observed.run_id,
        "family": observed.family,
        "batch_size": observed.batch_size,
        "document_max_abs_delta": float(np.max(document_delta)),
        "query_max_abs_delta": float(np.max(query_delta)),
        "document_min_cosine_similarity": float(np.min(document_cosine)),
        "query_min_cosine_similarity": float(np.min(query_cosine)),
        "ranking_change_count": changed_rankings,
        "ranking_change_fraction": changed_rankings / len(reference_observation.query_ids),
        "absolute_metric_delta": metric_delta,
    }


def _envelope(family: str, comparisons: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    family_comparisons = [
        comparison for comparison in comparisons if comparison["family"] == family
    ]
    if not family_comparisons:
        raise _fail(
            "MEASUREMENT_NULL_ENVELOPE_INVALID", f"no comparisons exist for family: {family}"
        )
    metric_names = set(family_comparisons[0]["absolute_metric_delta"])
    if any(
        set(comparison["absolute_metric_delta"]) != metric_names
        for comparison in family_comparisons
    ):
        raise _fail(
            "MEASUREMENT_NULL_ENVELOPE_INVALID", "metric sets differ across source comparisons"
        )
    return {
        "family": family,
        "comparison_count": len(family_comparisons),
        "observed_run_ids": [
            str(comparison["observed_run_id"]) for comparison in family_comparisons
        ],
        "empirical_maximum": {
            "document_max_abs_delta": max(
                float(comparison["document_max_abs_delta"]) for comparison in family_comparisons
            ),
            "query_max_abs_delta": max(
                float(comparison["query_max_abs_delta"]) for comparison in family_comparisons
            ),
            "ranking_change_count": max(
                int(comparison["ranking_change_count"]) for comparison in family_comparisons
            ),
            "ranking_change_fraction": max(
                float(comparison["ranking_change_fraction"]) for comparison in family_comparisons
            ),
            "absolute_metric_delta": {
                name: max(
                    float(comparison["absolute_metric_delta"][name])
                    for comparison in family_comparisons
                )
                for name in sorted(metric_names)
            },
        },
        "empirical_minimum": {
            "document_min_cosine_similarity": min(
                float(comparison["document_min_cosine_similarity"])
                for comparison in family_comparisons
            ),
            "query_min_cosine_similarity": min(
                float(comparison["query_min_cosine_similarity"])
                for comparison in family_comparisons
            ),
        },
    }


def _report(runs: Sequence[SourceRun], top_k: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(runs) < 2:
        raise _fail("MEASUREMENT_NULL_OBSERVATION_INVALID", "at least two source runs are required")
    rankings_by_run: dict[str, list[dict[str, Any]]] = {}
    metrics_by_run: dict[str, dict[str, Any]] = {}
    serialized_rankings: list[dict[str, Any]] = []
    for run in runs:
        rankings, metrics = _rank_and_measure(run.observation, top_k)
        rankings_by_run[run.run_id] = rankings
        metrics_by_run[run.run_id] = metrics
        serialized_rankings.extend(
            {
                "run_id": run.run_id,
                "family": run.family,
                "batch_size": run.batch_size,
                **ranking,
            }
            for ranking in rankings
        )
    reference = runs[0]
    comparisons = [
        _comparison(
            reference,
            run,
            rankings_by_run[reference.run_id],
            rankings_by_run[run.run_id],
            metrics_by_run[reference.run_id],
            metrics_by_run[run.run_id],
        )
        for run in runs[1:]
    ]
    report = {
        "measurement_null_format_version": MEASUREMENT_NULL_FORMAT_VERSION,
        "source_only": True,
        "reference_run_id": reference.run_id,
        "top_k": top_k,
        "source_runs": _run_metadata(runs),
        "per_run_metrics": [
            {"run_id": run.run_id, "metrics": metrics_by_run[run.run_id]} for run in runs
        ],
        "comparisons": comparisons,
        "empirical_envelopes": {
            REPEATED_INFERENCE_FAMILY: _envelope(REPEATED_INFERENCE_FAMILY, comparisons),
            BATCH_SIZE_VARIATION_FAMILY: _envelope(BATCH_SIZE_VARIATION_FAMILY, comparisons),
        },
        "measurement_null_status": "CAPTURED_NOT_DECIDED",
        "transition_a_decision": "NOT_APPLICABLE",
        "operational_tolerance": "NOT_SELECTED",
    }
    return report, serialized_rankings


def _write_observations(path: Path, runs: Sequence[SourceRun]) -> None:
    reference = runs[0].observation
    for run in runs[1:]:
        observation = run.observation
        if (
            observation.document_ids != reference.document_ids
            or observation.query_ids != reference.query_ids
            or observation.query_roles != reference.query_roles
            or observation.relevant_document_ids != reference.relevant_document_ids
        ):
            raise _fail(
                "MEASUREMENT_NULL_OBSERVATION_INVALID",
                "source observations do not share a canonical identity",
                "EXECUTION_ERROR",
            )
    np.savez_compressed(
        path,
        run_ids=np.asarray([run.run_id for run in runs]),
        families=np.asarray([run.family for run in runs]),
        batch_sizes=np.asarray([run.batch_size for run in runs], dtype=np.int64),
        document_ids=np.asarray(reference.document_ids),
        query_ids=np.asarray(reference.query_ids),
        document_embeddings=np.stack([run.observation.document_embeddings for run in runs]),
        query_embeddings=np.stack([run.observation.query_embeddings for run in runs]),
    )


def write_measurement_null_package(
    output_directory: str | Path,
    dataset: MaterializedDataset,
    runs: Sequence[SourceRun],
    teacher_manifest: Mapping[str, Any],
    evidence_scope: Mapping[str, Any],
    config_sha256: str,
    top_k: int,
) -> dict[str, Any]:
    output_path = Path(output_directory).resolve()
    if output_path.exists():
        raise _fail("OUTPUT_ALREADY_EXISTS", f"output already exists: {output_path}")
    if not runs or len({run.run_id for run in runs}) != len(runs):
        raise _fail(
            "MEASUREMENT_NULL_OBSERVATION_INVALID", "source run IDs are empty or duplicated"
        )
    report, rankings = _report(runs, top_k)
    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.building-", dir=output_path.parent)
    )
    try:
        observations_path = temporary_path / "source-null-observations.npz"
        _write_observations(observations_path, runs)
        rankings_path = temporary_path / "source-null-rankings.jsonl"
        _write_bytes(rankings_path, _jsonl_bytes(rankings))
        report_path = temporary_path / "comparison-report.json"
        _write_bytes(report_path, canonical_json_bytes(report) + b"\n")
        teacher_manifest_path = temporary_path / "teacher-manifest.json"
        _write_bytes(teacher_manifest_path, canonical_json_bytes(dict(teacher_manifest)) + b"\n")
        reference = runs[0].observation
        replay_bundle = {
            "replay_format_version": REPLAY_FORMAT_VERSION,
            "measurement_null_format_version": MEASUREMENT_NULL_FORMAT_VERSION,
            "evidence_scope": dict(evidence_scope),
            "configuration_sha256": config_sha256,
            "dataset": {
                "dataset_id": dataset.dataset_id,
                "materialization_manifest_sha256": dataset.manifest_sha256,
                "materialization_policy_sha256": dataset.materialization_policy_sha256,
                "partition_policy_sha256": dataset.partition_policy_sha256,
            },
            "observation": {
                "path": "source-null-observations.npz",
                "embedding_dtype": "float32",
                "embedding_dimension": int(reference.document_embeddings.shape[1]),
                "document_count": len(reference.document_ids),
                "query_count": len(reference.query_ids),
                "output_normalization": "l2_unit_after_encode",
            },
            "required_source_runs": _run_metadata(runs),
            "evaluation": {
                "top_k": top_k,
                "ranking_path": "source-null-rankings.jsonl",
                "comparison_report_path": "comparison-report.json",
            },
            "qrels": {
                query_id: reference.relevant_document_ids[query_id]
                for query_id in sorted(reference.query_ids, key=str.encode)
            },
            "replay_requires_model_execution": False,
            "measurement_null_status": "CAPTURED_NOT_DECIDED",
            "transition_a_decision": "NOT_APPLICABLE",
        }
        bundle_path = temporary_path / "replay-bundle.json"
        _write_bytes(bundle_path, canonical_json_bytes(replay_bundle) + b"\n")
        artifacts = [
            _artifact_entry(temporary_path, path)
            for path in (
                observations_path,
                rankings_path,
                report_path,
                teacher_manifest_path,
                bundle_path,
            )
        ]
        artifacts.sort(key=lambda artifact: artifact["path"])
        evidence_manifest = {
            "evidence_format_version": EVIDENCE_FORMAT_VERSION,
            "evidence_status": "CAPTURED_PENDING_REPLAY",
            "qualifying_m1_evidence": bool(evidence_scope.get("qualifying_m1_evidence")),
            "dataset_id": dataset.dataset_id,
            "evidence_kind": "real_teacher_measurement_null",
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
            "dataset_id": dataset.dataset_id,
            "document_count": len(reference.document_ids),
            "query_count": len(reference.query_ids),
            "source_run_count": len(runs),
            "evidence_status": "CAPTURED_PENDING_REPLAY",
        }
    except Exception:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise


def capture_measurement_null(
    config_path: str | Path, dataset_directory: str | Path, output_directory: str | Path
) -> dict[str, Any]:
    config, config_sha256 = _load_config(config_path)
    dataset = load_materialized_dataset(dataset_directory)
    expected_dataset_id = _require_string(
        _require_mapping(config.get("dataset"), "dataset").get("dataset_id"), "dataset.dataset_id"
    )
    if dataset.dataset_id != expected_dataset_id:
        raise _fail(
            "DATASET_ID_MISMATCH",
            f"expected dataset {expected_dataset_id}, got {dataset.dataset_id}",
        )
    top_k = _positive_int(
        _require_mapping(config.get("evaluation"), "evaluation").get("top_k"), "evaluation.top_k"
    )
    repeat_count, repeated_batch_size, batch_sizes = _measurement_config(config)
    model, teacher_manifest = _load_teacher(config)
    role_data = dataset.roles.get(MEASUREMENT_NULL_ROLE)
    if role_data is None:
        raise _fail("MEASUREMENT_NULL_ROLE_MISSING", "materialized dataset lacks measurement_null")
    query_texts = [
        text
        for _, text in sorted(
            zip(role_data.query_ids, role_data.query_texts, strict=True),
            key=lambda item: item[0].encode(),
        )
    ]
    conditions = [
        (f"repeated-inference-{index:03d}", REPEATED_INFERENCE_FAMILY, repeated_batch_size)
        for index in range(1, repeat_count + 1)
    ] + [
        (f"batch-size-{batch_size:04d}", BATCH_SIZE_VARIATION_FAMILY, batch_size)
        for batch_size in batch_sizes
    ]
    runs: list[SourceRun] = []
    for run_id, family, batch_size in conditions:
        document_embeddings = _encode(
            model, dataset.document_texts, batch_size, f"{run_id} documents"
        )
        query_embeddings = _encode(
            model, query_texts, batch_size, f"{run_id} measurement-null queries"
        )
        runs.append(
            SourceRun(
                run_id=run_id,
                family=family,
                batch_size=batch_size,
                observation=_measurement_observation(
                    dataset, document_embeddings, query_embeddings
                ),
            )
        )
    teacher_manifest = {
        **teacher_manifest,
        "configuration_sha256": config_sha256,
        "dataset_id": dataset.dataset_id,
        "materialization_manifest_sha256": dataset.manifest_sha256,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
    }
    return write_measurement_null_package(
        output_directory=output_directory,
        dataset=dataset,
        runs=runs,
        teacher_manifest=teacher_manifest,
        evidence_scope=_require_mapping(config.get("evidence_scope"), "evidence_scope"),
        config_sha256=config_sha256,
        top_k=top_k,
    )


def _load_replayed_runs(bundle: Mapping[str, Any], root: Path) -> list[SourceRun]:
    observation_info = _require_mapping(bundle.get("observation"), "observation")
    observation_path = _safe_artifact_path(
        root, _require_string(observation_info.get("path"), "observation.path")
    )
    raw_required_runs = bundle.get("required_source_runs")
    if not isinstance(raw_required_runs, list) or not raw_required_runs:
        raise _fail(
            "MISSING_DECLARED_SOURCE_OBSERVATION",
            "replay bundle does not declare its source observations",
        )
    required_runs: list[dict[str, Any]] = []
    for index, raw_run in enumerate(raw_required_runs):
        run = _require_mapping(raw_run, f"required_source_runs[{index}]")
        family = _require_string(run.get("family"), f"required_source_runs[{index}].family")
        if family not in {REPEATED_INFERENCE_FAMILY, BATCH_SIZE_VARIATION_FAMILY}:
            raise _fail("REPLAY_SOURCE_RUN_INVALID", f"unknown source run family: {family}")
        required_runs.append(
            {
                "run_id": _require_string(
                    run.get("run_id"), f"required_source_runs[{index}].run_id"
                ),
                "family": family,
                "batch_size": _positive_int(
                    run.get("batch_size"), f"required_source_runs[{index}].batch_size"
                ),
            }
        )
    if len({run["run_id"] for run in required_runs}) != len(required_runs):
        raise _fail("REPLAY_SOURCE_RUN_INVALID", "replay bundle duplicates source run IDs")
    try:
        with np.load(observation_path, allow_pickle=False) as archive:
            run_ids = [str(value) for value in archive["run_ids"].tolist()]
            families = [str(value) for value in archive["families"].tolist()]
            batch_sizes = [int(value) for value in archive["batch_sizes"].tolist()]
            document_ids = [str(value) for value in archive["document_ids"].tolist()]
            query_ids = [str(value) for value in archive["query_ids"].tolist()]
            document_embeddings = np.ascontiguousarray(
                archive["document_embeddings"], dtype=np.float32
            )
            query_embeddings = np.ascontiguousarray(archive["query_embeddings"], dtype=np.float32)
    except (OSError, KeyError, ValueError) as exc:
        raise _fail(
            "REPLAY_OBSERVATION_INVALID", f"cannot load source observations: {exc}"
        ) from exc
    if not run_ids or len(run_ids) != len(set(run_ids)):
        raise _fail("REPLAY_OBSERVATION_INVALID", "source observation run IDs are invalid")
    recorded_metadata = [
        {"run_id": run_id, "family": family, "batch_size": batch_size}
        for run_id, family, batch_size in zip(run_ids, families, batch_sizes, strict=True)
    ]
    if recorded_metadata != required_runs:
        raise _fail(
            "MISSING_DECLARED_SOURCE_OBSERVATION",
            "declared source observations are missing or have mismatched metadata",
        )
    run_count = len(required_runs)
    if (
        document_embeddings.ndim != 3
        or query_embeddings.ndim != 3
        or document_embeddings.shape[0] != run_count
        or query_embeddings.shape[0] != run_count
        or document_embeddings.shape[1] != len(document_ids)
        or query_embeddings.shape[1] != len(query_ids)
        or document_embeddings.shape[2] != query_embeddings.shape[2]
    ):
        raise _fail("REPLAY_OBSERVATION_INVALID", "source observation array shapes are invalid")
    if document_embeddings.dtype != np.float32 or query_embeddings.dtype != np.float32:
        raise _fail("REPLAY_OBSERVATION_INVALID", "source observation dtype must be float32")
    if document_embeddings.shape[2] != _require_int(
        observation_info.get("embedding_dimension"), "observation.embedding_dimension"
    ):
        raise _fail("REPLAY_OBSERVATION_INVALID", "embedding dimension mismatch")
    if len(document_ids) != _require_int(
        observation_info.get("document_count"), "observation.document_count"
    ) or len(query_ids) != _require_int(
        observation_info.get("query_count"), "observation.query_count"
    ):
        raise _fail("REPLAY_OBSERVATION_INVALID", "source observation count mismatch")
    qrels = _require_mapping(bundle.get("qrels"), "qrels")
    if set(qrels) != set(query_ids):
        raise _fail("REPLAY_QRELS_INVALID", "replay qrels do not match measurement-null queries")
    relevant_document_ids: dict[str, list[str]] = {}
    for query_id in query_ids:
        values = qrels[query_id]
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise _fail("REPLAY_QRELS_INVALID", f"invalid replay qrels for query: {query_id}")
        relevant_document_ids[query_id] = values
    return [
        SourceRun(
            run_id=metadata["run_id"],
            family=metadata["family"],
            batch_size=metadata["batch_size"],
            observation=TeacherObservation(
                document_ids=document_ids,
                document_embeddings=document_embeddings[index],
                query_ids=query_ids,
                query_embeddings=query_embeddings[index],
                query_roles=[MEASUREMENT_NULL_ROLE] * len(query_ids),
                relevant_document_ids=relevant_document_ids,
            ),
        )
        for index, metadata in enumerate(required_runs)
    ]


def replay_measurement_null(bundle_path: str | Path) -> dict[str, Any]:
    bundle_file = Path(bundle_path).resolve()
    root = bundle_file.parent
    bundle = _load_json(bundle_file, "REPLAY_BUNDLE_INVALID")
    if bundle.get("replay_format_version") != REPLAY_FORMAT_VERSION:
        raise _fail("REPLAY_FORMAT_INVALID", "replay bundle format is not authoritative")
    if bundle.get("measurement_null_format_version") != MEASUREMENT_NULL_FORMAT_VERSION:
        raise _fail("REPLAY_FORMAT_INVALID", "measurement-null format is not authoritative")
    evidence_manifest = _load_json(root / "evidence-manifest.json", "EVIDENCE_MANIFEST_INVALID")
    _verify_artifacts(root, evidence_manifest, "artifacts")
    runs = _load_replayed_runs(bundle, root)
    evaluation = _require_mapping(bundle.get("evaluation"), "evaluation")
    top_k = _positive_int(evaluation.get("top_k"), "evaluation.top_k")
    report, rankings = _report(runs, top_k)
    rankings_path = _safe_artifact_path(
        root, _require_string(evaluation.get("ranking_path"), "evaluation.ranking_path")
    )
    report_path = _safe_artifact_path(
        root,
        _require_string(
            evaluation.get("comparison_report_path"), "evaluation.comparison_report_path"
        ),
    )
    if rankings_path.read_bytes() != _jsonl_bytes(rankings):
        raise _fail("REPLAY_RANKING_MISMATCH", "replayed source rankings do not match evidence")
    if report_path.read_bytes() != canonical_json_bytes(report) + b"\n":
        raise _fail(
            "REPLAY_COMPARISON_MISMATCH", "replayed source envelope does not match evidence"
        )
    if bundle.get("replay_requires_model_execution") is not False:
        raise _fail("REPLAY_POLICY_INVALID", "replay must not execute the teacher model")
    if bundle.get("measurement_null_status") != "CAPTURED_NOT_DECIDED":
        raise _fail(
            "REPLAY_POLICY_INVALID",
            "measurement-null capture must not contain a transition decision",
        )
    return {
        "status": "PASS",
        "replay_verified": True,
        "model_execution_used": False,
        "dataset_id": _require_string(
            _require_mapping(bundle.get("dataset"), "dataset").get("dataset_id"),
            "dataset.dataset_id",
        ),
        "source_run_count": len(runs),
        "query_count": len(runs[0].observation.query_ids),
        "document_count": len(runs[0].observation.document_ids),
        "measurement_null_status": "CAPTURED_NOT_DECIDED",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture and replay source-only M1 measurement-null evidence without ONNX execution."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture", help="Capture real-teacher source measurement null.")
    capture.add_argument("--config", required=True)
    capture.add_argument("--dataset", required=True)
    capture.add_argument("--output", required=True)
    replay = subparsers.add_parser("replay", help="Replay captured source measurement null.")
    replay.add_argument("--bundle", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "capture":
            result = capture_measurement_null(args.config, args.dataset, args.output)
        else:
            result = replay_measurement_null(args.bundle)
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
