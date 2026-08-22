from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.m1_b.onnx_fp32_observation import (
    replay_fp32_source_observation,
)
from neural_continuity.m1_b.onnx_int8_observation import (
    replay_int8_target_observation,
)
from neural_continuity.m1_diagnostics.stage0_authority import (
    Stage0ControlError,
)
from neural_continuity.m1_onnx_transition import (
    PairedRun,
    _comparison,
    _metric_values,
)
from neural_continuity.m1_teacher_evidence import (
    TeacherObservation,
    _load_json,
    _rank_and_measure,
    _safe_artifact_path,
    _verify_artifacts,
)

ROLE_ORDER = (
    "measurement_null",
    "quantization_calibration",
    "contract_development",
    "validation",
    "frozen_critical",
    "final_holdout",
)


def _package(
    bundle_path: str | Path,
    expected_kind: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, np.ndarray]]:
    bundle = Path(bundle_path).resolve()
    root = bundle.parent
    manifest = _load_json(root / "evidence-manifest.json", "STAGE0_OBSERVATION_INVALID")
    if manifest.get("package_kind") != expected_kind:
        raise Stage0ControlError(
            "STAGE0_OBSERVATION_KIND_MISMATCH",
            f"unexpected observation package kind: {expected_kind}",
        )
    _verify_artifacts(root, manifest, "artifacts")
    replay = _load_json(bundle, "STAGE0_OBSERVATION_INVALID")
    if replay.get("replay_requires_model_execution") is not False:
        raise Stage0ControlError(
            "STAGE0_OBSERVATION_INVALID",
            "observation replay is not model-free",
        )
    metadata = _load_json(
        _safe_artifact_path(root, str(replay.get("metadata_path", ""))),
        "STAGE0_OBSERVATION_INVALID",
    )
    observation_path = _safe_artifact_path(
        root,
        str(replay.get("observation_path", "")),
    )
    try:
        with np.load(observation_path, allow_pickle=False) as archive:
            values = {key: archive[key] for key in archive.files}
    except Exception as exc:
        raise Stage0ControlError(
            "STAGE0_OBSERVATION_INVALID",
            f"cannot load observation arrays: {exc}",
        ) from exc
    return manifest, metadata, values


def _observation(
    values: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    index: int,
) -> TeacherObservation:
    query_ids = values["query_ids"].astype(str).tolist()
    qrels = metadata.get("qrels")
    if not isinstance(qrels, Mapping) or set(qrels) != set(query_ids):
        raise Stage0ControlError(
            "STAGE0_OBSERVATION_IDENTITY_MISMATCH",
            "qrels do not match canonical query identities",
        )
    return TeacherObservation(
        document_ids=values["document_ids"].astype(str).tolist(),
        document_embeddings=np.asarray(
            values["document_embeddings"][index],
            dtype=np.float32,
        ),
        query_ids=query_ids,
        query_embeddings=np.asarray(
            values["query_embeddings"][index],
            dtype=np.float32,
        ),
        query_roles=values["query_roles"].astype(str).tolist(),
        relevant_document_ids={query_id: list(qrels[query_id]) for query_id in query_ids},
    )


def _identity_fields(kind: str) -> tuple[str, ...]:
    return (
        ("dataset", "teacher_tokenizer_identity", "source_identity")
        if kind == "m1_onnx_fp32_source_observation"
        else ("dataset", "teacher_tokenizer_identity", "candidate_identity")
    )


def _require_compatible_repeat(
    baseline_manifest: Mapping[str, Any],
    fresh_manifest: Mapping[str, Any],
    baseline_metadata: Mapping[str, Any],
    fresh_metadata: Mapping[str, Any],
    kind: str,
) -> None:
    for field in _identity_fields(kind):
        if baseline_manifest.get(field) != fresh_manifest.get(field):
            raise Stage0ControlError(
                "STAGE0_OBSERVATION_IDENTITY_MISMATCH",
                f"repeat observation differs at manifest.{field}",
            )
    metadata_fields = (
        "dataset_id",
        "document_count",
        "query_count",
        "embedding_dimension",
        "embedding_dtype",
        "output_normalization",
        "query_roles",
        "qrels",
        "required_runs",
    )
    for field in metadata_fields:
        if baseline_metadata.get(field) != fresh_metadata.get(field):
            raise Stage0ControlError(
                "STAGE0_OBSERVATION_IDENTITY_MISMATCH",
                f"repeat observation differs at metadata.{field}",
            )


def _envelope(null_report: Mapping[str, Any]) -> Mapping[str, Any]:
    envelopes = null_report.get("empirical_envelopes")
    if not isinstance(envelopes, Mapping):
        raise Stage0ControlError(
            "STAGE0_MEASUREMENT_NULL_INVALID",
            "empirical envelopes are missing",
        )
    repeated = envelopes.get("repeated_inference")
    if (
        not isinstance(repeated, Mapping)
        or repeated.get("family") != "repeated_inference"
        or repeated.get("comparison_count") != 2
    ):
        raise Stage0ControlError(
            "STAGE0_MEASUREMENT_NULL_INVALID",
            "repeated-inference envelope is not authoritative",
        )
    return repeated


def _run_report(
    baseline: TeacherObservation,
    fresh: TeacherObservation,
    run_id: str,
    batch_size: int,
    envelope: Mapping[str, Any],
    top_k: int,
) -> dict[str, Any]:
    baseline_rankings, baseline_metrics = _rank_and_measure(baseline, top_k)
    fresh_rankings, fresh_metrics = _rank_and_measure(fresh, top_k)
    comparison = _comparison(
        PairedRun(
            run_id=run_id,
            batch_size=batch_size,
            source=baseline,
            target=fresh,
        ),
        baseline_rankings,
        fresh_rankings,
        baseline_metrics,
        fresh_metrics,
    )
    exact_documents = np.array_equal(
        baseline.document_embeddings,
        fresh.document_embeddings,
    )
    exact_queries = np.array_equal(
        baseline.query_embeddings,
        fresh.query_embeddings,
    )
    exact_identity = exact_documents and exact_queries
    maximum = envelope.get("empirical_maximum")
    minimum = envelope.get("empirical_minimum")
    if not isinstance(maximum, Mapping) or not isinstance(minimum, Mapping):
        raise Stage0ControlError(
            "STAGE0_MEASUREMENT_NULL_INVALID",
            "empirical envelope bounds are invalid",
        )
    maximum_metrics = maximum.get("absolute_metric_delta")
    if not isinstance(maximum_metrics, Mapping):
        raise Stage0ControlError(
            "STAGE0_MEASUREMENT_NULL_INVALID",
            "empirical metric bounds are invalid",
        )
    reasons: list[str] = []
    functional = comparison["functional"]
    if not exact_identity:
        if float(functional["document_max_abs_delta"]) > float(maximum["document_max_abs_delta"]):
            reasons.append("document_delta_exceeds_detection_limit")
        if float(functional["query_max_abs_delta"]) > float(maximum["query_max_abs_delta"]):
            reasons.append("query_delta_exceeds_detection_limit")
        if float(functional["document_min_cosine_similarity"]) < float(
            minimum["document_min_cosine_similarity"]
        ):
            reasons.append("document_cosine_below_detection_limit")
        if float(functional["query_min_cosine_similarity"]) < float(
            minimum["query_min_cosine_similarity"]
        ):
            reasons.append("query_cosine_below_detection_limit")
    role_reports: dict[str, Any] = {}
    for role in ROLE_ORDER:
        observed = comparison["roles"][role]
        baseline_values = _metric_values(baseline_metrics, role)
        fresh_values = _metric_values(fresh_metrics, role)
        absolute_delta = {
            name: abs(float(baseline_values[name]) - float(fresh_values[name]))
            for name in baseline_values
        }
        role_reasons: list[str] = []
        if int(observed["ranking_change_count"]) > int(maximum["ranking_change_count"]):
            role_reasons.append("ranking_change_count_exceeds_detection_limit")
        if float(observed["ranking_change_fraction"]) > float(maximum["ranking_change_fraction"]):
            role_reasons.append("ranking_change_fraction_exceeds_detection_limit")
        for metric, value in absolute_delta.items():
            if value > float(maximum_metrics[metric]):
                role_reasons.append(f"{metric}_delta_exceeds_detection_limit")
        role_reports[role] = {
            **observed,
            "absolute_metric_delta": absolute_delta,
            "outcome": "PASS" if not role_reasons else "BLOCKED",
            "reasons": role_reasons,
        }
        reasons.extend(f"{role}:{reason}" for reason in role_reasons)
    return {
        "run_id": run_id,
        "batch_size": batch_size,
        "exact_document_identity": exact_documents,
        "exact_query_identity": exact_queries,
        "exact_embedding_identity": exact_identity,
        "functional": functional,
        "roles": role_reports,
        "outcome": "PASS" if not reasons else "BLOCKED",
        "reasons": reasons,
    }


def compare_repeat_control(
    baseline_bundle: str | Path,
    fresh_bundle: str | Path,
    package_kind: str,
    null_report: Mapping[str, Any],
    top_k: int = 10,
) -> dict[str, Any]:
    replay = (
        replay_fp32_source_observation(fresh_bundle)
        if package_kind == "m1_onnx_fp32_source_observation"
        else replay_int8_target_observation(fresh_bundle)
    )
    if replay.get("replay_verified") is not True:
        raise Stage0ControlError(
            "STAGE0_FRESH_REPLAY_FAILED",
            "fresh observation replay failed",
        )
    baseline_manifest, baseline_metadata, baseline_values = _package(
        baseline_bundle,
        package_kind,
    )
    fresh_manifest, fresh_metadata, fresh_values = _package(
        fresh_bundle,
        package_kind,
    )
    _require_compatible_repeat(
        baseline_manifest,
        fresh_manifest,
        baseline_metadata,
        fresh_metadata,
        package_kind,
    )
    required = {"run_ids", "batch_sizes", "document_embeddings", "query_embeddings"}
    if (
        not required.issubset(baseline_values)
        or not required.issubset(fresh_values)
        or any(baseline_values[key].shape != fresh_values[key].shape for key in required)
        or not np.array_equal(
            baseline_values["run_ids"],
            fresh_values["run_ids"],
        )
        or not np.array_equal(
            baseline_values["batch_sizes"],
            fresh_values["batch_sizes"],
        )
    ):
        raise Stage0ControlError(
            "STAGE0_OBSERVATION_IDENTITY_MISMATCH",
            "repeat observation array schema or runs differ",
        )
    envelope = _envelope(null_report)
    reports = []
    run_ids = baseline_values["run_ids"].astype(str).tolist()
    batch_sizes = baseline_values["batch_sizes"].astype(int).tolist()
    for index, (run_id, batch_size) in enumerate(zip(run_ids, batch_sizes, strict=True)):
        reports.append(
            _run_report(
                _observation(baseline_values, baseline_metadata, index),
                _observation(fresh_values, fresh_metadata, index),
                run_id,
                batch_size,
                envelope,
                top_k,
            )
        )
    outcome = "PASS" if all(report["outcome"] == "PASS" for report in reports) else "BLOCKED"
    return {
        "package_kind": package_kind,
        "outcome": outcome,
        "run_count": len(reports),
        "runs": reports,
        "detection_limit_family": "repeated_inference",
        "exact_identity_is_sufficient": True,
        "scientific_fail_recorded": False,
    }


def build_stage0_control_report(
    baseline_fp32_bundle: str | Path,
    fresh_fp32_bundle: str | Path,
    baseline_int8_bundle: str | Path,
    fresh_int8_bundle: str | Path,
    null_report: Mapping[str, Any],
) -> dict[str, Any]:
    fp32 = compare_repeat_control(
        baseline_fp32_bundle,
        fresh_fp32_bundle,
        "m1_onnx_fp32_source_observation",
        null_report,
    )
    int8 = compare_repeat_control(
        baseline_int8_bundle,
        fresh_int8_bundle,
        "m1_onnx_int8_target_observation",
        null_report,
    )
    aggregate = "PASS" if fp32["outcome"] == int8["outcome"] == "PASS" else "BLOCKED"
    return {
        "kind": "m1-diagnostic-stage0-control-report",
        "status": aggregate,
        "controls": {
            "verified_onnx_fp32_reference": fp32,
            "frozen_int8_exact_replay": int8,
        },
        "stage_1_gate_status": aggregate,
        "stage_1_execution_started": False,
        "frozen_transition_b_v1_scientific_decision": "FAIL",
        "scientific_decision_recomputed": False,
        "causal_claim_made": False,
    }
