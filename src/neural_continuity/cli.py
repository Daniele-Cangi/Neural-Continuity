from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from . import FAIL, INCONCLUSIVE, PASS
from .bootstrap import build_envelopes
from .datasets import (
    RetrievalFixture,
    fixture_from_payload,
    fixture_identity,
    fixture_payload,
    load_retrieval_fixture,
)
from .decisions import evaluate_comparison
from .evidence import (
    build_environment_manifest,
    canonical_json_bytes,
    get_git_commit_sha,
    save_replay_bundle,
    sha256_file,
    write_artifacts,
)
from .metrics import (
    METRIC_POLICIES,
    METRIC_POLICIES_VERSION,
    REQUIRED_METRICS,
    MetricPolicy,
    compare_observations,
    load_metric_policies_from_mapping,
    load_metric_policies_from_path,
    metric_policies_payload,
)
from .models import PerturbedModel, SentenceTransformerModel, ToyEmbeddingModel
from .observations import (
    ModelObservation,
    evaluate_model,
    observation_from_manifest,
    save_raw_observations_parquet,
)
from .perturbations import perturbation_from_config

CONTROL_EXPECTED_STATUS = {
    "exact_repeat": PASS,
    "negative": FAIL,
    "boundary": INCONCLUSIVE,
}


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class CommandError(RuntimeError):
    pass


class ExecutionFailure(CommandError):
    def __init__(
        self,
        *,
        reason_code: str,
        reason: str,
        execution_status: str,
        exit_code: int,
    ) -> None:
        super().__init__(reason)
        self.reason_code = reason_code
        self.reason = reason
        self.execution_status = execution_status
        self.exit_code = exit_code


class ModelUnavailableError(ExecutionFailure):
    pass


class ReplayEvidenceError(ExecutionFailure):
    def __init__(self, reason_code: str, reason: str) -> None:
        super().__init__(
            reason_code=reason_code,
            reason=reason,
            execution_status="EXECUTION_ERROR",
            exit_code=2,
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise CommandError("configuration must be a mapping")
    return payload


def _load_contract(contract_path: Path) -> dict[str, Any]:
    if not contract_path.exists():
        raise CommandError(f"contract not found: {contract_path}")
    with contract_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise CommandError("contract must be a mapping")
    return payload


def _validate_config(config: dict[str, Any], contract: dict[str, Any]) -> None:
    required_top = set(contract.get("required_top_level_fields", []))
    if missing := sorted(required_top - set(config)):
        raise CommandError(f"missing required config fields: {missing}")

    for field in contract.get("required_dataset_fields", []):
        if field not in config["dataset"]:
            raise CommandError(f"dataset missing required field: {field}")

    for field in contract.get("required_null_fields", []):
        if field not in config["null"]:
            raise CommandError(f"null missing required field: {field}")

    for field in contract.get("required_controls", []):
        if field not in config["controls"]:
            raise CommandError(f"controls missing required control: {field}")

    if config["model"]["kind"] not in {"toy", "sentence-transformers"}:
        raise CommandError(f"unsupported model kind: {config['model']['kind']}")


def _build_model(config: dict[str, Any]) -> tuple[Any, dict[str, Any], bool]:
    model_cfg = config["model"]
    kind = str(model_cfg["kind"])
    if kind == "toy":
        toy_model: ToyEmbeddingModel = ToyEmbeddingModel(
            dimension=int(model_cfg.get("dimension", 32)),
            seed=int(model_cfg.get("seed", 0)),
        )
        toy_manifest: dict[str, Any] = {
            "model_type": "toy",
            "dimension": int(model_cfg.get("dimension", 32)),
            "seed": int(model_cfg.get("seed", 0)),
        }
        return toy_model, toy_manifest, False

    try:
        normalize_embeddings = bool(model_cfg.get("normalize_embeddings", False))
        output_dtype = str(model_cfg.get("output_dtype", "float32"))
        prompt_name = model_cfg.get("prompt_name")
        prompt = model_cfg.get("prompt")
        max_sequence_length = model_cfg.get("max_sequence_length")
        sentence_model = SentenceTransformerModel(
            model_id=str(model_cfg["model_id"]),
            device=str(model_cfg.get("device", "auto")),
            cache_only=bool(model_cfg.get("cache_only", True)),
            normalize_embeddings=normalize_embeddings,
            output_dtype=output_dtype,
            prompt_name=prompt_name if isinstance(prompt_name, str) else None,
            prompt=prompt if isinstance(prompt, str) else None,
            max_sequence_length=_safe_int(max_sequence_length),
        )
    except RuntimeError as exc:
        reason = str(exc)
        blocked_prefixes = (
            "teacher_dependency_unavailable:",
            "teacher_not_available_from_local_cache:",
        )
        if reason.startswith(blocked_prefixes):
            raise ModelUnavailableError(
                reason_code="MODEL_UNAVAILABLE",
                reason=reason,
                execution_status="BLOCKED",
                exit_code=3,
            ) from exc
        raise ModelUnavailableError(
            reason_code="MODEL_EXECUTION_ERROR",
            reason=reason,
            execution_status="EXECUTION_ERROR",
            exit_code=2,
        ) from exc

    sentence_manifest: dict[str, Any] = {
        "model_type": "sentence-transformers",
        "model_id": str(model_cfg["model_id"]),
        "device": str(model_cfg.get("device", "auto")),
        "cache_only": bool(model_cfg.get("cache_only", True)),
        "model_manifest": sentence_model.manifest(),
        "requested_configuration": {
            "normalize_embeddings": normalize_embeddings,
            "output_dtype": output_dtype,
            "prompt_name": prompt_name if isinstance(prompt_name, str) else None,
            "prompt": prompt if isinstance(prompt, str) else None,
            "max_sequence_length": _safe_int(max_sequence_length),
        },
    }
    return sentence_model, sentence_manifest, True


def _build_perturbed(
    base_model: Any, base_manifest: dict[str, Any], cfg: dict[str, Any]
) -> tuple[PerturbedModel, dict[str, Any]]:
    perturbation = perturbation_from_config(cfg)
    model = PerturbedModel(
        base_model=base_model,
        perturbation=perturbation,
        seed=int(cfg.get("seed", 0)),
        perturbation_manifest=perturbation.manifest(),
        model_id=f"{base_manifest.get('model_type')}::perturbed::{int(cfg.get('seed', 0))}",
    )
    manifest = {
        "model_type": "perturbed",
        "base": base_manifest,
        "perturbation": model.perturbation_manifest,
        "seed": int(cfg.get("seed", 0)),
    }
    return model, manifest


def _run_observation(
    model: Any,
    manifest: dict[str, Any],
    fixture: RetrievalFixture,
    batch_size: int,
    label: str,
) -> ModelObservation:
    return evaluate_model(
        model=model,
        fixture=fixture,
        batch_size=batch_size,
        run_label=label,
        model_manifest=manifest,
    )


def _run_null(
    model: Any,
    model_manifest: dict[str, Any],
    fixture: RetrievalFixture,
    null_cfg: dict[str, Any],
    topology_k: int,
) -> tuple[ModelObservation, list[ModelObservation], list[dict[str, Any]]]:
    batch_sizes = [int(v) for v in null_cfg.get("batch_sizes", [1])]
    repeats = int(null_cfg.get("repeats", 2))
    base_batch = batch_sizes[0]
    null_seed = int(null_cfg.get("random_seed", 0))

    baseline = _run_observation(
        model=model,
        manifest=model_manifest,
        fixture=fixture,
        batch_size=base_batch,
        label=f"null-baseline-b{base_batch}",
    )

    observations = [baseline]
    comparisons: list[dict[str, Any]] = []
    for repeat in range(repeats):
        for batch_size in batch_sizes:
            if repeat == 0 and batch_size == base_batch:
                continue

            obs = _run_observation(
                model=model,
                manifest=model_manifest,
                fixture=fixture,
                batch_size=batch_size,
                label=f"null-r{repeat}-b{batch_size}",
            )
            observations.append(obs)
            comparison = compare_observations(
                baseline,
                obs,
                fixture,
                topology_k=topology_k,
                metric_bootstrap_samples=int(null_cfg.get("bootstrap_samples", 500)),
                metric_bootstrap_seed=null_seed + repeat,
                confidence_level=float(null_cfg.get("confidence_level", 0.99)),
                metric_policies=METRIC_POLICIES,
            )
            comparison["control"] = "null"
            comparison["seed"] = null_seed + repeat
            comparison["batch_size"] = batch_size
            if batch_size == base_batch and repeat > 0:
                comparison["noise_source"] = "repeated_inference"
            elif batch_size != base_batch:
                comparison["noise_source"] = "batch_size_variation"
            else:
                comparison["noise_source"] = "runtime_or_hardware"

            comparison["runtime_hardware"] = obs.system_metrics["hardware"]
            comparison["sample_count"] = len(fixture.queries)
            comparisons.append(comparison)

    return baseline, observations, comparisons


def _run_control(
    control_name: str,
    model: Any,
    model_manifest: dict[str, Any],
    baseline: ModelObservation,
    fixture: RetrievalFixture,
    topology_k: int,
    envelopes: dict[str, Any],
    batch_size: int,
    metric_bootstrap_samples: int,
    metric_bootstrap_seed: int,
    candidate_confidence_level: float,
) -> tuple[dict[str, Any], dict[str, Any], ModelObservation]:
    observation = _run_observation(
        model=model,
        manifest=model_manifest,
        fixture=fixture,
        batch_size=batch_size,
        label=control_name,
    )
    comparison = compare_observations(
        source=baseline,
        candidate=observation,
        fixture=fixture,
        topology_k=topology_k,
        metric_bootstrap_samples=metric_bootstrap_samples,
        metric_bootstrap_seed=metric_bootstrap_seed,
        confidence_level=candidate_confidence_level,
        metric_policies=METRIC_POLICIES,
    )
    comparison["control"] = control_name
    comparison["batch_size"] = batch_size
    comparison["sample_count"] = len(fixture.queries)

    decision = evaluate_comparison(
        comparison,
        envelopes,
        required_metrics=REQUIRED_METRICS,
        metric_policies=METRIC_POLICIES,
    )
    return comparison, decision.as_dict(), observation


def _expected_status(control: str) -> str:
    return CONTROL_EXPECTED_STATUS.get(control, PASS)


def _control_record(
    control: str,
    control_name: str,
    decision: dict[str, Any],
    comparison: dict[str, Any] | None,
    enabled: bool,
    expected_status: str,
    comparison_run: int | None = None,
) -> dict[str, Any]:
    return {
        "control": control,
        "run": control_name,
        "expected_status": expected_status,
        "enabled": enabled,
        "decision": decision,
        "decision_matches_expected": decision.get("status") == expected_status,
        "sample_count": comparison.get("sample_count") if comparison else 0,
        "comparison_seed": comparison.get("seed") if comparison else None,
        "run_index": comparison_run,
    }


def _ordered_metric_ids(policies: list[MetricPolicy]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for policy in policies:
        if policy.metric_id not in seen:
            seen.add(policy.metric_id)
            ordered.append(policy.metric_id)
    return ordered


def _resolve_metric_policies(
    config_path: Path, config: dict[str, Any], contract_path: Path
) -> tuple[list[MetricPolicy], str]:
    configured = config.get("metric_policies")
    if isinstance(configured, list):
        return (
            load_metric_policies_from_mapping(
                {
                    "metric_policies": configured,
                    "version": str(config.get("metric_policies_version", "1.0.0")),
                }
            ),
            str(config.get("metric_policies_version", "1.0.0")),
        )

    if isinstance(configured, dict):
        return (
            load_metric_policies_from_mapping(configured),
            str(configured.get("version", config.get("metric_policies_version", "1.0.0"))),
        )

    if isinstance(config.get("metric_policies_path"), str):
        policy_path = (config_path.parent / str(config["metric_policies_path"])).resolve()
    else:
        fallback = contract_path.parent / "m0-metric-policies-v1.json"
        if fallback.exists():
            policy_path = fallback
        else:
            policy_path = config_path.parent / "contracts" / "m0-metric-policies-v1.json"
    return load_metric_policies_from_path(policy_path)


def _load_control_meta(value: Any, default: Any) -> Any:
    return value if isinstance(value, dict) else default


def _status_match(expected: str, actual: str) -> str:
    if actual == expected:
        return PASS
    if expected == INCONCLUSIVE:
        return PASS if actual == INCONCLUSIVE else INCONCLUSIVE
    if expected == PASS:
        return INCONCLUSIVE if actual == INCONCLUSIVE else FAIL
    if expected == FAIL:
        return INCONCLUSIVE if actual == INCONCLUSIVE else FAIL
    return INCONCLUSIVE


def _compute_control_health(control_records: list[dict[str, Any]]) -> tuple[str, str]:
    raw_control_status = PASS
    for record in control_records:
        if not record.get("enabled", False):
            continue
        actual = str(record.get("decision", {}).get("status", INCONCLUSIVE))
        expected = str(record.get("expected_status", PASS))
        if actual not in {PASS, FAIL, INCONCLUSIVE}:
            raise CommandError(f"invalid scientific decision status: {actual}")
        if record.get("decision", {}).get("reason") == "declared_control_observation_missing":
            status = FAIL
        else:
            status = _status_match(expected=expected, actual=actual)
        record["control_status"] = status
        if status == FAIL:
            raw_control_status = FAIL
        elif status == INCONCLUSIVE and raw_control_status == PASS:
            raw_control_status = INCONCLUSIVE

    measurement_integrity_status = raw_control_status
    return measurement_integrity_status, raw_control_status


def _replay_control_outcome_checks(
    control_records: list[dict[str, Any]], recorded_control_decisions: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], bool]:
    recorded_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for recorded in recorded_control_decisions:
        if not isinstance(recorded, dict):
            continue
        key = (str(recorded.get("control", "")), str(recorded.get("run", "")))
        recorded_by_key[key] = recorded

    checks: list[dict[str, Any]] = []
    match = True
    control_keys = set()

    for record in control_records:
        key = (str(record.get("control", "")), str(record.get("run", "")))
        control_keys.add(key)
        recorded = recorded_by_key.get(key, {})
        recorded_status = recorded.get("status") if recorded else None
        recomputed_status = record.get("decision", {}).get("status")
        status_match = recorded_status == recomputed_status
        if not status_match:
            match = False
        checks.append(
            {
                "control": record.get("control"),
                "run": record.get("run"),
                "recorded_status": recorded_status,
                "recomputed_status": recomputed_status,
                "matches": status_match,
            }
        )

    for key, recorded in recorded_by_key.items():
        if key not in control_keys:
            checks.append(
                {
                    "control": key[0],
                    "run": key[1],
                    "recorded_status": recorded.get("status"),
                    "recomputed_status": None,
                    "matches": False,
                }
            )
            match = False

    return checks, match


def _run_null_v2(
    model: Any,
    model_manifest: dict[str, Any],
    fixture: RetrievalFixture,
    null_cfg: dict[str, Any],
    topology_k: int,
    metric_policies: list[MetricPolicy],
) -> tuple[ModelObservation, list[ModelObservation], list[dict[str, Any]], list[dict[str, Any]]]:
    batch_sizes = [int(v) for v in null_cfg.get("batch_sizes", [1])]
    repeats = int(null_cfg.get("repeats", 2))
    base_batch = batch_sizes[0]
    null_seed = int(null_cfg.get("random_seed", 0))

    baseline = _run_observation(
        model=model,
        manifest=model_manifest,
        fixture=fixture,
        batch_size=base_batch,
        label=f"null-baseline-b{base_batch}",
    )

    observations: list[ModelObservation] = [baseline]
    comparisons: list[dict[str, Any]] = []
    null_plan: list[dict[str, Any]] = []
    for repeat in range(repeats):
        for batch_size in batch_sizes:
            if repeat == 0 and batch_size == base_batch:
                continue

            run_label = f"null-r{repeat}-b{batch_size}"
            obs = _run_observation(
                model=model,
                manifest=model_manifest,
                fixture=fixture,
                batch_size=batch_size,
                label=run_label,
            )
            observations.append(obs)

            comparison = compare_observations(
                baseline,
                obs,
                fixture,
                topology_k=topology_k,
                metric_bootstrap_samples=int(null_cfg.get("bootstrap_samples", 500)),
                metric_bootstrap_seed=null_seed + repeat,
                confidence_level=float(null_cfg.get("confidence_level", 0.99)),
                metric_policies=metric_policies,
            )
            comparison["control"] = "null"
            comparison["run_label"] = run_label
            comparison["seed"] = null_seed + repeat
            comparison["batch_size"] = batch_size
            if batch_size == base_batch and repeat > 0:
                comparison["noise_source"] = "repeated_inference"
            elif batch_size != base_batch:
                comparison["noise_source"] = "batch_size_variation"
            else:
                comparison["noise_source"] = "runtime_or_hardware"
            comparison["sample_count"] = len(fixture.queries)
            comparison["runtime_hardware"] = obs.system_metrics["hardware"]

            comparisons.append(comparison)
            null_plan.append(
                {
                    "run_label": run_label,
                    "seed": null_seed + repeat,
                    "batch_size": batch_size,
                    "noise_source": comparison["noise_source"],
                }
            )

    return baseline, observations, comparisons, null_plan


def _run_control_v2(
    control_name: str,
    model: Any,
    model_manifest: dict[str, Any],
    baseline: ModelObservation,
    fixture: RetrievalFixture,
    topology_k: int,
    envelopes: dict[str, Any],
    batch_size: int,
    metric_bootstrap_samples: int,
    metric_bootstrap_seed: int,
    candidate_confidence_level: float,
    metric_policies: list[MetricPolicy],
    required_metrics: list[str],
) -> tuple[dict[str, Any], dict[str, Any], ModelObservation]:
    observation = _run_observation(
        model=model,
        manifest=model_manifest,
        fixture=fixture,
        batch_size=batch_size,
        label=control_name,
    )
    comparison = compare_observations(
        source=baseline,
        candidate=observation,
        fixture=fixture,
        topology_k=topology_k,
        metric_bootstrap_samples=metric_bootstrap_samples,
        metric_bootstrap_seed=metric_bootstrap_seed,
        confidence_level=candidate_confidence_level,
        metric_policies=metric_policies,
    )
    comparison["control"] = control_name
    comparison["run_label"] = control_name
    comparison["batch_size"] = batch_size
    comparison["sample_count"] = len(fixture.queries)

    decision = evaluate_comparison(
        comparison,
        envelopes,
        required_metrics=required_metrics,
        metric_policies=metric_policies,
    )
    return comparison, decision.as_dict(), observation


def _run_boundary_v2(
    boundary_cfg: dict[str, Any],
    model: Any,
    model_manifest: dict[str, Any],
    baseline: ModelObservation,
    fixture: RetrievalFixture,
    topology_k: int,
    envelopes: dict[str, Any],
    metric_bootstrap_samples: int,
    metric_bootstrap_seed: int,
    candidate_confidence_level: float,
    metric_policies: list[MetricPolicy],
    required_metrics: list[str],
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any], ModelObservation, dict[str, Any]]:
    attempts = max(2, int(boundary_cfg.get("attempts", 8)))
    base_strength = float(boundary_cfg.get("strength", 0.1))
    base_seed = int(boundary_cfg.get("seed", 0))
    perturbation_type = str(boundary_cfg.get("type", "gaussian_noise"))
    min_strength = float(boundary_cfg.get("min_strength", 0.0))
    max_strength = float(
        boundary_cfg.get("max_strength", base_strength * 4.0 if base_strength > 0 else 1.0)
    )
    if perturbation_type in {"dimension_mask", "mask", "output_corruption", "corrupt"}:
        min_strength = max(0.0, min(1.0, min_strength))
        max_strength = max(min_strength, max_strength)
        max_strength = min(1.0, max_strength)
        if max_strength == min_strength:
            max_strength = 1.0
    base_batch = int(batch_size)

    attempt_results: list[dict[str, Any]] = []
    selected_attempt: dict[str, Any] = {}
    selected_payload: tuple[dict[str, Any], dict[str, Any], ModelObservation] | None = None
    crossing_detected = False
    crossing_point: dict[str, Any] = {}
    previous_status = None

    if attempts <= 1:
        attempts = 2
    step = (max_strength - min_strength) / (attempts - 1)
    if step == 0:
        step = 1.0

    for attempt in range(attempts):
        attempt_strength = min_strength + step * attempt
        attempt_cfg = dict(boundary_cfg)
        attempt_seed = base_seed + attempt
        attempt_cfg["seed"] = attempt_seed
        attempt_cfg["strength"] = attempt_strength
        attempt_cfg["batch_size"] = base_batch
        boundary_model, boundary_manifest = _build_perturbed(
            base_model=model,
            base_manifest=model_manifest,
            cfg=attempt_cfg,
        )
        comparison, decision, observation = _run_control_v2(
            control_name="boundary",
            model=boundary_model,
            model_manifest=boundary_manifest,
            baseline=baseline,
            fixture=fixture,
            topology_k=topology_k,
            envelopes=envelopes,
            batch_size=base_batch,
            metric_bootstrap_samples=metric_bootstrap_samples,
            metric_bootstrap_seed=metric_bootstrap_seed + 200 + attempt,
            candidate_confidence_level=candidate_confidence_level,
            metric_policies=metric_policies,
            required_metrics=required_metrics,
        )
        status = decision.get("status")
        attempt_payload = {
            "run_label": "boundary",
            "attempt": attempt,
            "seed": attempt_seed,
            "strength": attempt_strength,
            "status": status,
            "metric_bootstrap_seed": metric_bootstrap_seed + 200 + attempt,
        }
        attempt_results.append(attempt_payload)

        if selected_payload is None:
            selected_payload = (comparison, decision, observation)
            selected_attempt = attempt_payload

        if previous_status is not None and status != previous_status:
            crossing_detected = True
            if not crossing_point:
                crossing_point = {
                    "from_attempt": attempt - 1,
                    "to_attempt": attempt,
                    "from_status": previous_status,
                    "to_status": status,
                    "from_strength": attempt_strength - step,
                    "to_strength": attempt_strength,
                }
            if status != PASS and previous_status == PASS:
                selected_payload = (comparison, decision, observation)
                selected_attempt = attempt_payload
        previous_status = status

    if selected_payload is None:
        raise CommandError("boundary search produced no attempt payload")
    selected_comparison, selected_decision, selected_observation = selected_payload
    if crossing_detected and selected_decision.get("status") != INCONCLUSIVE:
        selected_decision = {
            **selected_decision,
            "status": INCONCLUSIVE,
            "status_reason": "boundary_synthetic_crossing_detected",
        }

    return (
        selected_comparison,
        selected_decision,
        selected_observation,
        {
            "attempt_count": attempts,
            "attempts": attempt_results,
            "selected_attempt": selected_attempt,
            "crossing_detected": crossing_detected,
            "crossing_point": crossing_point,
        },
    )


def _control_record_payload(
    control_records: list[dict[str, Any]],
    measurement_integrity_status: str,
    raw_control_status: str,
) -> list[dict[str, Any]]:
    return [
        {
            "control": item["control"],
            "status": item["decision"].get("status"),
            "control_status": item.get("control_status"),
            "expected_status": item["expected_status"],
            "enabled": item.get("enabled", False),
            "decision_matches_expected": item["decision_matches_expected"],
            "run": item.get("run"),
        }
        for item in control_records
    ]


def _rebuild_control_plan(
    control_records: list[dict[str, Any]], metric_policies_payload: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "exact_repeat": [item for item in control_records if item["control"] == "exact_repeat"],
        "negative": [item for item in control_records if item["control"] == "negative"],
        "boundary": [item for item in control_records if item["control"] == "boundary"],
        "metric_policies": metric_policies_payload,
    }


def run_m0(config_path: Path, output_root: Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    contract_path = config_path.parent / str(
        _load_yaml(config_path).get("contract", "contracts/m0-measurement-integrity-v1.json")
    )
    config = _load_yaml(config_path)
    contract = _load_contract(contract_path)
    _validate_config(config, contract)

    metric_policies, metric_policies_version = _resolve_metric_policies(
        config_path=config_path, config=config, contract_path=contract_path
    )
    metric_policies_payload_data = metric_policies_payload(metric_policies)
    required_metrics = _ordered_metric_ids(metric_policies)

    fixture = load_retrieval_fixture(config_path.parent / config["dataset"]["path"])
    model, model_manifest, is_real_teacher = _build_model(config)
    topology_k = int(config.get("runtime", {}).get("topology_k", 5))
    metric_bootstrap_samples = int(
        config["null"].get(
            "candidate_bootstrap_samples", config["null"].get("bootstrap_samples", 500)
        )
    )
    metric_bootstrap_seed = int(
        config["null"].get("candidate_random_seed", config["null"].get("random_seed", 17))
    )
    candidate_confidence_level = float(
        config["null"].get(
            "candidate_confidence_level", config["null"].get("confidence_level", 0.99)
        )
    )

    baseline, null_observations, null_comparisons, null_plan = _run_null_v2(
        model=model,
        model_manifest=model_manifest,
        fixture=fixture,
        null_cfg=config["null"],
        topology_k=topology_k,
        metric_policies=metric_policies,
    )
    all_observations: list[ModelObservation] = list(null_observations)

    envelopes = build_envelopes(
        null_comparisons,
        metric_policies=metric_policies_payload_data,
        bootstrap_samples=int(config["null"].get("bootstrap_samples", 500)),
        confidence_level=float(config["null"].get("confidence_level", 0.99)),
        seed=int(config["null"].get("random_seed", 17)),
    )

    null_payload = {
        "comparisons": null_comparisons,
        "envelope_config": {
            "bootstrap_samples": int(config["null"].get("bootstrap_samples", 500)),
            "confidence_level": float(config["null"].get("confidence_level", 0.99)),
            "random_seed": int(config["null"].get("random_seed", 17)),
        },
        "metric_policies": {
            "version": metric_policies_version,
            "metric_policies": metric_policies_payload_data,
        },
    }

    control_records: list[dict[str, Any]] = []
    exact_outputs: list[dict[str, Any]] = []
    exact_plan: list[dict[str, Any]] = []
    exact_runs = int(config["controls"]["exact_repeat"].get("repeats", 1))
    exact_enabled = bool(config["controls"]["exact_repeat"].get("enabled", True))
    base_batch = int(config["null"]["batch_sizes"][0])
    for i in range(exact_runs if exact_enabled else 0):
        comparison, decision, obs = _run_control_v2(
            control_name=f"exact_repeat_{i}",
            model=model,
            model_manifest=model_manifest,
            baseline=baseline,
            fixture=fixture,
            topology_k=topology_k,
            envelopes=envelopes,
            batch_size=base_batch,
            metric_bootstrap_samples=metric_bootstrap_samples,
            metric_bootstrap_seed=metric_bootstrap_seed + i,
            candidate_confidence_level=candidate_confidence_level,
            metric_policies=metric_policies,
            required_metrics=required_metrics,
        )
        exact_outputs.append({**comparison, "decision": decision, "expected_status": PASS})
        all_observations.append(obs)
        control_records.append(
            _control_record(
                control="exact_repeat",
                control_name=f"exact_repeat_{i}",
                decision=decision,
                comparison=comparison,
                enabled=True,
                expected_status=PASS,
                comparison_run=i,
            )
        )
        exact_plan.append(
            {
                "run_label": f"exact_repeat_{i}",
                "seed": metric_bootstrap_seed + i,
                "batch_size": base_batch,
                "expected_status": PASS,
                "run_index": i,
                "enabled": True,
            }
        )

    negative_output: dict[str, Any] = {"enabled": False}
    negative_plan: dict[str, Any] = {"enabled": False}
    if config["controls"]["negative"].get("enabled", True):
        negative_model, negative_manifest = _build_perturbed(
            base_model=model,
            base_manifest=model_manifest,
            cfg=config["controls"]["negative"],
        )
        comparison, decision, obs = _run_control_v2(
            control_name="negative",
            model=negative_model,
            model_manifest=negative_manifest,
            baseline=baseline,
            fixture=fixture,
            topology_k=topology_k,
            envelopes=envelopes,
            batch_size=base_batch,
            metric_bootstrap_samples=metric_bootstrap_samples,
            metric_bootstrap_seed=metric_bootstrap_seed + 100,
            candidate_confidence_level=candidate_confidence_level,
            metric_policies=metric_policies,
            required_metrics=required_metrics,
        )
        all_observations.append(obs)
        negative_output = {
            "enabled": True,
            "expected_status": FAIL,
            "comparison": comparison,
            "decision": decision,
        }
        negative_plan = {
            "run_label": "negative",
            "enabled": True,
            "expected_status": FAIL,
            "seed": metric_bootstrap_seed + 100,
            "batch_size": base_batch,
            "run_index": 0,
        }
        control_records.append(
            _control_record(
                control="negative",
                control_name="negative",
                decision=decision,
                comparison=comparison,
                enabled=True,
                expected_status=FAIL,
            )
        )

    boundary_output: dict[str, Any] = {"enabled": False}
    boundary_plan: dict[str, Any] = {"enabled": False}
    if config["controls"]["boundary"].get("enabled", True):
        comparison, decision, obs, boundary_meta = _run_boundary_v2(
            boundary_cfg=config["controls"]["boundary"],
            model=model,
            model_manifest=model_manifest,
            baseline=baseline,
            fixture=fixture,
            topology_k=topology_k,
            envelopes=envelopes,
            metric_bootstrap_samples=metric_bootstrap_samples,
            metric_bootstrap_seed=metric_bootstrap_seed,
            candidate_confidence_level=candidate_confidence_level,
            metric_policies=metric_policies,
            required_metrics=required_metrics,
            batch_size=base_batch,
        )
        all_observations.append(obs)
        boundary_output = {
            "enabled": True,
            "expected_status": INCONCLUSIVE,
            "comparison": comparison,
            "decision": decision,
            "attempts": boundary_meta.get("attempts", []),
            "selected_attempt": boundary_meta.get("selected_attempt"),
            "attempt_count": boundary_meta.get("attempt_count"),
            "crossing_detected": boundary_meta.get("crossing_detected", False),
            "crossing_point": boundary_meta.get("crossing_point", {}),
        }
        boundary_plan = {
            "run_label": "boundary",
            "expected_status": INCONCLUSIVE,
            "enabled": True,
            "batch_size": base_batch,
            "attempts": boundary_meta.get("attempts", []),
            "selected_attempt": boundary_meta.get("selected_attempt"),
            "attempt_count": boundary_meta.get("attempt_count"),
            "crossing_detected": boundary_meta.get("crossing_detected", False),
            "crossing_point": boundary_meta.get("crossing_point", {}),
        }
        control_records.append(
            _control_record(
                control="boundary",
                control_name="boundary",
                decision=decision,
                comparison=comparison,
                enabled=True,
                expected_status=INCONCLUSIVE,
            )
        )

    measurement_integrity_status, raw_control_status = _compute_control_health(control_records)
    control_counts = {
        "passed": len([item for item in control_records if item["decision"].get("status") == PASS]),
        "failed": len([item for item in control_records if item["decision"].get("status") == FAIL]),
        "inconclusive": len(
            [item for item in control_records if item["decision"].get("status") == INCONCLUSIVE]
        ),
    }

    control_status_payload = _control_record_payload(
        control_records,
        measurement_integrity_status,
        raw_control_status,
    )

    decision_payload = {
        "raw_control_status": raw_control_status,
        "measurement_integrity_status": measurement_integrity_status,
        "control_decisions": control_status_payload,
        "control_counts": control_counts,
        "required_evidence": {
            "null_metric_count": len(null_comparisons),
            "enabled_controls": len([rec for rec in control_records if rec.get("enabled")]),
        },
        "metric_policies": {
            "version": metric_policies_version,
            "metric_policies": metric_policies_payload_data,
        },
        "boundary_evidence": boundary_output,
    }

    run_id = time.strftime("%Y%m%d_%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = (output_root / run_id).resolve()

    comparison_report = {
        "exact_repeat": {
            "enabled": exact_enabled,
            "comparisons": [
                {
                    "run": f"exact_repeat_{idx}",
                    "comparison": comp,
                    "decision": comp.get("decision", {}),
                    "expected_status": PASS,
                    "decision_matches_expected": comp.get("decision", {}).get("status") == PASS,
                }
                for idx, comp in enumerate(exact_outputs)
            ],
        },
        "negative": negative_output,
        "boundary": boundary_output,
        "null": null_payload,
    }

    dataset_payload = {
        "fixture_path": str(
            (config_path.parent / config["dataset"]["path"]).relative_to(config_path.parent)
        ),
        "fixture_id": fixture.fixture_id,
        "fixture_identity_sha256": fixture_identity(fixture),
        "fixture_payload": fixture_payload(fixture),
        "query_count": len(fixture.queries),
        "name": fixture.name,
        "description": fixture.description,
    }

    model_payload = {
        "source": model_manifest,
        "negative_control": (
            negative_output.get("comparison", {}).get("run_label")
            if negative_output.get("enabled")
            else None
        ),
        "boundary_control": (
            boundary_output.get("comparison", {}).get("run_label")
            if boundary_output.get("enabled")
            else None
        ),
    }
    metrics_payload = {
        "required_metrics": required_metrics,
        "metric_policies": metric_policies_payload_data,
        "null": null_payload,
        "exact_repeat": comparison_report["exact_repeat"],
        "negative": comparison_report["negative"],
        "boundary": comparison_report["boundary"],
    }

    environment_payload = build_environment_manifest()
    environment_payload["git_commit"] = get_git_commit_sha()

    artifacts = {
        "model-manifest.json": model_payload,
        "dataset-manifest.json": dataset_payload,
        "environment-manifest.json": {
            **environment_payload,
            "commands": [f"neural-continuity m0-run --config {str(config_path)}"],
        },
        "experiment-config.json": {
            **config,
            "_resolved_path": str(config_path),
        },
        "metrics.json": metrics_payload,
        "noise-envelope.json": null_payload,
        "comparison-report.json": comparison_report,
        "decision.json": decision_payload,
    }

    artifact_manifest = write_artifacts(run_dir, artifacts)["artifact-manifest"]
    raw_observations_path = run_dir / "raw-observations.parquet"
    save_raw_observations_parquet(all_observations, raw_observations_path)
    control_plan = {
        "baseline_observation": {"run_label": baseline.run_label},
        "null_observations": null_plan,
        "null_settings": {
            "batch_sizes": [int(v) for v in config["null"].get("batch_sizes", [1])],
            "repeats": int(config["null"].get("repeats", 2)),
            "bootstrap_samples": int(config["null"].get("bootstrap_samples", 500)),
            "confidence_level": float(config["null"].get("confidence_level", 0.99)),
            "random_seed": int(config["null"].get("random_seed", 17)),
        },
        "exact_repeat": exact_plan,
        "negative": negative_plan,
        "boundary": boundary_plan,
    }
    replay_path = save_replay_bundle(
        run_dir,
        all_observations,
        dataset_identity=dataset_payload,
        config={
            "topology_k": topology_k,
            "metric_bootstrap_seed": metric_bootstrap_seed,
            "metric_bootstrap_samples": metric_bootstrap_samples,
            "candidate_confidence_level": candidate_confidence_level,
            "required_metrics": required_metrics,
            "metric_policies": {
                "version": metric_policies_version,
                "metric_policies": metric_policies_payload_data,
            },
            "control_plan": control_plan,
            "recorded_decision": decision_payload,
        },
    )

    artifact_manifest["artifacts"]["raw-observations.parquet"] = sha256_file(raw_observations_path)
    artifact_manifest["artifacts"]["replay-bundle.json"] = sha256_file(Path(replay_path))
    artifact_manifest["commands"] = [
        f"neural-continuity m0-run --config {str(config_path)}",
        f"neural-continuity m0-replay --bundle {str(replay_path)}",
    ]
    (run_dir / "artifact-manifest.json").write_bytes(canonical_json_bytes(artifact_manifest))

    return {
        "run_id": run_id,
        "status": measurement_integrity_status,
        "run_dir": str(run_dir),
        "config": str(config_path),
        "artifact_hierarchy": [
            "model-manifest.json",
            "dataset-manifest.json",
            "environment-manifest.json",
            "experiment-config.json",
            "raw-observations.parquet",
            "metrics.json",
            "noise-envelope.json",
            "comparison-report.json",
            "decision.json",
            "replay-bundle.json",
            "artifact-manifest.json",
        ],
        "artifact_manifest": artifact_manifest,
        "decision": decision_payload,
        "real_teacher_executed": (
            is_real_teacher and model_manifest["model_type"] == "sentence-transformers"
        ),
        "null_metric_count": len(null_comparisons),
        "measurement_integrity_status": measurement_integrity_status,
        "raw_control_status": raw_control_status,
        "control_metrics": {
            "exact_repeat_runs": len(exact_outputs),
            "negative_runs": 1 if negative_output.get("enabled") else 0,
            "boundary_runs": 1 if boundary_output.get("enabled") else 0,
        },
        "control_records": control_records,
        "control_plan": control_plan,
    }


def _load_replay_metadata(bundle: dict[str, Any]) -> dict[str, Any]:
    experiment = bundle.get("experiment")
    if isinstance(experiment, dict):
        return experiment
    return {}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _load_json_object(path: Path, *, reason_code: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReplayEvidenceError(reason_code, f"invalid JSON artifact: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ReplayEvidenceError(reason_code, f"JSON artifact is not an object: {path.name}")
    return payload


def _verify_replay_artifacts(path: Path, bundle_bytes: bytes) -> dict[str, Any]:
    manifest_path = path.parent / "artifact-manifest.json"
    if not manifest_path.exists():
        return {
            "status": "UNVERIFIED_STANDALONE",
            "artifact_manifest_sha256": None,
            "verified_artifact_count": 0,
        }

    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReplayEvidenceError(
            "ARTIFACT_MANIFEST_INVALID", "artifact-manifest.json is not valid JSON"
        ) from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("artifacts"), dict):
        raise ReplayEvidenceError(
            "ARTIFACT_MANIFEST_INVALID", "artifact manifest is missing the artifacts mapping"
        )

    artifacts = manifest["artifacts"]
    expected_bundle_hash = artifacts.get(path.name)
    if expected_bundle_hash is None:
        raise ReplayEvidenceError(
            "REPLAY_BUNDLE_UNDECLARED",
            f"artifact manifest does not declare {path.name}",
        )

    base_dir = path.parent.resolve()
    verified_count = 0
    for artifact_name, expected_hash in sorted(artifacts.items()):
        if not isinstance(artifact_name, str) or not _is_sha256(expected_hash):
            raise ReplayEvidenceError(
                "ARTIFACT_MANIFEST_INVALID",
                f"invalid artifact declaration: {artifact_name}",
            )
        artifact_path = (base_dir / artifact_name).resolve()
        if artifact_path.parent != base_dir:
            raise ReplayEvidenceError(
                "ARTIFACT_PATH_INVALID", f"artifact path escapes run directory: {artifact_name}"
            )
        if not artifact_path.is_file():
            raise ReplayEvidenceError(
                "DECLARED_ARTIFACT_MISSING", f"declared artifact is missing: {artifact_name}"
            )
        actual_hash = (
            hashlib.sha256(bundle_bytes).hexdigest()
            if artifact_path == path.resolve()
            else sha256_file(artifact_path)
        )
        if actual_hash.lower() != expected_hash.lower():
            raise ReplayEvidenceError(
                "ARTIFACT_HASH_MISMATCH", f"artifact hash mismatch: {artifact_name}"
            )
        verified_count += 1

    return {
        "status": "VERIFIED",
        "artifact_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "verified_artifact_count": verified_count,
    }


def _load_replay_bundle(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        bundle_bytes = Path(path).read_bytes()
    except OSError as exc:
        raise ReplayEvidenceError("REPLAY_BUNDLE_MISSING", "replay bundle is unavailable") from exc
    artifact_integrity = _verify_replay_artifacts(Path(path), bundle_bytes)
    try:
        payload = json.loads(bundle_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReplayEvidenceError(
            "REPLAY_BUNDLE_INVALID", "replay bundle is not valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise ReplayEvidenceError("REPLAY_BUNDLE_INVALID", "replay payload must be a JSON object")
    return payload, artifact_integrity


def _verify_embedded_artifact_coherence(
    replay_bundle_path: Path, payload: dict[str, Any], artifact_integrity: dict[str, Any]
) -> None:
    if artifact_integrity["status"] != "VERIFIED":
        return
    experiment = _load_replay_metadata(payload)
    comparisons = (
        ("dataset-manifest.json", payload.get("dataset")),
        ("decision.json", experiment.get("recorded_decision")),
    )
    for artifact_name, embedded in comparisons:
        external = _load_json_object(
            replay_bundle_path.parent / artifact_name,
            reason_code="ARTIFACT_COHERENCE_ERROR",
        )
        if not isinstance(embedded, dict) or canonical_json_bytes(external) != canonical_json_bytes(
            embedded
        ):
            raise ReplayEvidenceError(
                "ARTIFACT_COHERENCE_ERROR",
                f"embedded evidence differs from {artifact_name}",
            )


def run_m0_replay(replay_bundle_path: Path, output_root: Path | None = None) -> dict[str, Any]:
    payload, artifact_integrity = _load_replay_bundle(Path(replay_bundle_path))
    _verify_embedded_artifact_coherence(Path(replay_bundle_path), payload, artifact_integrity)
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise ReplayEvidenceError("DATASET_EVIDENCE_MISSING", "replay payload is missing dataset")

    fixture_data = dataset.get("fixture_payload")
    if isinstance(fixture_data, dict) and fixture_data.get("queries"):
        fixture = fixture_from_payload(fixture_data)
    else:
        path_value = dataset.get("fixture_path")
        if not isinstance(path_value, str):
            raise ReplayEvidenceError(
                "FIXTURE_EVIDENCE_MISSING", "replay payload is missing fixture evidence"
            )
        fixture = load_retrieval_fixture((replay_bundle_path.parent / path_value).resolve())

    expected_fixture_identity = dataset.get("fixture_identity_sha256")
    if not isinstance(expected_fixture_identity, str) or not _is_sha256(expected_fixture_identity):
        raise ReplayEvidenceError(
            "FIXTURE_IDENTITY_MISSING", "fixture identity must be a SHA-256 value"
        )
    if fixture_identity(fixture) != expected_fixture_identity.lower():
        raise ReplayEvidenceError(
            "FIXTURE_IDENTITY_MISMATCH", "fixture payload does not match its declared identity"
        )
    query_count = dataset.get("query_count")
    if not isinstance(query_count, int) or isinstance(query_count, bool):
        raise ReplayEvidenceError("FIXTURE_QUERY_COUNT_INVALID", "fixture query_count is invalid")
    if query_count != len(fixture.queries):
        raise ReplayEvidenceError(
            "FIXTURE_QUERY_COUNT_MISMATCH", "fixture query_count does not match the payload"
        )

    raw_observations = payload.get("observations", [])
    if not isinstance(raw_observations, list) or not raw_observations:
        raise ReplayEvidenceError("OBSERVATIONS_MISSING", "replay payload is missing observations")
    if not all(isinstance(raw, dict) for raw in raw_observations):
        raise ReplayEvidenceError("OBSERVATION_INVALID", "every observation must be an object")
    try:
        observations = [observation_from_manifest(raw) for raw in raw_observations]
    except Exception as exc:
        raise ReplayEvidenceError(
            "OBSERVATION_INVALID", "observation payload is malformed"
        ) from exc
    labels = [obs.run_label for obs in observations]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ReplayEvidenceError(
            "OBSERVATION_LABEL_INVALID", "observation run labels must be non-empty and unique"
        )
    by_label = {obs.run_label: obs for obs in observations}

    experiment = _load_replay_metadata(payload)
    topology_k = int(experiment.get("topology_k", 5))
    metric_bootstrap_samples = int(experiment.get("metric_bootstrap_samples", 500))
    metric_bootstrap_seed = int(experiment.get("metric_bootstrap_seed", 17))
    candidate_confidence_level = float(experiment.get("candidate_confidence_level", 0.99))
    required_metrics = experiment.get("required_metrics") or [
        policy.metric_id for policy in METRIC_POLICIES
    ]
    if not required_metrics:
        required_metrics = [policy.metric_id for policy in METRIC_POLICIES]

    metric_policies_payload_raw = experiment.get(
        "metric_policies",
        {
            "version": METRIC_POLICIES_VERSION,
            "metric_policies": metric_policies_payload(METRIC_POLICIES),
        },
    )
    metric_policies = load_metric_policies_from_mapping(metric_policies_payload_raw)
    metric_policies_payload_for_build = metric_policies_payload(metric_policies)

    control_plan = experiment.get("control_plan", {})
    if not isinstance(control_plan, dict):
        raise ReplayEvidenceError("CONTROL_PLAN_INVALID", "replay control plan is missing")
    baseline_plan = control_plan.get("baseline_observation")
    if not isinstance(baseline_plan, dict):
        raise ReplayEvidenceError(
            "DECLARED_BASELINE_MISSING", "baseline observation declaration is missing"
        )
    baseline_label = baseline_plan.get("run_label")
    if not isinstance(baseline_label, str) or not baseline_label or baseline_label not in by_label:
        raise ReplayEvidenceError(
            "DECLARED_BASELINE_MISSING", "declared baseline observation is missing"
        )
    baseline = by_label[baseline_label]

    null_plan = control_plan.get("null_observations", [])
    if not isinstance(null_plan, list) or not null_plan:
        raise ReplayEvidenceError("NULL_PLAN_INVALID", "declared null observation plan is empty")
    null_labels: list[str] = []
    null_comparisons: list[dict[str, Any]] = []
    for idx, item in enumerate(null_plan):
        if not isinstance(item, dict):
            raise ReplayEvidenceError("NULL_PLAN_INVALID", "null plan entry must be an object")
        label = item.get("run_label")
        if not isinstance(label, str) or not label:
            raise ReplayEvidenceError("NULL_PLAN_INVALID", "null plan run_label is missing")
        if label in null_labels:
            raise ReplayEvidenceError("NULL_PLAN_INVALID", "null plan run_label is duplicated")
        null_labels.append(label)
        if label not in by_label:
            raise ReplayEvidenceError(
                "DECLARED_NULL_OBSERVATION_MISSING",
                f"declared null observation is missing: {label}",
            )
        obs = by_label[label]
        comparison = compare_observations(
            source=baseline,
            candidate=obs,
            fixture=fixture,
            topology_k=topology_k,
            metric_bootstrap_samples=metric_bootstrap_samples,
            metric_bootstrap_seed=metric_bootstrap_seed + idx,
            confidence_level=float(item.get("confidence_level", candidate_confidence_level)),
            metric_policies=metric_policies,
        )
        comparison["control"] = "null"
        comparison["run_label"] = label
        comparison["seed"] = item.get("seed", metric_bootstrap_seed + idx)
        comparison["batch_size"] = item.get("batch_size", obs.batch_size)
        comparison["sample_count"] = len(fixture.queries)
        comparison["noise_source"] = item.get("noise_source", "unknown")
        null_comparisons.append(comparison)

    null_settings = control_plan.get("null_settings", {})
    envelopes = build_envelopes(
        null_comparisons,
        metric_policies=metric_policies_payload_for_build,
        bootstrap_samples=int(null_settings.get("bootstrap_samples", metric_bootstrap_samples)),
        confidence_level=float(null_settings.get("confidence_level", candidate_confidence_level)),
        seed=int(null_settings.get("random_seed", 17)),
    )

    control_records: list[dict[str, Any]] = []
    for idx, row in enumerate(
        control_plan.get("exact_repeat", [])
        if isinstance(control_plan.get("exact_repeat"), list)
        else []
    ):
        if not isinstance(row, dict) or not row.get("enabled", True):
            continue
        label = str(row.get("run_label", f"exact_repeat_{idx}"))
        exact_obs: ModelObservation | None = by_label.get(label)
        if exact_obs is None:
            raise ReplayEvidenceError(
                "DECLARED_CONTROL_OBSERVATION_MISSING",
                f"declared exact-repeat observation is missing: {label}",
            )

        comparison = compare_observations(
            source=baseline,
            candidate=exact_obs,
            fixture=fixture,
            topology_k=topology_k,
            metric_bootstrap_samples=metric_bootstrap_samples,
            metric_bootstrap_seed=int(row.get("seed", metric_bootstrap_seed + idx)),
            confidence_level=float(row.get("confidence_level", candidate_confidence_level)),
            metric_policies=metric_policies,
        )
        comparison["control"] = "exact_repeat"
        comparison["run_label"] = label
        comparison["sample_count"] = len(fixture.queries)
        comparison["seed"] = row.get("seed", metric_bootstrap_seed + idx)
        decision = evaluate_comparison(
            comparison,
            envelopes,
            required_metrics=required_metrics,
            metric_policies=metric_policies,
        )
        control_records.append(
            _control_record(
                control="exact_repeat",
                control_name=label,
                decision=decision.as_dict(),
                comparison=comparison,
                enabled=True,
                expected_status=row.get("expected_status", PASS),
                comparison_run=idx,
            )
        )

    negative_plan = _load_control_meta(
        control_plan.get("negative"), {"enabled": False, "expected_status": FAIL}
    )
    if negative_plan.get("enabled"):
        negative_label = str(negative_plan.get("run_label", "negative"))
        negative_obs = by_label.get(negative_label)
        if negative_obs is None:
            raise ReplayEvidenceError(
                "DECLARED_CONTROL_OBSERVATION_MISSING",
                f"declared negative-control observation is missing: {negative_label}",
            )
        else:
            comparison = compare_observations(
                source=baseline,
                candidate=negative_obs,
                fixture=fixture,
                topology_k=topology_k,
                metric_bootstrap_samples=metric_bootstrap_samples,
                metric_bootstrap_seed=int(negative_plan.get("seed", metric_bootstrap_seed + 100)),
                confidence_level=float(
                    negative_plan.get("confidence_level", candidate_confidence_level)
                ),
                metric_policies=metric_policies,
            )
            comparison["control"] = "negative"
            comparison["run_label"] = negative_label
            comparison["sample_count"] = len(fixture.queries)
            comparison["seed"] = negative_plan.get("seed", metric_bootstrap_seed + 100)
            decision = evaluate_comparison(
                comparison,
                envelopes,
                required_metrics=required_metrics,
                metric_policies=metric_policies,
            )
            control_records.append(
                _control_record(
                    control="negative",
                    control_name=negative_label,
                    decision=decision.as_dict(),
                    comparison=comparison,
                    enabled=True,
                    expected_status=negative_plan.get("expected_status", FAIL),
                    comparison_run=0,
                )
            )
    else:
        control_records.append(
            _control_record(
                control="negative",
                control_name="negative",
                decision={"status": INCONCLUSIVE},
                comparison=None,
                enabled=False,
                expected_status=negative_plan.get("expected_status", FAIL),
            )
        )

    boundary_plan = _load_control_meta(
        control_plan.get("boundary"),
        {"enabled": False, "expected_status": INCONCLUSIVE},
    )
    if boundary_plan.get("enabled"):
        boundary_label = str(boundary_plan.get("run_label", "boundary"))
        boundary_obs = by_label.get(boundary_label)
        if boundary_obs is None:
            raise ReplayEvidenceError(
                "DECLARED_CONTROL_OBSERVATION_MISSING",
                f"declared boundary-control observation is missing: {boundary_label}",
            )
        else:
            selected_attempt = boundary_plan.get("selected_attempt", {})
            comparison = compare_observations(
                source=baseline,
                candidate=boundary_obs,
                fixture=fixture,
                topology_k=topology_k,
                metric_bootstrap_samples=metric_bootstrap_samples,
                metric_bootstrap_seed=int(
                    selected_attempt.get("metric_bootstrap_seed", metric_bootstrap_seed + 200)
                ),
                confidence_level=float(
                    selected_attempt.get("confidence_level", candidate_confidence_level)
                ),
                metric_policies=metric_policies,
            )
            comparison["control"] = "boundary"
            comparison["run_label"] = boundary_label
            comparison["sample_count"] = len(fixture.queries)
            comparison["seed"] = selected_attempt.get("seed", metric_bootstrap_seed + 200)
            decision = evaluate_comparison(
                comparison,
                envelopes,
                required_metrics=required_metrics,
                metric_policies=metric_policies,
            )
            if boundary_plan.get("crossing_detected") and decision.status != INCONCLUSIVE:
                boundary_decision = decision.as_dict()
                if boundary_decision.get("status") != INCONCLUSIVE:
                    boundary_decision = {
                        **boundary_decision,
                        "status": INCONCLUSIVE,
                        "status_reason": "boundary_synthetic_crossing_detected",
                    }
            else:
                boundary_decision = decision.as_dict()
            control_records.append(
                _control_record(
                    control="boundary",
                    control_name=boundary_label,
                    decision=boundary_decision,
                    comparison=comparison,
                    enabled=True,
                    expected_status=boundary_plan.get("expected_status", INCONCLUSIVE),
                    comparison_run=0,
                )
            )
    else:
        control_records.append(
            _control_record(
                control="boundary",
                control_name="boundary",
                decision={"status": INCONCLUSIVE},
                comparison=None,
                enabled=False,
                expected_status=boundary_plan.get("expected_status", INCONCLUSIVE),
            )
        )

    measurement_integrity_status, raw_control_status = _compute_control_health(control_records)
    decision_payload = {
        "artifact_integrity": artifact_integrity,
        "raw_control_status": raw_control_status,
        "measurement_integrity_status": measurement_integrity_status,
        "control_decisions": _control_record_payload(
            control_records, measurement_integrity_status, raw_control_status
        ),
        "metric_policies": experiment.get(
            "metric_policies",
            {
                "version": METRIC_POLICIES_VERSION,
                "metric_policies": metric_policies_payload(METRIC_POLICIES),
            },
        ),
        "boundary_evidence": boundary_plan,
    }

    recorded_decision = experiment.get("recorded_decision")
    recorded_control_decisions = []
    recorded_status = None
    if isinstance(recorded_decision, dict):
        recorded_status = recorded_decision.get("measurement_integrity_status")
        status_match = recorded_status == measurement_integrity_status
        recorded_control_decisions = recorded_decision.get("control_decisions", [])
        if not isinstance(recorded_control_decisions, list):
            recorded_control_decisions = []
    else:
        status_match = False
    control_outcome_checks, control_outcome_match = _replay_control_outcome_checks(
        control_records, recorded_control_decisions
    )
    decision_payload["control_outcome_checks"] = control_outcome_checks
    decision_payload["control_outcome_match"] = control_outcome_match

    return {
        "run": str(replay_bundle_path),
        "status": measurement_integrity_status,
        "measurement_integrity_status": measurement_integrity_status,
        "raw_control_status": raw_control_status,
        "recorded_status": recorded_status,
        "status_match": status_match,
        "control_outcome_checks": control_outcome_checks,
        "control_outcome_match": control_outcome_match,
        "artifact_integrity": artifact_integrity,
        "recorded_decision": recorded_decision,
        "control_records": control_records,
        "decision": decision_payload,
        "control_counts": {
            "passed": len(
                [item for item in control_records if item["decision"].get("status") == PASS]
            ),
            "failed": len(
                [item for item in control_records if item["decision"].get("status") == FAIL]
            ),
            "inconclusive": len(
                [item for item in control_records if item["decision"].get("status") == INCONCLUSIVE]
            ),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neural-continuity")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    run = subparsers.add_parser("m0-run")
    run.add_argument("--config", required=True)
    run.add_argument("--output", default="runs")

    replay = subparsers.add_parser("m0-replay")
    replay.add_argument("--bundle", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "m0-run":
            summary = run_m0(config_path=Path(args.config), output_root=Path(args.output))
        else:
            summary = run_m0_replay(replay_bundle_path=Path(args.bundle))
    except ExecutionFailure as exc:
        payload = {
            "status": exc.execution_status,
            "execution_status": exc.execution_status,
            "reason_code": exc.reason_code,
            "reason": exc.reason,
            "real_teacher_executed": False,
            "measurement_integrity_status": None,
            "scientific_decision": None,
        }
        print(json.dumps(payload, sort_keys=True, indent=2))
        return exc.exit_code
    except CommandError as exc:
        print(f"ERROR: {exc}")
        return 2
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2

    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
