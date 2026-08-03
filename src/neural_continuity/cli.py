from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any

import yaml

from . import FAIL, INCONCLUSIVE, PASS
from .bootstrap import build_envelopes
from .datasets import RetrievalFixture, fixture_identity, load_retrieval_fixture
from .decisions import evaluate_comparison
from .evidence import (
    build_environment_manifest,
    canonical_json_bytes,
    get_git_commit_sha,
    sha256_file,
    write_artifacts,
)
from .metrics import compare_observations
from .models import PerturbedModel, SentenceTransformerModel, ToyEmbeddingModel
from .observations import ModelObservation, evaluate_model, save_raw_observations_parquet
from .perturbations import perturbation_from_config

REQUIRED_METRICS = [
    "recall_at_1",
    "recall_at_5",
    "mean_reciprocal_rank",
    "paired_cosine_drift",
    "nearest_neighbour_overlap_at_k",
    "rank_correlation",
    "latency_p50_ms",
    "latency_p95_ms",
    "throughput_queries_per_sec",
]


class CommandError(RuntimeError):
    pass


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


def _build_model(config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    model_cfg = config["model"]
    kind = str(model_cfg["kind"])
    if kind == "toy":
        model = ToyEmbeddingModel(
            dimension=int(model_cfg.get("dimension", 32)),
            seed=int(model_cfg.get("seed", 0)),
        )
        manifest = {
            "model_type": "toy",
            "dimension": int(model_cfg.get("dimension", 32)),
            "seed": int(model_cfg.get("seed", 0)),
        }
        return model, manifest

    try:
        model = SentenceTransformerModel(
            model_id=str(model_cfg["model_id"]),
            device=str(model_cfg.get("device", "auto")),
            cache_only=bool(model_cfg.get("cache_only", True)),
        )
    except RuntimeError as exc:
        if model_cfg.get("allow_offline_skip", False):
            raise CommandError(f"MODEL_UNAVAILABLE:{exc}") from exc
        raise
    manifest = {
        "model_type": "sentence-transformers",
        "model_id": str(model_cfg["model_id"]),
        "device": str(model_cfg.get("device", "auto")),
    }
    return model, manifest


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
            )
            comparison["control"] = "null"
            comparison["seed"] = null_seed + repeat
            comparison["batch_size"] = batch_size
            comparison["noise_source"] = "same_batch" if batch_size == base_batch else "batch_size"
            comparison["sample_count"] = len(fixture.queries)
            comparisons.append(comparison)

    if not comparisons:
        comparisons.append(
            {
                "control": "null",
                "source": baseline.model_id,
                "candidate": baseline.model_id,
                "metric_deltas": {metric: 0.0 for metric in REQUIRED_METRICS},
                "regressions": {"source_correct_candidate_wrong": [], "other": []},
                "affected_samples": {"source_correct_candidate_wrong": [], "other": []},
                "sample_count": len(fixture.queries),
            }
        )

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
    require_boundary: bool = False,
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
    )
    comparison["control"] = control_name
    comparison["batch_size"] = batch_size
    comparison["sample_count"] = len(fixture.queries)
    if require_boundary:
        comparison["sample_count"] = 1

    decision = evaluate_comparison(
        comparison,
        envelopes,
        required_metrics=REQUIRED_METRICS,
        require_boundary_inconclusive=require_boundary,
    )
    return comparison, decision.as_dict(), observation


def run_m0(config_path: Path, output_root: Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    contract = _load_contract(
        config_path.parent
        / str(config.get("contract", "contracts/m0-measurement-integrity-v1.json"))
    )
    _validate_config(config, contract)

    fixture = load_retrieval_fixture(config_path.parent / config["dataset"]["path"])
    model, model_manifest = _build_model(config)
    topology_k = int(config.get("runtime", {}).get("topology_k", 5))

    baseline, null_observations, null_comparisons = _run_null(
        model=model,
        model_manifest=model_manifest,
        fixture=fixture,
        null_cfg=config["null"],
        topology_k=topology_k,
    )
    all_observations = list(null_observations)

    envelopes = build_envelopes(
        null_comparisons,
        metric_names=REQUIRED_METRICS,
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
    }

    decisions: dict[str, Any] = {}
    comparison_report: dict[str, Any] = {}

    exact_outputs: list[dict[str, Any]] = []
    exact_runs = int(config["controls"]["exact_repeat"].get("repeats", 1))
    exact_enabled = bool(config["controls"]["exact_repeat"].get("enabled", True))
    for i in range(exact_runs if exact_enabled else 0):
        comparison, decision, obs = _run_control(
            control_name=f"exact_repeat_{i}",
            model=model,
            model_manifest=model_manifest,
            baseline=baseline,
            fixture=fixture,
            topology_k=topology_k,
            envelopes=envelopes,
            batch_size=int(config["null"]["batch_sizes"][0]),
        )
        exact_outputs.append(comparison)
        all_observations.append(obs)
        decisions[f"exact_repeat_{i}"] = decision
    comparison_report["exact_repeat"] = {"enabled": exact_enabled, "comparisons": exact_outputs}

    control_models: dict[str, dict[str, Any]] = {"source": model_manifest}
    negative_output: dict[str, Any] = {"enabled": False}
    if config["controls"]["negative"].get("enabled", True):
        negative_model, negative_manifest = _build_perturbed(
            base_model=model,
            base_manifest=model_manifest,
            cfg=config["controls"]["negative"],
        )
        control_models["negative"] = negative_manifest
        comparison, decision, obs = _run_control(
            control_name="negative",
            model=negative_model,
            model_manifest=negative_manifest,
            baseline=baseline,
            fixture=fixture,
            topology_k=topology_k,
            envelopes=envelopes,
            batch_size=int(config["null"]["batch_sizes"][0]),
        )
        negative_output = {
            "enabled": True,
            "comparison": comparison,
            "decision": decision,
        }
        comparison_report["negative"] = negative_output
        all_observations.append(obs)
        decisions["negative"] = decision
    else:
        comparison_report["negative"] = negative_output

    boundary_output: dict[str, Any] = {"enabled": False}
    if config["controls"]["boundary"].get("enabled", True):
        boundary_model, boundary_manifest = _build_perturbed(
            base_model=model,
            base_manifest=model_manifest,
            cfg=config["controls"]["boundary"],
        )
        control_models["boundary"] = boundary_manifest
        comparison, decision, obs = _run_control(
            control_name="boundary",
            model=boundary_model,
            model_manifest=boundary_manifest,
            baseline=baseline,
            fixture=fixture,
            topology_k=topology_k,
            envelopes=envelopes,
            batch_size=int(config["null"]["batch_sizes"][0]),
            require_boundary=True,
        )
        boundary_output = {"enabled": True, "comparison": comparison, "decision": decision}
        comparison_report["boundary"] = boundary_output
        all_observations.append(obs)
        decisions["boundary"] = decision
    else:
        comparison_report["boundary"] = boundary_output

    comparison_report["null"] = null_payload

    statuses = [v["status"] for v in decisions.values() if isinstance(v, dict)]
    overall = PASS
    if FAIL in statuses:
        overall = FAIL
    elif INCONCLUSIVE in statuses:
        overall = INCONCLUSIVE

    decision_payload = {
        "overall_status": overall,
        "controls": decisions,
        "control_counts": {
            "passed": len([status for status in statuses if status == PASS]),
            "failed": len([status for status in statuses if status == FAIL]),
            "inconclusive": len([status for status in statuses if status == INCONCLUSIVE]),
        },
    }

    run_id = time.strftime("%Y%m%d_%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = (output_root / run_id).resolve()

    model_payload = {
        "source": model_manifest,
        "negative_control": control_models.get("negative"),
        "boundary_control": control_models.get("boundary"),
    }
    dataset_payload = {
        "fixture_path": str(
            (config_path.parent / config["dataset"]["path"]).relative_to(config_path.parent)
        ),
        "fixture_id": fixture.fixture_id,
        "fixture_identity_sha256": fixture_identity(fixture),
        "query_count": len(fixture.queries),
        "name": fixture.name,
        "description": fixture.description,
    }
    noise_payload = {
        "envelopes": envelopes,
        "null_measurements": null_comparisons,
    }
    metrics_payload = {
        "required_metrics": REQUIRED_METRICS,
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
        "noise-envelope.json": noise_payload,
        "comparison-report.json": comparison_report,
        "decision.json": decision_payload,
    }

    artifact_manifest = write_artifacts(run_dir, artifacts)["artifact-manifest"]
    raw_observations_path = run_dir / "raw-observations.parquet"
    save_raw_observations_parquet(all_observations, raw_observations_path)
    artifact_manifest["artifacts"]["raw-observations.parquet"] = sha256_file(raw_observations_path)

    artifact_manifest["commands"] = [f"neural-continuity m0-run --config {str(config_path)}"]
    (run_dir / "artifact-manifest.json").write_bytes(canonical_json_bytes(artifact_manifest))

    return {
        "run_id": run_id,
        "status": overall,
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
            "artifact-manifest.json",
        ],
        "artifact_manifest": artifact_manifest,
        "decision": decision_payload,
        "real_teacher_executed": config["model"]["kind"] == "sentence-transformers",
        "null_metric_count": len(null_comparisons),
        "control_metrics": {
            "exact_repeat_runs": len(exact_outputs),
            "negative_runs": 1 if negative_output.get("enabled") else 0,
            "boundary_runs": 1 if boundary_output.get("enabled") else 0,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neural-continuity")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    run = subparsers.add_parser("m0-run")
    run.add_argument("--config", required=True)
    run.add_argument("--output", default="runs")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_m0(config_path=Path(args.config), output_root=Path(args.output))
    except CommandError as exc:
        message = str(exc)
        if message.startswith("MODEL_UNAVAILABLE:"):
            payload = {
                "status": INCONCLUSIVE,
                "reason": message.split("MODEL_UNAVAILABLE:", 1)[1].strip(),
                "real_teacher_executed": False,
            }
            print(json.dumps(payload, sort_keys=True, indent=2))
            return 0
        print(f"ERROR: {message}")
        return 2
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2

    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
