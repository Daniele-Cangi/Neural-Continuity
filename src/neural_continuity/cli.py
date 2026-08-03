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
    save_replay_bundle,
    sha256_file,
    write_artifacts,
)
from .metrics import (
    METRIC_POLICIES,
    METRIC_POLICIES_VERSION,
    REQUIRED_METRICS,
    compare_observations,
)
from .models import PerturbedModel, SentenceTransformerModel, ToyEmbeddingModel
from .observations import (
    ModelObservation,
    evaluate_model,
    save_raw_observations_parquet,
)
from .perturbations import perturbation_from_config

CONTROL_EXPECTED_STATUS = {
    "exact_repeat": PASS,
    "negative": FAIL,
    "boundary": INCONCLUSIVE,
}


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


def _build_model(config: dict[str, Any]) -> tuple[Any, dict[str, Any], bool]:
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
        return model, manifest, False

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
        "cache_only": bool(model_cfg.get("cache_only", True)),
        "model_manifest": model.manifest(),
    }
    return model, manifest, True


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


def run_m0(config_path: Path, output_root: Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    contract = _load_contract(
        config_path.parent
        / str(config.get("contract", "contracts/m0-measurement-integrity-v1.json"))
    )
    _validate_config(config, contract)

    fixture = load_retrieval_fixture(config_path.parent / config["dataset"]["path"])
    model, model_manifest, is_real_teacher = _build_model(config)
    topology_k = int(config.get("runtime", {}).get("topology_k", 5))
    metric_bootstrap_samples = int(
        config["null"].get(
            "candidate_bootstrap_samples",
            config["null"].get("bootstrap_samples", 500),
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

    baseline, null_observations, null_comparisons = _run_null(
        model=model,
        model_manifest=model_manifest,
        fixture=fixture,
        null_cfg=config["null"],
        topology_k=topology_k,
    )
    all_observations = list(null_observations)

    metric_policies_payload = [
        {
            "metric_id": policy.metric_id,
            "family": policy.family,
            "orientation": policy.orientation,
            "may_block_promotion": policy.may_block_promotion,
            "minimum_null_observations": policy.minimum_null_observations,
            "minimum_candidate_sample_size": policy.minimum_candidate_sample_size,
            "comparison_method": policy.comparison_method,
        }
        for policy in METRIC_POLICIES
    ]

    envelopes = build_envelopes(
        null_comparisons,
        metric_policies=metric_policies_payload,
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
            "version": METRIC_POLICIES_VERSION,
            "policies": metric_policies_payload,
        },
    }

    control_records: list[dict[str, Any]] = []

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
            metric_bootstrap_samples=metric_bootstrap_samples,
            metric_bootstrap_seed=metric_bootstrap_seed + i,
            candidate_confidence_level=candidate_confidence_level,
        )
        exact_outputs.append(
            {
                **comparison,
                "decision": decision,
                "expected_status": PASS,
            }
        )
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

    negative_output: dict[str, Any] = {"enabled": False}
    if config["controls"]["negative"].get("enabled", True):
        negative_model, negative_manifest = _build_perturbed(
            base_model=model,
            base_manifest=model_manifest,
            cfg=config["controls"]["negative"],
        )
        comparison, decision, obs = _run_control(
            control_name="negative",
            model=negative_model,
            model_manifest=negative_manifest,
            baseline=baseline,
            fixture=fixture,
            topology_k=topology_k,
            envelopes=envelopes,
            batch_size=int(config["null"]["batch_sizes"][0]),
            metric_bootstrap_samples=metric_bootstrap_samples,
            metric_bootstrap_seed=metric_bootstrap_seed + 100,
            candidate_confidence_level=candidate_confidence_level,
        )
        all_observations.append(obs)
        negative_output = {
            "enabled": True,
            "expected_status": FAIL,
            "comparison": comparison,
            "decision": decision,
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
    else:
        control_records.append(
            _control_record(
                control="negative",
                control_name="negative",
                decision={"status": INCONCLUSIVE},
                comparison=None,
                enabled=False,
                expected_status=FAIL,
            )
        )

    boundary_output: dict[str, Any] = {"enabled": False}
    if config["controls"]["boundary"].get("enabled", True):
        boundary_model, boundary_manifest = _build_perturbed(
            base_model=model,
            base_manifest=model_manifest,
            cfg=config["controls"]["boundary"],
        )
        comparison, decision, obs = _run_control(
            control_name="boundary",
            model=boundary_model,
            model_manifest=boundary_manifest,
            baseline=baseline,
            fixture=fixture,
            topology_k=topology_k,
            envelopes=envelopes,
            batch_size=int(config["null"]["batch_sizes"][0]),
            metric_bootstrap_samples=metric_bootstrap_samples,
            metric_bootstrap_seed=metric_bootstrap_seed + 200,
            candidate_confidence_level=candidate_confidence_level,
        )
        all_observations.append(obs)
        boundary_output = {
            "enabled": True,
            "expected_status": INCONCLUSIVE,
            "comparison": comparison,
            "decision": decision,
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
    else:
        control_records.append(
            _control_record(
                control="boundary",
                control_name="boundary",
                decision={"status": INCONCLUSIVE},
                comparison=None,
                enabled=False,
                expected_status=INCONCLUSIVE,
            )
        )

    control_statuses = [
        item["decision"].get("status") for item in control_records if item.get("enabled", False)
    ]
    measurement_integrity_status = PASS
    if any(
        item["decision"].get("status") == INCONCLUSIVE
        for item in control_records
        if item.get("enabled")
    ):
        measurement_integrity_status = INCONCLUSIVE
    if any(
        item["decision"].get("status") != item["expected_status"]
        for item in control_records
        if item.get("enabled")
    ):
        measurement_integrity_status = (
            FAIL
            if any(
                item["decision"].get("status") != item["expected_status"]
                and item["decision"].get("status") != INCONCLUSIVE
                for item in control_records
                if item.get("enabled")
            )
            else INCONCLUSIVE
        )

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

    control_decisions_only = [
        {
            "control": rec["control"],
            "status": rec["decision"].get("status"),
            "expected_status": rec["expected_status"],
            "enabled": rec.get("enabled", False),
        }
        for rec in control_records
    ]

    overall_status = PASS
    if FAIL in control_statuses:
        overall_status = FAIL
    elif INCONCLUSIVE in control_statuses:
        overall_status = INCONCLUSIVE

    decision_payload = {
        "overall_status": overall_status,
        "measurement_integrity_status": measurement_integrity_status,
        "control_decisions": control_decisions_only,
        "control_counts": {
            "passed": len([status for status in control_statuses if status == PASS]),
            "failed": len([status for status in control_statuses if status == FAIL]),
            "inconclusive": len([status for status in control_statuses if status == INCONCLUSIVE]),
        },
        "required_evidence": {
            "null_metric_count": len(null_comparisons),
            "enabled_controls": len([rec for rec in control_records if rec.get("enabled")]),
        },
        "metric_policies": {
            "version": METRIC_POLICIES_VERSION,
            "policies": metric_policies_payload,
        },
    }

    run_id = time.strftime("%Y%m%d_%H%M%S") + "-" + uuid.uuid4().hex[:8]
    run_dir = (output_root / run_id).resolve()

    model_payload = {
        "source": model_manifest,
        "negative_control": (
            negative_output.get("comparison", {}).get("control")
            if negative_output.get("enabled")
            else None
        ),
        "boundary_control": (
            boundary_output.get("comparison", {}).get("control")
            if boundary_output.get("enabled")
            else None
        ),
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

    metrics_payload = {
        "required_metrics": REQUIRED_METRICS,
        "metric_policies": metric_policies_payload,
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
    replay_path = save_replay_bundle(
        run_dir,
        all_observations,
        dataset_identity=dataset_payload,
        config={
            "topology_k": topology_k,
            "metric_bootstrap_seed": metric_bootstrap_seed,
            "metric_bootstrap_samples": metric_bootstrap_samples,
            "candidate_confidence_level": candidate_confidence_level,
        },
    )

    artifact_manifest["artifacts"]["raw-observations.parquet"] = sha256_file(raw_observations_path)
    artifact_manifest["artifacts"]["replay-bundle.json"] = sha256_file(Path(replay_path))
    artifact_manifest["commands"] = [f"neural-continuity m0-run --config {str(config_path)}"]
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
        "real_teacher_executed": is_real_teacher
        and model_manifest["model_type"] == "sentence-transformers",
        "null_metric_count": len(null_comparisons),
        "measurement_integrity_status": measurement_integrity_status,
        "control_metrics": {
            "exact_repeat_runs": len(exact_outputs),
            "negative_runs": 1 if negative_output.get("enabled") else 0,
            "boundary_runs": 1 if boundary_output.get("enabled") else 0,
        },
        "control_records": control_records,
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
                "measurement_integrity_status": INCONCLUSIVE,
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
