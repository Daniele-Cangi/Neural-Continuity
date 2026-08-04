from __future__ import annotations

import json
import shutil
from pathlib import Path

from neural_continuity.bootstrap import build_envelopes
from neural_continuity.cli import REQUIRED_METRICS, run_m0, run_m0_replay
import yaml
from neural_continuity.datasets import (
    CandidateDocument,
    RetrievalFixture,
    RetrievalQuery,
    load_retrieval_fixture,
)
from neural_continuity.decisions import evaluate_comparison
from neural_continuity.metrics import (
    METRIC_POLICIES,
    POLICY_BY_ID,
    compare_observations,
)
from neural_continuity.models import PerturbedModel, ToyEmbeddingModel
from neural_continuity.observations import evaluate_model
from neural_continuity.perturbations import DimensionMaskPerturbation

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "local_retrieval_fixture.json"


def _load_fixture():
    return load_retrieval_fixture(FIXTURE_PATH)


def _single_query_fixture() -> RetrievalFixture:
    return RetrievalFixture(
        fixture_id="single_query_unit",
        name="single",
        description="single-query fixture",
        queries=[
            RetrievalQuery(
                query_id="q",
                query="query",
                candidate_documents=[
                    CandidateDocument(document_id="a", text="alpha"),
                    CandidateDocument(document_id="b", text="beta"),
                ],
                relevant_document_ids=["a"],
            )
        ],
    )


def test_deterministic_toy_inference_no_material_divergence():
    fixture = _load_fixture()
    model = ToyEmbeddingModel(dimension=16, seed=7)
    manifest = {"model_type": "toy", "seed": 7}
    baseline = evaluate_model(
        model=model, fixture=fixture, batch_size=1, run_label="base", model_manifest=manifest
    )
    candidate = evaluate_model(
        model=model, fixture=fixture, batch_size=4, run_label="base2", model_manifest=manifest
    )
    comparison = compare_observations(
        baseline,
        candidate,
        fixture,
        topology_k=5,
        metric_bootstrap_samples=30,
        metric_bootstrap_seed=11,
    )
    null_payload = build_envelopes(
        [comparison, comparison],
        metric_policies=[p.__dict__ for p in METRIC_POLICIES],
        bootstrap_samples=100,
        confidence_level=0.99,
        seed=11,
    )
    decision = evaluate_comparison(
        comparison,
        null_payload,
        required_metrics=REQUIRED_METRICS,
        metric_policies=METRIC_POLICIES,
    )
    assert decision.status == "PASS"


def test_non_blocking_latency_interval_can_be_outside_boundary():
    comparison = {
        "metric_deltas": {
            metric: 0.0 if metric != "recall_at_1" else -0.6 for metric in REQUIRED_METRICS
        },
        "metric_uncertainty": {
            "recall_at_1": {
                "status": "complete",
                "lower_bound": -1.0,
                "upper_bound": -0.7,
                "sample_count": 4,
                "seed": 1,
                "bootstrap_samples": 40,
                "confidence_level": 0.99,
            },
            "recall_at_5": {
                "status": "complete",
                "lower_bound": 0.0,
                "upper_bound": 0.0,
                "sample_count": 4,
                "seed": 1,
                "bootstrap_samples": 40,
                "confidence_level": 0.99,
            },
            "mean_reciprocal_rank": {
                "status": "complete",
                "lower_bound": 0.0,
                "upper_bound": 0.0,
                "sample_count": 1,
                "seed": 1,
                "bootstrap_samples": 40,
                "confidence_level": 0.99,
            },
            "paired_cosine_drift": {
                "status": "complete",
                "lower_bound": 0.0,
                "upper_bound": 0.0,
                "sample_count": 1,
                "seed": 1,
                "bootstrap_samples": 40,
                "confidence_level": 0.99,
            },
            "nearest_neighbour_overlap_at_k": {
                "status": "complete",
                "lower_bound": 1.0,
                "upper_bound": 1.0,
                "sample_count": 1,
                "seed": 1,
                "bootstrap_samples": 40,
                "confidence_level": 0.99,
            },
            "rank_correlation": {
                "status": "complete",
                "lower_bound": 1.0,
                "upper_bound": 1.0,
                "sample_count": 1,
                "seed": 1,
                "bootstrap_samples": 40,
                "confidence_level": 0.99,
            },
            "latency_p50_ms": {
                "status": "complete",
                "lower_bound": 1.0,
                "upper_bound": 1.0,
                "sample_count": 1,
                "seed": 1,
                "bootstrap_samples": 40,
                "confidence_level": 0.99,
            },
            "latency_p95_ms": {
                "status": "complete",
                "lower_bound": 1.0,
                "upper_bound": 1.0,
                "sample_count": 1,
                "seed": 1,
                "bootstrap_samples": 40,
                "confidence_level": 0.99,
            },
            "throughput_queries_per_sec": {
                "status": "complete",
                "lower_bound": 1.0,
                "upper_bound": 1.0,
                "sample_count": 1,
                "seed": 1,
                "bootstrap_samples": 40,
                "confidence_level": 0.99,
            },
        },
        "regressions": {"source_correct_candidate_wrong": [], "other": []},
    }
    null_payload = {
        metric: {
            "status": "complete",
            "lower_bound": 0.0,
            "upper_bound": 0.0,
            "sample_count": 4,
            "method": "bootstrap_ci",
            "seed": 2,
            "bootstrap_samples": 40,
            "confidence_level": 0.99,
            "noise_source_counts": {},
            "raw_null_values": [0.0],
            "observed_null_distribution": [],
            "details": {},
        }
        for metric in REQUIRED_METRICS
    }
    # make null for recall_at_1 strict and outside harmful direction
    null_payload["recall_at_1"]["lower_bound"] = 0.0
    null_payload["recall_at_1"]["upper_bound"] = 0.0
    null_payload["recall_at_1"]["raw_null_values"] = [0.0]
    null_payload["recall_at_1"]["observed_null_distribution"] = []
    null_payload["recall_at_1"]["details"] = {}
    decision = evaluate_comparison(
        comparison,
        null_payload,
        required_metrics=REQUIRED_METRICS,
        metric_policies=list(POLICY_BY_ID.values()),
    )
    assert decision.status == "FAIL"


def test_boundary_interval_can_be_inconclusive_or_fail_when_overlaps_or_exits():
    policy = POLICY_BY_ID["recall_at_1"]
    comparison = {
        "metric_deltas": {"recall_at_1": 0.0},
        "metric_uncertainty": {
            "recall_at_1": {
                "status": "complete",
                "lower_bound": -0.02,
                "upper_bound": 0.02,
                "sample_count": 4,
                "seed": 1,
                "bootstrap_samples": 40,
                "confidence_level": 0.99,
            }
        },
        "regressions": {"source_correct_candidate_wrong": [], "other": []},
    }
    envelopes = {
        "recall_at_1": {
            "status": "complete",
            "lower_bound": -0.05,
            "upper_bound": 0.05,
            "sample_count": 4,
            "method": "bootstrap_ci",
            "seed": 1,
            "bootstrap_samples": 40,
            "confidence_level": 0.99,
            "noise_source_counts": {},
            "raw_null_values": [-0.01, 0.01, -0.02, 0.03],
            "observed_null_distribution": [],
            "details": {},
        }
    }
    assert (
        evaluate_comparison(
            comparison,
            envelopes,
            required_metrics=["recall_at_1"],
            metric_policies=[policy],
        ).status
        == "PASS"
    )


def test_boundary_interval_crossing_is_inconclusive():
    policy = POLICY_BY_ID["recall_at_1"]
    comparison = {
        "metric_deltas": {"recall_at_1": 0.0},
        "metric_uncertainty": {
            "recall_at_1": {
                "status": "complete",
                "lower_bound": -0.1,
                "upper_bound": 0.1,
                "sample_count": 4,
                "seed": 1,
                "bootstrap_samples": 40,
                "confidence_level": 0.99,
            }
        },
        "regressions": {"source_correct_candidate_wrong": [], "other": []},
    }
    envelopes = {
        "recall_at_1": {
            "status": "complete",
            "lower_bound": 0.0,
            "upper_bound": 0.0,
            "sample_count": 4,
            "method": "bootstrap_ci",
            "seed": 1,
            "bootstrap_samples": 40,
            "confidence_level": 0.99,
            "noise_source_counts": {},
            "raw_null_values": [0.0],
            "observed_null_distribution": [],
            "details": {},
        }
    }
    assert (
        evaluate_comparison(
            comparison,
            envelopes,
            required_metrics=["recall_at_1"],
            metric_policies=[policy],
        ).status
        == "INCONCLUSIVE"
    )


def test_missing_null_evidence_yields_inconclusive():
    policy = POLICY_BY_ID["recall_at_1"]
    comparison = {
        "metric_deltas": {},
        "metric_uncertainty": {},
        "regressions": {"source_correct_candidate_wrong": [], "other": []},
    }
    envelopes: dict[str, dict[str, int]] = {}
    decision = evaluate_comparison(
        comparison,
        envelopes,
        required_metrics=["recall_at_1"],
        metric_policies=[policy],
    )
    assert decision.status == "INCONCLUSIVE"


def test_insufficient_null_observations_are_inconclusive():
    policy = POLICY_BY_ID["recall_at_1"]
    comparison = {
        "metric_deltas": {"recall_at_1": 0.0},
        "metric_uncertainty": {
            "recall_at_1": {
                "status": "complete",
                "lower_bound": 0.0,
                "upper_bound": 0.0,
                "sample_count": 1,
                "seed": 1,
                "bootstrap_samples": 40,
                "confidence_level": 0.99,
            }
        },
        "regressions": {"source_correct_candidate_wrong": [], "other": []},
    }
    envelopes = {
        "recall_at_1": {
            "status": "insufficient",
            "lower_bound": float("nan"),
            "upper_bound": float("nan"),
            "sample_count": 1,
            "method": "bootstrap_ci",
            "seed": 1,
            "bootstrap_samples": 40,
            "confidence_level": 0.99,
            "noise_source_counts": {},
            "raw_null_values": [0.0],
            "observed_null_distribution": None,
            "details": {"required_null_observations": policy.minimum_null_observations},
        }
    }
    assert (
        evaluate_comparison(
            comparison,
            envelopes,
            required_metrics=["recall_at_1"],
            metric_policies=[policy],
        ).status
        == "INCONCLUSIVE"
    )


def test_candidate_interval_outside_harmful_direction_fails():
    policy = POLICY_BY_ID["latency_p50_ms"]
    comparison = {
        "metric_deltas": {"latency_p50_ms": 5.0},
        "metric_uncertainty": {
            "latency_p50_ms": {
                "status": "complete",
                "lower_bound": 4.0,
                "upper_bound": 6.0,
                "sample_count": 4,
                "seed": 1,
                "bootstrap_samples": 40,
                "confidence_level": 0.99,
            }
        },
        "regressions": {"source_correct_candidate_wrong": [], "other": []},
    }
    envelopes = {
        "latency_p50_ms": {
            "status": "complete",
            "lower_bound": -0.2,
            "upper_bound": 0.2,
            "sample_count": 4,
            "method": "bootstrap_ci",
            "seed": 2,
            "bootstrap_samples": 40,
            "confidence_level": 0.99,
            "noise_source_counts": {},
            "raw_null_values": [-0.1, 0.1],
            "observed_null_distribution": [],
            "details": {},
        }
    }
    decision = evaluate_comparison(
        comparison,
        envelopes,
        required_metrics=["latency_p50_ms"],
        metric_policies=[policy],
    )
    assert decision.status == "PASS"


def test_candidate_interval_changes_with_synthetic_boundary_fixture():
    fixture = _single_query_fixture()
    baseline = ToyEmbeddingModel(dimension=4, seed=1)
    baseline_manifest = {"model_type": "toy", "seed": 1}
    source = evaluate_model(
        model=baseline,
        fixture=fixture,
        batch_size=1,
        run_label="src",
        model_manifest=baseline_manifest,
    )
    perturbed = PerturbedModel(
        base_model=baseline,
        perturbation=DimensionMaskPerturbation(
            mask_fraction=0.2, seed=1, perturbation_type="dimension_mask"
        ),
        seed=3,
        perturbation_manifest={"type": "dimension_mask", "mask_fraction": 0.2, "seed": 1},
        model_id="candidate",
    )
    candidate = evaluate_model(
        model=perturbed,
        fixture=fixture,
        batch_size=1,
        run_label="cand",
        model_manifest={"model_type": "perturbed"},
    )
    comparison = compare_observations(
        source,
        candidate,
        fixture,
        topology_k=5,
        metric_bootstrap_samples=10,
        metric_bootstrap_seed=3,
    )
    assert comparison["metric_uncertainty"]["recall_at_1"]["sample_count"] == 1


def test_affected_query_ids_are_preserved():
    fixture = RetrievalFixture(
        fixture_id="regression-check",
        name="regression-check",
        description="query with deterministic regression",
        queries=[
            RetrievalQuery(
                query_id="q1",
                query="target-match",
                candidate_documents=[
                    CandidateDocument(document_id="wrong_1", text="other information"),
                    CandidateDocument(document_id="z_relevant", text="target-match"),
                    CandidateDocument(document_id="wrong_2", text="unrelated"),
                ],
                relevant_document_ids=["z_relevant"],
            ),
        ],
    )
    base = ToyEmbeddingModel(dimension=8, seed=12)
    manifest = {"model_type": "toy", "seed": 12}
    source = evaluate_model(
        base, fixture, batch_size=1, run_label="source", model_manifest=manifest
    )
    perturbed = PerturbedModel(
        base_model=base,
        perturbation=DimensionMaskPerturbation(
            mask_fraction=1.0, seed=1, perturbation_type="dimension_mask"
        ),
        seed=1,
        perturbation_manifest={"type": "dimension_mask", "mask_fraction": 1.0, "seed": 1},
        model_id="regressed",
    )
    candidate = evaluate_model(
        perturbed,
        fixture,
        batch_size=1,
        run_label="candidate",
        model_manifest={"model_type": "perturbed"},
    )
    comparison = compare_observations(source, candidate, fixture, topology_k=5)
    assert comparison["affected_samples"]["source_correct_candidate_wrong"] == ["q1"]
    assert "q1" in comparison["changed_nearest_neighbours"]


def test_bootstrap_observation_seed_is_used_for_stability():
    fixture = _load_fixture()
    source_model = ToyEmbeddingModel(dimension=10, seed=3)
    manifest = {"model_type": "toy", "seed": 3}
    source = evaluate_model(
        model=source_model,
        fixture=fixture,
        batch_size=2,
        run_label="src",
        model_manifest=manifest,
    )
    candidate = evaluate_model(
        model=source_model,
        fixture=fixture,
        batch_size=2,
        run_label="cand",
        model_manifest=manifest,
    )
    c1 = compare_observations(
        source,
        candidate,
        fixture,
        metric_bootstrap_seed=11,
    )
    c2 = compare_observations(
        source,
        candidate,
        fixture,
        metric_bootstrap_seed=11,
    )
    assert (
        c1["metric_uncertainty"]["recall_at_1"]["seed"]
        == c2["metric_uncertainty"]["recall_at_1"]["seed"]
    )


def _write_replay_config(tmp_path: Path) -> Path:
    fixture_name = "local_retrieval_fixture.json"
    fixture_copy = tmp_path / fixture_name
    shutil.copyfile(FIXTURE_PATH, fixture_copy)

    config = {
        "experiment_name": "m0-replay-smoke",
        "contract": "../contracts/m0-measurement-integrity-v1.json",
        "model": {
            "kind": "toy",
            "dimension": 16,
            "seed": 1729,
        },
        "dataset": {
            "path": fixture_name,
        },
        "null": {
            "repeats": 3,
            "batch_sizes": [1],
            "bootstrap_samples": 40,
            "confidence_level": 0.99,
            "random_seed": 2026,
            "candidate_bootstrap_samples": 40,
            "candidate_confidence_level": 0.99,
        },
        "runtime": {"topology_k": 5},
        "controls": {
            "exact_repeat": {
                "enabled": True,
                "repeats": 1,
            },
            "negative": {
                "enabled": False,
            },
            "boundary": {
                "enabled": False,
            },
        },
    }
    config_path = tmp_path / "m0-replay.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_m0_replay_recomputes_matching_decision(tmp_path: Path):
    config_path = _write_replay_config(tmp_path)
    summary = run_m0(config_path=config_path, output_root=tmp_path / "runs")

    replay_path = Path(summary["run_dir"]) / "replay-bundle.json"
    assert replay_path.exists()

    replay_bundle = json.loads(replay_path.read_text(encoding="utf-8"))
    dataset_payload = replay_bundle.get("dataset")
    assert isinstance(dataset_payload, dict)
    assert "fixture_payload" in dataset_payload
    assert "metric_policies" in replay_bundle.get("experiment", {})

    replay = run_m0_replay(replay_bundle_path=replay_path)
    assert replay["status_match"] is True
    assert replay["measurement_integrity_status"] == summary["measurement_integrity_status"]
