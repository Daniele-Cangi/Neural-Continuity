from __future__ import annotations

import copy
from pathlib import Path

from neural_continuity.bootstrap import build_envelopes
from neural_continuity.cli import REQUIRED_METRICS
from neural_continuity.datasets import (
    CandidateDocument,
    RetrievalFixture,
    RetrievalQuery,
    load_retrieval_fixture,
)
from neural_continuity.decisions import evaluate_comparison
from neural_continuity.metrics import compare_observations
from neural_continuity.models import PerturbedModel, ToyEmbeddingModel
from neural_continuity.observations import chunked, evaluate_model
from neural_continuity.perturbations import DimensionMaskPerturbation, GaussianNoisePerturbation

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "local_retrieval_fixture.json"


def _load_fixture():
    return load_retrieval_fixture(FIXTURE_PATH)


def test_deterministic_toy_inference_no_material_divergence():
    fixture = _load_fixture()
    model = ToyEmbeddingModel(dimension=16, seed=7)
    manifest = {"model_type": "toy", "seed": 7}
    source = evaluate_model(
        model=model, fixture=fixture, batch_size=1, run_label="base", model_manifest=manifest
    )
    candidate = evaluate_model(
        model=model, fixture=fixture, batch_size=4, run_label="base2", model_manifest=manifest
    )
    comparison = compare_observations(source, candidate, fixture, topology_k=5)
    envelopes = build_envelopes(
        [comparison],
        metric_names=REQUIRED_METRICS,
        bootstrap_samples=100,
        confidence_level=0.99,
        seed=11,
    )
    decision = evaluate_comparison(comparison, envelopes, required_metrics=REQUIRED_METRICS)
    assert decision.status == "PASS"


def test_strong_perturbation_fails():
    fixture = _load_fixture()
    base_model = ToyEmbeddingModel(dimension=16, seed=9)
    manifest = {"model_type": "toy", "seed": 9}
    source = evaluate_model(
        model=base_model,
        fixture=fixture,
        batch_size=2,
        run_label="source",
        model_manifest=manifest,
    )
    perturbed = PerturbedModel(
        base_model=base_model,
        perturbation=DimensionMaskPerturbation(
            mask_fraction=1.0, seed=0, perturbation_type="dimension_mask"
        ),
        seed=0,
        perturbation_manifest={"type": "dimension_mask", "mask_fraction": 1.0},
        model_id="perturbed",
    )
    candidate = evaluate_model(
        model=perturbed,
        fixture=fixture,
        batch_size=2,
        run_label="perturbed",
        model_manifest={"model_type": "perturbed"},
    )
    comparison = compare_observations(source, candidate, fixture, topology_k=5)
    envelopes = build_envelopes(
        [
            {
                "metric_deltas": {metric: 0.0 for metric in REQUIRED_METRICS},
                "source": comparison["source"],
                "candidate": comparison["candidate"],
                "regressions": {"source_correct_candidate_wrong": [], "other": []},
                "sample_count": len(fixture.queries),
            }
        ],
        metric_names=REQUIRED_METRICS,
        bootstrap_samples=100,
        confidence_level=0.99,
        seed=13,
    )
    decision = evaluate_comparison(comparison, envelopes, required_metrics=REQUIRED_METRICS)
    assert decision.status == "FAIL"


def test_boundary_control_is_inconclusive():
    fixture = _load_fixture()
    base_model = ToyEmbeddingModel(dimension=16, seed=11)
    manifest = {"model_type": "toy", "seed": 11}
    source = evaluate_model(
        model=base_model, fixture=fixture, batch_size=2, run_label="source", model_manifest=manifest
    )
    candidate_model = PerturbedModel(
        base_model=base_model,
        perturbation=GaussianNoisePerturbation(
            strength=0.001, seed=5, perturbation_type="gaussian_noise"
        ),
        seed=5,
        perturbation_manifest={"type": "gaussian_noise", "strength": 0.001, "seed": 5},
        model_id="mild",
    )
    candidate = evaluate_model(
        model=candidate_model,
        fixture=fixture,
        batch_size=2,
        run_label="boundary",
        model_manifest={"model_type": "boundary"},
    )
    comparison = compare_observations(source, candidate, fixture, topology_k=5)
    comparison["sample_count"] = 1
    envelopes = build_envelopes(
        [comparison],
        metric_names=REQUIRED_METRICS,
        bootstrap_samples=20,
        confidence_level=0.99,
        seed=15,
    )
    decision = evaluate_comparison(
        comparison,
        envelopes,
        required_metrics=REQUIRED_METRICS,
        require_boundary_inconclusive=True,
    )
    assert decision.status == "INCONCLUSIVE"


def test_missing_required_evidence_is_inconclusive():
    fixture = _load_fixture()
    model = ToyEmbeddingModel(dimension=16, seed=3)
    manifest = {"model_type": "toy", "seed": 3}
    source = evaluate_model(
        model=model, fixture=fixture, batch_size=1, run_label="source", model_manifest=manifest
    )
    candidate = evaluate_model(
        model=model, fixture=fixture, batch_size=1, run_label="candidate", model_manifest=manifest
    )
    comparison = compare_observations(source, candidate, fixture, topology_k=5)
    broken = copy.deepcopy(comparison)
    broken["metric_deltas"].pop("recall_at_1")
    envelopes = build_envelopes(
        [comparison],
        metric_names=[m for m in REQUIRED_METRICS if m != "recall_at_1"],
        bootstrap_samples=20,
        confidence_level=0.99,
        seed=14,
    )
    decision = evaluate_comparison(broken, envelopes, required_metrics=REQUIRED_METRICS)
    assert decision.status == "INCONCLUSIVE"


def test_metric_results_keep_affected_query_ids():
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


def test_batch_chunks_do_not_reorder():
    values = list(range(6))
    chunks = chunked(values, 2)
    assert chunks == [[0, 1], [2, 3], [4, 5]]


def test_decision_output_stable_for_identical_inputs():
    fixture = _load_fixture()
    model = ToyEmbeddingModel(dimension=16, seed=5)
    manifest = {"model_type": "toy", "seed": 5}
    source = evaluate_model(
        model=model, fixture=fixture, batch_size=2, run_label="source", model_manifest=manifest
    )
    candidate = evaluate_model(
        model=model, fixture=fixture, batch_size=2, run_label="candidate", model_manifest=manifest
    )
    comparison = compare_observations(source, candidate, fixture, topology_k=5)
    envelopes = build_envelopes(
        [comparison],
        metric_names=REQUIRED_METRICS,
        bootstrap_samples=60,
        confidence_level=0.99,
        seed=19,
    )
    first = evaluate_comparison(comparison, envelopes, required_metrics=REQUIRED_METRICS)
    second = evaluate_comparison(comparison, envelopes, required_metrics=REQUIRED_METRICS)
    assert first == second


def test_cuda_metrics_are_skipped_when_cuda_unavailable(monkeypatch):
    from neural_continuity import observations as obs

    monkeypatch.setattr(obs.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(obs.torch.cuda, "reset_peak_memory_stats", lambda *args, **kwargs: None)
    monkeypatch.setattr(obs.torch.cuda, "max_memory_allocated", lambda *args, **kwargs: 0)

    model = ToyEmbeddingModel(dimension=4, seed=2)
    fixture = _load_fixture()
    run = evaluate_model(
        model=model,
        fixture=fixture,
        batch_size=1,
        run_label="cpu-only",
        model_manifest={"model_type": "toy", "seed": 2},
    )
    assert run.system_metrics["cuda_peak_allocated_bytes"] is None
    assert run.system_metrics["hardware"] == "cpu"
