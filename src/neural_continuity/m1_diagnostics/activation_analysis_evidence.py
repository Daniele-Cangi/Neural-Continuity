from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.activation_analysis_authority import (
    ActivationAnalysisError,
    VerifiedActivationAnalysisInput,
    verify_activation_analysis_input,
)
from neural_continuity.m1_diagnostics.activation_analysis_metrics import (
    ProgressCallback,
    analyze_activation_batches,
)

ANALYSIS_ARTIFACTS = (
    "analysis-plan.json",
    "probe-diagnostics.json",
    "diagnostic-report.json",
    "replay-bundle.json",
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActivationAnalysisError(
            "ANALYSIS_ARTIFACT_INVALID",
            f"cannot load {path.name}: {exc}",
        ) from exc
    if not isinstance(payload, Mapping):
        raise ActivationAnalysisError(
            "ANALYSIS_ARTIFACT_INVALID",
            f"{path.name} must contain an object",
        )
    return payload


def _safe_artifact(root: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ActivationAnalysisError(
            "ANALYSIS_ARTIFACT_INVALID",
            "analysis artifact path is missing",
        )
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ActivationAnalysisError(
            "ANALYSIS_ARTIFACT_INVALID",
            "analysis artifact path must be relative",
        )
    candidate = (root / relative).resolve()
    if root != candidate and root not in candidate.parents:
        raise ActivationAnalysisError(
            "ANALYSIS_ARTIFACT_INVALID",
            "analysis artifact path escapes the package",
        )
    if not candidate.is_file():
        raise ActivationAnalysisError(
            "ANALYSIS_ARTIFACT_MISSING",
            f"analysis artifact is missing: {relative_path}",
        )
    return candidate


def _analysis_plan(
    authority: VerifiedActivationAnalysisInput,
) -> dict[str, Any]:
    return {
        "kind": "m1-diagnostic-activation-analysis-plan",
        "version": "1.0.0",
        "status": "READY",
        "source_activation_bundle": str(authority.bundle_path),
        "source_activation_manifest_sha256": authority.manifest_sha256,
        "source_capture_plan_sha256": sha256_file(authority.root / "capture-plan.json"),
        "query_count": authority.capture_plan["query_count"],
        "probe_count": authority.capture_plan["probe_count"],
        "integer_probe_count": authority.capture_plan["integer_probe_count"],
        "paired_numeric_metrics": [
            "bitwise_difference_count",
            "mean_absolute_delta",
            "rmse",
            "relative_l2_error",
            "maximum_absolute_delta",
            "cosine_similarity",
        ],
        "paired_dtype_domains": ["float32", "int64", "bool"],
        "integer_metric": "dtype_extreme_rate_descriptive_proxy",
        "first_divergence_rule": ("first_probe_with_nonzero_bitwise_difference_count"),
        "numerical_threshold_selected": False,
        "scientific_decision_recomputed": False,
        "frozen_transition_b_v1_scientific_decision": "FAIL",
        "model_execution_used": False,
        "onnx_graph_loaded": False,
    }


def _documents(
    authority: VerifiedActivationAnalysisInput,
    analysis: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = _analysis_plan(authority)
    diagnostics = {
        "kind": "m1-diagnostic-activation-probe-diagnostics",
        "status": "COMPLETE",
        "probe_count": analysis["probe_count"],
        "probes": analysis["probes"],
    }
    report = {
        "kind": "m1-diagnostic-activation-report",
        "status": "COMPLETE",
        "diagnostic_status": "DESCRIPTIVE_ONLY",
        "source_activation_manifest_sha256": authority.manifest_sha256,
        "batch_count": analysis["batch_count"],
        "query_count": analysis["query_count"],
        "probe_count": analysis["probe_count"],
        "integer_probe_count": analysis["integer_probe_count"],
        "summary": analysis["summary"],
        "scientific_decision_recomputed": False,
        "frozen_transition_b_v1_scientific_decision": "FAIL",
        "model_execution_used": False,
        "causal_claim_made": False,
    }
    return plan, diagnostics, report


def _write_manifest(build_root: Path) -> str:
    artifacts = [
        {
            "path": name,
            "sha256": sha256_file(build_root / name),
            "size_bytes": (build_root / name).stat().st_size,
        }
        for name in ANALYSIS_ARTIFACTS
    ]
    manifest = {
        "kind": "m1-diagnostic-activation-analysis-manifest",
        "status": "COMPLETE",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "tamper_evident": True,
        "model_execution_used": False,
        "replay_requires_model_execution": False,
    }
    _write_json(build_root / "artifact-manifest.json", manifest)
    return sha256_file(build_root / "artifact-manifest.json")


def _verify_manifest(root: Path, expected_sha256: str) -> Mapping[str, Any]:
    manifest_path = root / "artifact-manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected_sha256:
        raise ActivationAnalysisError(
            "ANALYSIS_MANIFEST_HASH_MISMATCH",
            "analysis manifest hash does not match",
        )
    manifest = _load_json(manifest_path)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or manifest.get("artifact_count") != len(ANALYSIS_ARTIFACTS):
        raise ActivationAnalysisError(
            "ANALYSIS_MANIFEST_INVALID",
            "analysis manifest declaration is invalid",
        )
    declared: set[str] = set()
    for record in artifacts:
        if not isinstance(record, Mapping):
            raise ActivationAnalysisError(
                "ANALYSIS_MANIFEST_INVALID",
                "analysis artifact record is invalid",
            )
        path = record.get("path")
        artifact = _safe_artifact(root, path)
        if path in declared or sha256_file(artifact) != record.get("sha256"):
            raise ActivationAnalysisError(
                "ANALYSIS_ARTIFACT_HASH_MISMATCH",
                f"analysis artifact hash does not match: {path}",
            )
        if artifact.stat().st_size != record.get("size_bytes"):
            raise ActivationAnalysisError(
                "ANALYSIS_ARTIFACT_SIZE_MISMATCH",
                f"analysis artifact size does not match: {path}",
            )
        declared.add(str(path))
    if declared != set(ANALYSIS_ARTIFACTS):
        raise ActivationAnalysisError(
            "ANALYSIS_MANIFEST_INVALID",
            "analysis artifact set is incomplete",
        )
    return manifest


def create_activation_analysis_package(
    source_bundle: str | Path,
    source_manifest_sha256: str,
    output_directory: str | Path,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    authority = verify_activation_analysis_input(source_bundle, source_manifest_sha256)
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise ActivationAnalysisError(
            "ANALYSIS_OUTPUT_EXISTS",
            "analysis output directory already exists",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    build_root = Path(
        tempfile.mkdtemp(
            prefix=".activation-analysis-build-",
            dir=destination.parent,
        )
    ).resolve()
    try:
        analysis = analyze_activation_batches(
            authority.root, authority.capture_plan, authority.batch_index, progress
        )
        plan, diagnostics, report = _documents(authority, analysis)
        _write_json(build_root / "analysis-plan.json", plan)
        _write_json(build_root / "probe-diagnostics.json", diagnostics)
        _write_json(build_root / "diagnostic-report.json", report)
        replay_bundle = {
            "replay_format_version": "1.0.0",
            "analysis_plan_path": "analysis-plan.json",
            "probe_diagnostics_path": "probe-diagnostics.json",
            "diagnostic_report_path": "diagnostic-report.json",
            "source_activation_bundle": str(authority.bundle_path),
            "source_activation_manifest_sha256": authority.manifest_sha256,
            "expected_status": "COMPLETE",
            "replay_requires_model_execution": False,
        }
        _write_json(build_root / "replay-bundle.json", replay_bundle)
        manifest_sha256 = _write_manifest(build_root)
        build_root.rename(destination)
    except Exception:
        if build_root.exists() and build_root.parent == destination.parent:
            shutil.rmtree(build_root)
        raise
    return {
        "status": "COMPLETE",
        "diagnostic_status": "DESCRIPTIVE_ONLY",
        "output_directory": str(destination),
        "artifact_manifest_sha256": manifest_sha256,
        "query_count": analysis["query_count"],
        "probe_count": analysis["probe_count"],
        "integer_probe_count": analysis["integer_probe_count"],
        "first_bitwise_divergence": analysis["summary"]["first_bitwise_divergence"],
        "scientific_decision_recomputed": False,
        "model_execution_used": False,
    }


def replay_activation_analysis(
    bundle_path: str | Path,
    expected_manifest_sha256: str,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    bundle = Path(bundle_path).resolve()
    root = bundle.parent
    _verify_manifest(root, expected_manifest_sha256)
    replay_bundle = _load_json(bundle)
    if replay_bundle.get("replay_requires_model_execution") is not False:
        raise ActivationAnalysisError(
            "MODEL_EXECUTION_REQUIRED",
            "analysis replay is not model-free",
        )
    source_bundle = replay_bundle.get("source_activation_bundle")
    source_manifest = replay_bundle.get("source_activation_manifest_sha256")
    if not isinstance(source_bundle, str) or not isinstance(source_manifest, str):
        raise ActivationAnalysisError(
            "ANALYSIS_REPLAY_INVALID",
            "source activation authority is missing",
        )
    authority = verify_activation_analysis_input(source_bundle, source_manifest)
    analysis = analyze_activation_batches(
        authority.root, authority.capture_plan, authority.batch_index, progress
    )
    plan, diagnostics, report = _documents(authority, analysis)
    stored_plan = _load_json(_safe_artifact(root, replay_bundle.get("analysis_plan_path")))
    stored_diagnostics = _load_json(
        _safe_artifact(root, replay_bundle.get("probe_diagnostics_path"))
    )
    stored_report = _load_json(_safe_artifact(root, replay_bundle.get("diagnostic_report_path")))
    plan_match = canonical_json_bytes(plan) == canonical_json_bytes(stored_plan)
    diagnostics_match = canonical_json_bytes(diagnostics) == canonical_json_bytes(
        stored_diagnostics
    )
    report_match = canonical_json_bytes(report) == canonical_json_bytes(stored_report)
    if not plan_match or not diagnostics_match or not report_match:
        raise ActivationAnalysisError(
            "ANALYSIS_REPLAY_MISMATCH",
            "recomputed diagnostic evidence does not match",
        )
    return {
        "status": "COMPLETE",
        "diagnostic_status": "DESCRIPTIVE_ONLY",
        "replay_verified": True,
        "plan_match": plan_match,
        "probe_diagnostics_match": diagnostics_match,
        "report_match": report_match,
        "probe_count": analysis["probe_count"],
        "integer_probe_count": analysis["integer_probe_count"],
        "query_count": analysis["query_count"],
        "scientific_decision_recomputed": False,
        "model_execution_used": False,
    }
