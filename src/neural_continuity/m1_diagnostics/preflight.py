"""CLI orchestration for the deterministic, model-free M1 diagnostic preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from neural_continuity.m1_diagnostics.authority import (
    DiagnosticPreflightError,
    FrozenAuthorityPaths,
    verify_frozen_authority_set,
)


def _json_bytes(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def _write_static_package(
    output_directory: Path, artifacts: dict[str, dict[str, object]]
) -> tuple[Path, str]:
    if output_directory.exists():
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="OUTPUT_DIRECTORY_ALREADY_EXISTS",
            message="Static preflight output directory must not already exist",
            details={"path": str(output_directory)},
        )
    output_directory.mkdir(parents=True)
    manifest_entries: list[dict[str, object]] = []
    for name in sorted(artifacts):
        encoded = _json_bytes(artifacts[name])
        (output_directory / name).write_bytes(encoded)
        manifest_entries.append(
            {
                "path": name,
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "size_bytes": len(encoded),
            }
        )
    manifest = {
        "kind": "m1_transition_b_v2_static_preflight_manifest",
        "artifacts": manifest_entries,
        "artifact_count": len(manifest_entries),
        "tamper_evident": True,
        "model_execution_used": False,
    }
    manifest_path = output_directory / "artifact-manifest.json"
    manifest_bytes = _json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    return manifest_path, hashlib.sha256(manifest_bytes).hexdigest()


def run_static_preflight(paths: FrozenAuthorityPaths, output_directory: Path) -> dict[str, object]:
    authorities = verify_frozen_authority_set(paths)

    # These imports are intentionally deferred until every frozen authority is verified.
    from neural_continuity.m1_diagnostics.graph_inventory import (
        build_graph_inventory,
        load_verified_graph,
    )
    from neural_continuity.m1_diagnostics.probe_plan import build_probe_plan
    from neural_continuity.m1_diagnostics.quantization_audit import (
        audit_quantization_parameters,
    )

    source_graph = load_verified_graph(authorities, "onnx_fp32_source")
    target_graph = load_verified_graph(authorities, "onnx_int8_candidate")
    source_inventory = build_graph_inventory(source_graph)
    target_inventory = build_graph_inventory(target_graph)
    quantization_audit = audit_quantization_parameters(target_graph)
    probe_plan = build_probe_plan(source_inventory, target_inventory)
    status = (
        "STATIC_PREFLIGHT_COMPLETE" if quantization_audit.integrity_status == "PASS" else "BLOCKED"
    )
    report: dict[str, object] = {
        "kind": "m1_transition_b_v2_static_preflight_report",
        "status": status,
        "frozen_transition_b_v1_scientific_decision": "FAIL",
        "scientific_decision_recomputed": False,
        "source_node_count": len(source_inventory.nodes),
        "target_node_count": len(target_inventory.nodes),
        "quantization_audit_status": quantization_audit.status,
        "quantization_audit_integrity_status": quantization_audit.integrity_status,
        "quantization_diagnostic_anomaly_count": len(quantization_audit.diagnostic_anomalies),
        "quantization_blocking_finding_count": len(quantization_audit.blocking_findings),
        "probe_count": len(probe_plan.probes),
        "probe_plan_sha256": probe_plan.sha256,
        "onnx_runtime_session_created": False,
        "activations_read": False,
        "model_execution_used": False,
    }
    artifacts = {
        "diagnostic-authority.json": authorities.to_dict(),
        "probe-plan.json": probe_plan.to_dict(),
        "quantization-parameter-audit.json": quantization_audit.to_dict(),
        "source-graph-inventory.json": source_inventory.to_dict(),
        "static-preflight-report.json": report,
        "target-graph-inventory.json": target_inventory.to_dict(),
    }
    manifest_path, manifest_sha256 = _write_static_package(output_directory, artifacts)
    return {
        **report,
        "output_directory": str(output_directory.resolve()),
        "artifact_manifest": str(manifest_path.resolve()),
        "artifact_manifest_sha256": manifest_sha256,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic static M1 Transition B v2 diagnostic artifacts."
    )
    parser.add_argument("--fp32-model", type=Path, required=True)
    parser.add_argument("--int8-model", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--fp32-evidence-manifest", type=Path, required=True)
    parser.add_argument("--int8-evidence-manifest", type=Path, required=True)
    parser.add_argument("--transition-b-decision-manifest", type=Path, required=True)
    parser.add_argument("--transition-a-contract", type=Path, required=True)
    parser.add_argument("--transition-b-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = FrozenAuthorityPaths(
        onnx_fp32_source=args.fp32_model,
        onnx_int8_candidate=args.int8_model,
        calibration_manifest=args.calibration_manifest,
        paired_fp32_evidence=args.fp32_evidence_manifest,
        int8_target_evidence=args.int8_evidence_manifest,
        transition_b_decision=args.transition_b_decision_manifest,
        transition_a_contract=args.transition_a_contract,
        transition_b_v1_contract=args.transition_b_contract,
    )
    try:
        result = run_static_preflight(paths, args.output)
    except DiagnosticPreflightError as exc:
        print(json.dumps(exc.to_dict(), indent=2, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2 if result["status"] == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
