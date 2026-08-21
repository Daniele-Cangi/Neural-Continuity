"""CLI for deterministic derivation of M1 diagnostic instrumented graph copies."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from neural_continuity.m1_diagnostics.authority import DiagnosticPreflightError
from neural_continuity.m1_diagnostics.instrumentation import derive_instrumented_graphs
from neural_continuity.m1_diagnostics.static_package import (
    verify_static_preflight_package,
)


def _json_bytes(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_instrumentation_preflight(
    static_package: Path, output_directory: Path
) -> dict[str, object]:
    if output_directory.exists():
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="OUTPUT_DIRECTORY_ALREADY_EXISTS",
            message="Instrumentation output directory must not already exist",
            details={"path": str(output_directory)},
        )
    verified = verify_static_preflight_package(static_package)
    output_directory.mkdir(parents=True)
    result = derive_instrumented_graphs(verified, output_directory)
    documents = {
        "instrumentation-authority.json": {
            "kind": "m1_transition_b_v2_instrumentation_authority",
            "status": "PASS",
            "static_preflight": verified.to_dict(),
            "model_execution_used": False,
        },
        "instrumentation-plan.json": result.to_dict(),
        "instrumentation-report.json": {
            "kind": "m1_transition_b_v2_instrumentation_report",
            "status": "READY_FOR_FIDELITY_CONTROL",
            "source_instrumented_sha256": result.source.sha256,
            "target_instrumented_sha256": result.target.sha256,
            "probe_count": len(result.probe_mappings),
            "frozen_models_overwritten": False,
            "onnx_runtime_session_created": False,
            "activations_read": False,
            "model_execution_used": False,
            "scientific_decision_recomputed": False,
            "frozen_transition_b_v1_scientific_decision": "FAIL",
        },
    }
    artifact_entries: list[dict[str, object]] = []
    for name in sorted(documents):
        encoded = _json_bytes(documents[name])
        path = output_directory / name
        path.write_bytes(encoded)
        artifact_entries.append(
            {
                "path": name,
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "size_bytes": len(encoded),
            }
        )
    for path in (result.source.path, result.target.path):
        artifact_entries.append(
            {
                "path": path.name,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    artifact_entries.sort(key=lambda entry: str(entry["path"]))
    manifest = {
        "kind": "m1_transition_b_v2_instrumentation_manifest",
        "status": "READY_FOR_FIDELITY_CONTROL",
        "artifacts": artifact_entries,
        "artifact_count": len(artifact_entries),
        "tamper_evident": True,
        "model_execution_used": False,
    }
    manifest_path = output_directory / "artifact-manifest.json"
    manifest_bytes = _json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    return {
        "status": "READY_FOR_FIDELITY_CONTROL",
        "output_directory": str(output_directory.resolve()),
        "artifact_manifest": str(manifest_path.resolve()),
        "artifact_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_instrumented_sha256": result.source.sha256,
        "target_instrumented_sha256": result.target.sha256,
        "probe_count": len(result.probe_mappings),
        "onnx_runtime_session_created": False,
        "activations_read": False,
        "model_execution_used": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive M1 diagnostic instrumented ONNX copies without model execution."
    )
    parser.add_argument("--static-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_instrumentation_preflight(args.static_package, args.output)
    except DiagnosticPreflightError as exc:
        print(json.dumps(exc.to_dict(), indent=2, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
