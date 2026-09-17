"""Source-only CUDA technical preflight capture and model-free replay."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path

from neural_continuity.m1_diagnostics.cuda_null_runtime_authority import CudaNullRuntimeBlocked
from neural_continuity.m1_diagnostics.cuda_null_source_preflight_authority import (
    CudaNullSourcePreflightBlocked,
)
from neural_continuity.m1_diagnostics.cuda_null_source_preflight_evidence import (
    replay_source_preflight,
    write_source_preflight_package,
)
from neural_continuity.m1_diagnostics.cuda_preflight_authority import CudaPreflightBlocked
from neural_continuity.m1_teacher_evidence import TeacherEvidenceError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Source-only CUDA technical preflight")
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    for name in (
        "authorization-spec",
        "preflight-spec",
        "static-bundle",
        "config",
        "dataset",
        "transition-a-bundle",
        "teacher-snapshot",
        "cpu-extension-bundle",
        "historical-cuda-bundle",
        "output",
    ):
        capture.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "reviewed-spec-sha256",
        "preflight-spec-sha256",
        "static-manifest-sha256",
        "config-sha256",
    ):
        capture.add_argument(f"--{name}", required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--bundle", type=Path, required=True)
    replay.add_argument("--artifact-manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "replay":
        result = replay_source_preflight(arguments.bundle, arguments.artifact_manifest_sha256)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["replay_status"] == "PASS" else 2

    from neural_continuity.m1_diagnostics.cuda_null_source_preflight_runtime import (
        capture_source_preflight,
    )

    try:
        with tempfile.TemporaryDirectory(prefix="nc-m1-cuda-source-preflight-") as working:
            capture = capture_source_preflight(
                authorization_spec=arguments.authorization_spec,
                external_reviewed_spec_sha256=arguments.reviewed_spec_sha256,
                preflight_spec=arguments.preflight_spec,
                external_preflight_spec_sha256=arguments.preflight_spec_sha256,
                static_bundle=arguments.static_bundle,
                external_static_manifest_sha256=arguments.static_manifest_sha256,
                config_path=arguments.config,
                external_config_sha256=arguments.config_sha256,
                dataset_root=arguments.dataset,
                transition_a_bundle=arguments.transition_a_bundle,
                teacher_snapshot_root=arguments.teacher_snapshot,
                cpu_extension_bundle=arguments.cpu_extension_bundle,
                historical_cuda_bundle=arguments.historical_cuda_bundle,
                working_directory=Path(working),
            )
    except (CudaNullSourcePreflightBlocked, CudaNullRuntimeBlocked, CudaPreflightBlocked) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, sort_keys=True))
        return 2
    except TeacherEvidenceError as exc:
        status = "BLOCKED" if exc.status == "BLOCKED" else "EXECUTION_ERROR"
        print(json.dumps({"status": status, "reason": str(exc)}, sort_keys=True))
        return 2 if status == "BLOCKED" else 3
    except Exception as exc:
        print(
            json.dumps(
                {"status": "EXECUTION_ERROR", "reason": f"{type(exc).__name__}: {exc}"},
                sort_keys=True,
            )
        )
        return 3
    try:
        output, manifest_sha256 = write_source_preflight_package(capture, arguments.output)
        replay = replay_source_preflight(output / "replay-bundle.json", manifest_sha256)
        if replay.get("replay_status") != "PASS":
            raise ValueError("new source-only package did not replay")
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": "TECHNICAL_PREFLIGHT_PASS",
                "scientific_decision": "NOT_EVALUATED",
                "qualifying_m1_evidence": False,
                "output_directory": str(output),
                "artifact_manifest_sha256": manifest_sha256,
                "replay_status": replay["replay_status"],
                "run_timings": replay["run_timings"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
