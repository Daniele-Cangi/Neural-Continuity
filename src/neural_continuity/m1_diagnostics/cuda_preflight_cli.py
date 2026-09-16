"""Command line interface for capture and model-free CUDA preflight replay."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence

from neural_continuity.m1_diagnostics.cuda_preflight_authority import (
    CudaPreflightBlocked,
    authority_record,
    verify_cuda_preflight_authority,
)
from neural_continuity.m1_diagnostics.cuda_preflight_evidence import (
    replay_cuda_preflight,
    write_cuda_preflight_package,
)
from neural_continuity.m1_diagnostics.cuda_preflight_runtime import (
    run_cuda_preflight,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture or replay the non-qualifying M1 hybrid CUDA preflight."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--config", required=True)
    capture.add_argument("--config-sha256", required=True)
    capture.add_argument("--dataset-directory", required=True)
    capture.add_argument("--transition-a-bundle", required=True)
    capture.add_argument("--extension-plan-bundle", required=True)
    capture.add_argument("--extension-plan-manifest-sha256", required=True)
    capture.add_argument("--candidate-package", required=True)
    capture.add_argument("--output", required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--bundle", required=True)
    replay.add_argument("--artifact-manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "replay":
        result = replay_cuda_preflight(args.bundle, args.artifact_manifest_sha256)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["replay_status"] == "PASS" else 2

    try:
        authority = verify_cuda_preflight_authority(
            config_path=args.config,
            dataset_directory=args.dataset_directory,
            transition_a_bundle=args.transition_a_bundle,
            extension_plan_bundle=args.extension_plan_bundle,
            extension_plan_manifest_sha256=args.extension_plan_manifest_sha256,
            candidate_package=args.candidate_package,
            expected_config_sha256=args.config_sha256,
        )
        with tempfile.TemporaryDirectory(
            prefix="nc-m1-cuda-preflight-", ignore_cleanup_errors=True
        ) as working:
            try:
                runtime_result = run_cuda_preflight(authority, working)
            except CudaPreflightBlocked:
                raise
            except Exception as exc:
                raise CudaPreflightBlocked(
                    f"runtime execution error ({type(exc).__name__}): {exc}"
                ) from exc
        output, manifest_sha256 = write_cuda_preflight_package(
            authority_record(authority), runtime_result, args.output
        )
        result = {
            "preflight_status": "PASS",
            "scientific_decision": "NOT_EVALUATED",
            "qualifying_m1_evidence": False,
            "full_corpus_authorized": False,
            "output_directory": str(output),
            "artifact_manifest_sha256": manifest_sha256,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (CudaPreflightBlocked, FileExistsError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "preflight_status": "BLOCKED",
                    "scientific_decision": "NOT_EVALUATED",
                    "reason": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
