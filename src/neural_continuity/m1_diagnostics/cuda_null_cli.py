"""Non-executable static authority and replay entry point for CUDA null."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_continuity.m1_diagnostics.cuda_null_authority import (
    CudaNullAuthorityBlocked,
    build_static_authority,
)
from neural_continuity.m1_diagnostics.cuda_null_evidence import (
    replay_static_package,
    write_static_package,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify CUDA-null static authority without loading ONNX"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument(
        "--config-sha256", required=True, help="SHA-256 supplied outside the draft config"
    )
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--transition-a-bundle", type=Path, required=True)
    prepare.add_argument("--teacher-snapshot", type=Path, required=True)
    prepare.add_argument("--cpu-extension-bundle", type=Path, required=True)
    prepare.add_argument("--historical-cuda-bundle", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    replay = subcommands.add_parser("replay")
    replay.add_argument("--bundle", type=Path, required=True)
    replay.add_argument(
        "--manifest-sha256", required=True, help="SHA-256 supplied outside the replay bundle"
    )
    arguments = parser.parse_args()
    try:
        if arguments.command == "prepare":
            authority = build_static_authority(
                config_path=arguments.config,
                external_config_sha256=arguments.config_sha256,
                dataset_root=arguments.dataset,
                transition_a_bundle=arguments.transition_a_bundle,
                teacher_snapshot_root=arguments.teacher_snapshot,
                cpu_extension_bundle=arguments.cpu_extension_bundle,
                historical_cuda_bundle=arguments.historical_cuda_bundle,
            )
            result = write_static_package(authority, arguments.output)
        else:
            result = replay_static_package(arguments.bundle, arguments.manifest_sha256)
    except CudaNullAuthorityBlocked as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
