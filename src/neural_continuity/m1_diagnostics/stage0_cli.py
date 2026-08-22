from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from typing import Any

from neural_continuity.m1_diagnostics.stage0_authority import (
    Stage0ControlError,
)
from neural_continuity.m1_diagnostics.stage0_evidence import (
    create_stage0_control_package,
    replay_stage0_control_package,
)


def _emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neural-continuity-m1-diagnostic-stage0-controls")
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--causal-plan-bundle", required=True)
    capture.add_argument("--causal-plan-manifest-sha256", required=True)
    capture.add_argument("--archive-manifest", required=True)
    capture.add_argument("--archive-manifest-sha256", required=True)
    capture.add_argument("--runtime-manifest", required=True)
    capture.add_argument("--runtime-manifest-sha256", required=True)
    capture.add_argument("--output", required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("--bundle", required=True)
    replay.add_argument("--manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "capture":
            result = create_stage0_control_package(
                arguments.causal_plan_bundle,
                arguments.causal_plan_manifest_sha256,
                arguments.archive_manifest,
                arguments.archive_manifest_sha256,
                arguments.runtime_manifest,
                arguments.runtime_manifest_sha256,
                arguments.output,
            )
        else:
            result = replay_stage0_control_package(
                arguments.bundle,
                arguments.manifest_sha256,
            )
    except Stage0ControlError as exc:
        _emit(exc.as_dict())
        return 2
    except Exception as exc:
        _emit(
            {
                "code": "STAGE0_CONTROL_EXECUTION_ERROR",
                "message": str(exc),
                "status": "EXECUTION_ERROR",
            }
        )
        return 2
    _emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
