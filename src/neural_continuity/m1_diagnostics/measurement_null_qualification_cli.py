from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from neural_continuity.m1_diagnostics.measurement_null_qualification_authority import (
    QualificationPreflightError,
)
from neural_continuity.m1_diagnostics.measurement_null_qualification_evidence import (
    capture_qualification_preflight_package,
    replay_qualification_preflight_package,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neural-continuity-m1-null-qualification",
        description="Verify and package the model-free M1 null-qualification preflight.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare",
        help="capture a qualification-preflight package",
    )
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--transition-a-bundle", required=True)
    prepare.add_argument("--extension-plan-bundle", required=True)
    prepare.add_argument("--extension-plan-manifest-sha256", required=True)
    prepare.add_argument("--sentinel-run", required=True)
    prepare.add_argument("--sentinel-root-manifest-sha256", required=True)
    prepare.add_argument("--sentinel-checkpoint-sha256", required=True)
    prepare.add_argument("--output", required=True)
    replay = commands.add_parser(
        "replay",
        help="replay a qualification-preflight package",
    )
    replay.add_argument("--bundle", required=True)
    replay.add_argument("--manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "prepare":
            result = capture_qualification_preflight_package(
                arguments.config,
                arguments.dataset,
                arguments.transition_a_bundle,
                arguments.extension_plan_bundle,
                arguments.extension_plan_manifest_sha256,
                arguments.sentinel_run,
                arguments.sentinel_root_manifest_sha256,
                arguments.sentinel_checkpoint_sha256,
                arguments.output,
            )
        else:
            result = replay_qualification_preflight_package(
                arguments.bundle,
                arguments.manifest_sha256,
            )
    except QualificationPreflightError as exc:
        print(
            json.dumps(
                {
                    "status": exc.status,
                    "error_code": exc.code,
                    "message": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
