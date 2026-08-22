from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from neural_continuity.m1_diagnostics.measurement_null_extension_authority import (
    MeasurementNullPlanError,
)
from neural_continuity.m1_diagnostics.measurement_null_extension_evidence import (
    capture_measurement_null_extension_plan,
    replay_measurement_null_extension_plan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neural-continuity-m1-null-extension-plan")
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--provenance-bundle", required=True)
    capture.add_argument("--provenance-manifest-sha256", required=True)
    capture.add_argument("--output", required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("--bundle", required=True)
    replay.add_argument("--manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "capture":
            result = capture_measurement_null_extension_plan(
                args.provenance_bundle, args.provenance_manifest_sha256, args.output
            )
        else:
            result = replay_measurement_null_extension_plan(args.bundle, args.manifest_sha256)
    except MeasurementNullPlanError as exc:
        print(
            json.dumps(
                {"status": exc.status, "code": exc.code, "message": str(exc)}, sort_keys=True
            )
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
