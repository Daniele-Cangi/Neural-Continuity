from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from neural_continuity.m1_diagnostics.runtime_provenance_authority import RuntimeProvenanceError
from neural_continuity.m1_diagnostics.runtime_provenance_evidence import (
    capture_runtime_provenance_package,
    replay_runtime_provenance_package,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neural-continuity-m1-runtime-provenance")
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--stage0-bundle", required=True)
    capture.add_argument("--stage0-manifest-sha256", required=True)
    capture.add_argument("--output", required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("--bundle", required=True)
    replay.add_argument("--manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "capture":
            result = capture_runtime_provenance_package(
                args.stage0_bundle, args.stage0_manifest_sha256, args.output
            )
        else:
            result = replay_runtime_provenance_package(args.bundle, args.manifest_sha256)
    except RuntimeProvenanceError as exc:
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
