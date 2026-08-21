from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from neural_continuity.m1_diagnostics.fidelity_authority import FidelityGateError
from neural_continuity.m1_diagnostics.fidelity_control import capture_fidelity
from neural_continuity.m1_diagnostics.fidelity_evidence import replay_fidelity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture or replay the M1 instrumented-graph final-output fidelity control."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--config", required=True)
    capture.add_argument("--dataset", required=True)
    capture.add_argument("--instrumentation", required=True)
    capture.add_argument("--output", required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--bundle", required=True)
    replay.add_argument("--manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "capture":
            result = capture_fidelity(
                args.config,
                args.dataset,
                args.instrumentation,
                args.output,
            )
        else:
            result = replay_fidelity(args.bundle, args.manifest_sha256)
    except FidelityGateError as exc:
        print(
            json.dumps(
                {"status": exc.status, "error": {"code": exc.code, "message": exc.message}},
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
