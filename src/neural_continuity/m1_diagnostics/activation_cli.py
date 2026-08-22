from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from typing import Any

from neural_continuity.m1_diagnostics.activation_capture import capture_activations
from neural_continuity.m1_diagnostics.activation_evidence import replay_activation_capture
from neural_continuity.m1_diagnostics.fidelity_authority import FidelityGateError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture or replay M1 paired diagnostic activations."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--config", required=True)
    capture.add_argument("--dataset", required=True)
    capture.add_argument("--instrumentation", required=True)
    capture.add_argument("--fidelity", required=True)
    capture.add_argument("--output", required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--bundle", required=True)
    replay.add_argument("--manifest-sha256", required=True)
    return parser


def _progress(event: Mapping[str, Any]) -> None:
    print(json.dumps(dict(event), sort_keys=True), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "capture":
            result = capture_activations(
                args.config,
                args.dataset,
                args.instrumentation,
                args.fidelity,
                args.output,
                progress=_progress,
            )
        else:
            result = replay_activation_capture(args.bundle, args.manifest_sha256)
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
