"""CLI for the bounded profile-storage probe and its model-free replay."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--spec-sha256", required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--bundle", type=Path, required=True)
    replay.add_argument("--manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "capture":
        from neural_continuity.m1_diagnostics.cuda_null_full_profile_probe_capture import (
            capture_full_profile_probe,
        )

        result = capture_full_profile_probe(args.spec_sha256)
    else:
        from neural_continuity.m1_diagnostics.cuda_null_full_profile_probe_replay import (
            replay_full_profile_probe,
        )

        result = replay_full_profile_probe(args.bundle, args.manifest_sha256)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("replay_status", "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
