"""CLI for deterministic compression and replay of the profile probe."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from neural_continuity.m1_diagnostics.cuda_null_full_profile_compression_package import (
    build_compressed_profile_package,
    replay_compressed_profile_package,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--source-bundle", type=Path, required=True)
    build.add_argument("--source-manifest-sha256", required=True)
    build.add_argument("--output", type=Path, required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--bundle", type=Path, required=True)
    replay.add_argument("--manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        result = build_compressed_profile_package(
            args.source_bundle,
            args.source_manifest_sha256,
            args.output,
        )
    else:
        result = replay_compressed_profile_package(
            args.bundle,
            args.manifest_sha256,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("replay_status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
