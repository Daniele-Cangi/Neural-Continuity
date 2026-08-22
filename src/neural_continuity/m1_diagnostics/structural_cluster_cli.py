from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from typing import Any

from neural_continuity.m1_diagnostics.structural_cluster_authority import (
    StructuralClusterError,
)
from neural_continuity.m1_diagnostics.structural_cluster_evidence import (
    create_structural_cluster_package,
    replay_structural_cluster_package,
)


def _emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neural-continuity-m1-diagnostic-clusters")
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--bundle", required=True)
    analyze.add_argument("--manifest-sha256", required=True)
    analyze.add_argument("--output", required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("--bundle", required=True)
    replay.add_argument("--manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "analyze":
            result = create_structural_cluster_package(
                arguments.bundle,
                arguments.manifest_sha256,
                arguments.output,
            )
        else:
            result = replay_structural_cluster_package(
                arguments.bundle,
                arguments.manifest_sha256,
            )
    except StructuralClusterError as exc:
        _emit(exc.as_dict())
        return 2
    except Exception as exc:
        _emit(
            {
                "code": "STRUCTURAL_CLUSTER_EXECUTION_ERROR",
                "message": str(exc),
                "status": "EXECUTION_ERROR",
            }
        )
        return 2
    _emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
