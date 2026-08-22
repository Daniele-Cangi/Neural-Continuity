from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from neural_continuity.m1_diagnostics.measurement_null_sentinel_executor import (
    prepare_sentinel_run,
    run_next_sentinel_epoch,
    sentinel_status,
)
from neural_continuity.m1_teacher_evidence import TeacherEvidenceError


def _authority_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--transition-a-bundle", required=True)
    parser.add_argument("--extension-plan-bundle", required=True)
    parser.add_argument("--extension-plan-manifest-sha256", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and execute one process-isolated M1 measurement-null " "sentinel epoch."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare",
        help="Verify all authorities and create an unexecuted sentinel run.",
    )
    _authority_arguments(prepare)
    prepare.add_argument("--output", required=True)

    run_next = commands.add_parser(
        "run-next",
        help="Verify all authorities and execute exactly one next sentinel epoch.",
    )
    _authority_arguments(run_next)
    run_next.add_argument("--run", required=True)
    run_next.add_argument("--root-manifest-sha256", required=True)
    run_next.add_argument("--expected-checkpoint-sha256", required=True)

    status = commands.add_parser(
        "status",
        help="Replay checkpoint integrity without loading a model.",
    )
    status.add_argument("--run", required=True)
    status.add_argument("--root-manifest-sha256", required=True)
    status.add_argument("--expected-checkpoint-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_sentinel_run(
                config_path=args.config,
                dataset_directory=args.dataset,
                transition_a_bundle=args.transition_a_bundle,
                extension_plan_bundle=args.extension_plan_bundle,
                extension_plan_manifest_sha256=(args.extension_plan_manifest_sha256),
                output_directory=args.output,
            )
        elif args.command == "run-next":
            result = run_next_sentinel_epoch(
                config_path=args.config,
                dataset_directory=args.dataset,
                transition_a_bundle=args.transition_a_bundle,
                extension_plan_bundle=args.extension_plan_bundle,
                extension_plan_manifest_sha256=(args.extension_plan_manifest_sha256),
                run_directory=args.run,
                root_manifest_sha256=args.root_manifest_sha256,
                expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            )
        else:
            result = sentinel_status(
                run_directory=args.run,
                root_manifest_sha256=args.root_manifest_sha256,
                expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            )
    except TeacherEvidenceError as exc:
        print(
            json.dumps(
                {
                    "status": exc.status,
                    "error": {"code": exc.code, "message": exc.message},
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
