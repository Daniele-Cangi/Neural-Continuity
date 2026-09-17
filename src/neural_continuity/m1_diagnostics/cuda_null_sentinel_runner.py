"""Strict 120-child-process sentinel orchestrator; no scientific qualification."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.cuda_null_sentinel_capture import capture_epoch
from neural_continuity.m1_diagnostics.cuda_null_sentinel_chain import append_checkpoint
from neural_continuity.m1_diagnostics.cuda_null_sentinel_chain_replay import (
    replay_cuda_sentinel_chain_packages,
)
from neural_continuity.m1_diagnostics.cuda_null_sentinel_epoch_package import (
    replay_cuda_sentinel_epoch,
)
from neural_continuity.m1_diagnostics.cuda_null_sentinel_execution_authority import (
    SentinelExecutionBlocked,
    verify_sentinel_execution_authority,
)
from neural_continuity.m1_diagnostics.cuda_null_source_preflight_inputs import (
    CORPUS_SHA256,
    QUERY_SHA256,
    _load_jsonl,
)
from neural_continuity.m1_diagnostics.measurement_null_sentinel_authority import (
    _select_sentinel_ids,
)


def _replay_bound_epochs(
    root: Path, count: int, tip: str, authority_sha256: str, dataset: Path
) -> None:
    replay = replay_cuda_sentinel_chain_packages(
        root, expected_count=count, external_tip_sha256=tip
    )
    if replay.get("status") != "TECHNICAL_REPLAY_PASS_NOT_QUALIFYING":
        raise SentinelExecutionBlocked("sentinel chain package replay blocked")
    if (
        sha256_file(dataset / "corpus.jsonl") != CORPUS_SHA256
        or sha256_file(dataset / "roles" / "measurement_null.queries.jsonl") != QUERY_SHA256
    ):
        raise SentinelExecutionBlocked("frozen sentinel population changed before replay")
    documents = _load_jsonl(dataset / "corpus.jsonl", "document_id", 5183)
    queries = _load_jsonl(dataset / "roles" / "measurement_null.queries.jsonl", "query_id", 81)
    expected_documents = list(_select_sentinel_ids([item[0] for item in documents]))
    expected_queries = [item[0] for item in queries]
    prior = authority_sha256
    process_ids: set[str] = set()
    for number in range(1, count + 1):
        epoch = root / f"epoch-{number:04d}"
        plan = json.loads((epoch / "epoch-plan.json").read_text(encoding="utf-8"))
        runtime = json.loads((epoch / "runtime-inventory.json").read_text(encoding="utf-8"))
        if (
            plan["document_ids"] != expected_documents
            or plan["query_ids"] != expected_queries
            or plan["previous_checkpoint_sha256"] != prior
            or runtime["process_instance_id"] in process_ids
        ):
            raise SentinelExecutionBlocked(f"epoch {number}: authority binding differs")
        process_ids.add(runtime["process_instance_id"])
        prior = sha256_file(root / "chain" / f"epoch-{number:04d}.json")
    if prior != tip:
        raise SentinelExecutionBlocked("chain tip changed during authority-bound replay")


def run_sentinel(authority_sha256: str) -> dict[str, Any]:
    authority = verify_sentinel_execution_authority(authority_sha256)
    root = authority.paths["output_root"]
    if root.exists():
        raise SentinelExecutionBlocked(
            "output root already exists; refusing overwrite or unanchored resume"
        )
    if root.parent.drive.upper() != "D:":
        raise SentinelExecutionBlocked("sentinel output must remain on declared D drive")
    root.mkdir()
    (root / "run-authority.json").write_bytes(
        canonical_json_bytes(
            {"authority_sha256": authority_sha256, "epoch_count": 120, "scope": "sentinel_only"}
        )
        + b"\n"
    )
    previous: str | None = None
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    for epoch_number in range(1, 121):
        command = [
            sys.executable,
            "-m",
            "neural_continuity.m1_diagnostics.cuda_null_sentinel_runner",
            "capture-one",
            "--authority-sha256",
            authority_sha256,
            "--epoch",
            str(epoch_number),
        ]
        if previous is not None:
            command.extend(("--previous-checkpoint-sha256", previous))
        completed = subprocess.run(
            command, capture_output=True, text=True, env=environment, timeout=3600
        )
        if completed.returncode != 0:
            raise SentinelExecutionBlocked(
                f"epoch {epoch_number} failed closed: {completed.stderr[-4000:]}"
            )
        result = json.loads(completed.stdout)
        directory = root / f"epoch-{epoch_number:04d}"
        manifest = sha256_file(directory / "artifact-manifest.json")
        if result.get("manifest_sha256") != manifest:
            raise SentinelExecutionBlocked(f"epoch {epoch_number}: child manifest differs")
        replay = replay_cuda_sentinel_epoch(directory / "replay-bundle.json", manifest)
        if replay.get("replay_status") != "PASS":
            raise SentinelExecutionBlocked(f"epoch {epoch_number}: child replay blocked")
        previous = append_checkpoint(
            root,
            epoch_number=epoch_number,
            external_epoch_manifest_sha256=manifest,
            external_previous_checkpoint_sha256=previous,
        )
        print(json.dumps({"epoch": epoch_number, "checkpoint_sha256": previous}), flush=True)
    assert previous is not None
    _replay_bound_epochs(root, 120, previous, authority_sha256, authority.paths["dataset_root"])
    return {
        "status": "TECHNICAL_REPLAY_PASS_NOT_QUALIFYING",
        "epoch_count": 120,
        "tip_sha256": previous,
        "scientific_decision": "NOT_EVALUATED",
        "full_corpus_started": False,
        "int8_executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "capture-one"))
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--previous-checkpoint-sha256")
    arguments = parser.parse_args()
    if arguments.mode == "run":
        if arguments.epoch is not None or arguments.previous_checkpoint_sha256 is not None:
            parser.error("run mode does not accept an epoch or predecessor")
        result = run_sentinel(arguments.authority_sha256)
    else:
        if arguments.epoch is None:
            parser.error("capture-one requires --epoch")
        result = capture_epoch(
            external_authority_sha256=arguments.authority_sha256,
            epoch_number=arguments.epoch,
            previous_checkpoint_sha256=arguments.previous_checkpoint_sha256,
        )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
