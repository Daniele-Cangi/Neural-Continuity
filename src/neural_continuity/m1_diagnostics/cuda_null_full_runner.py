"""Resumable isolated-process controller for the authorized CUDA full corpus."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes
from neural_continuity.m1_diagnostics.cuda_null_full_attempt_journal import (
    append_attempt_event,
    verify_attempt_journal,
)
from neural_continuity.m1_diagnostics.cuda_null_full_epoch_capture import capture_full_epoch
from neural_continuity.m1_diagnostics.cuda_null_full_epoch_package import (
    replay_full_epoch_package,
)
from neural_continuity.m1_diagnostics.cuda_null_full_execution_authority import (
    FullCorpusExecutionAuthority,
    FullCorpusExecutionBlocked,
    verify_full_corpus_execution_authority,
)
from neural_continuity.m1_diagnostics.cuda_null_paths import has_linked_ancestor

PackageVerifier = Callable[[int, str], bool]
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    if has_linked_ancestor(path.absolute()):
        raise FullCorpusExecutionBlocked("external checkpoint path contains a link")
    descriptor, pending = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    finally:
        if os.path.exists(pending):
            os.unlink(pending)


def _write_tip(path: Path, authority_sha256: str, tip_sha256: str) -> None:
    _write_json_atomic(
        path,
        {
            "kind": "m1_cuda_full_corpus_external_checkpoint_tip",
            "version": "1.0.0",
            "authority_sha256": authority_sha256,
            "checkpoint_tip_sha256": tip_sha256,
        },
    )


def _read_tip(path: Path, authority_sha256: str) -> str:
    try:
        if not path.is_file() or has_linked_ancestor(path):
            raise ValueError("external checkpoint path invalid")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise FullCorpusExecutionBlocked("external checkpoint tip cannot be verified") from exc
    expected = {
        "kind": "m1_cuda_full_corpus_external_checkpoint_tip",
        "version": "1.0.0",
        "authority_sha256": authority_sha256,
        "checkpoint_tip_sha256": (
            value.get("checkpoint_tip_sha256") if isinstance(value, dict) else None
        ),
    }
    tip = expected["checkpoint_tip_sha256"]
    if value != expected or not isinstance(tip, str) or _SHA256.fullmatch(tip) is None:
        raise FullCorpusExecutionBlocked("external checkpoint tip schema differs")
    return tip


def _package_verifier(root: Path, authority_sha256: str) -> PackageVerifier:
    def verify(epoch: int, manifest_sha256: str) -> bool:
        replay = replay_full_epoch_package(
            root / f"epoch-{epoch:04d}" / "replay-bundle.json",
            manifest_sha256,
            authority_sha256,
        )
        return replay.get("replay_status") == "PASS" and replay.get("epoch_number") == epoch

    return verify


def _initialize(authority: FullCorpusExecutionAuthority) -> tuple[Path, Path, str]:
    root = authority.paths["output_root"]
    checkpoint = authority.paths["external_checkpoint_tip"]
    root.mkdir()
    _write_json_atomic(
        root / "run-authority.json",
        {
            "kind": "m1_cuda_full_corpus_run_authority",
            "version": "1.0.0",
            "authority_sha256": authority.spec_sha256,
            "epoch_count": 120,
            "scientific_decision": "NOT_EVALUATED",
        },
    )
    _write_tip(checkpoint, authority.spec_sha256, authority.spec_sha256)
    return root, checkpoint, authority.spec_sha256


def _resume(authority: FullCorpusExecutionAuthority) -> tuple[Path, Path, str]:
    root = authority.paths["output_root"]
    checkpoint = authority.paths["external_checkpoint_tip"]
    expected = {
        "kind": "m1_cuda_full_corpus_run_authority",
        "version": "1.0.0",
        "authority_sha256": authority.spec_sha256,
        "epoch_count": 120,
        "scientific_decision": "NOT_EVALUATED",
    }
    try:
        run_authority = root / "run-authority.json"
        if not run_authority.is_file() or has_linked_ancestor(run_authority):
            raise OSError("run authority path invalid")
        observed = json.loads(run_authority.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FullCorpusExecutionBlocked("run authority cannot be verified") from exc
    if observed != expected:
        raise FullCorpusExecutionBlocked("run authority differs")
    return root, checkpoint, _read_tip(checkpoint, authority.spec_sha256)


def _persist_event(
    journal: Path,
    checkpoint: Path,
    tip: str,
    authority_sha256: str,
    event: dict[str, Any],
    package_verifier: PackageVerifier,
) -> str:
    new_tip = append_attempt_event(
        journal,
        external_tip_sha256=tip,
        authority_sha256=authority_sha256,
        event=event,
        package_verifier=package_verifier,
    )
    _write_tip(checkpoint, authority_sha256, new_tip)
    return new_tip


def _persist_failure(
    journal: Path,
    checkpoint: Path,
    tip: str,
    authority_sha256: str,
    epoch: int,
    attempt: int,
    summary: str,
    package_verifier: PackageVerifier,
) -> str:
    return _persist_event(
        journal,
        checkpoint,
        tip,
        authority_sha256,
        {
            "kind": "attempt_failed",
            "epoch": epoch,
            "attempt": attempt,
            "technical_status": "EXECUTION_ERROR",
            "error_summary": summary[-1024:],
        },
        package_verifier,
    )


def _run_child(
    authority_sha256: str,
    epoch: int,
    attempt: int,
    process_instance_id: str,
    predecessor: str | None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "neural_continuity.m1_diagnostics.cuda_null_full_runner",
        "capture-one",
        "--authority-sha256",
        authority_sha256,
        "--epoch",
        str(epoch),
        "--attempt",
        str(attempt),
        "--process-instance-id",
        process_instance_id,
    ]
    if predecessor is not None:
        command.extend(("--previous-completed-manifest-sha256", predecessor))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
        timeout=3600,
    )


def run_full_corpus(authority_sha256: str, *, resume: bool) -> dict[str, Any]:
    authority = verify_full_corpus_execution_authority(authority_sha256, resume=resume)
    root, checkpoint, tip = _resume(authority) if resume else _initialize(authority)
    journal = root / "attempt-journal"
    package_verifier = _package_verifier(root, authority_sha256)
    state = verify_attempt_journal(
        journal,
        external_tip_sha256=tip,
        authority_sha256=authority_sha256,
        package_verifier=package_verifier,
    )
    if state["open_attempt"] is not None:
        epoch, attempt = state["open_attempt"]
        tip = _persist_event(
            journal,
            checkpoint,
            tip,
            authority_sha256,
            {
                "kind": "attempt_interrupted",
                "epoch": epoch,
                "attempt": attempt,
                "technical_status": "EXECUTION_ERROR",
                "error_summary": "controller resumed after an unclosed child attempt",
            },
            package_verifier,
        )
        state = verify_attempt_journal(
            journal,
            external_tip_sha256=tip,
            authority_sha256=authority_sha256,
            package_verifier=package_verifier,
        )

    while state["next_epoch"] <= 120:
        epoch = state["next_epoch"]
        attempt = state["next_attempt"]
        process_instance_id = str(uuid.uuid4())
        completed = state["completed_epoch_manifests"]
        predecessor = completed.get(epoch - 1) if epoch > 1 else None
        tip = _persist_event(
            journal,
            checkpoint,
            tip,
            authority_sha256,
            {
                "kind": "attempt_started",
                "epoch": epoch,
                "attempt": attempt,
                "process_instance_id": process_instance_id,
                "runtime_identity_sha256": authority.sentinel.runtime_identity_sha256,
            },
            package_verifier,
        )
        try:
            child = _run_child(
                authority_sha256,
                epoch,
                attempt,
                process_instance_id,
                predecessor,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            child = subprocess.CompletedProcess([], 1, "", str(exc))
        if child.returncode != 0:
            summary = (child.stderr or "child process failed without stderr")[-1024:]
            tip = _persist_failure(
                journal,
                checkpoint,
                tip,
                authority_sha256,
                epoch,
                attempt,
                summary,
                package_verifier,
            )
            raise FullCorpusExecutionBlocked(f"epoch {epoch} attempt {attempt} failed: {summary}")
        try:
            result = json.loads(child.stdout)
        except (TypeError, json.JSONDecodeError) as exc:
            summary = "child result cannot be decoded"
            tip = _persist_failure(
                journal,
                checkpoint,
                tip,
                authority_sha256,
                epoch,
                attempt,
                summary,
                package_verifier,
            )
            raise FullCorpusExecutionBlocked("child result cannot be decoded") from exc
        manifest = result.get("manifest_sha256") if isinstance(result, dict) else None
        if (
            not isinstance(result, dict)
            or result.get("epoch_number") != epoch
            or result.get("attempt_number") != attempt
            or result.get("process_instance_id") != process_instance_id
            or not isinstance(manifest, str)
            or not package_verifier(epoch, manifest)
        ):
            summary = "child result or package replay differs"
            tip = _persist_failure(
                journal,
                checkpoint,
                tip,
                authority_sha256,
                epoch,
                attempt,
                summary,
                package_verifier,
            )
            raise FullCorpusExecutionBlocked("child result or package replay differs")
        tip = _persist_event(
            journal,
            checkpoint,
            tip,
            authority_sha256,
            {
                "kind": "epoch_completed",
                "epoch": epoch,
                "attempt": attempt,
                "package_manifest_sha256": manifest,
            },
            package_verifier,
        )
        print(json.dumps({"epoch": epoch, "checkpoint_tip_sha256": tip}), flush=True)
        state = verify_attempt_journal(
            journal,
            external_tip_sha256=tip,
            authority_sha256=authority_sha256,
            package_verifier=package_verifier,
        )
    return {
        "status": "FULL_CORPUS_CAPTURE_COMPLETE_NOT_DECIDED",
        "epoch_count": 120,
        "checkpoint_tip_sha256": tip,
        "all_epoch_packages_replayed": True,
        "scientific_decision": "NOT_EVALUATED",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "resume", "capture-one"))
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--process-instance-id")
    parser.add_argument("--previous-completed-manifest-sha256")
    args = parser.parse_args()
    if args.mode in {"run", "resume"}:
        if any(
            value is not None
            for value in (
                args.epoch,
                args.attempt,
                args.process_instance_id,
                args.previous_completed_manifest_sha256,
            )
        ):
            parser.error("run and resume accept only the authority SHA-256")
        result = run_full_corpus(args.authority_sha256, resume=args.mode == "resume")
    else:
        if args.epoch is None or args.attempt is None or args.process_instance_id is None:
            parser.error("capture-one requires epoch, attempt, and process identity")
        result = capture_full_epoch(
            external_authority_sha256=args.authority_sha256,
            epoch_number=args.epoch,
            attempt_number=args.attempt,
            process_instance_id=args.process_instance_id,
            previous_completed_epoch_manifest_sha256=(args.previous_completed_manifest_sha256),
        )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
