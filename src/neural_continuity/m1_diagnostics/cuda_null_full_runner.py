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

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.cuda_null_full_attempt_journal import (
    FullCorpusJournalBlocked,
    append_attempt_event,
    verify_attempt_journal,
)
from neural_continuity.m1_diagnostics.cuda_null_full_corpus_package import (
    finalize_full_corpus_package,
    replay_full_corpus_package,
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


def _verify_or_recover_journal(
    journal: Path,
    checkpoint: Path,
    tip: str,
    authority_sha256: str,
    package_verifier: PackageVerifier,
) -> tuple[dict[str, Any], str]:
    """Recover only the single fsynced event allowed before tip replacement."""
    try:
        state = verify_attempt_journal(
            journal,
            external_tip_sha256=tip,
            authority_sha256=authority_sha256,
            package_verifier=package_verifier,
        )
        return state, tip
    except FullCorpusJournalBlocked as original:
        checkpoints = sorted(journal.glob("*.json")) if journal.is_dir() else []
        if not checkpoints:
            raise original
        latest = checkpoints[-1]
        try:
            record = json.loads(latest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise FullCorpusExecutionBlocked(
                "unanchored journal checkpoint cannot be decoded"
            ) from exc
        if not isinstance(record, dict) or record.get("previous_sha256") != tip:
            raise original
        recovered_tip = sha256_file(latest)
        state = verify_attempt_journal(
            journal,
            external_tip_sha256=recovered_tip,
            authority_sha256=authority_sha256,
            package_verifier=package_verifier,
        )
        _write_tip(checkpoint, authority_sha256, recovered_tip)
        return state, recovered_tip


def _open_attempt_process(journal: Path, state: dict[str, Any]) -> str:
    sequence = state["checkpoint_count"]
    try:
        record = json.loads((journal / f"{sequence:08d}.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FullCorpusExecutionBlocked("open attempt intent cannot be decoded") from exc
    process = record.get("process_instance_id") if isinstance(record, dict) else None
    if not isinstance(record, dict) or record.get("kind") != "attempt_started":
        raise FullCorpusExecutionBlocked("open attempt intent differs")
    if not isinstance(process, str):
        raise FullCorpusExecutionBlocked("open attempt process identity differs")
    return process


def _published_manifest(
    root: Path,
    authority_sha256: str,
    epoch: int,
    attempt: int,
    process_instance_id: str,
    predecessor: str | None,
) -> str | None:
    output = root / f"epoch-{epoch:04d}"
    if not output.exists():
        return None
    manifest_path = output / "artifact-manifest.json"
    if not manifest_path.is_file() or has_linked_ancestor(manifest_path):
        raise FullCorpusExecutionBlocked("published epoch manifest path is invalid")
    manifest = sha256_file(manifest_path)
    replay = replay_full_epoch_package(
        output / "replay-bundle.json",
        manifest,
        authority_sha256,
    )
    if (
        replay.get("replay_status") != "PASS"
        or replay.get("epoch_number") != epoch
        or replay.get("attempt_number") != attempt
        or replay.get("process_instance_id") != process_instance_id
        or replay.get("previous_completed_epoch_manifest_sha256") != predecessor
    ):
        raise FullCorpusExecutionBlocked("published epoch does not match open attempt")
    return manifest


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


def _finalize_or_replay_corpus(
    root: Path,
    checkpoint: Path,
    authority_sha256: str,
    final_checkpoint_tip_sha256: str,
    completed: dict[int, str],
) -> tuple[str, dict[str, Any]]:
    declarations = [
        {"epoch": epoch, "manifest_sha256": completed[epoch]} for epoch in range(1, 121)
    ]
    package = root / "full-corpus-package"
    if package.exists():
        manifest_path = package / "artifact-manifest.json"
        if (
            not package.is_dir()
            or has_linked_ancestor(package)
            or not manifest_path.is_file()
            or has_linked_ancestor(manifest_path)
        ):
            raise FullCorpusExecutionBlocked("existing full-corpus package path is invalid")
        try:
            manifest_sha256 = sha256_file(manifest_path)
        except OSError as exc:
            raise FullCorpusExecutionBlocked(
                "existing full-corpus manifest cannot be read"
            ) from exc
    else:
        package, manifest_sha256 = finalize_full_corpus_package(
            root,
            authority_sha256,
            final_checkpoint_tip_sha256,
            declarations,
        )
    replay = replay_full_corpus_package(
        package / "replay-bundle.json",
        manifest_sha256,
        authority_sha256,
        final_checkpoint_tip_sha256,
    )
    if replay.get("replay_status") != "PASS":
        raise FullCorpusExecutionBlocked(
            f"complete full-corpus replay blocked: {replay.get('reason', 'unknown reason')}"
        )
    final_anchor = checkpoint.with_name(f"{checkpoint.stem}.final-manifest.json")
    expected_anchor = {
        "kind": "m1_cuda_full_corpus_external_final_manifest",
        "version": "1.0.0",
        "authority_sha256": authority_sha256,
        "final_checkpoint_tip_sha256": final_checkpoint_tip_sha256,
        "artifact_manifest_sha256": manifest_sha256,
    }
    if final_anchor.exists():
        try:
            if not final_anchor.is_file() or has_linked_ancestor(final_anchor):
                raise OSError("external final anchor path invalid")
            observed_anchor = json.loads(final_anchor.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise FullCorpusExecutionBlocked("external final anchor cannot be decoded") from exc
        if observed_anchor != expected_anchor:
            raise FullCorpusExecutionBlocked("external final anchor differs")
    else:
        _write_json_atomic(final_anchor, expected_anchor)
    return manifest_sha256, replay


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
    state, tip = _verify_or_recover_journal(
        journal,
        checkpoint,
        tip,
        authority_sha256,
        package_verifier,
    )
    if state["open_attempt"] is not None:
        epoch, attempt = state["open_attempt"]
        process_instance_id = _open_attempt_process(journal, state)
        completed = state["completed_epoch_manifests"]
        predecessor = completed.get(epoch - 1) if epoch > 1 else None
        published = _published_manifest(
            root,
            authority_sha256,
            epoch,
            attempt,
            process_instance_id,
            predecessor,
        )
        if published is not None:
            tip = _persist_event(
                journal,
                checkpoint,
                tip,
                authority_sha256,
                {
                    "kind": "epoch_completed",
                    "epoch": epoch,
                    "attempt": attempt,
                    "package_manifest_sha256": published,
                },
                package_verifier,
            )
        else:
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
        manifest = _published_manifest(
            root,
            authority_sha256,
            epoch,
            attempt,
            process_instance_id,
            predecessor,
        )
        if manifest is None:
            summary = (
                child.stderr
                or ("child completed without publishing an epoch" if child.returncode == 0 else "")
                or "child process failed without stderr"
            )[-1024:]
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
        )
    completed = state["completed_epoch_manifests"]
    if set(completed) != set(range(1, 121)):
        raise FullCorpusExecutionBlocked("complete epoch manifest coverage differs")
    final_manifest, corpus_replay = _finalize_or_replay_corpus(
        root,
        checkpoint,
        authority_sha256,
        tip,
        completed,
    )
    return {
        "status": "CAPTURED_NOT_DECIDED",
        "epoch_count": 120,
        "checkpoint_tip_sha256": tip,
        "all_epoch_packages_replayed": True,
        "process_restart_variation_status": "REPLAY_VERIFIED",
        "family_unit_counts": corpus_replay["family_unit_counts"],
        "artifact_manifest_sha256": final_manifest,
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
