from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from neural_continuity.m1_diagnostics.cuda_null_full_attempt_journal import (
    FullCorpusJournalBlocked,
    append_attempt_event,
    verify_attempt_journal,
)

AUTHORITY = "a" * 64
RUNTIME = "b" * 64
MANIFEST = "c" * 64


def _start(epoch: int, attempt: int) -> dict[str, object]:
    return {
        "kind": "attempt_started",
        "epoch": epoch,
        "attempt": attempt,
        "process_instance_id": str(uuid.uuid4()),
        "runtime_identity_sha256": RUNTIME,
    }


def _append(
    root: Path,
    tip: str,
    event: dict[str, object],
    *,
    verifier=None,
) -> str:
    return append_attempt_event(
        root,
        external_tip_sha256=tip,
        authority_sha256=AUTHORITY,
        event=event,
        package_verifier=verifier,
    )


def test_failed_attempt_can_resume_only_same_epoch_with_next_attempt(tmp_path: Path) -> None:
    root = tmp_path / "attempt-journal"
    tip = _append(root, AUTHORITY, _start(1, 1))
    tip = _append(
        root,
        tip,
        {
            "kind": "attempt_failed",
            "epoch": 1,
            "attempt": 1,
            "technical_status": "EXECUTION_ERROR",
            "error_summary": "synthetic interruption",
        },
    )
    tip = _append(root, tip, _start(1, 2))
    result = verify_attempt_journal(
        root,
        external_tip_sha256=tip,
        authority_sha256=AUTHORITY,
    )
    assert result["next_epoch"] == 1
    assert result["next_attempt"] == 2
    assert result["open_attempt"] == (1, 2)


def test_stale_external_tip_blocks_append(tmp_path: Path) -> None:
    root = tmp_path / "attempt-journal"
    _append(root, AUTHORITY, _start(1, 1))
    with pytest.raises(FullCorpusJournalBlocked, match="external checkpoint tip differs"):
        _append(root, AUTHORITY, _start(1, 1))


def test_outcome_without_matching_intent_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "attempt-journal"
    with pytest.raises(FullCorpusJournalBlocked, match="matching intent"):
        _append(
            root,
            AUTHORITY,
            {
                "kind": "attempt_failed",
                "epoch": 1,
                "attempt": 1,
                "technical_status": "EXECUTION_ERROR",
                "error_summary": "synthetic failure",
            },
        )


def test_completion_requires_package_replay_and_advances_epoch(tmp_path: Path) -> None:
    root = tmp_path / "attempt-journal"
    tip = _append(root, AUTHORITY, _start(1, 1))
    event = {
        "kind": "epoch_completed",
        "epoch": 1,
        "attempt": 1,
        "package_manifest_sha256": MANIFEST,
    }
    with pytest.raises(FullCorpusJournalBlocked, match="requires package replay"):
        _append(root, tip, event)

    def verifier(epoch: int, manifest: str) -> bool:
        return epoch == 1 and manifest == MANIFEST

    tip = _append(root, tip, event, verifier=verifier)
    result = verify_attempt_journal(
        root,
        external_tip_sha256=tip,
        authority_sha256=AUTHORITY,
        package_verifier=verifier,
    )
    assert result["next_epoch"] == 2
    assert result["next_attempt"] == 1
    assert result["completed_epoch_manifests"] == {1: MANIFEST}
    assert result["completed_packages_replayed"] is True


def test_declared_completed_package_failure_blocks_replay(tmp_path: Path) -> None:
    root = tmp_path / "attempt-journal"
    tip = _append(root, AUTHORITY, _start(1, 1))

    def passing(epoch: int, manifest: str) -> bool:
        return epoch == 1 and manifest == MANIFEST

    tip = _append(
        root,
        tip,
        {
            "kind": "epoch_completed",
            "epoch": 1,
            "attempt": 1,
            "package_manifest_sha256": MANIFEST,
        },
        verifier=passing,
    )
    with pytest.raises(FullCorpusJournalBlocked, match="completed epoch replay blocked"):
        verify_attempt_journal(
            root,
            external_tip_sha256=tip,
            authority_sha256=AUTHORITY,
            package_verifier=lambda _epoch, _manifest: False,
        )
