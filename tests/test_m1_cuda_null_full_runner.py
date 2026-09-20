from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from neural_continuity.m1_diagnostics import cuda_null_full_runner as runner
from neural_continuity.m1_diagnostics.cuda_null_full_attempt_journal import (
    verify_attempt_journal,
)

AUTHORITY = "a" * 64
RUNTIME = "b" * 64
MANIFEST = "c" * 64


def _authority(tmp_path: Path):
    return SimpleNamespace(
        spec_sha256=AUTHORITY,
        paths={
            "output_root": tmp_path / "run",
            "external_checkpoint_tip": tmp_path / "checkpoint.json",
        },
        sentinel=SimpleNamespace(runtime_identity_sha256=RUNTIME),
    )


def test_external_tip_round_trips_and_rejects_wrong_authority(tmp_path: Path) -> None:
    path = tmp_path / "tip.json"
    runner._write_tip(path, AUTHORITY, MANIFEST)
    assert runner._read_tip(path, AUTHORITY) == MANIFEST
    with pytest.raises(runner.FullCorpusExecutionBlocked):
        runner._read_tip(path, "d" * 64)


def test_attempt_intent_is_durable_before_child_and_completion_replays(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    authority = _authority(tmp_path)
    root, checkpoint, tip = runner._initialize(authority)
    journal = root / "attempt-journal"

    def verifier(epoch: int, manifest: str) -> bool:
        return epoch == 1 and manifest == MANIFEST

    monkeypatch.setattr(runner, "_package_verifier", lambda *_args: verifier)

    def child(*_args):
        current_tip = runner._read_tip(checkpoint, AUTHORITY)
        state = verify_attempt_journal(
            journal,
            external_tip_sha256=current_tip,
            authority_sha256=AUTHORITY,
            package_verifier=verifier,
        )
        assert state["open_attempt"] == (1, 1)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "epoch_number": 1,
                    "attempt_number": 1,
                    "process_instance_id": _args[3],
                    "manifest_sha256": MANIFEST,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(runner, "_run_child", child)
    monkeypatch.setattr(runner, "_published_manifest", lambda *_args: MANIFEST)
    original_verify = runner.verify_attempt_journal
    calls = 0

    def bounded_verify(*args, **kwargs):
        nonlocal calls
        result = original_verify(*args, **kwargs)
        calls += 1
        if result["next_epoch"] == 2:
            result["next_epoch"] = 121
        return result

    monkeypatch.setattr(runner, "verify_attempt_journal", bounded_verify)
    monkeypatch.setattr(
        runner,
        "verify_full_corpus_execution_authority",
        lambda *_args, **_kwargs: authority,
    )
    result = runner.run_full_corpus(AUTHORITY, resume=True)
    assert result["status"] == "FULL_CORPUS_EPOCH_CAPTURE_COMPLETE_AGGREGATION_PENDING"
    assert calls >= 2
    final_tip = runner._read_tip(checkpoint, AUTHORITY)
    state = original_verify(
        journal,
        external_tip_sha256=final_tip,
        authority_sha256=AUTHORITY,
        package_verifier=verifier,
    )
    assert state["completed_epoch_manifests"] == {1: MANIFEST}


def test_resume_closes_unfinished_attempt_as_execution_error(tmp_path: Path) -> None:
    authority = _authority(tmp_path)
    root, checkpoint, tip = runner._initialize(authority)
    journal = root / "attempt-journal"
    process = "12345678-1234-5678-1234-567812345678"
    tip = runner._persist_event(
        journal,
        checkpoint,
        tip,
        AUTHORITY,
        {
            "kind": "attempt_started",
            "epoch": 1,
            "attempt": 1,
            "process_instance_id": process,
            "runtime_identity_sha256": RUNTIME,
        },
        lambda _epoch, _manifest: True,
    )
    state = verify_attempt_journal(
        journal,
        external_tip_sha256=tip,
        authority_sha256=AUTHORITY,
    )
    assert state["open_attempt"] == (1, 1)


def test_invalid_child_result_closes_attempt_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    authority = _authority(tmp_path)
    runner._initialize(authority)
    monkeypatch.setattr(
        runner,
        "verify_full_corpus_execution_authority",
        lambda *_args, **_kwargs: authority,
    )
    monkeypatch.setattr(
        runner,
        "_run_child",
        lambda *_args: SimpleNamespace(returncode=0, stdout="not-json", stderr=""),
    )
    monkeypatch.setattr(runner, "_published_manifest", lambda *_args: None)
    with pytest.raises(
        runner.FullCorpusExecutionBlocked,
        match="child completed without publishing an epoch",
    ):
        runner.run_full_corpus(AUTHORITY, resume=True)
    tip = runner._read_tip(authority.paths["external_checkpoint_tip"], AUTHORITY)
    state = verify_attempt_journal(
        authority.paths["output_root"] / "attempt-journal",
        external_tip_sha256=tip,
        authority_sha256=AUTHORITY,
    )
    assert state["open_attempt"] is None
    assert state["next_epoch"] == 1
    assert state["next_attempt"] == 2


def test_resume_recovers_fsynced_checkpoint_before_external_tip(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path)
    root, checkpoint, old_tip = runner._initialize(authority)
    journal = root / "attempt-journal"
    new_tip = runner.append_attempt_event(
        journal,
        external_tip_sha256=old_tip,
        authority_sha256=AUTHORITY,
        event={
            "kind": "attempt_started",
            "epoch": 1,
            "attempt": 1,
            "process_instance_id": "12345678-1234-5678-1234-567812345678",
            "runtime_identity_sha256": RUNTIME,
        },
    )
    state, recovered_tip = runner._verify_or_recover_journal(
        journal,
        checkpoint,
        old_tip,
        AUTHORITY,
        lambda _epoch, _manifest: True,
    )
    assert recovered_tip == new_tip
    assert runner._read_tip(checkpoint, AUTHORITY) == new_tip
    assert state["open_attempt"] == (1, 1)
