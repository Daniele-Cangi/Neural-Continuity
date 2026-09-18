"""Fail-closed attempt journal for the proposed CUDA full-corpus executor.

This module never loads a model and does not authorize a qualifying epoch.
The journal tip must be retained outside its directory by the caller.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from neural_continuity.m1_diagnostics.cuda_null_paths import has_linked_ancestor

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CHECKPOINT = re.compile(r"[0-9]{8}\.json\Z")
_VERSION = "1.0.0"
_EPOCH_COUNT = 120
_BASE_FIELDS = {
    "version",
    "sequence",
    "previous_sha256",
    "authority_sha256",
    "kind",
    "epoch",
    "attempt",
}
_EXTRA_FIELDS = {
    "attempt_started": {"process_instance_id", "runtime_identity_sha256"},
    "attempt_failed": {"technical_status", "error_summary"},
    "attempt_interrupted": {"technical_status", "error_summary"},
    "epoch_completed": {"package_manifest_sha256"},
}
PackageVerifier = Callable[[int, str], bool]


class FullCorpusJournalBlocked(ValueError):
    """The journal or its external anchor did not verify."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise FullCorpusJournalBlocked(reason)


def _digest(value: str, label: str) -> None:
    _require(_SHA256.fullmatch(value) is not None, f"{label} must be a SHA-256 digest")


def _canonical(record: dict[str, Any]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _root_check(root: Path) -> None:
    _require(root.name == "attempt-journal", "journal directory name differs")
    try:
        _require(not has_linked_ancestor(root.parent), "journal parent contains a link")
        if root.exists() or root.is_symlink():
            _require(root.is_dir(), "journal path is not a directory")
            _require(not has_linked_ancestor(root), "journal path contains a link")
    except OSError as exc:
        raise FullCorpusJournalBlocked("journal path cannot be verified") from exc


def _new_state() -> dict[str, Any]:
    return {
        "next_epoch": 1,
        "next_attempt": 1,
        "open_attempt": None,
        "completed": {},
        "processes": set(),
    }


def _reduce(record: dict[str, Any], state: dict[str, Any]) -> None:
    kind = record.get("kind")
    _require(kind in _EXTRA_FIELDS, "unknown journal event")
    _require(set(record) == _BASE_FIELDS | _EXTRA_FIELDS[kind], "journal field set differs")
    _require(record["version"] == _VERSION, "journal version differs")
    _require(
        type(record["epoch"]) is int and 1 <= record["epoch"] <= _EPOCH_COUNT,
        "journal epoch invalid",
    )
    _require(type(record["attempt"]) is int and record["attempt"] >= 1, "attempt invalid")
    _digest(record["previous_sha256"], "previous checkpoint")
    _digest(record["authority_sha256"], "authority")
    epoch = record["epoch"]
    attempt = record["attempt"]
    _require(epoch == state["next_epoch"], "journal epoch order differs")
    if kind == "attempt_started":
        _require(state["open_attempt"] is None, "prior attempt has no outcome")
        _require(attempt == state["next_attempt"], "attempt order differs")
        process = record["process_instance_id"]
        _require(isinstance(process, str), "process identity missing")
        try:
            parsed = uuid.UUID(process)
        except (ValueError, AttributeError) as exc:
            raise FullCorpusJournalBlocked("process identity malformed") from exc
        _require(str(parsed) == process, "process identity is not canonical")
        _require(process not in state["processes"], "process reused across attempts")
        state["processes"].add(process)
        _digest(record["runtime_identity_sha256"], "runtime identity")
        state["open_attempt"] = (epoch, attempt)
    else:
        _require(state["open_attempt"] == (epoch, attempt), "outcome has no matching intent")
        if kind == "epoch_completed":
            _digest(record["package_manifest_sha256"], "epoch package manifest")
            state["completed"][epoch] = record["package_manifest_sha256"]
            state["next_epoch"] += 1
            state["next_attempt"] = 1
        else:
            _require(
                record["technical_status"] == "EXECUTION_ERROR",
                "technical failure was recorded as a scientific result",
            )
            summary = record["error_summary"]
            _require(
                isinstance(summary, str) and 0 < len(summary) <= 1024,
                "technical error summary invalid",
            )
            state["next_attempt"] += 1
        state["open_attempt"] = None


def _read_chain(root: Path, authority_sha256: str) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    _digest(authority_sha256, "authority")
    _root_check(root)
    state = _new_state()
    if not root.exists():
        return [], authority_sha256, state
    entries = sorted(root.iterdir(), key=lambda item: item.name)
    _require(all(entry.is_file() and not entry.is_symlink() for entry in entries), "invalid journal entry")
    _require(all(_CHECKPOINT.fullmatch(entry.name) for entry in entries), "unexpected journal entry")
    records: list[dict[str, Any]] = []
    previous = authority_sha256
    for sequence, path in enumerate(entries, start=1):
        _require(path.name == f"{sequence:08d}.json", "missing or duplicated checkpoint")
        try:
            raw = path.read_bytes()
            record = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise FullCorpusJournalBlocked("checkpoint cannot be decoded") from exc
        _require(isinstance(record, dict), "checkpoint must be an object")
        _require(raw == _canonical(record), "checkpoint encoding is not canonical")
        _require(
            type(record.get("sequence")) is int and record["sequence"] == sequence,
            "checkpoint sequence differs",
        )
        _require(record.get("previous_sha256") == previous, "checkpoint chain is broken")
        _require(record.get("authority_sha256") == authority_sha256, "authority changed")
        _reduce(record, state)
        previous = hashlib.sha256(raw).hexdigest()
        records.append(record)
    return records, previous, state


def verify_attempt_journal(
    root: Path,
    *,
    external_tip_sha256: str,
    authority_sha256: str,
    package_verifier: PackageVerifier | None = None,
) -> dict[str, Any]:
    """Replay the exact chain; optionally replay every completed epoch package."""
    _digest(external_tip_sha256, "external checkpoint tip")
    records, tip, state = _read_chain(root, authority_sha256)
    _require(tip == external_tip_sha256, "external checkpoint tip differs")
    if package_verifier is not None:
        for epoch, manifest in state["completed"].items():
            _require(package_verifier(epoch, manifest) is True, "completed epoch replay blocked")
    return {
        "status": "ATTEMPT_JOURNAL_REPLAY_PASS",
        "checkpoint_count": len(records),
        "checkpoint_tip_sha256": tip,
        "next_epoch": state["next_epoch"],
        "next_attempt": state["next_attempt"],
        "open_attempt": state["open_attempt"],
        "completed_epoch_manifests": dict(state["completed"]),
        "completed_packages_replayed": package_verifier is not None,
        "model_loaded": False,
        "execution_authorized": False,
    }


def append_attempt_event(
    root: Path,
    *,
    external_tip_sha256: str,
    authority_sha256: str,
    event: dict[str, Any],
    package_verifier: PackageVerifier | None = None,
) -> str:
    """Atomically append one event after checking its external chain anchor.

    The returned SHA-256 is the new external tip. Persist it outside the journal
    before any further event or resume. A stale lock or partial temp file blocks.
    """
    _require(isinstance(event, dict), "journal event must be an object")
    _root_check(root)
    lock = root.parent / ".full-corpus-journal.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except OSError as exc:
        raise FullCorpusJournalBlocked("journal writer lock unavailable") from exc
    os.close(descriptor)
    try:
        verified = verify_attempt_journal(
            root,
            external_tip_sha256=external_tip_sha256,
            authority_sha256=authority_sha256,
            package_verifier=package_verifier,
        )
        if event.get("kind") == "epoch_completed":
            _require(package_verifier is not None, "epoch completion requires package replay")
            epoch = event.get("epoch")
            manifest = event.get("package_manifest_sha256")
            _require(type(epoch) is int and isinstance(manifest, str), "completion fields invalid")
            _digest(manifest, "epoch package manifest")
            _require(package_verifier(epoch, manifest) is True, "new epoch package replay blocked")
        record = {
            **event,
            "version": _VERSION,
            "sequence": verified["checkpoint_count"] + 1,
            "previous_sha256": external_tip_sha256,
            "authority_sha256": authority_sha256,
        }
        state = _new_state()
        records, _tip, _old_state = _read_chain(root, authority_sha256)
        for prior in records:
            _reduce(prior, state)
        _reduce(record, state)
        raw = _canonical(record)
        root.mkdir(exist_ok=True)
        target = root / f"{record['sequence']:08d}.json"
        _require(not target.exists(), "checkpoint already exists")
        descriptor, pending = tempfile.mkstemp(prefix=".pending-", suffix=".tmp", dir=root)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(pending, target)
        except OSError as exc:
            raise FullCorpusJournalBlocked("checkpoint could not be created exclusively") from exc
        os.unlink(pending)
        return hashlib.sha256(raw).hexdigest()
    finally:
        lock.unlink()
