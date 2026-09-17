"""Non-executing, tamper-evident index for CUDA sentinel epoch checkpoints.

This checks the sequence and external file hashes, not the epoch package semantics.
It cannot authorize CUDA execution or qualify measurement-null evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

MAX_EPOCHS = 120
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = {
    "kind",
    "version",
    "epoch_number",
    "epoch_manifest",
    "epoch_manifest_sha256",
    "previous_checkpoint_sha256",
    "epoch_replay_verified",
    "execution_authorized",
    "scientific_decision",
}


class CudaNullSentinelChainBlocked(ValueError):
    """The declared checkpoint sequence or external root cannot be verified."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise CudaNullSentinelChainBlocked(reason)


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CudaNullSentinelChainBlocked(f"{label} invalid")
    return value


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CudaNullSentinelChainBlocked("artifact unreadable during hashing") from exc
    return digest.hexdigest()


def _safe_root(root: Path) -> None:
    _require(root.is_dir(), "checkpoint root missing")
    _require(not any(part.is_symlink() for part in (root, *root.parents)), "linked checkpoint root")


def _manifest_path(root: Path, epoch: int) -> Path:
    directory = root / f"epoch-{epoch:04d}"
    path = directory / "artifact-manifest.json"
    _require(directory.is_dir() and not directory.is_symlink(), "epoch directory missing or linked")
    _require(path.is_file() and not path.is_symlink(), "epoch manifest missing or linked")
    return path


def _checkpoint_path(root: Path, epoch: int) -> Path:
    return root / "chain" / f"epoch-{epoch:04d}.json"


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, "duplicate checkpoint field")
        result[key] = value
    return result


def _read_checkpoint(path: Path, epoch: int, previous: str | None) -> str:
    _require(path.is_file() and not path.is_symlink(), "checkpoint missing or linked")
    try:
        record = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CudaNullSentinelChainBlocked("checkpoint unreadable") from exc
    _require(isinstance(record, dict) and set(record) == _FIELDS, "checkpoint schema mismatch")
    _require(
        record["kind"] == "m1_cuda_null_sentinel_epoch_chain"
        and record["version"] == "1.0.0"
        and type(record["epoch_number"]) is int
        and record["epoch_number"] == epoch
        and record["epoch_manifest"] == f"epoch-{epoch:04d}/artifact-manifest.json"
        and record["previous_checkpoint_sha256"] == previous
        and record["epoch_replay_verified"] is False
        and record["execution_authorized"] is False
        and record["scientific_decision"] == "NOT_EVALUATED",
        "checkpoint declaration mismatch",
    )
    _digest(record["epoch_manifest_sha256"], "epoch manifest SHA-256")
    manifest = _manifest_path(path.parent.parent, epoch)
    _require(
        _file_hash(manifest) == record["epoch_manifest_sha256"],
        "epoch manifest SHA-256 mismatch",
    )
    return _file_hash(path)


def replay_checkpoint_chain(
    root: Path, *, expected_count: int, external_tip_sha256: str
) -> dict[str, Any]:
    """Verify a contiguous chain and every epoch-manifest hash, without a model."""
    _safe_root(root)
    _require(
        type(expected_count) is int and 1 <= expected_count <= MAX_EPOCHS, "epoch count invalid"
    )
    _digest(external_tip_sha256, "external chain tip SHA-256")
    chain_dir = root / "chain"
    _require(chain_dir.is_dir() and not chain_dir.is_symlink(), "chain directory missing or linked")
    expected_names = {f"epoch-{epoch:04d}.json" for epoch in range(1, expected_count + 1)}
    _require(
        {entry.name for entry in chain_dir.iterdir()} == expected_names,
        "checkpoint set missing or unexpected",
    )
    previous: str | None = None
    for epoch in range(1, expected_count + 1):
        previous = _read_checkpoint(_checkpoint_path(root, epoch), epoch, previous)
    _require(previous == external_tip_sha256, "external chain tip SHA-256 mismatch")
    return {
        "status": "CHAIN_STRUCTURE_VERIFIED_EPOCH_REPLAY_REQUIRED",
        "checkpoint_count": expected_count,
        "tip_sha256": previous,
        "epoch_replay_verified": False,
        "execution_authorized": False,
        "scientific_decision": "NOT_EVALUATED",
    }


def append_checkpoint(
    root: Path,
    *,
    epoch_number: int,
    external_epoch_manifest_sha256: str,
    external_previous_checkpoint_sha256: str | None,
) -> str:
    """Atomically append one manifest-bound checkpoint; never overwrite an epoch."""
    _safe_root(root)
    _require(type(epoch_number) is int and 1 <= epoch_number <= MAX_EPOCHS, "epoch number invalid")
    _digest(external_epoch_manifest_sha256, "external epoch manifest SHA-256")
    manifest = _manifest_path(root, epoch_number)
    _require(
        _file_hash(manifest) == external_epoch_manifest_sha256, "epoch manifest SHA-256 mismatch"
    )
    chain_dir = root / "chain"
    if epoch_number == 1:
        _require(external_previous_checkpoint_sha256 is None, "first epoch has predecessor")
        chain_dir.mkdir(exist_ok=True)
        _require(not chain_dir.is_symlink() and not any(chain_dir.iterdir()), "chain is not empty")
    else:
        previous = _digest(external_previous_checkpoint_sha256, "external predecessor SHA-256")
        replay_checkpoint_chain(root, expected_count=epoch_number - 1, external_tip_sha256=previous)
    target = _checkpoint_path(root, epoch_number)
    _require(not target.exists(), "checkpoint already exists")
    record = {
        "kind": "m1_cuda_null_sentinel_epoch_chain",
        "version": "1.0.0",
        "epoch_number": epoch_number,
        "epoch_manifest": f"epoch-{epoch_number:04d}/artifact-manifest.json",
        "epoch_manifest_sha256": external_epoch_manifest_sha256,
        "previous_checkpoint_sha256": external_previous_checkpoint_sha256,
        "epoch_replay_verified": False,
        "execution_authorized": False,
        "scientific_decision": "NOT_EVALUATED",
    }
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=chain_dir)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return _file_hash(target)
