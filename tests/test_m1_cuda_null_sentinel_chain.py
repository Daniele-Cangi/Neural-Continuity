"""Synthetic-only checks for the non-executing CUDA sentinel chain index."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from neural_continuity.m1_diagnostics.cuda_null_sentinel_chain import (
    CudaNullSentinelChainBlocked,
    append_checkpoint,
    replay_checkpoint_chain,
)


def _manifest(root: Path, epoch: int) -> str:
    directory = root / f"epoch-{epoch:04d}"
    directory.mkdir()
    path = directory / "artifact-manifest.json"
    path.write_text(json.dumps({"synthetic_epoch": epoch}), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _append(root: Path, epoch: int, previous: str | None) -> str:
    return append_checkpoint(
        root,
        epoch_number=epoch,
        external_epoch_manifest_sha256=_manifest(root, epoch),
        external_previous_checkpoint_sha256=previous,
    )


def test_contiguous_chain_is_non_qualifying(tmp_path: Path) -> None:
    first = _append(tmp_path, 1, None)
    second = _append(tmp_path, 2, first)
    replay = replay_checkpoint_chain(tmp_path, expected_count=2, external_tip_sha256=second)
    assert replay == {
        "status": "CHAIN_STRUCTURE_VERIFIED_EPOCH_REPLAY_REQUIRED",
        "checkpoint_count": 2,
        "tip_sha256": second,
        "epoch_replay_verified": False,
        "execution_authorized": False,
        "scientific_decision": "NOT_EVALUATED",
    }


def test_modified_epoch_manifest_blocks_replay(tmp_path: Path) -> None:
    tip = _append(tmp_path, 1, None)
    (tmp_path / "epoch-0001" / "artifact-manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CudaNullSentinelChainBlocked, match="manifest SHA-256 mismatch"):
        replay_checkpoint_chain(tmp_path, expected_count=1, external_tip_sha256=tip)


def test_missing_checkpoint_blocks_replay(tmp_path: Path) -> None:
    first = _append(tmp_path, 1, None)
    second = _append(tmp_path, 2, first)
    (tmp_path / "chain" / "epoch-0001.json").unlink()
    with pytest.raises(CudaNullSentinelChainBlocked, match="checkpoint set"):
        replay_checkpoint_chain(tmp_path, expected_count=2, external_tip_sha256=second)


def test_wrong_external_tip_blocks_replay(tmp_path: Path) -> None:
    _append(tmp_path, 1, None)
    with pytest.raises(CudaNullSentinelChainBlocked, match="external chain tip"):
        replay_checkpoint_chain(tmp_path, expected_count=1, external_tip_sha256="0" * 64)


def test_append_rejects_wrong_predecessor(tmp_path: Path) -> None:
    _append(tmp_path, 1, None)
    with pytest.raises(CudaNullSentinelChainBlocked, match="external chain tip"):
        _append(tmp_path, 2, "0" * 64)


def test_duplicate_epoch_cannot_be_overwritten(tmp_path: Path) -> None:
    tip = _append(tmp_path, 1, None)
    manifest = tmp_path / "epoch-0001" / "artifact-manifest.json"
    with pytest.raises(CudaNullSentinelChainBlocked, match="chain is not empty"):
        append_checkpoint(
            tmp_path,
            epoch_number=1,
            external_epoch_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            external_previous_checkpoint_sha256=None,
        )
    assert hashlib.sha256((tmp_path / "chain" / "epoch-0001.json").read_bytes()).hexdigest() == tip


def test_extra_checkpoint_blocks_replay(tmp_path: Path) -> None:
    tip = _append(tmp_path, 1, None)
    (tmp_path / "chain" / "epoch-9999.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CudaNullSentinelChainBlocked, match="checkpoint set"):
        replay_checkpoint_chain(tmp_path, expected_count=1, external_tip_sha256=tip)
