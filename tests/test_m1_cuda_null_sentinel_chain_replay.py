"""End-to-end synthetic package replay through the non-executing chain gate."""

from __future__ import annotations

import importlib.util
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from neural_continuity.m1_diagnostics import cuda_null_sentinel_chain_replay as gate
from neural_continuity.m1_diagnostics.cuda_null_sentinel_chain import append_checkpoint
from neural_continuity.m1_diagnostics.cuda_null_sentinel_epoch_package import (
    write_cuda_sentinel_epoch,
)


def _synthetic_observations() -> tuple[object, ...]:
    fixture_path = Path(__file__).with_name("test_m1_cuda_null_sentinel_epoch.py")
    specification = importlib.util.spec_from_file_location("synthetic_epoch_fixture", fixture_path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module._capture()


@pytest.fixture
def captured_chain() -> Iterator[tuple[Path, str]]:
    with tempfile.TemporaryDirectory(dir=Path.home()) as temporary:
        root = Path(temporary)
        plan, runtime, documents, queries, profiles = _synthetic_observations()
        _, manifest_sha256 = write_cuda_sentinel_epoch(
            root / "epoch-0001",
            plan=plan,
            runtime=runtime,
            document_embeddings=documents,
            query_embeddings=queries,
            profiles=profiles,
        )
        tip = append_checkpoint(
            root,
            epoch_number=1,
            external_epoch_manifest_sha256=manifest_sha256,
            external_previous_checkpoint_sha256=None,
        )
        yield root, tip


def test_real_synthetic_package_replays_through_chain(captured_chain: tuple[Path, str]) -> None:
    root, tip = captured_chain
    result = gate.replay_cuda_sentinel_chain_packages(
        root, expected_count=1, external_tip_sha256=tip
    )
    assert result["status"] == "TECHNICAL_REPLAY_PASS_NOT_QUALIFYING"
    assert result["chain_structure_verified"] is True
    assert result["epoch_package_replay_verified"] is True
    assert result["epoch_outcomes"][0]["replay_status"] == "PASS"
    assert result["authority_verified"] is False
    assert result["execution_authorized"] is False
    assert result["scientific_decision"] == "NOT_EVALUATED"


def test_missing_declared_bundle_blocks_chain_replay(captured_chain: tuple[Path, str]) -> None:
    root, tip = captured_chain
    (root / "epoch-0001" / "replay-bundle.json").unlink()
    with pytest.raises(gate.CudaNullSentinelChainReplayBlocked, match="bundle missing"):
        gate.replay_cuda_sentinel_chain_packages(root, expected_count=1, external_tip_sha256=tip)


def test_package_replay_mismatch_blocks_chain(
    captured_chain: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, tip = captured_chain
    monkeypatch.setattr(gate, "replay_cuda_sentinel_epoch", lambda *_: {"replay_status": "BLOCKED"})
    with pytest.raises(gate.CudaNullSentinelChainReplayBlocked, match="package replay blocked"):
        gate.replay_cuda_sentinel_chain_packages(root, expected_count=1, external_tip_sha256=tip)


def test_wrong_external_tip_blocks_before_package_replay(
    captured_chain: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = captured_chain
    monkeypatch.setattr(
        gate,
        "replay_cuda_sentinel_epoch",
        lambda *_: pytest.fail("epoch replay must not run before chain verification"),
    )
    with pytest.raises(ValueError, match="external chain tip"):
        gate.replay_cuda_sentinel_chain_packages(
            root, expected_count=1, external_tip_sha256="0" * 64
        )
