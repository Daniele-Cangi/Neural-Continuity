from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from neural_continuity.m1_diagnostics import cuda_null_sentinel_capture as capture
from neural_continuity.m1_diagnostics import cuda_null_sentinel_execution_authority as authority
from neural_continuity.m1_diagnostics import cuda_null_sentinel_runner as runner


def test_owner_authority_rejects_unanchored_hash() -> None:
    with pytest.raises(ValueError):
        authority.verify_sentinel_execution_authority("0" * 64)


@pytest.mark.parametrize("epoch", [0, 121, True])
def test_epoch_outside_frozen_range_blocks_before_loading_model(
    monkeypatch: pytest.MonkeyPatch, epoch: int
) -> None:
    monkeypatch.setattr(
        capture,
        "verify_sentinel_execution_authority",
        lambda _digest: pytest.fail("authority must not be consulted for an invalid epoch"),
    )
    with pytest.raises(capture.SentinelExecutionBlocked):
        capture.capture_epoch(
            external_authority_sha256="a" * 64,
            epoch_number=epoch,
            previous_checkpoint_sha256=None,
        )


def test_existing_output_blocks_without_spawning_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        runner,
        "verify_sentinel_execution_authority",
        lambda _digest: SimpleNamespace(paths={"output_root": tmp_path}),
    )
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("no child process may be spawned"),
    )
    with pytest.raises(runner.SentinelExecutionBlocked, match="already exists"):
        runner.run_sentinel("a" * 64)
