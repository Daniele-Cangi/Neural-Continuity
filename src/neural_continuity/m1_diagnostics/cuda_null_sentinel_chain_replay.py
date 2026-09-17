"""Model-free replay of every package named by a CUDA sentinel checkpoint chain.

Passing this technical replay never grants execution or scientific qualification.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from neural_continuity.m1_diagnostics.cuda_null_sentinel_chain import (
    CudaNullSentinelChainBlocked,
    replay_checkpoint_chain,
)
from neural_continuity.m1_diagnostics.cuda_null_sentinel_epoch_package import (
    replay_cuda_sentinel_epoch,
)


class CudaNullSentinelChainReplayBlocked(CudaNullSentinelChainBlocked):
    """One declared epoch package did not pass model-free technical replay."""


def _manifest_hash(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CudaNullSentinelChainReplayBlocked("epoch manifest unreadable") from exc
    return digest.hexdigest()


def replay_cuda_sentinel_chain_packages(
    root: Path, *, expected_count: int, external_tip_sha256: str
) -> dict[str, Any]:
    """Verify the externally anchored chain, then replay each exact epoch bundle."""
    chain = replay_checkpoint_chain(
        root, expected_count=expected_count, external_tip_sha256=external_tip_sha256
    )
    if chain.get("status") != "CHAIN_STRUCTURE_VERIFIED_EPOCH_REPLAY_REQUIRED":
        raise CudaNullSentinelChainReplayBlocked("chain structure did not verify")

    outcomes: list[dict[str, Any]] = []
    for epoch in range(1, expected_count + 1):
        directory = root / f"epoch-{epoch:04d}"
        bundle = directory / "replay-bundle.json"
        if not bundle.is_file() or bundle.is_symlink():
            raise CudaNullSentinelChainReplayBlocked(
                f"epoch {epoch}: replay bundle missing or linked"
            )
        manifest_sha256 = _manifest_hash(directory / "artifact-manifest.json")
        try:
            replay = replay_cuda_sentinel_epoch(bundle, manifest_sha256)
        except (OSError, ValueError) as exc:
            raise CudaNullSentinelChainReplayBlocked(
                f"epoch {epoch}: package replay could not complete"
            ) from exc
        if not (
            replay.get("replay_status") == "PASS"
            and replay.get("technical_structure_status") == "PASS"
            and replay.get("summary_match") is True
            and replay.get("artifact_manifest_sha256") == manifest_sha256
            and type(replay.get("epoch_number")) is int
            and replay.get("epoch_number") == epoch
            and replay.get("authority_verified") is False
            and replay.get("checkpoint_chain_verified") is False
            and replay.get("qualifying_detection_evidence") is False
            and replay.get("scientific_decision") == "NOT_EVALUATED"
            and replay.get("model_loaded") is False
            and replay.get("onnx_graph_loaded") is False
        ):
            raise CudaNullSentinelChainReplayBlocked(f"epoch {epoch}: package replay blocked")
        outcomes.append(
            {
                "epoch_number": epoch,
                "artifact_manifest_sha256": manifest_sha256,
                "replay_status": "PASS",
            }
        )

    # Detect any checkpoint or manifest mutation during the per-epoch replay pass.
    replay_checkpoint_chain(
        root, expected_count=expected_count, external_tip_sha256=external_tip_sha256
    )
    return {
        "status": "TECHNICAL_REPLAY_PASS_NOT_QUALIFYING",
        "checkpoint_count": expected_count,
        "tip_sha256": external_tip_sha256,
        "chain_structure_verified": True,
        "epoch_package_replay_verified": True,
        "epoch_outcomes": outcomes,
        "authority_verified": False,
        "execution_authorized": False,
        "qualifying_detection_evidence": False,
        "scientific_decision": "NOT_EVALUATED",
        "model_loaded": False,
        "onnx_graph_loaded": False,
    }
