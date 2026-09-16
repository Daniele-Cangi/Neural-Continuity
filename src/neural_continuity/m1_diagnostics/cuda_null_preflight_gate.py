"""Re-verify frozen static inputs immediately before any CUDA preflight.

This module imports neither ONNX nor ONNX Runtime. Its positive result is not
permission to load a graph: runtime identity and formal authorization remain
separate gates.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes
from neural_continuity.m1_diagnostics.cuda_null_authority import (
    CudaNullAuthorityBlocked,
    build_static_authority,
)
from neural_continuity.m1_diagnostics.cuda_null_evidence import replay_static_package


def verify_live_static_authority(
    *,
    static_bundle: Path,
    external_static_manifest_sha256: str,
    config_path: Path,
    external_config_sha256: str,
    dataset_root: Path,
    transition_a_bundle: Path,
    teacher_snapshot_root: Path,
    cpu_extension_bundle: Path,
    historical_cuda_bundle: Path,
) -> dict[str, Any]:
    """Bind a model-free replay to a fresh verification of the same inputs."""
    replay = replay_static_package(static_bundle, external_static_manifest_sha256)
    if (
        replay.get("integrity_replay_status") != "PASS"
        or replay.get("status_match") is not True
        or replay.get("decision_match") is not True
        or replay.get("execution_authorized") is not False
        or replay.get("manifest_sha256") != external_static_manifest_sha256
    ):
        raise CudaNullAuthorityBlocked("static replay did not verify all declared fields")

    live = build_static_authority(
        config_path=config_path,
        external_config_sha256=external_config_sha256,
        dataset_root=dataset_root,
        transition_a_bundle=transition_a_bundle,
        teacher_snapshot_root=teacher_snapshot_root,
        cpu_extension_bundle=cpu_extension_bundle,
        historical_cuda_bundle=historical_cuda_bundle,
    )
    live_sha256 = hashlib.sha256(canonical_json_bytes(live) + b"\n").hexdigest()
    if live_sha256 != replay.get("authority_sha256"):
        raise CudaNullAuthorityBlocked("live authority differs from the frozen static package")
    if live.get("config_sha256") != external_config_sha256:
        raise CudaNullAuthorityBlocked("live configuration identity mismatch")
    if live.get("execution_authorized") is not False:
        raise CudaNullAuthorityBlocked("static authority cannot authorize execution")

    return {
        "status": "FROZEN_STATIC_INPUTS_REVERIFIED",
        "static_manifest_sha256": external_static_manifest_sha256,
        "static_authority_sha256": live_sha256,
        "config_sha256": external_config_sha256,
        "runtime_verified": False,
        "onnx_graph_loaded": False,
        "session_created": False,
        "execution_authorized": False,
    }
