from __future__ import annotations

from pathlib import Path

import pytest

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.fidelity_authority import (
    FidelityGateError,
    verify_artifact_manifest,
)


def _manifest(root: Path) -> str:
    artifact = root / "artifact.bin"
    artifact.write_bytes(b"frozen")
    manifest = {
        "artifact_count": 1,
        "artifacts": [
            {
                "path": artifact.name,
                "sha256": sha256_file(artifact),
                "size_bytes": artifact.stat().st_size,
            }
        ],
    }
    manifest_path = root / "artifact-manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return sha256_file(manifest_path)


def test_artifact_manifest_verification_fails_closed_on_tamper(tmp_path: Path) -> None:
    digest = _manifest(tmp_path)
    verified = verify_artifact_manifest(tmp_path, digest)
    assert verified["artifact_count"] == 1

    (tmp_path / "artifact.bin").write_bytes(b"changed")
    with pytest.raises(FidelityGateError, match="size mismatch") as error:
        verify_artifact_manifest(tmp_path, digest)
    assert error.value.status == "BLOCKED"
