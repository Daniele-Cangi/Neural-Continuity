from __future__ import annotations

from pathlib import Path

import pytest

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.stage0_authority import (
    Stage0ControlError,
    _verify_active_teacher_runtime,
    _verify_runtime_authority,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def test_runtime_authority_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    manifest = tmp_path / "authority-manifest.json"
    _write_json(manifest, {"kind": "wrong"})

    with pytest.raises(Stage0ControlError) as error:
        _verify_runtime_authority(manifest, "a" * 64)
    assert error.value.code == "STAGE0_RUNTIME_AUTHORITY_HASH_MISMATCH"


def test_runtime_authority_rejects_absolute_artifact_path(tmp_path: Path) -> None:
    artifact = tmp_path / "contract.json"
    artifact.write_text("{}")
    manifest = tmp_path / "authority-manifest.json"
    _write_json(
        manifest,
        {
            "kind": "m1-stage0-frozen-runtime-authority",
            "version": "1.0.0",
            "status": "VERIFIED",
            "extraction_source": "exact_git_blob_bytes",
            "mutable_checkout_used": False,
            "files": [
                {
                    "path": str(artifact.resolve()),
                    "raw_sha256": sha256_file(artifact),
                    "size_bytes": artifact.stat().st_size,
                }
            ],
        },
    )

    with pytest.raises(Stage0ControlError) as error:
        _verify_runtime_authority(manifest, sha256_file(manifest))
    assert error.value.code == "STAGE0_RUNTIME_AUTHORITY_INVALID"


def test_active_teacher_runtime_rejects_dependency_version_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    _write_json(
        baseline / "evidence-manifest.json",
        {
            "teacher_tokenizer_identity": {
                "torch_version": "2.10.0+cpu",
                "sentence_transformers_version": "5.6.1",
            }
        },
    )
    monkeypatch.setattr(
        "neural_continuity.m1_diagnostics.stage0_authority.version",
        lambda distribution: {
            "torch": "2.13.0+cpu",
            "sentence-transformers": "5.6.1",
        }[distribution],
    )

    with pytest.raises(Stage0ControlError) as error:
        _verify_active_teacher_runtime(baseline / "replay-bundle.json")
    assert error.value.code == "STAGE0_RUNTIME_DEPENDENCY_MISMATCH"
