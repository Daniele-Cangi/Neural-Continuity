from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from neural_continuity.m1_diagnostics import activation_mask_evidence_v2 as mask_evidence
from neural_continuity.m1_diagnostics.authority import DiagnosticPreflightError


def test_derived_masks_are_hash_bound_and_replay_without_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "capture-plan.json").write_text("{}\n", encoding="utf-8")
    source = SimpleNamespace(
        root=source_root,
        bundle_path=source_root / "replay-bundle.json",
        manifest_sha256="a" * 64,
        capture_plan={"batch_size": 2, "query_count": 2},
        batch_index={"batches": [{"batch_id": "batch-0001", "query_ids": ["q-1", "q-2"]}]},
    )
    monkeypatch.setattr(mask_evidence, "BATCH_SIZE", 2)
    monkeypatch.setattr(mask_evidence, "QUERY_COUNT", 2)
    monkeypatch.setattr(mask_evidence, "verify_activation_analysis_input", lambda *_: source)
    monkeypatch.setattr(
        mask_evidence,
        "_verify_dataset_identity",
        lambda *_: (object(), {"membership_sha256": mask_evidence.ROLE_MEMBERSHIP_SHA256}),
    )
    monkeypatch.setattr(mask_evidence, "verify_fidelity_authority", lambda *_: object())
    monkeypatch.setattr(
        mask_evidence,
        "_verify_tokenizer",
        lambda *_: (object(), {"model_id": "frozen", "revision": "v1"}),
    )
    monkeypatch.setattr(
        mask_evidence, "_canonical_queries", lambda *_: (["q-1", "q-2"], ["one", "two"])
    )
    monkeypatch.setattr(
        mask_evidence,
        "_token_inputs",
        lambda *_: {"attention_mask": np.asarray([[1, 1, 0], [1, 0, 0]], dtype=np.int64)},
    )

    artifact = tmp_path / "attention-masks.json"
    captured = mask_evidence.capture_reconstructed_attention_masks(
        source_bundle=source.bundle_path,
        source_manifest_sha256=source.manifest_sha256,
        config_path=tmp_path / "config.yaml",
        dataset_directory=tmp_path / "dataset",
        instrumentation_directory=tmp_path / "instrumentation",
        output_path=artifact,
    )
    replayed = mask_evidence.replay_reconstructed_attention_masks(
        artifact, captured["artifact_sha256"]
    )
    assert replayed["replay_verified"] is True
    assert replayed["qualifying_m1_evidence"] is False
    assert mask_evidence.verify_reconstructed_attention_masks(
        artifact, captured["artifact_sha256"]
    ).masks_by_batch_id["batch-0001"].tolist() == [[1, 1, 0], [1, 0, 0]]

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["batches"][0]["attention_mask"][0][0] = 0
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DiagnosticPreflightError):
        mask_evidence.replay_reconstructed_attention_masks(artifact, captured["artifact_sha256"])
