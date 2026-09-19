from __future__ import annotations

from pathlib import Path

import pytest

from neural_continuity.evidence import sha256_file
from neural_continuity.m1_diagnostics import cuda_null_full_execution_authority as authority


def test_full_corpus_authority_schema_is_exact() -> None:
    digest = sha256_file(authority.SPEC_PATH)
    spec = authority._read_spec(digest)

    assert spec["execution_authorized"] is False
    assert spec["scientific_decision"] == "NOT_EVALUATED"
    assert spec["profile_storage"] == "deterministic_gzip_level_9"
    assert spec["design_pre_execution_review_completed"] is True
    assert spec["implementation_independent_review_completed"] is False


def test_full_corpus_authority_rejects_unanchored_hash() -> None:
    with pytest.raises(authority.FullCorpusExecutionBlocked):
        authority._read_spec("0" * 64)


def test_full_corpus_authority_requires_external_checkpoint_path() -> None:
    spec = authority._read_spec(sha256_file(authority.SPEC_PATH))
    paths = dict(spec["paths"])
    paths.pop("external_checkpoint_tip")
    spec["paths"] = paths

    with pytest.raises(authority.FullCorpusExecutionBlocked, match="path set differs"):
        authority._paths(spec)


def test_full_corpus_authority_paths_keep_evidence_on_d() -> None:
    spec = authority._read_spec(sha256_file(authority.SPEC_PATH))
    paths = authority._paths(spec)

    assert all(path.drive.upper() == "D:" for path in paths.values())
    assert paths["external_checkpoint_tip"].parent == paths["output_root"].parent
    assert paths["external_checkpoint_tip"] != Path(paths["output_root"])


def test_full_corpus_authority_blocks_before_prerequisite_replay_without_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(authority, "has_linked_ancestor", lambda _path: False)
    monkeypatch.setattr(
        authority,
        "verify_sentinel_execution_authority",
        lambda _digest: pytest.fail("blocked authority must not replay execution dependencies"),
    )
    with pytest.raises(authority.FullCorpusExecutionBlocked, match="implementation review"):
        authority.verify_full_corpus_execution_authority(sha256_file(authority.SPEC_PATH))
