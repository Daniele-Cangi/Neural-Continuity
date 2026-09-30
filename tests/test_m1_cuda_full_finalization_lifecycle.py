from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from neural_continuity.m1_diagnostics import cuda_null_full_corpus_package as package
from neural_continuity.m1_diagnostics import cuda_null_full_runner as runner

AUTHORITY = "a" * 64
TIP = "b" * 64
MANIFEST = "c" * 64
DECLARATIONS = [{"epoch": epoch, "manifest_sha256": MANIFEST} for epoch in range(1, 121)]


@pytest.fixture
def aggregate_stub(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []

    def recompute(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(1)
        return {
            "units": {
                "repeated_inference": [{"unit_number": i} for i in range(1, 121)],
                "batch_size_variation": [{"unit_number": i} for i in range(1, 121)],
                "process_restart_variation": [{"unit_number": i} for i in range(1, 61)],
            },
            "family_extrema": {},
        }

    monkeypatch.setattr(package, "_recompute", recompute)
    return calls


def _interrupt_staging(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    def interrupt(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise KeyboardInterrupt

    with monkeypatch.context() as scoped:
        scoped.setattr(package, "replay_full_corpus_package", interrupt)
        with pytest.raises(KeyboardInterrupt):
            package.finalize_full_corpus_package(root, AUTHORITY, TIP, DECLARATIONS)
    staging = list(root.glob(".full-corpus-package.tmp-*"))
    assert len(staging) == 1
    assert not (root / "full-corpus-package").exists()
    return staging[0]


def test_runner_uses_verified_finalization_without_duplicate_replay(
    tmp_path: Path, aggregate_stub: list[int], capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    digest, replay = runner._finalize_or_replay_corpus(
        tmp_path, checkpoint, AUTHORITY, TIP, {i: MANIFEST for i in range(1, 121)}
    )
    assert len(aggregate_stub) == 2  # Build summary, then independently replay it.
    assert replay["replay_status"] == "PASS"
    assert replay["model_loaded"] is False
    assert replay["scientific_decision"] == "NOT_EVALUATED"
    assert replay["family_unit_counts"] == {
        "repeated_inference": 120,
        "batch_size_variation": 120,
        "process_restart_variation": 60,
    }
    anchor = json.loads((tmp_path / "checkpoint.final-manifest.json").read_text())
    assert anchor["artifact_manifest_sha256"] == digest
    assert not list(tmp_path.glob(".full-corpus-package.tmp-*"))
    progress = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert [item["phase"] for item in progress] == ["corpus_publication", "final_anchor"]
    assert all(item["timestamp"] and item["event"] == "completed" for item in progress)


def test_existing_package_is_replayed_before_anchor(
    tmp_path: Path, aggregate_stub: list[int]
) -> None:
    package.finalize_full_corpus_package(tmp_path, AUTHORITY, TIP, DECLARATIONS)
    aggregate_stub.clear()
    runner._finalize_or_replay_corpus(
        tmp_path,
        tmp_path / "checkpoint.json",
        AUTHORITY,
        TIP,
        {i: MANIFEST for i in range(1, 121)},
    )
    assert len(aggregate_stub) == 1
    assert (tmp_path / "checkpoint.final-manifest.json").is_file()


def test_interrupted_staging_is_replayed_without_rebuilding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, aggregate_stub: list[int]
) -> None:
    staging = _interrupt_staging(tmp_path, monkeypatch)
    original = (staging / "artifact-manifest.json").read_bytes()
    aggregate_stub.clear()
    output, digest, replay = package.finalize_full_corpus_package(
        tmp_path, AUTHORITY, TIP, DECLARATIONS
    )
    assert len(aggregate_stub) == 1
    assert replay["replay_status"] == "PASS"
    assert replay["artifact_manifest_sha256"] == digest
    assert (output / "artifact-manifest.json").read_bytes() == original
    assert not staging.exists()


@pytest.mark.parametrize("mismatch", ["authority", "tip", "declarations", "bytes"])
def test_invalid_staging_fails_closed_and_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    aggregate_stub: list[int],
    mismatch: str,
) -> None:
    staging = _interrupt_staging(tmp_path, monkeypatch)
    authority = "d" * 64 if mismatch == "authority" else AUTHORITY
    tip = "d" * 64 if mismatch == "tip" else TIP
    declarations = DECLARATIONS[:-1] if mismatch == "declarations" else DECLARATIONS
    if mismatch == "bytes":
        with (staging / "corpus-summary.json").open("ab") as stream:
            stream.write(b" ")
    aggregate_stub.clear()
    with pytest.raises(package.FullCorpusPackageBlocked):
        package.finalize_full_corpus_package(tmp_path, authority, tip, declarations)
    assert staging.exists()
    assert not (tmp_path / "full-corpus-package").exists()
    assert not aggregate_stub


def test_ambiguous_staging_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, aggregate_stub: list[int]
) -> None:
    staging = _interrupt_staging(tmp_path, monkeypatch)
    (tmp_path / ".full-corpus-package.tmp-other").mkdir()
    aggregate_stub.clear()
    with pytest.raises(package.FullCorpusPackageBlocked, match="multiple"):
        package.finalize_full_corpus_package(tmp_path, AUTHORITY, TIP, DECLARATIONS)
    assert staging.exists()
    assert not aggregate_stub


def test_blocked_fresh_replay_does_not_publish_or_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, aggregate_stub: list[int]
) -> None:
    monkeypatch.setattr(
        package,
        "replay_full_corpus_package",
        lambda *args, **kwargs: {"replay_status": "BLOCKED", "reason": "test block"},
    )
    with pytest.raises(package.FullCorpusPackageBlocked):
        runner._finalize_or_replay_corpus(
            tmp_path,
            tmp_path / "checkpoint.json",
            AUTHORITY,
            TIP,
            {i: MANIFEST for i in range(1, 121)},
        )
    assert not (tmp_path / "full-corpus-package").exists()
    assert not (tmp_path / "checkpoint.final-manifest.json").exists()
    assert not list(tmp_path.glob(".full-corpus-package.tmp-*"))


def test_recovered_staging_recomputation_mismatch_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, aggregate_stub: list[int]
) -> None:
    staging = _interrupt_staging(tmp_path, monkeypatch)
    monkeypatch.setattr(
        package, "_recompute", lambda *args, **kwargs: {"units": {}, "family_extrema": {}}
    )
    with pytest.raises(package.FullCorpusPackageBlocked, match="recomputation differs"):
        package.finalize_full_corpus_package(tmp_path, AUTHORITY, TIP, DECLARATIONS)
    assert staging.exists()
    assert not (tmp_path / "full-corpus-package").exists()


def test_incomplete_staging_is_preserved_and_not_rebuilt(
    tmp_path: Path, aggregate_stub: list[int]
) -> None:
    staging = tmp_path / ".full-corpus-package.tmp-incomplete"
    staging.mkdir()
    with pytest.raises(package.FullCorpusPackageBlocked, match="cannot decode"):
        package.finalize_full_corpus_package(tmp_path, AUTHORITY, TIP, DECLARATIONS)
    assert staging.exists()
    assert not aggregate_stub


def test_existing_package_tampering_blocks_anchor(
    tmp_path: Path, aggregate_stub: list[int]
) -> None:
    package.finalize_full_corpus_package(tmp_path, AUTHORITY, TIP, DECLARATIONS)
    with (tmp_path / "full-corpus-package" / "corpus-summary.json").open("ab") as stream:
        stream.write(b" ")
    aggregate_stub.clear()
    with pytest.raises(runner.FullCorpusExecutionBlocked, match="replay blocked"):
        runner._finalize_or_replay_corpus(
            tmp_path,
            tmp_path / "checkpoint.json",
            AUTHORITY,
            TIP,
            {i: MANIFEST for i in range(1, 121)},
        )
    assert not aggregate_stub
    assert not (tmp_path / "checkpoint.final-manifest.json").exists()


def test_publication_bytes_are_checked_before_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, aggregate_stub: list[int]
) -> None:
    original_replace = Path.replace

    def tampered_replace(self: Path, target: Path) -> Path:
        result = original_replace(self, target)
        with (result / "corpus-summary.json").open("ab") as stream:
            stream.write(b" ")
        return result

    monkeypatch.setattr(Path, "replace", tampered_replace)
    with pytest.raises(package.FullCorpusPackageBlocked, match="published.*bytes differ"):
        runner._finalize_or_replay_corpus(
            tmp_path,
            tmp_path / "checkpoint.json",
            AUTHORITY,
            TIP,
            {i: MANIFEST for i in range(1, 121)},
        )
    assert len(aggregate_stub) == 2
    assert (tmp_path / "full-corpus-package").exists()
    assert not (tmp_path / "checkpoint.final-manifest.json").exists()
