"""Frozen UTF-8 source-only input selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_continuity.m1_diagnostics.cuda_null_source_preflight_inputs import _load_jsonl


def test_jsonl_selection_is_utf8_sorted_and_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    records = [
        {"document_id": "b", "text": "second"},
        {"document_id": "a", "text": "first"},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    assert _load_jsonl(path, "document_id", 2) == [("a", "first"), ("b", "second")]
    records[1]["document_id"] = "b"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        _load_jsonl(path, "document_id", 2)
