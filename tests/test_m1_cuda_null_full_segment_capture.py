from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from neural_continuity.m1_diagnostics import cuda_null_full_segment_capture as capture


class _Session:
    def __init__(self, prefix: Path) -> None:
        self.prefix = prefix
        self.call_count = 0

    def end_profiling(self) -> str:
        path = self.prefix.with_suffix(".json")
        events = [
            {"cat": "Session", "name": "model_run", "args": {}} for _ in range(self.call_count)
        ]
        events.append(
            {
                "cat": "Node",
                "name": "synthetic_kernel",
                "args": {"provider": "CUDAExecutionProvider", "op_name": "MatMul"},
            }
        )
        path.write_text(json.dumps(events), encoding="utf-8")
        return str(path)


def test_segmented_capture_covers_every_document_and_replays(tmp_path, monkeypatch) -> None:
    staging = tmp_path / "staging"
    scratch = tmp_path / "scratch"
    staging.mkdir()
    scratch.mkdir()
    ids = [f"d{index:05d}" for index in range(5183)]
    texts = [f"document {index}" for index in range(5183)]

    def session_factory(_source, prefix, _providers):
        return _Session(prefix)

    def encode(_teacher, session, segment, batch_size, _label):
        session.call_count = math.ceil(len(segment) / batch_size)
        values = np.ones((len(segment), 384), dtype=np.float32)
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        return values

    monkeypatch.setattr(capture, "_profiled_session", session_factory)
    monkeypatch.setattr(capture, "encode_onnx_source", encode)
    record = capture._capture_profiled_role(
        object(),
        tmp_path / "unused.onnx",
        ids=ids,
        texts=texts,
        run_label="batch_16_primary",
        role="documents",
        batch_size=16,
        staging=staging,
        scratch=scratch,
    )
    assert len(record["segments"]) == 21
    assert record["segments"][0]["start"] == 0
    assert record["segments"][-1]["end"] == 5183
    assert max(segment["end"] - segment["start"] for segment in record["segments"]) == 256
    assert all(segment["profile_path"].endswith(".json.gz") for segment in record["segments"])
    assert all(
        segment["profile_archive"]["compression"] == "gzip" for segment in record["segments"]
    )
    assert not list((staging / "profiles").rglob("*.json"))
