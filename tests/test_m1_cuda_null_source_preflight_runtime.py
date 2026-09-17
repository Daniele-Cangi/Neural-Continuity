"""The bounded source runner honors authority, six profiles, and cleanup without a model."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from neural_continuity.evidence import canonical_json_bytes
from neural_continuity.m1_diagnostics import cuda_null_source_preflight_runtime as runtime
from neural_continuity.m1_diagnostics.cuda_null_source_preflight_inputs import FROZEN_RUNS


def _stubbed_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str = "valid"
) -> tuple[dict[str, Any], list[object], list[object]]:
    events: list[object] = []
    sessions: list[object] = []
    inventory = {"status": "RUNTIME_IDENTITY_VERIFIED_EXECUTION_BLOCKED"}
    digest = hashlib.sha256(canonical_json_bytes(inventory) + b"\n").hexdigest()
    monkeypatch.setattr(runtime, "RUNTIME_IDENTITY_SHA256", digest)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def authority(**_kwargs: object) -> dict[str, object]:
        events.append("authority")
        if mode == "authority":
            raise ValueError("authority blocked")
        return {
            "status": "SOURCE_ONLY_PREFLIGHT_AUTHORITY_VERIFIED",
            "technical_preflight_permission": "GRANTED_AFTER_REVIEW",
            "source_only": True,
            "int8_allowed": False,
        }

    def verified_runtime(*_args: object) -> dict[str, object]:
        events.append("runtime")
        return inventory

    def reverify_source(_bundle: Path, _snapshot: Path) -> dict[str, object]:
        events.append("snapshot")
        if mode == "snapshot" or (mode == "snapshot_after" and events.count("snapshot") == 2):
            raise ValueError("snapshot mismatch")
        return {"status": "verified"}

    class Teacher:
        max_seq_length = 256

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            events.append("teacher")
            assert events[0] == "authority"

        def eval(self) -> None:
            return None

    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = Teacher  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    class Session:
        def __init__(self, name: str) -> None:
            self.name = name
            self.ended = False

        def end_profiling(self) -> str:
            self.ended = True
            events.append(("end", self.name))
            return str(tmp_path / f"{self.name}.profile.json")

    def profiled_session(_onnx: Path, destination: Path, providers: object) -> Session:
        assert providers == runtime.PROVIDERS
        session = Session(destination.name)
        sessions.append(session)
        events.append(("session", destination.name))
        return session

    def encode(
        _teacher: object, session: Session, texts: list[str], batch: int, name: str
    ) -> np.ndarray:
        events.append(("encode", name, batch, texts[0]))
        assert not session.ended
        if mode == "encode":
            raise ValueError("synthetic encode failure")
        rows = 63 if mode == "shape" else 64
        array = np.zeros((rows, 384), dtype=np.float32)
        array[:, 0] = 1.0
        return array

    def profile_summary(_path: Path, _providers: object) -> dict[str, object]:
        events.append(("profile", _path.stem))
        if mode == "profile":
            raise ValueError("CUDA provider activity absent")
        return {
            "provider_event_counts": {"CUDAExecutionProvider": 1},
            "operator_event_counts": {"CUDAExecutionProvider": {"MatMul": 1}},
            "cpu_fallback_operator_types": [],
            "unclassified_cpu_events": 0,
            "undeclared_providers": [],
        }

    inputs = SimpleNamespace(
        snapshot_root=tmp_path / "snapshot",
        source_onnx=tmp_path / "teacher.onnx",
        document_texts=["document"] * 64,
        query_texts=["query"] * 64,
        identity={"kind": "stubbed"},
    )
    monkeypatch.setattr(runtime, "verify_source_preflight_authority", authority)

    def load_inputs(*_args: object) -> SimpleNamespace:
        if mode == "inputs":
            raise ValueError("corrupt materialization")
        return inputs

    monkeypatch.setattr(runtime, "load_source_preflight_inputs", load_inputs)
    monkeypatch.setattr(runtime, "verify_runtime_identity", verified_runtime)
    monkeypatch.setattr(runtime, "_verify_source", reverify_source)
    monkeypatch.setattr(runtime, "_profiled_session", profiled_session)
    monkeypatch.setattr(runtime, "encode_onnx_source", encode)
    monkeypatch.setattr(runtime, "_profile_summary", profile_summary)
    arguments: dict[str, Any] = {
        "authorization_spec": tmp_path / "authorization.yaml",
        "external_reviewed_spec_sha256": "a" * 64,
        "preflight_spec": tmp_path / "preflight.yaml",
        "external_preflight_spec_sha256": "b" * 64,
        "static_bundle": tmp_path / "static.json",
        "external_static_manifest_sha256": "c" * 64,
        "config_path": tmp_path / "config.yaml",
        "external_config_sha256": "d" * 64,
        "dataset_root": tmp_path / "dataset",
        "transition_a_bundle": tmp_path / "transition-a.json",
        "teacher_snapshot_root": tmp_path / "snapshot",
        "cpu_extension_bundle": tmp_path / "cpu.json",
        "historical_cuda_bundle": tmp_path / "historical.json",
        "working_directory": tmp_path / "work",
    }
    return arguments, events, sessions


def test_source_runner_captures_six_independent_profiles_in_frozen_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    arguments, events, sessions = _stubbed_runner(monkeypatch, tmp_path)
    capture = runtime.capture_source_preflight(**arguments)
    names = [f"{role}_batch_{batch}" for role, batch in FROZEN_RUNS]
    assert events[:5] == ["authority", "runtime", "snapshot", "teacher", "snapshot"]
    assert [session.name for session in sessions] == names
    assert all(session.ended for session in sessions)
    assert [profile["observation_name"] for profile in capture["provider_profiles"]] == names
    assert [
        event[1] for event in events if isinstance(event, tuple) and event[0] == "encode"
    ] == names
    assert set(capture["observations"]) == set(names)
    assert all(
        profile["provider_event_counts"]["CUDAExecutionProvider"] > 0
        for profile in capture["provider_profiles"]
    )


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("authority", "authority blocked"),
        ("inputs", "frozen source inputs"),
        ("snapshot", "teacher snapshot or source changed"),
        ("snapshot_after", "teacher snapshot or source changed"),
        ("encode", "synthetic encode failure"),
        ("shape", "invalid source embedding shape"),
        ("profile", "CUDA provider activity absent"),
    ],
)
def test_source_runner_fails_closed_and_ends_started_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, message: str
) -> None:
    arguments, events, sessions = _stubbed_runner(monkeypatch, tmp_path, mode)
    with pytest.raises(ValueError, match=message):
        runtime.capture_source_preflight(**arguments)
    if mode in {"authority", "inputs", "snapshot", "snapshot_after"}:
        if mode == "authority":
            assert events == ["authority"]
        if mode == "snapshot":
            assert "teacher" not in events
        assert not sessions
    else:
        assert len(sessions) == 1
        assert sessions[0].ended
