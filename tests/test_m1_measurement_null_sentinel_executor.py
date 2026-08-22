from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from neural_continuity.m1_diagnostics import (
    measurement_null_sentinel_executor as executor,
)
from neural_continuity.m1_teacher_evidence import TeacherEvidenceError, _fail


def _authority() -> Any:
    document_ids = tuple(f"doc-{index:04d}" for index in range(256))
    query_ids = ("query-1", "query-2")
    return SimpleNamespace(
        authority_sha256="a" * 64,
        config={},
        config_sha256="b" * 64,
        extension_plan_bundle=Path("extension/replay-bundle.json"),
        extension_plan_manifest_sha256="c" * 64,
        extension_plan_sha256="d" * 64,
        source=SimpleNamespace(
            transition_a_manifest_sha256="e" * 64,
            artifact_sha256="f" * 64,
            execution_provider="CPUExecutionProvider",
        ),
        dataset=SimpleNamespace(
            dataset_id="dataset",
            manifest_sha256="1" * 64,
            materialization_policy_sha256="2" * 64,
            partition_policy_sha256="3" * 64,
        ),
        selected_document_ids=document_ids,
        selected_document_texts=tuple(f"text {value}" for value in document_ids),
        query_ids=query_ids,
        query_texts=("query one", "query two"),
        qrels={"query-1": ("doc-0001",), "query-2": ("doc-0002",)},
    )


class _FakeBackend:
    def __init__(self, process_instance_id: str) -> None:
        self.process_instance_id = process_instance_id
        self.encode_count = 0

    def encode(
        self,
        texts: tuple[str, ...],
        batch_size: int,
        label: str,
    ) -> np.ndarray:
        del batch_size, label
        self.encode_count += 1
        embeddings = np.zeros((len(texts), 4), dtype=np.float32)
        embeddings[:, self.encode_count % 4] = 1.0
        return embeddings

    def runtime_inventory(self) -> dict[str, object]:
        return {
            "model_execution_used": True,
            "onnx_graph_loaded": True,
            "activation_read": False,
            "execution_provider": "CPUExecutionProvider",
            "process_instance_id": self.process_instance_id,
        }


def test_run_next_executes_one_epoch_and_stops_before_full_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _authority()

    def verified(*args: object, **kwargs: object) -> Any:
        del args, kwargs
        return authority

    monkeypatch.setattr(executor, "verify_sentinel_authority", verified)
    run = tmp_path / "sentinel"
    prepared = executor.prepare_sentinel_run(
        config_path="config.yaml",
        dataset_directory="dataset",
        transition_a_bundle="transition-a.json",
        extension_plan_bundle="extension.json",
        extension_plan_manifest_sha256="c" * 64,
        output_directory=run,
    )
    factory_calls: list[_FakeBackend] = []

    def factory(_: Any) -> _FakeBackend:
        backend = _FakeBackend(f"process-{len(factory_calls) + 1}")
        factory_calls.append(backend)
        return backend

    first = executor.run_next_sentinel_epoch(
        config_path="config.yaml",
        dataset_directory="dataset",
        transition_a_bundle="transition-a.json",
        extension_plan_bundle="extension.json",
        extension_plan_manifest_sha256="c" * 64,
        run_directory=run,
        root_manifest_sha256=str(prepared["root_manifest_sha256"]),
        expected_checkpoint_sha256=str(prepared["latest_checkpoint_sha256"]),
        backend_factory=factory,
    )

    assert first["epoch_number"] == 1
    assert first["next_epoch_number"] == 2
    assert first["full_corpus_execution"] is False
    assert len(factory_calls) == 1
    assert factory_calls[0].encode_count == 8


def test_authority_failure_precedes_backend_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rejected(*args: object, **kwargs: object) -> Any:
        del args, kwargs
        raise _fail("SENTINEL_AUTHORITY_BLOCKED", "authority missing")

    monkeypatch.setattr(executor, "verify_sentinel_authority", rejected)
    backend_created = False

    def factory(_: Any) -> _FakeBackend:
        nonlocal backend_created
        backend_created = True
        return _FakeBackend("unexpected")

    with pytest.raises(TeacherEvidenceError) as error:
        executor.run_next_sentinel_epoch(
            config_path="config.yaml",
            dataset_directory="dataset",
            transition_a_bundle="transition-a.json",
            extension_plan_bundle="extension.json",
            extension_plan_manifest_sha256="c" * 64,
            run_directory=tmp_path / "missing",
            root_manifest_sha256="d" * 64,
            expected_checkpoint_sha256="d" * 64,
            backend_factory=factory,
        )

    assert error.value.status == "BLOCKED"
    assert backend_created is False
