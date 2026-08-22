from __future__ import annotations

import copy
import os
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from neural_continuity.m1_b.onnx_source import (
    encode_onnx_source,
    open_onnx_session,
)
from neural_continuity.m1_diagnostics.measurement_null_sentinel_authority import (
    SentinelAuthority,
)
from neural_continuity.m1_diagnostics.runtime_provenance_authority import (
    RuntimeProvenanceError,
)
from neural_continuity.m1_diagnostics.runtime_provenance_environment import (
    capture_runtime_inventory,
)
from neural_continuity.m1_teacher_evidence import _fail, _load_teacher

PROCESS_INSTANCE_ID = uuid.uuid4().hex


class OnnxSentinelBackend:
    def __init__(self, authority: SentinelAuthority) -> None:
        try:
            inventory = capture_runtime_inventory()
        except RuntimeProvenanceError as exc:
            raise _fail(
                "SENTINEL_RUNTIME_INVENTORY_FAILED",
                f"cannot capture runtime inventory: {exc}",
            ) from exc
        session = open_onnx_session(authority.source)
        teacher, _ = _load_teacher(authority.config)
        inventory["onnx_graph_loaded"] = True
        inventory["model_execution_used"] = False
        inventory["activation_read"] = False
        inventory["execution_provider"] = authority.source.execution_provider
        inventory["process_id"] = os.getpid()
        inventory["process_instance_id"] = PROCESS_INSTANCE_ID
        runtime = inventory.get("onnxruntime")
        if not isinstance(runtime, dict):
            raise _fail(
                "SENTINEL_RUNTIME_INVENTORY_FAILED",
                "ONNX Runtime inventory is not mutable",
            )
        runtime["session_created"] = True
        runtime["active_providers"] = list(session.get_providers())
        runtime["provider_options"] = session.get_provider_options()
        self._inventory = inventory
        self._session = session
        self._teacher = teacher

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int,
        label: str,
    ) -> np.ndarray:
        values = encode_onnx_source(
            self._teacher,
            self._session,
            texts,
            batch_size,
            label,
        )
        self._inventory["model_execution_used"] = True
        return values

    def runtime_inventory(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._inventory)


def create_onnx_sentinel_backend(
    authority: SentinelAuthority,
) -> OnnxSentinelBackend:
    return OnnxSentinelBackend(authority)
