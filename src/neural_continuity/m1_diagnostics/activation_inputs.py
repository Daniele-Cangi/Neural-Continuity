"""Record and replay the actual shared token inputs, without reconstructing masks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.evidence import sha256_file
from neural_continuity.m1_diagnostics.fidelity_authority import FidelityGateError
from neural_continuity.m1_diagnostics.fidelity_evidence import _write_deterministic_npz

INPUT_CAPTURE_VERSION = "1.1.0"
INPUT_NAMES = frozenset({"input_ids", "attention_mask", "token_type_ids"})


def snapshot_token_inputs(
    inputs: Mapping[str, np.ndarray], query_count: int
) -> dict[str, np.ndarray]:
    if set(inputs) != INPUT_NAMES:
        raise FidelityGateError("CAPTURE_INPUT_INVALID", "Token input identities do not match")
    snapshot: dict[str, np.ndarray] = {}
    shape = inputs["attention_mask"].shape
    for name, value in inputs.items():
        if value.dtype != np.int64 or value.ndim != 2 or value.shape != shape:
            raise FidelityGateError("CAPTURE_INPUT_INVALID", "Token input shape or dtype invalid")
        snapshot[name] = np.array(value, copy=True, order="C")
    if shape[0] != query_count or shape[1] == 0 or query_count == 0:
        raise FidelityGateError("CAPTURE_INPUT_INVALID", "Token input batch identity invalid")
    mask = snapshot["attention_mask"]
    if not np.all((mask == 0) | (mask == 1)) or not np.all(np.any(mask == 1, axis=1)):
        raise FidelityGateError(
            "CAPTURE_INPUT_INVALID", "Attention mask must be binary and nonempty"
        )
    return snapshot


def verify_token_inputs_unchanged(
    inputs: Mapping[str, np.ndarray], snapshot: Mapping[str, np.ndarray]
) -> None:
    if set(inputs) != set(snapshot) or any(
        inputs[name].dtype != value.dtype or not np.array_equal(inputs[name], value)
        for name, value in snapshot.items()
    ):
        raise FidelityGateError("CAPTURE_INPUT_MUTATED", "Shared inference inputs changed")


def write_token_input_batch(
    root: Path, batch_id: str, query_ids: Sequence[str], inputs: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    arrays = snapshot_token_inputs(inputs, len(query_ids))
    arrays["query_ids"] = np.asarray(query_ids, dtype=np.str_)
    path = root / f"{batch_id}-inputs.npz"
    if path.resolve().parent != root.resolve():
        raise FidelityGateError("CAPTURE_INPUT_INVALID", "Input archive path escapes package")
    _write_deterministic_npz(path, arrays)
    return {
        "inputs_path": path.name,
        "inputs_sha256": sha256_file(path),
        "inputs_size_bytes": path.stat().st_size,
    }


def load_token_input_batch(root: Path, record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    relative = record.get("inputs_path")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise FidelityGateError("CAPTURE_INPUT_MISSING", "Declared input archive is missing")
    path = (root / relative).resolve()
    if path.parent != root.resolve() or not path.is_file():
        raise FidelityGateError("CAPTURE_INPUT_MISSING", "Declared input archive is missing")
    if path.stat().st_size != record.get("inputs_size_bytes") or sha256_file(path) != record.get(
        "inputs_sha256"
    ):
        raise FidelityGateError("CAPTURE_INPUT_HASH_MISMATCH", "Input archive integrity mismatch")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != INPUT_NAMES | {"query_ids"}:
                raise FidelityGateError("CAPTURE_INPUT_INVALID", "Input archive keys invalid")
            ids = archive["query_ids"]
            if ids.ndim != 1 or ids.dtype.kind != "U" or ids.tolist() != record.get("query_ids"):
                raise FidelityGateError("CAPTURE_INPUT_INVALID", "Input archive query IDs mismatch")
            return snapshot_token_inputs({name: archive[name] for name in INPUT_NAMES}, len(ids))
    except (OSError, ValueError, KeyError) as exc:
        raise FidelityGateError(
            "CAPTURE_INPUT_INVALID", f"Cannot read input archive: {exc}"
        ) from exc
