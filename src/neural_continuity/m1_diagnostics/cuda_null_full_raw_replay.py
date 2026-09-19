"""Memory-bounded model-free replay for one full-corpus source run and role.

The caller must separately verify the frozen dataset, qrels, epoch package,
checkpoint chain, and the complete four-run layout.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.cuda_null_full_provider_coverage import (
    FullProviderCoverageBlocked,
    replay_full_provider_coverage,
)
from neural_continuity.m1_diagnostics.cuda_null_paths import has_linked_ancestor

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ROLE_COUNTS = {"documents": 5183, "measurement_null_queries": 81}
_RUNS = {"batch_1_primary", "batch_16_primary", "batch_16_repeat", "batch_64_primary"}
_DIMENSION = 384
_BLOCK_ROWS = 64
_NORMALIZATION_BOUND = 1e-4
_FIELDS = {
    "run_label",
    "role",
    "array_path",
    "array_sha256",
    "ordered_ids_sha256",
    "embedding_dtype",
    "embedding_dimension",
    "output_normalization",
    "segments",
}


class FullRawReplayBlocked(ValueError):
    """A declared raw observation or provider profile did not verify."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise FullRawReplayBlocked(reason)


def _sha256(value: Any, label: str) -> None:
    _require(
        isinstance(value, str) and _SHA256.fullmatch(value) is not None,
        f"{label} is not a SHA-256 digest",
    )


def replay_full_raw_run(
    root: Path,
    *,
    run_label: str,
    role: str,
    ordered_ids: Sequence[str],
    record: dict[str, Any],
) -> dict[str, Any]:
    """Verify all rows and their paired profile segments without loading a model."""
    _require(run_label in _RUNS and role in _ROLE_COUNTS, "run or role is not frozen")
    _require(isinstance(record, dict) and set(record) == _FIELDS, "raw-run schema differs")
    _require(record["run_label"] == run_label and record["role"] == role, "run identity differs")
    _require(
        len(ordered_ids) == _ROLE_COUNTS[role]
        and all(isinstance(item, str) and bool(item) for item in ordered_ids)
        and len(set(ordered_ids)) == len(ordered_ids),
        "ordered input identities differ",
    )
    expected_id_hash = hashlib.sha256(canonical_json_bytes(list(ordered_ids))).hexdigest()
    _require(record["ordered_ids_sha256"] == expected_id_hash, "run ID hash differs")
    _require(
        record["embedding_dtype"] == "float32_le"
        and type(record["embedding_dimension"]) is int
        and record["embedding_dimension"] == _DIMENSION
        and record["output_normalization"] == "l2_unit_after_encode",
        "embedding semantics differ",
    )
    _sha256(record["array_sha256"], "observation array")
    _require(not has_linked_ancestor(root), "observation package path contains a link")
    expected_path = f"observations/{run_label}/{role}.npy"
    _require(record["array_path"] == expected_path, "observation array path differs")
    array_path = root / expected_path
    _require(
        array_path.is_file() and not has_linked_ancestor(array_path),
        "observation array missing or linked",
    )
    _require(sha256_file(array_path) == record["array_sha256"], "observation array hash differs")
    try:
        array = np.load(array_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, TypeError) as exc:
        raise FullRawReplayBlocked("observation array cannot be mapped") from exc
    _require(isinstance(array, np.memmap), "observation array is not memory-mapped NPY")
    _require(
        array.dtype == np.dtype("<f4")
        and array.shape == (_ROLE_COUNTS[role], _DIMENSION)
        and bool(array.flags.c_contiguous),
        "observation array dtype, shape, or layout differs",
    )
    segments = record["segments"]
    try:
        provider = replay_full_provider_coverage(
            root,
            run_label=run_label,
            role=role,
            ordered_ids=ordered_ids,
            segments=segments,
        )
    except FullProviderCoverageBlocked as exc:
        raise FullRawReplayBlocked("full-input provider coverage blocked") from exc
    for start in range(0, len(ordered_ids), _BLOCK_ROWS):
        block = np.asarray(array[start : start + _BLOCK_ROWS], dtype=np.float64)
        _require(bool(np.isfinite(block).all()), "observation contains non-finite values")
        norms = np.linalg.norm(block, axis=1)
        _require(
            bool(np.isfinite(norms).all() and np.all(np.abs(norms - 1.0) <= _NORMALIZATION_BOUND)),
            "observation is not L2 normalized",
        )
    for segment in segments:
        start, end = segment["start"], segment["end"]
        block = np.ascontiguousarray(array[start:end], dtype="<f4")
        digest = hashlib.sha256(block.tobytes(order="C")).hexdigest()
        _require(digest == segment["embeddings_sha256"], "profiled segment embedding differs")
    return {
        "status": "FULL_RAW_RUN_REPLAY_PASS",
        "run_label": run_label,
        "role": role,
        "item_count": len(ordered_ids),
        "embedding_dimension": _DIMENSION,
        "array_sha256": record["array_sha256"],
        "provider_coverage": provider,
        "model_loaded": False,
        "scientific_decision": "NOT_EVALUATED",
        "execution_authorized": False,
    }
