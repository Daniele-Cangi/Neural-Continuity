"""Segmented full-input capture helper; not an executable authority or CLI."""

from __future__ import annotations

import hashlib
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_b.onnx_source import encode_onnx_source
from neural_continuity.m1_diagnostics.cuda_null_full_epoch_format import (
    EMBEDDING_DIMENSION,
    PROVIDERS,
)
from neural_continuity.m1_diagnostics.cuda_null_full_raw_replay import replay_full_raw_run
from neural_continuity.m1_diagnostics.cuda_null_paths import has_linked_ancestor
from neural_continuity.m1_diagnostics.cuda_preflight_runtime import (
    _profile_summary,
    _profiled_session,
)

_SEGMENT_ITEMS = 256


class FullSegmentCaptureBlocked(RuntimeError):
    """A profiled source segment could not be retained and replayed exactly."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise FullSegmentCaptureBlocked(reason)


def _capture_profiled_role(
    teacher: Any,
    source: Path,
    *,
    ids: list[str],
    texts: list[str],
    run_label: str,
    role: str,
    batch_size: int,
    staging: Path,
    scratch: Path,
) -> dict[str, Any]:
    """Capture one run/role only after the caller has verified execution authority."""
    _require(
        len(ids) == len(texts)
        and bool(ids)
        and len(set(ids)) == len(ids)
        and all(isinstance(item, str) and bool(item) for item in ids),
        "capture input identities differ",
    )
    _require(
        staging.is_dir()
        and scratch.is_dir()
        and not has_linked_ancestor(staging)
        and not has_linked_ancestor(scratch),
        "capture staging or scratch path is invalid",
    )
    observation_path = staging / "observations" / run_label / f"{role}.npy"
    profile_directory = staging / "profiles" / run_label / role
    _require(
        not observation_path.exists() and not profile_directory.exists(),
        "capture output already exists",
    )
    observation_path.parent.mkdir(parents=True, exist_ok=True)
    profile_directory.mkdir(parents=True)
    output = np.lib.format.open_memmap(
        observation_path,
        mode="w+",
        dtype="<f4",
        shape=(len(ids), EMBEDDING_DIMENSION),
    )
    segments: list[dict[str, Any]] = []
    for number, start in enumerate(range(0, len(ids), _SEGMENT_ITEMS), start=1):
        end = min(start + _SEGMENT_ITEMS, len(ids))
        label = f"{run_label}.{role}.segment-{number:04d}"
        session = _profiled_session(source, scratch / label, PROVIDERS)
        try:
            embeddings = encode_onnx_source(
                teacher,
                session,
                texts[start:end],
                batch_size,
                label,
            )
        finally:
            raw_profile = Path(session.end_profiling())
            del session
        _require(
            embeddings.dtype == np.dtype("float32")
            and embeddings.shape == (end - start, EMBEDDING_DIMENSION)
            and bool(np.isfinite(embeddings).all()),
            "profiled segment embedding differs",
        )
        block = np.ascontiguousarray(embeddings, dtype="<f4")
        output[start:end] = block
        relative_profile = f"profiles/{run_label}/{role}/segment-{number:04d}.json"
        retained_profile = staging / relative_profile
        _require(not retained_profile.exists(), "profile segment already exists")
        shutil.copyfile(raw_profile, retained_profile)
        profile_sha256 = sha256_file(retained_profile)
        summary = _profile_summary(raw_profile, PROVIDERS)
        _require(
            summary["unclassified_cpu_events"] == 0 and summary["undeclared_providers"] == [],
            "provider profile is incomplete",
        )
        segments.append(
            {
                "run_label": run_label,
                "role": role,
                "start": start,
                "end": end,
                "batch_size": batch_size,
                "ordered_ids_sha256": hashlib.sha256(
                    canonical_json_bytes(ids[start:end])
                ).hexdigest(),
                "embeddings_sha256": hashlib.sha256(block.tobytes(order="C")).hexdigest(),
                "profile_path": relative_profile,
                "profile_sha256": profile_sha256,
                "inference_call_count": math.ceil((end - start) / batch_size),
                "provider_event_counts": summary["provider_event_counts"],
                "operator_event_counts": summary["operator_event_counts"],
            }
        )
    output.flush()
    del output
    record = {
        "run_label": run_label,
        "role": role,
        "array_path": f"observations/{run_label}/{role}.npy",
        "array_sha256": sha256_file(observation_path),
        "ordered_ids_sha256": hashlib.sha256(canonical_json_bytes(ids)).hexdigest(),
        "embedding_dtype": "float32_le",
        "embedding_dimension": EMBEDDING_DIMENSION,
        "output_normalization": "l2_unit_after_encode",
        "segments": segments,
    }
    replay = replay_full_raw_run(
        staging,
        run_label=run_label,
        role=role,
        ordered_ids=ids,
        record=record,
    )
    _require(replay.get("status") == "FULL_RAW_RUN_REPLAY_PASS", "captured role did not replay")
    return record
