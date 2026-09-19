"""One bounded, nonqualifying raw-profile storage probe."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from neural_continuity.evidence import sha256_file
from neural_continuity.m1_b.onnx_source import encode_onnx_source
from neural_continuity.m1_diagnostics.cuda_null_full_profile_probe_authority import (
    BUDGET_MANIFEST_SHA256,
    DATASET_SHA256,
    GATE_MANIFEST_SHA256,
    PLAN_SHA256,
    REVIEW_SHA256,
    SOURCE_SHA256,
    FullProfileProbeBlocked,
    verify_full_profile_probe_authority,
)
from neural_continuity.m1_diagnostics.cuda_null_full_profile_probe_replay import (
    write_full_profile_probe,
)
from neural_continuity.m1_diagnostics.cuda_null_sentinel_capture import _population
from neural_continuity.m1_diagnostics.cuda_null_sentinel_readiness import EPOCH_LAYOUT
from neural_continuity.m1_diagnostics.cuda_null_source_preflight_runtime import (
    _reverify_teacher_source,
)
from neural_continuity.m1_diagnostics.cuda_preflight_runtime import (
    _profile_summary,
    _profiled_session,
)

_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")
_RAW_OBSERVATION_BYTES = 3_881_041_920
_METADATA_RESERVE_BYTES = 2 * 1024**3
_OPERATIONAL_FREE_RESERVE_BYTES = 20 * 1024**3


def _capture_profile(
    teacher: Any,
    source: Path,
    texts: list[str],
    *,
    run_label: str,
    role: str,
    batch_size: int,
    scratch: Path,
) -> tuple[dict[str, Any], Path]:
    label = f"{run_label}.{role}"
    session = _profiled_session(source, scratch / label, _PROVIDERS)
    try:
        embeddings = encode_onnx_source(teacher, session, texts, batch_size, label)
    finally:
        raw = Path(session.end_profiling())
        del session
    if embeddings.shape != (len(texts), 384):
        raise FullProfileProbeBlocked("profile probe embedding shape differs")
    embedding_sha256 = hashlib.sha256(embeddings.astype("<f4", copy=False).tobytes()).hexdigest()
    del embeddings
    retained = scratch / "retained" / f"{label}.json"
    retained.parent.mkdir(exist_ok=True)
    shutil.copyfile(raw, retained)
    summary = _profile_summary(raw, _PROVIDERS)
    if summary["unclassified_cpu_events"] != 0 or summary["undeclared_providers"] != []:
        raise FullProfileProbeBlocked("profile probe provider classification differs")
    relative = f"profiles/{label}.json"
    return (
        {
            "run_label": run_label,
            "role": role,
            "batch_size": batch_size,
            "item_count": len(texts),
            "inference_call_count": math.ceil(len(texts) / batch_size),
            "embedding_sha256": embedding_sha256,
            "profile_path": relative,
            "profile_sha256": sha256_file(retained),
            "size_bytes": retained.stat().st_size,
            "provider_event_counts": summary["provider_event_counts"],
            "operator_event_counts": summary["operator_event_counts"],
        },
        retained,
    )


def capture_full_profile_probe(external_spec_sha256: str) -> dict[str, Any]:
    """Run eight bounded profiles; this cannot create qualifying evidence."""
    authority = verify_full_profile_probe_authority(external_spec_sha256)
    paths = authority.sentinel.paths
    _document_ids, documents, _query_ids, queries, source = _population(paths)
    if len(documents) != 256 or len(queries) != 81:
        raise FullProfileProbeBlocked("profile probe population differs")
    _reverify_teacher_source(paths["transition_a_bundle"], paths["teacher_snapshot_root"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer

    teacher = SentenceTransformer(str(paths["teacher_snapshot_root"]), device="cpu")
    teacher.eval()
    if teacher.max_seq_length != 256:
        raise FullProfileProbeBlocked("teacher preprocessing length differs")
    free_bytes = shutil.disk_usage(authority.output_directory.parent).free
    profiles: list[dict[str, Any]] = []
    raw_profiles: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(
        prefix="cuda-full-profile-probe-", dir=paths["scratch_root"]
    ) as temporary:
        scratch = Path(temporary)
        for run_label, batch_size in EPOCH_LAYOUT:
            for role, texts in (("documents", documents), ("measurement_null_queries", queries)):
                profile, raw = _capture_profile(
                    teacher,
                    source,
                    texts,
                    run_label=run_label,
                    role=role,
                    batch_size=batch_size,
                    scratch=scratch,
                )
                profiles.append(profile)
                raw_profiles[profile["profile_path"]] = raw
        per_epoch = sum(
            profile["size_bytes"] * (math.ceil(5183 / 256) if profile["role"] == "documents" else 1)
            for profile in profiles
        )
        projected = per_epoch * 120
        required = _RAW_OBSERVATION_BYTES + 2 * projected + _METADATA_RESERVE_BYTES
        minimum_free = required + _OPERATIONAL_FREE_RESERVE_BYTES
        record = {
            "kind": "m1_cuda_full_profile_storage_probe_observation",
            "version": "1.0.0",
            "status": "TECHNICAL_PROFILE_STORAGE_CAPTURED_NOT_QUALIFYING",
            "review_sha256": REVIEW_SHA256,
            "execution_plan_sha256": PLAN_SHA256,
            "technical_gate_manifest_sha256": GATE_MANIFEST_SHA256,
            "budget_preflight_manifest_sha256": BUDGET_MANIFEST_SHA256,
            "dataset_manifest_sha256": DATASET_SHA256,
            "source_onnx_sha256": SOURCE_SHA256,
            "document_profile_item_count": 256,
            "query_profile_item_count": 81,
            "maximum_profile_segment_items": 256,
            "run_layout": [{"label": label, "batch_size": size} for label, size in EPOCH_LAYOUT],
            "ordered_providers": list(_PROVIDERS),
            "profiles": profiles,
            "raw_observation_bytes_120_epochs": _RAW_OBSERVATION_BYTES,
            "projected_profile_bytes_per_epoch": per_epoch,
            "projected_profile_bytes_120_epochs": projected,
            "profile_projection_safety_multiplier": 2,
            "metadata_reserve_bytes": _METADATA_RESERVE_BYTES,
            "operational_free_reserve_bytes": _OPERATIONAL_FREE_RESERVE_BYTES,
            "required_evidence_storage_bytes": required,
            "minimum_free_bytes_before_execution": minimum_free,
            "free_bytes_before_probe": free_bytes,
            "storage_budget_satisfied": free_bytes >= minimum_free,
            "projection_is_guaranteed_upper_bound": False,
            "raw_profiles_retained": True,
            "embeddings_retained": False,
            "rankings_or_metrics_retained": False,
            "qualifying_detection_evidence": False,
            "full_corpus_qualification_started": False,
            "full_corpus_execution_authorized": False,
            "candidate_or_int8_executed": False,
            "holdout_accessed": False,
            "scientific_decision": "NOT_EVALUATED",
        }
        output, manifest_sha256 = write_full_profile_probe(
            authority.output_directory, record, raw_profiles
        )
    return {
        "status": record["status"],
        "output": str(output),
        "artifact_manifest_sha256": manifest_sha256,
        "projected_profile_bytes_120_epochs": projected,
        "required_evidence_storage_bytes": required,
        "minimum_free_bytes_before_execution": minimum_free,
        "free_bytes_before_probe": free_bytes,
        "storage_budget_satisfied": record["storage_budget_satisfied"],
        "full_corpus_execution_authorized": False,
    }
