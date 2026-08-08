from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.m1_teacher_evidence import (
    TeacherEvidenceError,
    _fail,
    _load_json,
    _safe_artifact_path,
    _verify_artifacts,
)


def _package(
    bundle_path: str | Path, kind: str
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    bundle = Path(bundle_path).resolve()
    root = bundle.parent
    manifest = _load_json(root / "evidence-manifest.json", "PAIRING_MANIFEST_INVALID")
    if manifest.get("package_kind") != kind:
        raise _fail("PAIRING_PACKAGE_KIND_INVALID", f"expected {kind}")
    _verify_artifacts(root, manifest, "artifacts")
    replay = _load_json(bundle, "PAIRING_REPLAY_BUNDLE_INVALID")
    metadata = _load_json(
        _safe_artifact_path(root, str(replay.get("metadata_path", ""))),
        "PAIRING_METADATA_INVALID",
    )
    return root, manifest, replay, metadata


def verify_paired_observation_compatibility(
    source_bundle: str | Path, target_bundle: str | Path
) -> dict[str, Any]:
    source_root, source_manifest, source_replay, source_metadata = _package(
        source_bundle, "m1_onnx_fp32_source_observation"
    )
    target_root, target_manifest, target_replay, target_metadata = _package(
        target_bundle, "m1_onnx_int8_target_observation"
    )
    required_manifest_fields = ("dataset", "teacher_tokenizer_identity")
    if any(
        field not in source_manifest or field not in target_manifest
        for field in required_manifest_fields
    ):
        raise _fail("MISSING_DECLARED_SOURCE_OBSERVATION", "pairing identity is incomplete")
    if source_manifest["dataset"] != target_manifest["dataset"]:
        raise _fail("PAIRED_OBSERVATION_IDENTITY_MISMATCH", "dataset differs")
    tokenizer_fields = (
        "model_id",
        "revision",
        "snapshot_files",
        "device",
        "cache_only",
        "output_dtype",
        "encode_normalize_embeddings",
        "effective_output_normalization",
    )
    for field in tokenizer_fields:
        if source_manifest["teacher_tokenizer_identity"].get(field) != target_manifest[
            "teacher_tokenizer_identity"
        ].get(field):
            raise _fail("PAIRED_OBSERVATION_IDENTITY_MISMATCH", f"tokenizer.{field} differs")
    for field in (
        "dataset_id",
        "document_count",
        "query_count",
        "embedding_dimension",
        "embedding_dtype",
        "output_normalization",
        "query_roles",
        "qrels",
        "required_runs",
    ):
        if source_metadata.get(field) != target_metadata.get(field):
            raise _fail("PAIRED_OBSERVATION_IDENTITY_MISMATCH", f"metadata.{field} differs")
    if source_replay.get("required_runs") != target_replay.get("required_runs"):
        raise _fail("PAIRED_OBSERVATION_BATCH_MISMATCH", "required execution batches differ")
    arrays: list[dict[str, np.ndarray]] = []
    for root, replay in ((source_root, source_replay), (target_root, target_replay)):
        path = _safe_artifact_path(root, str(replay.get("observation_path", "")))
        try:
            with np.load(path, allow_pickle=False) as archive:
                arrays.append(
                    {key: archive[key] for key in ("document_ids", "query_ids", "query_roles")}
                )
        except Exception as exc:
            raise _fail("PAIRED_OBSERVATION_INVALID", f"cannot load observations: {exc}") from exc
    if any(not np.array_equal(arrays[0][key], arrays[1][key]) for key in arrays[0]):
        raise _fail("PAIRED_OBSERVATION_IDENTITY_MISMATCH", "canonical IDs or roles differ")
    return {
        "status": "PASS",
        "compatible": True,
        "comparison_state": "READY_FOR_PURE_COMPARISON",
        "transition_b_decision": "NOT_EVALUATED",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify M1 FP32/INT8 observation compatibility.")
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--target-bundle", required=True)
    args = parser.parse_args(argv)
    try:
        result = verify_paired_observation_compatibility(args.source_bundle, args.target_bundle)
    except TeacherEvidenceError as exc:
        print(json.dumps({"status": exc.status, "error": exc.__dict__}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
