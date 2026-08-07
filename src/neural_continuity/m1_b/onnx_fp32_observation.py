from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_b.onnx_int8_observation import (
    OBSERVATION_FORMAT_VERSION,
    TargetRun,
    _observation_config,
    _ordered_query_texts,
    _write_observations,
)
from neural_continuity.m1_b.onnx_source import (
    encode_onnx_source,
    load_verified_onnx_source,
    open_onnx_session,
)
from neural_continuity.m1_teacher_evidence import (
    EVIDENCE_FORMAT_VERSION,
    REPLAY_FORMAT_VERSION,
    TeacherEvidenceError,
    _artifact_entry,
    _fail,
    _load_config,
    _load_json,
    _load_teacher,
    _ordered_observation,
    _require_mapping,
    _require_string,
    _safe_artifact_path,
    _verify_artifacts,
    _write_bytes,
    load_materialized_dataset,
)


def capture_fp32_source_observation(
    config_path: str | Path,
    dataset_directory: str | Path,
    transition_a_bundle: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    config, config_sha256 = _load_config(config_path)
    contract_path = _require_string(config.get("contract_path"), "contract_path")
    contract = _load_json(Path(contract_path), "TRANSITION_B_CONTRACT_INVALID")
    if contract.get("contract_id") != "m1-transition-b-v1":
        raise _fail("TRANSITION_B_CONTRACT_INVALID", "contract_id must be m1-transition-b-v1")
    batch_sizes = _observation_config(config, contract)
    dataset = load_materialized_dataset(dataset_directory)
    expected_dataset_id = _require_string(
        _require_mapping(config.get("dataset"), "dataset").get("dataset_id"), "dataset.dataset_id"
    )
    if dataset.dataset_id != expected_dataset_id:
        raise _fail("DATASET_ID_MISMATCH", "materialized dataset identity differs")
    source = load_verified_onnx_source(transition_a_bundle, contract)
    session = open_onnx_session(source)
    teacher, teacher_manifest = _load_teacher(config)
    query_texts = _ordered_query_texts(dataset)
    runs = [
        TargetRun(
            run_id=f"batch-size-{batch_size:04d}",
            batch_size=batch_size,
            observation=_ordered_observation(
                dataset,
                encode_onnx_source(
                    teacher, session, dataset.document_texts, batch_size, "FP32 documents"
                ),
                encode_onnx_source(teacher, session, query_texts, batch_size, "FP32 queries"),
            ),
        )
        for batch_size in batch_sizes
    ]
    output = Path(output_directory).resolve()
    if output.exists():
        raise _fail("OUTPUT_ALREADY_EXISTS", f"output already exists: {output}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        artifact = temporary / "teacher-fp32.onnx"
        shutil.copyfile(source.artifact_path, artifact)
        observations = temporary / "source-observations.npz"
        _write_observations(observations, runs)
        reference = runs[0].observation
        metadata = {
            "format_version": OBSERVATION_FORMAT_VERSION,
            "dataset_id": dataset.dataset_id,
            "document_count": len(reference.document_ids),
            "query_count": len(reference.query_ids),
            "embedding_dimension": int(reference.document_embeddings.shape[1]),
            "embedding_dtype": "float32",
            "output_normalization": "l2_unit_after_encode",
            "query_roles": reference.query_roles,
            "qrels": reference.relevant_document_ids,
            "required_runs": [{"run_id": run.run_id, "batch_size": run.batch_size} for run in runs],
            "comparison_state": "TARGET_OBSERVATION_PENDING",
        }
        metadata_path = temporary / "observation-metadata.json"
        _write_bytes(metadata_path, canonical_json_bytes(metadata) + b"\n")
        bundle = {
            "replay_format_version": REPLAY_FORMAT_VERSION,
            "observation_path": observations.name,
            "metadata_path": metadata_path.name,
            "required_runs": metadata["required_runs"],
            "replay_requires_model_execution": False,
            "transition_b_decision": "NOT_EVALUATED",
        }
        bundle_path = temporary / "replay-bundle.json"
        _write_bytes(bundle_path, canonical_json_bytes(bundle) + b"\n")
        artifacts = [
            _artifact_entry(temporary, path)
            for path in (artifact, observations, metadata_path, bundle_path)
        ]
        manifest = {
            "evidence_format_version": EVIDENCE_FORMAT_VERSION,
            "package_kind": "m1_onnx_fp32_source_observation",
            "evidence_status": "CAPTURED_PENDING_TARGET_COMPARISON",
            "transition_b_decision": "NOT_EVALUATED",
            "contract_id": contract["contract_id"],
            "contract_sha256": sha256_file(Path(contract_path)),
            "configuration_sha256": config_sha256,
            "source_identity": {
                "transition_a_evidence_manifest_sha256": source.transition_a_manifest_sha256,
                "onnx_fp32_artifact_sha256": source.artifact_sha256,
                "execution_provider": source.execution_provider,
            },
            "artifacts": sorted(artifacts, key=lambda entry: entry["path"]),
        }
        _write_bytes(temporary / "evidence-manifest.json", canonical_json_bytes(manifest) + b"\n")
        os.replace(temporary, output)
        return {"status": "PASS", "output_directory": str(output), "target_run_count": len(runs)}
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def replay_fp32_source_observation(bundle_path: str | Path) -> dict[str, Any]:
    bundle = Path(bundle_path).resolve()
    root = bundle.parent
    replay = _load_json(bundle, "SOURCE_REPLAY_BUNDLE_INVALID")
    manifest = _load_json(root / "evidence-manifest.json", "SOURCE_EVIDENCE_MANIFEST_INVALID")
    if manifest.get("package_kind") != "m1_onnx_fp32_source_observation":
        raise _fail("SOURCE_EVIDENCE_MANIFEST_INVALID", "package kind is not authoritative")
    _verify_artifacts(root, manifest, "artifacts")
    observations = _safe_artifact_path(
        root, _require_string(replay.get("observation_path"), "observation_path")
    )
    metadata = _load_json(
        _safe_artifact_path(root, _require_string(replay.get("metadata_path"), "metadata_path")),
        "SOURCE_OBSERVATION_METADATA_INVALID",
    )
    required_runs = replay.get("required_runs")
    if not isinstance(required_runs, list) or required_runs != metadata.get("required_runs"):
        raise _fail("MISSING_DECLARED_SOURCE_OBSERVATION", "declared source runs are invalid")
    try:
        with np.load(observations, allow_pickle=False) as archive:
            run_ids = archive["run_ids"].astype(str).tolist()
            batch_sizes = archive["batch_sizes"].astype(int).tolist()
            document_embeddings = archive["document_embeddings"]
            query_embeddings = archive["query_embeddings"]
    except Exception as exc:
        raise _fail(
            "SOURCE_REPLAY_OBSERVATION_INVALID",
            f"cannot load observations: {exc}",
        ) from exc
    actual_runs = [
        {"run_id": run_id, "batch_size": batch_size}
        for run_id, batch_size in zip(run_ids, batch_sizes, strict=True)
    ]
    if actual_runs != required_runs or len(actual_runs) != 3:
        raise _fail("MISSING_DECLARED_SOURCE_OBSERVATION", "source runs do not match declaration")
    if (
        document_embeddings.ndim != 3
        or query_embeddings.ndim != 3
        or document_embeddings.shape[0] != 3
        or query_embeddings.shape[0] != 3
        or not np.isfinite(document_embeddings).all()
        or not np.isfinite(query_embeddings).all()
    ):
        raise _fail("SOURCE_REPLAY_OBSERVATION_INVALID", "source embeddings are invalid")
    return {
        "status": "PASS",
        "replay_verified": True,
        "model_execution_used": False,
        "source_run_count": 3,
        "transition_b_decision": "NOT_EVALUATED",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture M1 ONNX FP32 source observations.")
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--config", required=True)
    capture.add_argument("--dataset", required=True)
    capture.add_argument("--transition-a-bundle", required=True)
    capture.add_argument("--output", required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("--bundle", required=True)
    args = parser.parse_args(argv)
    try:
        result = (
            capture_fp32_source_observation(
                args.config, args.dataset, args.transition_a_bundle, args.output
            )
            if args.command == "capture"
            else replay_fp32_source_observation(args.bundle)
        )
    except TeacherEvidenceError as exc:
        print(json.dumps({"status": exc.status, "error": exc.__dict__}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
