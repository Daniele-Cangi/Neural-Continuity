from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_b.onnx_source import encode_onnx_source, open_onnx_session
from neural_continuity.m1_b.static_quantization import load_verified_static_quantized_candidate
from neural_continuity.m1_teacher_evidence import (
    EVIDENCE_FORMAT_VERSION,
    REPLAY_FORMAT_VERSION,
    ROLE_ORDER,
    MaterializedDataset,
    TeacherEvidenceError,
    TeacherObservation,
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

OBSERVATION_FORMAT_VERSION = "1.0.0"
_REQUIRED_BATCH_SIZES = (1, 16, 64)


@dataclass(frozen=True)
class TargetRun:
    run_id: str
    batch_size: int
    observation: TeacherObservation


def _ordered_query_texts(dataset: MaterializedDataset) -> list[str]:
    texts: list[str] = []
    for role in ROLE_ORDER:
        role_data = dataset.roles.get(role)
        if role_data is None:
            raise _fail("TARGET_OBSERVATION_ROLE_MISSING", f"materialized dataset lacks {role}")
        texts.extend(
            text
            for _, text in sorted(
                zip(role_data.query_ids, role_data.query_texts, strict=True),
                key=lambda item: item[0].encode(),
            )
        )
    return texts


def _observation_config(config: Mapping[str, Any], contract: Mapping[str, Any]) -> list[int]:
    observation = _require_mapping(config.get("observation"), "observation")
    if observation.get("execution_provider") != "CPUExecutionProvider":
        raise _fail("TARGET_OBSERVATION_CONFIG_INVALID", "only CPUExecutionProvider is authorized")
    batch_sizes = observation.get("batch_sizes")
    if not isinstance(batch_sizes, list) or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in batch_sizes
    ):
        raise _fail("TARGET_OBSERVATION_CONFIG_INVALID", "observation.batch_sizes are invalid")
    required = _require_mapping(
        _require_mapping(contract.get("preconditions"), "preconditions").get("target_capture"),
        "preconditions.target_capture",
    ).get("required_batch_sizes")
    if batch_sizes != required or tuple(batch_sizes) != _REQUIRED_BATCH_SIZES:
        raise _fail(
            "TARGET_OBSERVATION_CONFIG_INVALID",
            "batch sizes differ from the frozen contract",
        )
    return batch_sizes


def _write_observations(path: Path, runs: Sequence[TargetRun]) -> None:
    reference = runs[0].observation
    for run in runs:
        observation = run.observation
        if (
            observation.document_ids != reference.document_ids
            or observation.query_ids != reference.query_ids
            or observation.query_roles != reference.query_roles
            or observation.relevant_document_ids != reference.relevant_document_ids
        ):
            raise _fail("TARGET_OBSERVATION_INVALID", "target runs lack canonical identity")
    np.savez_compressed(
        path,
        run_ids=np.asarray([run.run_id for run in runs]),
        batch_sizes=np.asarray([run.batch_size for run in runs], dtype=np.int64),
        document_ids=np.asarray(reference.document_ids),
        query_ids=np.asarray(reference.query_ids),
        query_roles=np.asarray(reference.query_roles),
        document_embeddings=np.stack([run.observation.document_embeddings for run in runs]),
        query_embeddings=np.stack([run.observation.query_embeddings for run in runs]),
    )


def _capture_manifest(
    temporary: Path,
    dataset: MaterializedDataset,
    runs: Sequence[TargetRun],
    candidate: Any,
    contract_sha256: str,
    config_sha256: str,
    teacher_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    observation_path = temporary / "target-observations.npz"
    metadata_path = temporary / "observation-metadata.json"
    candidate_path = temporary / "teacher-int8-qdq.onnx"
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
        "comparison_state": "SOURCE_OBSERVATION_PENDING",
    }
    _write_bytes(metadata_path, canonical_json_bytes(metadata) + b"\n")
    replay_bundle = {
        "replay_format_version": REPLAY_FORMAT_VERSION,
        "observation_format_version": OBSERVATION_FORMAT_VERSION,
        "observation_path": observation_path.name,
        "metadata_path": metadata_path.name,
        "candidate_path": candidate_path.name,
        "required_runs": metadata["required_runs"],
        "replay_requires_model_execution": False,
        "transition_b_decision": "NOT_EVALUATED",
    }
    bundle_path = temporary / "replay-bundle.json"
    _write_bytes(bundle_path, canonical_json_bytes(replay_bundle) + b"\n")
    artifacts = [
        _artifact_entry(temporary, path)
        for path in (candidate_path, observation_path, metadata_path, bundle_path)
    ]
    artifacts.sort(key=lambda entry: entry["path"])
    return {
        "evidence_format_version": EVIDENCE_FORMAT_VERSION,
        "package_kind": "m1_onnx_int8_target_observation",
        "evidence_status": "CAPTURED_PENDING_SOURCE_COMPARISON",
        "transition_b_decision": "NOT_EVALUATED",
        "contract_id": "m1-transition-b-v1",
        "contract_sha256": contract_sha256,
        "configuration_sha256": config_sha256,
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "materialization_manifest_sha256": dataset.manifest_sha256,
            "materialization_policy_sha256": dataset.materialization_policy_sha256,
            "partition_policy_sha256": dataset.partition_policy_sha256,
        },
        "candidate_identity": {
            "candidate_manifest_sha256": candidate.candidate_manifest_sha256,
            "onnx_int8_artifact_sha256": candidate.artifact_sha256,
            "execution_provider": candidate.execution_provider,
        },
        "teacher_tokenizer_identity": dict(teacher_manifest),
        "artifacts": artifacts,
        "integrity": {
            "artifact_hash_algorithm": "SHA-256",
            "missing_evidence_behavior": "BLOCKED",
            "replay_without_model_execution_required": True,
        },
    }


def capture_int8_target_observation(
    config_path: str | Path,
    dataset_directory: str | Path,
    candidate_package: str | Path,
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
    candidate = load_verified_static_quantized_candidate(candidate_package, contract_path)
    session = open_onnx_session(candidate)
    teacher, teacher_manifest = _load_teacher(config)
    query_texts = _ordered_query_texts(dataset)
    runs = [
        TargetRun(
            run_id=f"batch-size-{batch_size:04d}",
            batch_size=batch_size,
            observation=_ordered_observation(
                dataset,
                encode_onnx_source(
                    teacher, session, dataset.document_texts, batch_size, "INT8 documents"
                ),
                encode_onnx_source(teacher, session, query_texts, batch_size, "INT8 queries"),
            ),
        )
        for batch_size in batch_sizes
    ]
    output = Path(output_directory).resolve()
    if output.exists():
        raise _fail("OUTPUT_ALREADY_EXISTS", f"output already exists: {output}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        shutil.copyfile(candidate.artifact_path, temporary / "teacher-int8-qdq.onnx")
        _write_observations(temporary / "target-observations.npz", runs)
        teacher_manifest = {
            **teacher_manifest,
            "configuration_sha256": config_sha256,
            "target_execution_used": "ONNX INT8 only",
        }
        manifest = _capture_manifest(
            temporary,
            dataset,
            runs,
            candidate,
            sha256_file(Path(contract_path)),
            config_sha256,
            teacher_manifest,
        )
        manifest_path = temporary / "evidence-manifest.json"
        _write_bytes(manifest_path, canonical_json_bytes(manifest) + b"\n")
        os.replace(temporary, output)
        return {
            "status": "PASS",
            "evidence_status": "CAPTURED_PENDING_SOURCE_COMPARISON",
            "transition_b_decision": "NOT_EVALUATED",
            "output_directory": str(output),
            "evidence_manifest_sha256": sha256_file(output / "evidence-manifest.json"),
            "target_run_count": len(runs),
        }
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def replay_int8_target_observation(bundle_path: str | Path) -> dict[str, Any]:
    bundle = Path(bundle_path).resolve()
    root = bundle.parent
    replay = _load_json(bundle, "TARGET_REPLAY_BUNDLE_INVALID")
    manifest = _load_json(root / "evidence-manifest.json", "TARGET_EVIDENCE_MANIFEST_INVALID")
    if manifest.get("package_kind") != "m1_onnx_int8_target_observation":
        raise _fail("TARGET_EVIDENCE_MANIFEST_INVALID", "package kind is not authoritative")
    _verify_artifacts(root, manifest, "artifacts")
    observation_path = _safe_artifact_path(
        root, _require_string(replay.get("observation_path"), "observation_path")
    )
    metadata_path = _safe_artifact_path(
        root, _require_string(replay.get("metadata_path"), "metadata_path")
    )
    metadata = _load_json(metadata_path, "TARGET_OBSERVATION_METADATA_INVALID")
    required_runs = replay.get("required_runs")
    if not isinstance(required_runs, list) or required_runs != metadata.get("required_runs"):
        raise _fail("MISSING_DECLARED_TARGET_OBSERVATION", "declared target runs are invalid")
    try:
        with np.load(observation_path, allow_pickle=False) as archive:
            run_ids = archive["run_ids"].astype(str).tolist()
            batch_sizes = archive["batch_sizes"].astype(int).tolist()
            document_embeddings = archive["document_embeddings"]
            query_embeddings = archive["query_embeddings"]
    except Exception as exc:
        raise _fail(
            "TARGET_REPLAY_OBSERVATION_INVALID",
            f"cannot load observations: {exc}",
        ) from exc
    actual_runs = [
        {"run_id": run_id, "batch_size": batch_size}
        for run_id, batch_size in zip(run_ids, batch_sizes, strict=True)
    ]
    if actual_runs != required_runs or len(actual_runs) != len(_REQUIRED_BATCH_SIZES):
        raise _fail("MISSING_DECLARED_TARGET_OBSERVATION", "target runs do not match declaration")
    if (
        document_embeddings.ndim != 3
        or query_embeddings.ndim != 3
        or document_embeddings.shape[0] != len(actual_runs)
        or query_embeddings.shape[0] != len(actual_runs)
        or not np.isfinite(document_embeddings).all()
        or not np.isfinite(query_embeddings).all()
    ):
        raise _fail("TARGET_REPLAY_OBSERVATION_INVALID", "target embeddings are invalid")
    return {
        "status": "PASS",
        "replay_verified": True,
        "model_execution_used": False,
        "target_run_count": len(actual_runs),
        "transition_b_decision": "NOT_EVALUATED",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture M1 ONNX INT8 target observations.")
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--config", required=True)
    capture.add_argument("--dataset", required=True)
    capture.add_argument("--candidate-package", required=True)
    capture.add_argument("--output", required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("--bundle", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = (
            capture_int8_target_observation(
                args.config, args.dataset, args.candidate_package, args.output
            )
            if args.command == "capture"
            else replay_int8_target_observation(args.bundle)
        )
    except TeacherEvidenceError as exc:
        print(json.dumps({"status": exc.status, "error": exc.__dict__}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
