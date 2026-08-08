from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.evidence import sha256_file
from neural_continuity.m1_b.onnx_source import (
    encode_onnx_source,
    load_verified_onnx_source,
    open_onnx_session,
)
from neural_continuity.m1_measurement_null import (
    BATCH_SIZE_VARIATION_FAMILY,
    REPEATED_INFERENCE_FAMILY,
    SourceRun,
    _measurement_config,
    _measurement_observation,
    replay_measurement_null,
    write_measurement_null_package,
)
from neural_continuity.m1_teacher_evidence import (
    TeacherEvidenceError,
    _fail,
    _load_config,
    _load_json,
    _load_teacher,
    _require_mapping,
    _require_string,
    load_materialized_dataset,
)


def _load_contract(path: str | Path) -> tuple[dict[str, Any], str]:
    contract_path = Path(path)
    contract = _load_json(contract_path, "TRANSITION_B_CONTRACT_INVALID")
    if contract.get("contract_id") != "m1-transition-b-v1":
        raise _fail("TRANSITION_B_CONTRACT_INVALID", "contract_id must be m1-transition-b-v1")
    return contract, sha256_file(contract_path)


def _measurement_query_texts(dataset: Any) -> list[str]:
    role = dataset.roles.get("measurement_null")
    if role is None:
        raise _fail("MEASUREMENT_NULL_ROLE_MISSING", "materialized dataset lacks measurement_null")
    return [
        text
        for _, text in sorted(
            zip(role.query_ids, role.query_texts, strict=True), key=lambda item: item[0].encode()
        )
    ]


def capture_onnx_measurement_null(
    config_path: str | Path,
    dataset_directory: str | Path,
    transition_a_bundle: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    config, config_sha256 = _load_config(config_path)
    contract, contract_sha256 = _load_contract(
        _require_string(config.get("contract_path"), "contract_path")
    )
    repeat_count, repeated_batch_size, batch_sizes = _measurement_config(config)
    dataset = load_materialized_dataset(dataset_directory)
    expected_dataset_id = _require_string(
        _require_mapping(config.get("dataset"), "dataset").get("dataset_id"), "dataset.dataset_id"
    )
    if dataset.dataset_id != expected_dataset_id:
        raise _fail(
            "DATASET_ID_MISMATCH",
            f"expected dataset {expected_dataset_id}, got {dataset.dataset_id}",
        )
    source = load_verified_onnx_source(transition_a_bundle, contract)
    session = open_onnx_session(source)
    teacher, teacher_manifest = _load_teacher(config)
    query_texts = _measurement_query_texts(dataset)
    conditions = [
        (f"repeated-inference-{index:03d}", REPEATED_INFERENCE_FAMILY, repeated_batch_size)
        for index in range(1, repeat_count + 1)
    ] + [
        (f"batch-size-{batch_size:04d}", BATCH_SIZE_VARIATION_FAMILY, batch_size)
        for batch_size in batch_sizes
    ]
    runs: list[SourceRun] = []
    for run_id, family, batch_size in conditions:
        document_embeddings = encode_onnx_source(
            teacher, session, dataset.document_texts, batch_size, f"{run_id} documents"
        )
        query_embeddings = encode_onnx_source(
            teacher, session, query_texts, batch_size, f"{run_id} measurement-null queries"
        )
        runs.append(
            SourceRun(
                run_id=run_id,
                family=family,
                batch_size=batch_size,
                observation=_measurement_observation(
                    dataset, document_embeddings, query_embeddings
                ),
            )
        )
    source_identity = {
        "transition_a_evidence_manifest_sha256": source.transition_a_manifest_sha256,
        "onnx_fp32_artifact_sha256": source.artifact_sha256,
        "execution_provider": source.execution_provider,
        "contract_id": contract["contract_id"],
        "contract_sha256": contract_sha256,
    }
    teacher_manifest = {
        **teacher_manifest,
        "configuration_sha256": config_sha256,
        "dataset_id": dataset.dataset_id,
        "materialization_manifest_sha256": dataset.manifest_sha256,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "source_execution_used": "ONNX FP32 only",
    }
    top_k = _require_mapping(config.get("evaluation"), "evaluation").get("top_k")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise _fail("MEASUREMENT_NULL_CONFIG_INVALID", "evaluation.top_k must be positive")
    return write_measurement_null_package(
        output_directory=output_directory,
        dataset=dataset,
        runs=runs,
        teacher_manifest=teacher_manifest,
        evidence_scope=_require_mapping(config.get("evidence_scope"), "evidence_scope"),
        config_sha256=config_sha256,
        top_k=top_k,
        source_identity=source_identity,
        evidence_kind="onnx_fp32_measurement_null",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture M1 ONNX FP32 source measurement null.")
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="Capture ONNX FP32 source null evidence.")
    capture.add_argument("--config", required=True)
    capture.add_argument("--dataset", required=True)
    capture.add_argument("--source-bundle", required=True)
    capture.add_argument("--output", required=True)
    replay = commands.add_parser("replay", help="Replay ONNX FP32 source null evidence.")
    replay.add_argument("--bundle", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "capture":
            result = capture_onnx_measurement_null(
                args.config, args.dataset, args.source_bundle, args.output
            )
        else:
            result = replay_measurement_null(args.bundle)
    except TeacherEvidenceError as exc:
        print(json.dumps({"status": exc.status, "error": exc.__dict__}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
