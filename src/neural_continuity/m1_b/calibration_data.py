from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_teacher_evidence import (
    TeacherEvidenceError,
    _artifact_entry,
    _fail,
    _load_json,
    _require_mapping,
    _verify_artifacts,
    _write_bytes,
    load_materialized_dataset,
)

_CANONICAL_ORDER = "query ID ascending by UTF-8 bytes"


@dataclass(frozen=True)
class VerifiedCalibrationInputs:
    query_ids: list[str]
    query_texts: list[str]
    manifest_sha256: str
    inputs_sha256: str


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise _fail("CALIBRATION_MANIFEST_INVALID", f"{field} must be a SHA-256 string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise _fail("CALIBRATION_MANIFEST_INVALID", f"{field} must be a SHA-256 string") from exc
    return value


def _load_contract(path: str | Path) -> tuple[dict[str, Any], str]:
    contract_path = Path(path)
    contract = _load_json(contract_path, "TRANSITION_B_CONTRACT_INVALID")
    if contract.get("contract_id") != "m1-transition-b-v1":
        raise _fail("TRANSITION_B_CONTRACT_INVALID", "contract_id must be m1-transition-b-v1")
    calibration = _require_mapping(contract.get("calibration"), "calibration")
    if calibration.get("data_role") != "quantization_calibration":
        raise _fail("CALIBRATION_CONTRACT_INVALID", "contract does not authorize calibration role")
    if calibration.get("canonical_order") != _CANONICAL_ORDER:
        raise _fail("CALIBRATION_CONTRACT_INVALID", "contract canonical order is not authoritative")
    return contract, sha256_file(contract_path)


def build_calibration_package(
    contract_path: str | Path, dataset_directory: str | Path, output_directory: str | Path
) -> dict[str, Any]:
    contract, contract_sha256 = _load_contract(contract_path)
    dataset = load_materialized_dataset(dataset_directory)
    calibration = _require_mapping(contract.get("calibration"), "calibration")
    try:
        role = dataset.roles["quantization_calibration"]
    except KeyError as exc:
        raise _fail(
            "CALIBRATION_IDENTITY_INVALID",
            "materialized dataset lacks quantization_calibration",
        ) from exc
    maximum = calibration.get("max_query_count")
    if not isinstance(maximum, int) or maximum <= 0 or len(role.query_ids) != maximum:
        raise _fail("CALIBRATION_IDENTITY_INVALID", "calibration query count differs from contract")
    pairs = sorted(
        zip(role.query_ids, role.query_texts, strict=True),
        key=lambda item: item[0].encode(),
    )
    if len({query_id for query_id, _ in pairs}) != len(pairs):
        raise _fail("CALIBRATION_IDENTITY_INVALID", "calibration query IDs are duplicated")
    if any(not isinstance(text, str) for _, text in pairs):
        raise _fail("CALIBRATION_IDENTITY_INVALID", "calibration query texts are invalid")
    output = Path(output_directory).resolve()
    if output.exists():
        raise _fail("OUTPUT_ALREADY_EXISTS", f"output already exists: {output}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        records = [{"query_id": query_id, "text": text} for query_id, text in pairs]
        inputs = temporary / "calibration-inputs.jsonl"
        _write_bytes(inputs, b"".join(canonical_json_bytes(record) + b"\n" for record in records))
        manifest = {
            "format_version": "1.0.0",
            "package_kind": "m1_static_quantization_calibration",
            "dataset_id": dataset.dataset_id,
            "contract_id": contract["contract_id"],
            "contract_sha256": contract_sha256,
            "data_role": "quantization_calibration",
            "query_count": len(records),
            "canonical_order": _CANONICAL_ORDER,
            "leakage_behavior": "BLOCKED",
            "source_identity": {
                "materialization_manifest_sha256": dataset.manifest_sha256,
                "partition_policy_sha256": dataset.partition_policy_sha256,
            },
            "artifacts": [_artifact_entry(temporary, inputs)],
        }
        _write_bytes(
            temporary / "calibration-manifest.json",
            canonical_json_bytes(manifest) + b"\n",
        )
        os.replace(temporary, output)
        return {
            "status": "PASS",
            "output_directory": str(output),
            "query_count": len(records),
            "model_execution_used": False,
        }
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def verify_calibration_package(package_directory: str | Path) -> dict[str, Any]:
    root = Path(package_directory).resolve()
    manifest = _load_json(root / "calibration-manifest.json", "CALIBRATION_MANIFEST_INVALID")
    if manifest.get("package_kind") != "m1_static_quantization_calibration":
        raise _fail("CALIBRATION_MANIFEST_INVALID", "package kind is not authoritative")
    if manifest.get("data_role") != "quantization_calibration":
        raise _fail("CALIBRATION_LEAKAGE", "package role is not quantization_calibration")
    if manifest.get("canonical_order") != _CANONICAL_ORDER:
        raise _fail("CALIBRATION_MANIFEST_INVALID", "package order is not authoritative")
    source_identity = _require_mapping(manifest.get("source_identity"), "source_identity")
    _require_sha256(
        source_identity.get("materialization_manifest_sha256"),
        "source_identity.materialization_manifest_sha256",
    )
    _require_sha256(
        source_identity.get("partition_policy_sha256"),
        "source_identity.partition_policy_sha256",
    )
    _verify_artifacts(root, manifest, "artifacts")
    path = root / "calibration-inputs.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    ids = [record.get("query_id") for record in records]
    if (
        not records
        or len(records) != manifest.get("query_count")
        or any(not isinstance(query_id, str) for query_id in ids)
        or len(ids) != len(set(ids))
        or ids != sorted(ids, key=lambda value: value.encode())
        or any(not isinstance(record.get("text"), str) for record in records)
    ):
        raise _fail(
            "CALIBRATION_IDENTITY_INVALID",
            "calibration inputs are invalid or non-canonical",
        )
    return {"status": "PASS", "query_count": len(records), "model_execution_used": False}


def load_verified_calibration_inputs(
    package_directory: str | Path, contract_path: str | Path
) -> VerifiedCalibrationInputs:
    contract, contract_sha256 = _load_contract(contract_path)
    root = Path(package_directory).resolve()
    verify_calibration_package(root)
    manifest_path = root / "calibration-manifest.json"
    manifest = _load_json(manifest_path, "CALIBRATION_MANIFEST_INVALID")
    if (
        manifest.get("contract_id") != contract["contract_id"]
        or manifest.get("contract_sha256") != contract_sha256
    ):
        raise _fail("CALIBRATION_CONTRACT_MISMATCH", "package contract identity differs")
    inputs_path = root / "calibration-inputs.jsonl"
    records = [json.loads(line) for line in inputs_path.read_text(encoding="utf-8").splitlines()]
    return VerifiedCalibrationInputs(
        query_ids=[record["query_id"] for record in records],
        query_texts=[record["text"] for record in records],
        manifest_sha256=sha256_file(manifest_path),
        inputs_sha256=sha256_file(inputs_path),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and verify M1 static calibration inputs.")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--contract", required=True)
    build.add_argument("--dataset", required=True)
    build.add_argument("--output", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--package", required=True)
    args = parser.parse_args(argv)
    try:
        result = (
            build_calibration_package(args.contract, args.dataset, args.output)
            if args.command == "build"
            else verify_calibration_package(args.package)
        )
    except TeacherEvidenceError as exc:
        print(json.dumps({"status": exc.status, "error": exc.__dict__}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
