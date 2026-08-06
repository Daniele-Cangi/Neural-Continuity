from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_b.calibration_data import load_verified_calibration_inputs
from neural_continuity.m1_b.onnx_source import load_verified_onnx_source, open_onnx_session
from neural_continuity.m1_teacher_evidence import (
    TeacherEvidenceError,
    _artifact_entry,
    _fail,
    _load_config,
    _load_json,
    _load_teacher,
    _require_mapping,
    _require_string,
    _write_bytes,
)

_EXPECTED_INPUTS = ("input_ids", "attention_mask", "token_type_ids")


class StaticCalibrationReader:
    def __init__(self, teacher: Any, texts: Sequence[str], batch_size: int) -> None:
        self._teacher = teacher
        self._texts = list(texts)
        self._batch_size = batch_size
        self._index = 0

    @staticmethod
    def _array(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.int64)

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._index >= len(self._texts):
            return None
        texts = self._texts[self._index : self._index + self._batch_size]
        self._index += len(texts)
        tokens = self._teacher.tokenize(texts)
        input_ids = tokens.get("input_ids")
        attention_mask = tokens.get("attention_mask")
        if input_ids is None or attention_mask is None:
            raise _fail("ONNX_TOKENIZATION_INVALID", "tokenizer lacks static calibration inputs")
        input_ids_array = self._array(input_ids)
        attention_mask_array = self._array(attention_mask)
        token_type_ids = tokens.get("token_type_ids")
        return {
            "input_ids": input_ids_array,
            "attention_mask": attention_mask_array,
            "token_type_ids": (
                np.zeros_like(input_ids_array)
                if token_type_ids is None
                else self._array(token_type_ids)
            ),
        }


def _positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise _fail("STATIC_QUANTIZATION_CONFIG_INVALID", f"{field} must be a positive integer")
    return value


def _quantization_settings(config: Mapping[str, Any], contract: Mapping[str, Any]) -> int:
    settings = _require_mapping(config.get("quantization"), "quantization")
    contract_settings = _require_mapping(contract.get("calibration"), "calibration")
    required = {
        "mode": "static",
        "quantization_format": "QDQ",
        "activation_type": "QUInt8",
        "weight_type": "QInt8",
        "per_channel": True,
        "calibration_method": "MinMax",
        "execution_provider": "CPUExecutionProvider",
    }
    for field, expected in required.items():
        if settings.get(field) != expected:
            raise _fail(
                "STATIC_QUANTIZATION_CONFIG_INVALID",
                f"quantization.{field} is unauthorized",
            )
    for field in (
        "mode",
        "quantization_format",
        "activation_type",
        "weight_type",
        "per_channel",
        "calibration_method",
    ):
        if settings[field] != contract_settings.get(field):
            raise _fail("STATIC_QUANTIZATION_CONTRACT_MISMATCH", f"quantization.{field} differs")
    return _positive_int(
        settings.get("calibration_batch_size"),
        "quantization.calibration_batch_size",
    )


def _run_static_quantization(
    source_path: Path, target_path: Path, reader: StaticCalibrationReader
) -> dict[str, str]:
    try:
        import onnx
        import onnxruntime
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ModuleNotFoundError as exc:
        raise _fail("ONNX_DEPENDENCY_MISSING", f"missing dependency: {exc.name}") from exc
    try:
        quantize_static(
            source_path,
            target_path,
            reader,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.MinMax,
            calibration_providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise _fail(
            "STATIC_QUANTIZATION_FAILED",
            f"cannot create ONNX INT8 candidate: {exc}",
            "EXECUTION_ERROR",
        ) from exc
    if not target_path.is_file() or target_path.stat().st_size == 0:
        raise _fail(
            "STATIC_QUANTIZATION_FAILED",
            "ONNX INT8 candidate was not produced",
            "EXECUTION_ERROR",
        )
    try:
        onnx.checker.check_model(str(target_path))
    except Exception as exc:
        raise _fail(
            "ONNX_INT8_INVALID",
            f"ONNX INT8 candidate failed structural validation: {exc}",
            "EXECUTION_ERROR",
        ) from exc
    return {
        "onnxruntime_version": onnxruntime.__version__,
        "execution_provider": "CPUExecutionProvider",
    }


def create_static_quantized_candidate(
    config_path: str | Path,
    transition_a_bundle: str | Path,
    calibration_package: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    config, config_sha256 = _load_config(config_path)
    contract_path = _require_string(config.get("contract_path"), "contract_path")
    contract = _load_json(Path(contract_path), "TRANSITION_B_CONTRACT_INVALID")
    if contract.get("contract_id") != "m1-transition-b-v1":
        raise _fail("TRANSITION_B_CONTRACT_INVALID", "contract_id must be m1-transition-b-v1")
    contract_sha256 = sha256_file(Path(contract_path))
    batch_size = _quantization_settings(config, contract)
    source = load_verified_onnx_source(transition_a_bundle, contract)
    session = open_onnx_session(source)
    if tuple(value.name for value in session.get_inputs()) != _EXPECTED_INPUTS:
        raise _fail("ONNX_IO_INVALID", "ONNX source graph input names are not authoritative")
    calibration = load_verified_calibration_inputs(calibration_package, contract_path)
    teacher, teacher_manifest = _load_teacher(config)
    output = Path(output_directory).resolve()
    if output.exists():
        raise _fail("OUTPUT_ALREADY_EXISTS", f"output already exists: {output}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        target_path = temporary / "teacher-int8-qdq.onnx"
        reader = StaticCalibrationReader(teacher, calibration.query_texts, batch_size)
        runtime = _run_static_quantization(source.artifact_path, target_path, reader)
        quantization_config = {
            "format_version": "1.0.0",
            "contract_id": contract["contract_id"],
            "contract_sha256": contract_sha256,
            "configuration_sha256": config_sha256,
            "mode": "static",
            "quantization_format": "QDQ",
            "activation_type": "QUInt8",
            "weight_type": "QInt8",
            "per_channel": True,
            "calibration_method": "MinMax",
            "calibration_batch_size": batch_size,
            "execution_provider": runtime["execution_provider"],
        }
        config_artifact = temporary / "quantization-config.json"
        _write_bytes(config_artifact, canonical_json_bytes(quantization_config) + b"\n")
        manifest = {
            "format_version": "1.0.0",
            "package_kind": "m1_onnx_int8_static_qdq_candidate",
            "contract_id": contract["contract_id"],
            "contract_sha256": contract_sha256,
            "candidate_status": "CAPTURED_PENDING_OBSERVATION",
            "source_identity": {
                "transition_a_evidence_manifest_sha256": source.transition_a_manifest_sha256,
                "onnx_fp32_artifact_sha256": source.artifact_sha256,
                "execution_provider": source.execution_provider,
            },
            "calibration_identity": {
                "manifest_sha256": calibration.manifest_sha256,
                "inputs_sha256": calibration.inputs_sha256,
                "query_count": len(calibration.query_ids),
            },
            "quantization_configuration": quantization_config,
            "runtime": runtime,
            "teacher_tokenizer_identity": teacher_manifest,
            "artifacts": [
                _artifact_entry(temporary, target_path),
                _artifact_entry(temporary, config_artifact),
            ],
        }
        manifest_path = temporary / "candidate-manifest.json"
        _write_bytes(manifest_path, canonical_json_bytes(manifest) + b"\n")
        os.replace(temporary, output)
        return {
            "status": "PASS",
            "candidate_status": "CAPTURED_PENDING_OBSERVATION",
            "output_directory": str(output),
            "candidate_manifest_sha256": sha256_file(output / "candidate-manifest.json"),
            "onnx_int8_artifact_sha256": sha256_file(output / "teacher-int8-qdq.onnx"),
            "calibration_query_count": len(calibration.query_ids),
        }
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create the M1 static QDQ ONNX INT8 candidate.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--calibration-package", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = create_static_quantized_candidate(
            args.config, args.source_bundle, args.calibration_package, args.output
        )
    except TeacherEvidenceError as exc:
        print(json.dumps({"status": exc.status, "error": exc.__dict__}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
