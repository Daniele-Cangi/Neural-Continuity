from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from neural_continuity.evidence import sha256_file
from neural_continuity.m1_b.onnx_fp32_observation import (
    replay_fp32_source_observation,
)
from neural_continuity.m1_b.onnx_int8_observation import (
    replay_int8_target_observation,
)
from neural_continuity.m1_b.paired_observation_compatibility import (
    verify_paired_observation_compatibility,
)
from neural_continuity.m1_diagnostics.causal_plan_evidence import (
    replay_causal_plan_package,
)
from neural_continuity.m1_measurement_null import replay_measurement_null
from neural_continuity.m1_teacher_evidence import (
    _load_config,
    _load_json,
    _verify_artifacts,
    load_materialized_dataset,
)

REQUIRED_PACKAGES = {
    "dataset": "nc-m1-scifact-v1-materialized-20260806",
    "transition_a": "nc-m1-transition-a-20260806T193853",
    "onnx_null": "nc-m1-onnx-null-20260806T211943",
    "candidate": "nc-m1-static-qdq-verified-20260806T230647",
    "int8_observation": "nc-m1-int8-observation-20260806T232512",
    "fp32_observation": "nc-m1-fp32-paired-source-20260807T101905",
    "transition_b": "nc-m1-transition-b-decision-20260808T014916",
}
REQUIRED_AUTHORITY_SHA256 = {
    "dataset": "beab716b9f322478ca3f2efd0e6e93e7d66a2b3483ed098941cd9f2275bcdcc2",
    "transition_a": "12566ccbcc7f3f74a799abca2189a9b0906efd44a0f038ce0dc7c44b7b87fc3a",
    "onnx_null": "506ab742aac5abbb8558ce20e714d4d5baf2c3785bb17de0d9fbdb22ad84c123",
    "candidate": "d11888e48e24a9e29f5bdfac48ad7ace4204fb7b101e3531faa0f11190ad562c",
    "int8_observation": "4027c1edf9f24254e6174ca79bc722c98758c8f97f5ad175b380866f64063a80",
    "fp32_observation": "cf03882df0913e84b456b61f02a1c00a14ec151cd0fd9cc07f7d0bf04745b4df",
    "transition_b": "eed7d7af553ae9aa77274104cc75f348de910df464d836272ab37e8760e78d4e",
}
EXPECTED_CONTRACT_SHA256 = "ad8c04574b3121eb69028e89f98f81cd1a68c34f15ecc23f9dc85c66b45273b0"
EXPECTED_FP32_CONFIG_SHA256 = "c22f142b36d27b6d59087369293b6f5be092fb40380a72e969200686f6969014"
EXPECTED_INT8_CONFIG_SHA256 = "19fccc3e7dd2df86e0a1a589bb9e2c3c40d6cdb8e646377c460b54d3e7e6d479"


class Stage0ControlError(RuntimeError):
    def __init__(self, code: str, message: str, status: str = "BLOCKED") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "status": self.status}


@dataclass(frozen=True)
class Stage0Authority:
    causal_plan_bundle: Path
    causal_plan_manifest_sha256: str
    archive_manifest_path: Path
    archive_manifest_sha256: str
    runtime_root: Path
    runtime_manifest_sha256: str
    contract_path: Path
    fp32_config_path: Path
    int8_config_path: Path
    dataset_root: Path
    transition_a_bundle: Path
    onnx_null_bundle: Path
    candidate_root: Path
    baseline_int8_bundle: Path
    baseline_fp32_bundle: Path
    transition_b_bundle: Path
    onnx_null_report: Mapping[str, Any]
    package_authority_sha256: Mapping[str, str]


def _json(path: Path, code: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage0ControlError(code, f"cannot load {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise Stage0ControlError(code, f"{path.name} must contain an object")
    return payload


def _safe(root: Path, relative_path: Any, code: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise Stage0ControlError(code, "declared path is missing")
    relative = Path(relative_path)
    if relative.is_absolute():
        raise Stage0ControlError(code, "declared path must be relative")
    candidate = (root / relative).resolve()
    if root != candidate and root not in candidate.parents:
        raise Stage0ControlError(code, "declared path escapes its authority root")
    if not candidate.is_file():
        raise Stage0ControlError(code, f"declared file is missing: {relative_path}")
    return candidate


def _verify_runtime_authority(
    manifest_path: Path,
    expected_sha256: str,
) -> tuple[Path, Path, Path, Path]:
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected_sha256:
        raise Stage0ControlError(
            "STAGE0_RUNTIME_AUTHORITY_HASH_MISMATCH",
            "runtime authority manifest hash does not match",
        )
    root = manifest_path.parent.resolve()
    manifest = _json(manifest_path, "STAGE0_RUNTIME_AUTHORITY_INVALID")
    required_header = {
        "kind": "m1-stage0-frozen-runtime-authority",
        "version": "1.0.0",
        "status": "VERIFIED",
        "extraction_source": "exact_git_blob_bytes",
        "mutable_checkout_used": False,
    }
    if any(manifest.get(key) != value for key, value in required_header.items()):
        raise Stage0ControlError(
            "STAGE0_RUNTIME_AUTHORITY_INVALID",
            "runtime authority header is invalid",
        )
    records = manifest.get("files")
    if not isinstance(records, list) or len(records) != 3:
        raise Stage0ControlError(
            "STAGE0_RUNTIME_AUTHORITY_INVALID",
            "runtime authority file declaration is incomplete",
        )
    declared: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise Stage0ControlError(
                "STAGE0_RUNTIME_AUTHORITY_INVALID",
                "runtime authority file record is invalid",
            )
        path = str(record["path"])
        artifact = _safe(root, path, "STAGE0_RUNTIME_AUTHORITY_INVALID")
        if path in declared or artifact.stat().st_size != record.get("size_bytes"):
            raise Stage0ControlError(
                "STAGE0_RUNTIME_AUTHORITY_INVALID",
                f"runtime authority size or identity is invalid: {path}",
            )
        raw_hash = record.get("raw_sha256")
        if not isinstance(raw_hash, str) or sha256_file(artifact) != raw_hash:
            raise Stage0ControlError(
                "STAGE0_RUNTIME_AUTHORITY_HASH_MISMATCH",
                f"runtime authority file hash does not match: {path}",
            )
        declared[path] = record
    expected_paths = {
        "contracts/m1-transition-b-v1.json",
        "experiments/m1-transition-b-fp32-source-observation.yaml",
        "experiments/m1-transition-b-int8-observation.yaml",
    }
    if set(declared) != expected_paths:
        raise Stage0ControlError(
            "STAGE0_RUNTIME_AUTHORITY_INVALID",
            "runtime authority file set differs from the frozen declaration",
        )
    contract = _safe(
        root,
        "contracts/m1-transition-b-v1.json",
        "STAGE0_RUNTIME_AUTHORITY_INVALID",
    )
    fp32 = _safe(
        root,
        "experiments/m1-transition-b-fp32-source-observation.yaml",
        "STAGE0_RUNTIME_AUTHORITY_INVALID",
    )
    int8 = _safe(
        root,
        "experiments/m1-transition-b-int8-observation.yaml",
        "STAGE0_RUNTIME_AUTHORITY_INVALID",
    )
    fp32_config, fp32_hash = _load_config(fp32)
    int8_config, int8_hash = _load_config(int8)
    if (
        sha256_file(contract) != EXPECTED_CONTRACT_SHA256
        or fp32_hash != EXPECTED_FP32_CONFIG_SHA256
        or int8_hash != EXPECTED_INT8_CONFIG_SHA256
        or declared["experiments/m1-transition-b-fp32-source-observation.yaml"].get(
            "semantic_configuration_sha256"
        )
        != fp32_hash
        or declared["experiments/m1-transition-b-int8-observation.yaml"].get(
            "semantic_configuration_sha256"
        )
        != int8_hash
        or fp32_config.get("contract_path") != "contracts/m1-transition-b-v1.json"
        or int8_config.get("contract_path") != "contracts/m1-transition-b-v1.json"
    ):
        raise Stage0ControlError(
            "STAGE0_RUNTIME_AUTHORITY_IDENTITY_MISMATCH",
            "runtime contract or semantic configuration identity differs",
        )
    return root, contract, fp32, int8


def _verify_archive(
    manifest_path: Path,
    expected_sha256: str,
) -> dict[str, tuple[Path, str]]:
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected_sha256:
        raise Stage0ControlError(
            "STAGE0_ARCHIVE_HASH_MISMATCH",
            "archive manifest hash does not match",
        )
    manifest = _json(manifest_path, "STAGE0_ARCHIVE_INVALID")
    if (
        manifest.get("schema_version") != "1.0.0"
        or manifest.get("package_count") != 11
        or manifest.get("package_file_count") != 53
        or manifest.get("restore_policy")
        != "restore packages to their recorded original_path before invoking frozen replay bundles"
    ):
        raise Stage0ControlError(
            "STAGE0_ARCHIVE_INVALID",
            "archive declaration is not authoritative",
        )
    packages = manifest.get("packages")
    if not isinstance(packages, list):
        raise Stage0ControlError("STAGE0_ARCHIVE_INVALID", "archive packages are missing")
    by_name = {
        record.get("package_name"): record
        for record in packages
        if isinstance(record, Mapping) and isinstance(record.get("package_name"), str)
    }
    resolved: dict[str, tuple[Path, str]] = {}
    for role, package_name in REQUIRED_PACKAGES.items():
        record = by_name.get(package_name)
        if not isinstance(record, Mapping):
            raise Stage0ControlError(
                "STAGE0_ARCHIVE_PACKAGE_MISSING",
                f"required package is absent: {package_name}",
            )
        authority_hash = record.get("authority_sha256")
        if authority_hash != REQUIRED_AUTHORITY_SHA256[role]:
            raise Stage0ControlError(
                "STAGE0_ARCHIVE_IDENTITY_MISMATCH",
                f"authority hash differs for {package_name}",
            )
        original = Path(str(record.get("original_path", ""))).resolve()
        if not original.is_dir():
            raise Stage0ControlError(
                "STAGE0_ARCHIVE_PACKAGE_MISSING",
                f"restored package is absent: {package_name}",
            )
        files = record.get("files")
        if not isinstance(files, list) or not files:
            raise Stage0ControlError(
                "STAGE0_ARCHIVE_INVALID",
                f"file inventory is absent: {package_name}",
            )
        for file_record in files:
            if not isinstance(file_record, Mapping):
                raise Stage0ControlError(
                    "STAGE0_ARCHIVE_INVALID",
                    f"file record is invalid: {package_name}",
                )
            artifact = _safe(
                original,
                file_record.get("path"),
                "STAGE0_ARCHIVE_PACKAGE_INVALID",
            )
            if artifact.stat().st_size != file_record.get("size_bytes") or sha256_file(
                artifact
            ) != file_record.get("sha256"):
                raise Stage0ControlError(
                    "STAGE0_ARCHIVE_PACKAGE_HASH_MISMATCH",
                    f"restored artifact differs: {package_name}/{file_record.get('path')}",
                )
        authority_file = _safe(
            original,
            record.get("authority_file"),
            "STAGE0_ARCHIVE_PACKAGE_INVALID",
        )
        if sha256_file(authority_file) != authority_hash:
            raise Stage0ControlError(
                "STAGE0_ARCHIVE_PACKAGE_HASH_MISMATCH",
                f"restored authority differs: {package_name}",
            )
        resolved[role] = (original, str(authority_hash))
    return resolved


def _verify_transition_b(root: Path) -> None:
    manifest = _load_json(root / "evidence-manifest.json", "STAGE0_TRANSITION_B_INVALID")
    _verify_artifacts(root, manifest, "artifacts")
    decision = _load_json(root / "decision.json", "STAGE0_TRANSITION_B_INVALID")
    if (
        manifest.get("package_kind") != "m1_transition_b_decision"
        or manifest.get("transition_b_status") != "FAIL"
        or decision.get("transition_b_status") != "FAIL"
    ):
        raise Stage0ControlError(
            "STAGE0_FROZEN_DECISION_MISMATCH",
            "frozen Transition B decision is not FAIL",
        )


def _verify_active_teacher_runtime(fp32_bundle: Path) -> None:
    manifest = _load_json(
        fp32_bundle.parent / "evidence-manifest.json",
        "STAGE0_BASELINE_MANIFEST_INVALID",
    )
    identity = manifest.get("teacher_tokenizer_identity")
    if not isinstance(identity, Mapping):
        raise Stage0ControlError(
            "STAGE0_BASELINE_MANIFEST_INVALID",
            "FP32 baseline teacher tokenizer identity is missing",
        )
    required_versions = {
        "torch": "torch_version",
        "sentence-transformers": "sentence_transformers_version",
    }
    for distribution, field in required_versions.items():
        expected = identity.get(field)
        if not isinstance(expected, str) or not expected:
            raise Stage0ControlError(
                "STAGE0_BASELINE_MANIFEST_INVALID",
                f"FP32 baseline {field} is missing",
            )
        try:
            actual = version(distribution)
        except PackageNotFoundError as exc:
            raise Stage0ControlError(
                "STAGE0_RUNTIME_DEPENDENCY_MISSING",
                f"required runtime dependency is missing: {distribution}",
            ) from exc
        if actual != expected:
            raise Stage0ControlError(
                "STAGE0_RUNTIME_DEPENDENCY_MISMATCH",
                f"{distribution} {actual} differs from frozen {expected}",
            )


def verify_stage0_authority(
    causal_plan_bundle: str | Path,
    causal_plan_manifest_sha256: str,
    archive_manifest_path: str | Path,
    archive_manifest_sha256: str,
    runtime_manifest_path: str | Path,
    runtime_manifest_sha256: str,
) -> Stage0Authority:
    causal_bundle = Path(causal_plan_bundle).resolve()
    try:
        causal_replay = replay_causal_plan_package(
            causal_bundle,
            causal_plan_manifest_sha256,
        )
    except Exception as exc:
        raise Stage0ControlError(
            "STAGE0_CAUSAL_PLAN_REPLAY_FAILED",
            f"causal plan replay failed: {exc}",
        ) from exc
    required_causal = {
        "status": "PRE_REGISTERED",
        "replay_verified": True,
        "plan_match": True,
        "hypotheses_match": True,
        "intervention_matrix_match": True,
        "intervention_execution_authorized": False,
        "causal_claim_made": False,
        "model_execution_used": False,
        "activation_artifact_loaded": False,
    }
    if any(causal_replay.get(key) != value for key, value in required_causal.items()):
        raise Stage0ControlError(
            "STAGE0_CAUSAL_PLAN_REPLAY_FAILED",
            "causal plan replay result is not authoritative",
        )
    runtime_root, contract, fp32_config, int8_config = _verify_runtime_authority(
        Path(runtime_manifest_path).resolve(),
        runtime_manifest_sha256,
    )
    archive_path = Path(archive_manifest_path).resolve()
    packages = _verify_archive(archive_path, archive_manifest_sha256)
    dataset_root = packages["dataset"][0]
    dataset = load_materialized_dataset(dataset_root)
    if dataset.manifest_sha256 != REQUIRED_AUTHORITY_SHA256["dataset"]:
        raise Stage0ControlError(
            "STAGE0_DATASET_IDENTITY_MISMATCH",
            "materialized dataset authority differs",
        )
    fp32_bundle = packages["fp32_observation"][0] / "replay-bundle.json"
    int8_bundle = packages["int8_observation"][0] / "replay-bundle.json"
    null_bundle = packages["onnx_null"][0] / "replay-bundle.json"
    if replay_fp32_source_observation(fp32_bundle).get("replay_verified") is not True:
        raise Stage0ControlError(
            "STAGE0_BASELINE_REPLAY_FAILED",
            "FP32 baseline replay failed",
        )
    if replay_int8_target_observation(int8_bundle).get("replay_verified") is not True:
        raise Stage0ControlError(
            "STAGE0_BASELINE_REPLAY_FAILED",
            "INT8 baseline replay failed",
        )
    _verify_active_teacher_runtime(fp32_bundle)
    compatibility = verify_paired_observation_compatibility(fp32_bundle, int8_bundle)
    if compatibility.get("compatible") is not True:
        raise Stage0ControlError(
            "STAGE0_BASELINE_COMPATIBILITY_FAILED",
            "baseline FP32 and INT8 packages are incompatible",
        )
    null_replay = replay_measurement_null(null_bundle)
    if null_replay.get("replay_verified") is not True:
        raise Stage0ControlError(
            "STAGE0_MEASUREMENT_NULL_REPLAY_FAILED",
            "ONNX measurement-null replay failed",
        )
    transition_b_root = packages["transition_b"][0]
    _verify_transition_b(transition_b_root)
    null_report = _load_json(
        packages["onnx_null"][0] / "comparison-report.json",
        "STAGE0_MEASUREMENT_NULL_INVALID",
    )
    return Stage0Authority(
        causal_plan_bundle=causal_bundle,
        causal_plan_manifest_sha256=causal_plan_manifest_sha256,
        archive_manifest_path=archive_path,
        archive_manifest_sha256=archive_manifest_sha256,
        runtime_root=runtime_root,
        runtime_manifest_sha256=runtime_manifest_sha256,
        contract_path=contract,
        fp32_config_path=fp32_config,
        int8_config_path=int8_config,
        dataset_root=dataset_root,
        transition_a_bundle=packages["transition_a"][0] / "replay-bundle.json",
        onnx_null_bundle=null_bundle,
        candidate_root=packages["candidate"][0],
        baseline_int8_bundle=int8_bundle,
        baseline_fp32_bundle=fp32_bundle,
        transition_b_bundle=transition_b_root / "replay-bundle.json",
        onnx_null_report=null_report,
        package_authority_sha256={
            role: authority_hash for role, (_, authority_hash) in packages.items()
        },
    )
