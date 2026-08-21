from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural_continuity.evidence import sha256_file

INSTRUMENTATION_MANIFEST_SHA256 = "6edccc8aa44cde9665dedbf82599eb59b26b41c6d40f2979b0b55d3f1e7c2765"
STATIC_MANIFEST_SHA256 = "efbd9f60588d4b7b080b41f48de3691860634778b0a4f255c3fc54e6d690e507"
PROBE_PLAN_SHA256 = "f432ed99a8d0747b2d763863397fd9e8e95569419937eaa12a716f6ff7e626e7"
SOURCE_INSTRUMENTED_SHA256 = "9c7a3454ffa147397e7b5a76acb124e6c238e511230a9a9d377afec34932ca5b"
TARGET_INSTRUMENTED_SHA256 = "9d795d346650ac99e94abfa9aaf6d99c391bf743ef57f26d5c3014ed0d1225bd"
EXPECTED_AUTHORITY_HASHES = {
    "onnx_fp32_source": "5c0d999bd6b5e64e36cad1f61a83ef8e7507d55be49086745780fabb7c648511",
    "onnx_int8_candidate": "8b28688438e249c42b523e276333a3a009ca30d0754a3ba6fcbb10d76de873e5",
    "calibration_manifest": "3ac7d68e01976ee444217cd80c5b4b7338f870d8c0ab5a350a960495baef0778",
    "paired_fp32_evidence": "cf03882df0913e84b456b61f02a1c00a14ec151cd0fd9cc07f7d0bf04745b4df",
    "int8_target_evidence": "4027c1edf9f24254e6174ca79bc722c98758c8f97f5ad175b380866f64063a80",
    "transition_b_decision": "eed7d7af553ae9aa77274104cc75f348de910df464d836272ab37e8760e78d4e",
    "transition_a_contract": "772e0df5133de09f6108cb42144e9b2ee69e47c0694bdf5b60ca4d88c18ee5c4",
    "transition_b_v1_contract": "ad8c04574b3121eb69028e89f98f81cd1a68c34f15ecc23f9dc85c66b45273b0",
}


class FidelityGateError(RuntimeError):
    def __init__(self, code: str, message: str, status: str = "BLOCKED") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


@dataclass(frozen=True)
class VerifiedFidelityAuthority:
    instrumentation_root: Path
    instrumentation_manifest_sha256: str
    static_manifest_sha256: str
    source_original_path: Path
    source_instrumented_path: Path
    target_original_path: Path
    target_instrumented_path: Path
    authority_records: tuple[dict[str, Any], ...]


def _load_json(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FidelityGateError(code, f"cannot load JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FidelityGateError(code, f"JSON artifact is not an object: {path}")
    return value


def _safe_artifact_path(root: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise FidelityGateError("ARTIFACT_PATH_INVALID", "artifact path must be a string")
    raw_path = root / relative_path
    if raw_path.is_symlink():
        raise FidelityGateError(
            "ARTIFACT_PATH_INVALID", f"artifact cannot be a symlink: {relative_path}"
        )
    candidate = raw_path.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise FidelityGateError(
            "ARTIFACT_PATH_INVALID", f"artifact escapes package root: {relative_path}"
        ) from exc
    if not candidate.is_file():
        raise FidelityGateError("ARTIFACT_MISSING", f"declared artifact missing: {relative_path}")
    return candidate


def verify_artifact_manifest(root: Path, expected_sha256: str) -> dict[str, Any]:
    package_root = root.resolve()
    manifest_path = package_root / "artifact-manifest.json"
    if not manifest_path.is_file():
        raise FidelityGateError("MANIFEST_MISSING", f"artifact manifest missing: {manifest_path}")
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != expected_sha256:
        raise FidelityGateError(
            "MANIFEST_HASH_MISMATCH",
            f"artifact manifest hash mismatch: {actual_manifest_sha256}",
        )
    manifest = _load_json(manifest_path, "MANIFEST_INVALID")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise FidelityGateError("MANIFEST_INVALID", "manifest artifacts must be a non-empty array")
    if manifest.get("artifact_count", len(artifacts)) != len(artifacts):
        raise FidelityGateError("MANIFEST_INVALID", "manifest artifact count mismatch")
    seen: set[str] = set()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            raise FidelityGateError("MANIFEST_INVALID", f"artifact {index} is not an object")
        relative_path = artifact.get("path")
        if not isinstance(relative_path, str) or relative_path in seen:
            raise FidelityGateError("MANIFEST_INVALID", "artifact paths are invalid or duplicated")
        seen.add(relative_path)
        path = _safe_artifact_path(package_root, relative_path)
        if path.stat().st_size != artifact.get("size_bytes"):
            raise FidelityGateError("ARTIFACT_SIZE_MISMATCH", f"size mismatch: {relative_path}")
        if sha256_file(path) != artifact.get("sha256"):
            raise FidelityGateError("ARTIFACT_HASH_MISMATCH", f"hash mismatch: {relative_path}")
    return manifest


def _lf_normalized_sha256(path: Path) -> str:
    normalized = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def _verify_frozen_authorities(payload: Mapping[str, Any]) -> dict[str, Path]:
    authorities = payload.get("authorities")
    if (
        payload.get("status") != "PASS"
        or payload.get("all_authorities_verified") is not True
        or payload.get("authority_count") != len(EXPECTED_AUTHORITY_HASHES)
        or not isinstance(authorities, list)
        or len(authorities) != len(EXPECTED_AUTHORITY_HASHES)
    ):
        raise FidelityGateError("AUTHORITY_SET_INVALID", "frozen authority set is incomplete")
    verified: dict[str, Path] = {}
    for record in authorities:
        if not isinstance(record, Mapping):
            raise FidelityGateError("AUTHORITY_SET_INVALID", "authority record is not an object")
        role = record.get("role")
        raw_path = record.get("path")
        method = record.get("verification_method")
        if role not in EXPECTED_AUTHORITY_HASHES or not isinstance(raw_path, str):
            raise FidelityGateError("AUTHORITY_SET_INVALID", "authority role or path is invalid")
        if role in verified:
            raise FidelityGateError("AUTHORITY_SET_INVALID", f"duplicated authority: {role}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FidelityGateError("AUTHORITY_MISSING", f"authority missing: {role}")
        if path.stat().st_size != record.get("size_bytes"):
            raise FidelityGateError("AUTHORITY_SIZE_MISMATCH", f"authority size mismatch: {role}")
        if method == "byte_sha256":
            digest = sha256_file(path)
        elif method == "lf_normalized_sha256":
            digest = _lf_normalized_sha256(path)
        else:
            raise FidelityGateError("AUTHORITY_METHOD_INVALID", f"unknown authority method: {role}")
        if digest != EXPECTED_AUTHORITY_HASHES[role] or digest != record.get("sha256"):
            raise FidelityGateError("AUTHORITY_HASH_MISMATCH", f"authority hash mismatch: {role}")
        verified[str(role)] = path
    if set(verified) != set(EXPECTED_AUTHORITY_HASHES):
        raise FidelityGateError("AUTHORITY_SET_INVALID", "authority roles do not match frozen set")
    return verified


def verify_fidelity_authority(
    instrumentation_directory: str | Path,
) -> VerifiedFidelityAuthority:
    root = Path(instrumentation_directory).resolve()
    manifest = verify_artifact_manifest(root, INSTRUMENTATION_MANIFEST_SHA256)
    if (
        manifest.get("kind") != "m1_transition_b_v2_instrumentation_manifest"
        or manifest.get("status") != "READY_FOR_FIDELITY_CONTROL"
        or manifest.get("tamper_evident") is not True
        or manifest.get("model_execution_used") is not False
    ):
        raise FidelityGateError(
            "INSTRUMENTATION_MANIFEST_INVALID", "instrumentation is not authorized"
        )
    authority = _load_json(root / "instrumentation-authority.json", "AUTHORITY_INVALID")
    static = authority.get("static_preflight")
    if authority.get("status") != "PASS" or not isinstance(static, Mapping):
        raise FidelityGateError("AUTHORITY_INVALID", "instrumentation authority did not pass")
    if (
        static.get("manifest_sha256") != STATIC_MANIFEST_SHA256
        or static.get("probe_plan_sha256") != PROBE_PLAN_SHA256
        or static.get("probe_count") != 283
        or static.get("static_preflight_status") != "STATIC_PREFLIGHT_COMPLETE"
    ):
        raise FidelityGateError("STATIC_AUTHORITY_MISMATCH", "static preflight identity mismatch")
    static_root_raw = static.get("package_directory")
    if not isinstance(static_root_raw, str):
        raise FidelityGateError("STATIC_AUTHORITY_MISMATCH", "static package path is missing")
    verify_artifact_manifest(Path(static_root_raw), STATIC_MANIFEST_SHA256)
    frozen_payload = static.get("authorities")
    if not isinstance(frozen_payload, Mapping):
        raise FidelityGateError("AUTHORITY_SET_INVALID", "frozen authority payload missing")
    frozen_paths = _verify_frozen_authorities(frozen_payload)

    plan = _load_json(root / "instrumentation-plan.json", "INSTRUMENTATION_PLAN_INVALID")
    if (
        plan.get("probe_count") != 283
        or plan.get("onnx_runtime_session_created") is not False
        or plan.get("activations_read") is not False
        or plan.get("model_execution_used") is not False
        or plan.get("frozen_models_overwritten") is not False
    ):
        raise FidelityGateError(
            "INSTRUMENTATION_PLAN_INVALID", "instrumentation plan is not pristine"
        )

    source_instrumented = root / "source-instrumented.onnx"
    target_instrumented = root / "target-instrumented.onnx"
    if sha256_file(source_instrumented) != SOURCE_INSTRUMENTED_SHA256:
        raise FidelityGateError("INSTRUMENTED_MODEL_MISMATCH", "source instrumented hash mismatch")
    if sha256_file(target_instrumented) != TARGET_INSTRUMENTED_SHA256:
        raise FidelityGateError("INSTRUMENTED_MODEL_MISMATCH", "target instrumented hash mismatch")
    return VerifiedFidelityAuthority(
        instrumentation_root=root,
        instrumentation_manifest_sha256=INSTRUMENTATION_MANIFEST_SHA256,
        static_manifest_sha256=STATIC_MANIFEST_SHA256,
        source_original_path=frozen_paths["onnx_fp32_source"],
        source_instrumented_path=source_instrumented,
        target_original_path=frozen_paths["onnx_int8_candidate"],
        target_instrumented_path=target_instrumented,
        authority_records=tuple(dict(record) for record in frozen_payload["authorities"]),
    )
