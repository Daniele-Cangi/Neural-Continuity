"""Fail-closed, non-executing readiness gate for the CUDA-null source preflight.

The pinned specification has not granted technical-preflight permission. This
module can attest prerequisites for independent review, never authorize a graph,
session, sentinel, or full-corpus run.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes
from neural_continuity.m1_diagnostics.cuda_null_authority import CudaNullAuthorityBlocked
from neural_continuity.m1_diagnostics.cuda_null_preflight_gate import (
    verify_live_static_authority,
)
from neural_continuity.m1_diagnostics.cuda_null_runtime_authority import (
    CudaNullRuntimeBlocked,
    verify_runtime_identity,
)

_ROOT = Path(__file__).resolve().parents[3]
PREFLIGHT_SPEC_PATH = _ROOT / "experiments" / "m1-cuda-null-preflight-v1.yaml"
PREFLIGHT_SPEC_SHA256 = "31b517886f221c1577596eff97431bb8c6a0422ec9321b12e1a9ef8732267e4a"
PROTOCOL_PATH = _ROOT / "docs" / "M1_CUDA_NULL_QUALIFICATION_PROTOCOL.md"
PROTOCOL_SHA256 = "eb3db027c73d989349bf3a11a8c734a7c4c175701f214acb288e39247676c7d3"
CONFIG_PATH = _ROOT / "experiments" / "m1-cuda-null-v1.yaml"
CONFIG_SHA256 = "df1a171d93e7fa07909e3f24b552baccd592a7239c5e6a5c4e3f972c475551d3"
STATIC_MANIFEST_SHA256 = "1e762db2ef7fb52c20a2d49f454161c1e4e945bc14bfe5f2cf7cb8916457856f"
STATIC_AUTHORITY_SHA256 = "f621b08bd1e4f9bc5697c0825dcab07301010f7e41bee8e992ee825f191a0cd6"


class CudaNullPreflightBlocked(ValueError):
    """A prerequisite or frozen identity did not verify."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise CudaNullPreflightBlocked(reason)


def _pinned_file(path: Path, canonical_path: Path, digest: str, label: str) -> None:
    _require(".." not in path.parts, f"{label} path traverses a parent")
    lexical_path = Path(os.path.abspath(path))
    _require(lexical_path == canonical_path, f"{label} path is not canonical")
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    for component in (lexical_path, *lexical_path.parents):
        try:
            info = os.lstat(component)
        except OSError as exc:
            raise CudaNullPreflightBlocked(f"{label} path is unavailable") from exc
        _require(
            not stat.S_ISLNK(info.st_mode)
            and not (getattr(info, "st_file_attributes", 0) & reparse_flag),
            f"{label} path contains a link or reparse point",
        )
    _require(lexical_path.is_file(), f"{label} is not a regular file")
    try:
        observed = hashlib.sha256(lexical_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise CudaNullPreflightBlocked(f"{label} is unreadable") from exc
    _require(observed == digest, f"{label} hash mismatch")


def verify_preflight_readiness(
    *,
    preflight_spec: Path,
    external_preflight_spec_sha256: str,
    static_bundle: Path,
    external_static_manifest_sha256: str,
    config_path: Path,
    external_config_sha256: str,
    dataset_root: Path,
    transition_a_bundle: Path,
    teacher_snapshot_root: Path,
    cpu_extension_bundle: Path,
    historical_cuda_bundle: Path,
    runtime_inventory_out: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attest frozen prerequisites without granting any execution permission."""
    _require(
        runtime_inventory_out is None or not runtime_inventory_out,
        "runtime inventory output must be empty",
    )
    _require(
        external_preflight_spec_sha256 == PREFLIGHT_SPEC_SHA256,
        "external preflight specification hash mismatch",
    )
    _require(
        external_static_manifest_sha256 == STATIC_MANIFEST_SHA256,
        "external static manifest hash mismatch",
    )
    _require(
        external_config_sha256 == CONFIG_SHA256,
        "external CUDA-null configuration hash mismatch",
    )
    _pinned_file(
        preflight_spec, PREFLIGHT_SPEC_PATH, PREFLIGHT_SPEC_SHA256, "preflight specification"
    )
    _pinned_file(PROTOCOL_PATH, PROTOCOL_PATH, PROTOCOL_SHA256, "qualification protocol")
    _pinned_file(config_path, CONFIG_PATH, CONFIG_SHA256, "CUDA-null configuration")

    try:
        static = verify_live_static_authority(
            static_bundle=static_bundle,
            external_static_manifest_sha256=external_static_manifest_sha256,
            config_path=config_path,
            external_config_sha256=external_config_sha256,
            dataset_root=dataset_root,
            transition_a_bundle=transition_a_bundle,
            teacher_snapshot_root=teacher_snapshot_root,
            cpu_extension_bundle=cpu_extension_bundle,
            historical_cuda_bundle=historical_cuda_bundle,
        )
    except (CudaNullAuthorityBlocked, OSError, ValueError) as exc:
        raise CudaNullPreflightBlocked("live static authority did not verify") from exc
    _require(isinstance(static, dict), "live static authority record is malformed")
    _require(
        static.get("status") == "FROZEN_STATIC_INPUTS_REVERIFIED"
        and static.get("static_manifest_sha256") == STATIC_MANIFEST_SHA256
        and static.get("static_authority_sha256") == STATIC_AUTHORITY_SHA256
        and static.get("config_sha256") == CONFIG_SHA256
        and static.get("runtime_verified") is False
        and static.get("onnx_graph_loaded") is False
        and static.get("session_created") is False
        and static.get("execution_authorized") is False,
        "live static authority is incomplete or inconsistent",
    )

    try:
        runtime = verify_runtime_identity(config_path, external_config_sha256)
    except (CudaNullRuntimeBlocked, OSError, ValueError) as exc:
        raise CudaNullPreflightBlocked("CUDA runtime identity did not verify") from exc
    _require(isinstance(runtime, dict), "CUDA runtime identity record is malformed")
    _require(
        runtime.get("status") == "RUNTIME_IDENTITY_VERIFIED_EXECUTION_BLOCKED"
        and runtime.get("config_sha256") == CONFIG_SHA256
        and runtime.get("onnx_graph_loaded") is False
        and runtime.get("session_created") is False
        and runtime.get("execution_authorized") is False,
        "CUDA runtime identity is incomplete or inconsistent",
    )

    record: dict[str, Any] = {
        "status": "PREFLIGHT_PREREQUISITES_VERIFIED_NOT_AUTHORIZED",
        "preflight_spec_sha256": PREFLIGHT_SPEC_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "config_sha256": CONFIG_SHA256,
        "static_manifest_sha256": STATIC_MANIFEST_SHA256,
        "static_authority_sha256": STATIC_AUTHORITY_SHA256,
        "runtime_identity_sha256": hashlib.sha256(
            canonical_json_bytes(runtime) + b"\n"
        ).hexdigest(),
        "technical_preflight_permission": "NOT_GRANTED",
        "qualifying_m1_evidence": False,
        "sentinel_permission": False,
        "full_corpus_permission": False,
        "int8_permission": False,
        "holdout_permission": False,
        "onnx_graph_loaded": False,
        "session_created": False,
        "execution_authorized": False,
    }
    result = {
        **record,
        "record_sha256": hashlib.sha256(canonical_json_bytes(record) + b"\n").hexdigest(),
    }
    if runtime_inventory_out is not None:
        runtime_inventory_out.update(runtime)
    return result
