"""Owner-authorized, sentinel-only gate; the frozen v1 draft remains unchanged."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.cuda_null_authority import build_static_authority
from neural_continuity.m1_diagnostics.cuda_null_evidence import replay_static_package
from neural_continuity.m1_diagnostics.cuda_null_preflight_readiness import (
    CONFIG_PATH,
    CONFIG_SHA256,
    PROTOCOL_PATH,
    PROTOCOL_SHA256,
    _pinned_file,
)
from neural_continuity.m1_diagnostics.cuda_null_runtime_authority import verify_runtime_identity
from neural_continuity.m1_diagnostics.cuda_null_sentinel_readiness import EPOCH_LAYOUT
from neural_continuity.m1_diagnostics.cuda_null_source_preflight_evidence import (
    replay_source_preflight,
)

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "experiments" / "m1-cuda-null-sentinel-authority-v2.json"
EXPECTED_STATIC_MANIFEST = "1e762db2ef7fb52c20a2d49f454161c1e4e945bc14bfe5f2cf7cb8916457856f"
EXPECTED_STATIC_AUTHORITY = "f621b08bd1e4f9bc5697c0825dcab07301010f7e41bee8e992ee825f191a0cd6"
EXPECTED_SOURCE_MANIFEST = "9340a7a52ff6502d9ee6639dc3c15d41325c3521c77629778e44f8d7043bd8d1"


class SentinelExecutionBlocked(ValueError):
    """A declared authority or bounded execution prerequisite did not verify."""


@dataclass(frozen=True)
class SentinelExecutionAuthority:
    spec_sha256: str
    paths: dict[str, Path]
    runtime_identity_sha256: str


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise SentinelExecutionBlocked(reason)


def _load_spec(external_sha256: str) -> dict[str, Any]:
    _require(len(external_sha256) == 64, "external authority SHA-256 is required")
    _pinned_file(SPEC_PATH, SPEC_PATH, external_sha256, "owner sentinel authority")
    try:
        spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SentinelExecutionBlocked("owner authority cannot be read") from exc
    _require(isinstance(spec, dict), "owner authority is malformed")
    expected = {
        "kind": "m1_cuda_null_sentinel_owner_authority",
        "version": "2.0.0",
        "status": "OWNER_AUTHORIZED_SENTINEL_ONLY",
        "independent_review_complete": False,
        "parent_config_sha256": CONFIG_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "fresh_static_manifest_sha256": EXPECTED_STATIC_MANIFEST,
        "fresh_static_authority_sha256": EXPECTED_STATIC_AUTHORITY,
        "source_preflight_manifest_sha256": EXPECTED_SOURCE_MANIFEST,
        "dataset_manifest_sha256": (
            "0746d98f5e69c6a0ee48ca3f47b342de1d968a877c90df26ffe8f893437fd5de"
        ),
        "source_onnx_sha256": "5c0d999bd6b5e64e36cad1f61a83ef8e7507d55be49086745780fabb7c648511",
        "phase_id": "tensor_sentinel_preflight",
        "epoch_count": 120,
        "document_count": 256,
        "query_count": 81,
        "selection_domain": "neural-continuity:m1:null-extension:v1:document",
        "run_layout": [{"label": label, "batch_size": size} for label, size in EPOCH_LAYOUT],
        "independent_process_per_epoch": True,
        "early_stopping_allowed": False,
        "adaptive_sample_size_allowed": False,
        "qualifying_detection_evidence": False,
        "full_corpus_execution_allowed": False,
        "int8_execution_allowed": False,
        "holdout_access_allowed": False,
        "operational_tolerance_change_allowed": False,
        "scientific_decision": "NOT_EVALUATED",
    }
    _require(set(spec) == set(expected) | {"approval_basis", "paths"}, "authority fields differ")
    _require(
        all(spec.get(key) == value for key, value in expected.items()),
        "authority scope differs",
    )
    _require(
        isinstance(spec.get("approval_basis"), str) and bool(spec["approval_basis"]),
        "owner approval basis is missing",
    )
    return spec


def _paths(spec: dict[str, Any]) -> dict[str, Path]:
    raw = spec.get("paths")
    names = {
        "dataset_root",
        "transition_a_bundle",
        "teacher_snapshot_root",
        "cpu_extension_bundle",
        "historical_cuda_bundle",
        "fresh_static_bundle",
        "source_preflight_bundle",
        "output_root",
        "scratch_root",
    }
    if not isinstance(raw, dict) or set(raw) != names:
        raise SentinelExecutionBlocked("authority path set differs")
    _require(
        all(isinstance(raw[name], str) and raw[name] for name in names),
        "authority path invalid",
    )
    return {name: Path(raw[name]) for name in names}


def verify_sentinel_execution_authority(external_sha256: str) -> SentinelExecutionAuthority:
    """Verify the complete frozen set before any ONNX graph or activation is read."""
    spec = _load_spec(external_sha256)
    paths = _paths(spec)
    _pinned_file(CONFIG_PATH, CONFIG_PATH, CONFIG_SHA256, "frozen v1 configuration")
    _pinned_file(PROTOCOL_PATH, PROTOCOL_PATH, PROTOCOL_SHA256, "frozen protocol")
    static_bundle = paths["fresh_static_bundle"]
    static_replay = replay_static_package(static_bundle, EXPECTED_STATIC_MANIFEST)
    _require(
        static_replay.get("integrity_replay_status") == "PASS"
        and static_replay.get("authority_sha256") == EXPECTED_STATIC_AUTHORITY
        and static_replay.get("decision_match") is True
        and static_replay.get("status_match") is True,
        "fresh static package replay blocked",
    )
    live = build_static_authority(
        config_path=CONFIG_PATH,
        external_config_sha256=CONFIG_SHA256,
        dataset_root=paths["dataset_root"],
        transition_a_bundle=paths["transition_a_bundle"],
        teacher_snapshot_root=paths["teacher_snapshot_root"],
        cpu_extension_bundle=paths["cpu_extension_bundle"],
        historical_cuda_bundle=paths["historical_cuda_bundle"],
    )
    recorded_path = static_bundle.parent / "static-authority.json"
    _require(
        live == json.loads(recorded_path.read_text(encoding="utf-8")),
        "live static authority differs",
    )
    _require(
        sha256_file(recorded_path) == EXPECTED_STATIC_AUTHORITY
        and live.get("status") == "STATIC_VERIFIED_EXECUTION_BLOCKED",
        "live static authority is not the frozen record",
    )
    source = replay_source_preflight(paths["source_preflight_bundle"], EXPECTED_SOURCE_MANIFEST)
    _require(
        source.get("replay_status") == "PASS"
        and source.get("technical_preflight_status") == "PASS"
        and source.get("decision_match") is True
        and source.get("scientific_decision") == "NOT_EVALUATED",
        "source-only preflight replay blocked",
    )
    runtime = verify_runtime_identity(CONFIG_PATH, CONFIG_SHA256)
    _require(
        runtime.get("status") == "RUNTIME_IDENTITY_VERIFIED_EXECUTION_BLOCKED",
        "runtime identity mismatch",
    )
    runtime_digest = hashlib.sha256(canonical_json_bytes(runtime)).hexdigest()
    return SentinelExecutionAuthority(external_sha256, paths, runtime_digest)
