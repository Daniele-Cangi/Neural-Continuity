from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural_continuity.m1_diagnostics.activation_evidence import (
    replay_activation_capture,
)

EXPECTED_BATCH_COUNT = 23
EXPECTED_FLOATING_PROBE_COUNT = 283
EXPECTED_INTEGER_PROBE_COUNT = 248
EXPECTED_QUERY_COUNT = 364


class ActivationAnalysisError(RuntimeError):
    def __init__(self, code: str, message: str, status: str = "BLOCKED") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "status": self.status}


@dataclass(frozen=True)
class VerifiedActivationAnalysisInput:
    root: Path
    bundle_path: Path
    manifest_sha256: str
    capture_plan: Mapping[str, Any]
    batch_index: Mapping[str, Any]
    replay_result: Mapping[str, Any]


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ActivationAnalysisError(
            "ACTIVATION_AUTHORITY_INVALID", f"cannot load {path.name}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ActivationAnalysisError(
            "ACTIVATION_AUTHORITY_INVALID", f"{path.name} must contain an object"
        )
    return payload


def _safe_artifact(root: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ActivationAnalysisError("ACTIVATION_AUTHORITY_INVALID", "artifact path is missing")
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ActivationAnalysisError(
            "ACTIVATION_AUTHORITY_INVALID", "artifact path must be relative"
        )
    candidate = (root / relative).resolve()
    if root != candidate and root not in candidate.parents:
        raise ActivationAnalysisError(
            "ACTIVATION_AUTHORITY_INVALID", "artifact path escapes the package"
        )
    if not candidate.is_file():
        raise ActivationAnalysisError(
            "ACTIVATION_AUTHORITY_MISSING",
            f"declared artifact is missing: {relative_path}",
        )
    return candidate


def _require_exact_counts(capture_plan: Mapping[str, Any], batch_index: Mapping[str, Any]) -> None:
    expected = {
        "probe_count": EXPECTED_FLOATING_PROBE_COUNT,
        "integer_probe_count": EXPECTED_INTEGER_PROBE_COUNT,
        "query_count": EXPECTED_QUERY_COUNT,
    }
    for field, value in expected.items():
        if capture_plan.get(field) != value:
            raise ActivationAnalysisError(
                "ACTIVATION_AUTHORITY_IDENTITY_MISMATCH",
                f"capture plan {field} does not match the frozen evidence",
            )
    if batch_index.get("batch_count") != EXPECTED_BATCH_COUNT:
        raise ActivationAnalysisError(
            "ACTIVATION_AUTHORITY_IDENTITY_MISMATCH",
            "batch count does not match the frozen evidence",
        )
    if batch_index.get("query_count") != EXPECTED_QUERY_COUNT:
        raise ActivationAnalysisError(
            "ACTIVATION_AUTHORITY_IDENTITY_MISMATCH",
            "batch-index query count does not match the frozen evidence",
        )


def _require_batch_semantics(batch_index: Mapping[str, Any]) -> None:
    batches = batch_index.get("batches")
    if not isinstance(batches, list) or len(batches) != EXPECTED_BATCH_COUNT:
        raise ActivationAnalysisError(
            "ACTIVATION_AUTHORITY_INVALID", "batch records are incomplete"
        )
    query_ids: list[str] = []
    for record in batches:
        if not isinstance(record, Mapping) or not isinstance(record.get("query_ids"), list):
            raise ActivationAnalysisError(
                "ACTIVATION_AUTHORITY_INVALID", "batch query identities are invalid"
            )
        query_ids.extend(str(value) for value in record["query_ids"])
    if len(query_ids) != EXPECTED_QUERY_COUNT or len(set(query_ids)) != EXPECTED_QUERY_COUNT:
        raise ActivationAnalysisError(
            "ACTIVATION_AUTHORITY_INVALID", "batch query identities are incomplete"
        )
    if query_ids != sorted(query_ids, key=lambda value: value.encode("utf-8")):
        raise ActivationAnalysisError(
            "QUERY_ORDER_MISMATCH", "batch queries violate UTF-8 byte ordering"
        )


def _require_plan_semantics(capture_plan: Mapping[str, Any]) -> None:
    if capture_plan.get("frozen_transition_b_v1_scientific_decision") != "FAIL":
        raise ActivationAnalysisError(
            "FROZEN_DECISION_MISMATCH",
            "the frozen Transition B v1 decision is not FAIL",
        )
    if capture_plan.get("scientific_decision_recomputed") is not False:
        raise ActivationAnalysisError(
            "SCIENTIFIC_SCOPE_VIOLATION",
            "activation capture recomputed a scientific decision",
        )
    if capture_plan.get("execution_provider") != "CPUExecutionProvider":
        raise ActivationAnalysisError(
            "ACTIVATION_AUTHORITY_IDENTITY_MISMATCH",
            "activation package does not use the frozen CPU execution provider",
        )
    exceptions = capture_plan.get("exceptions")
    if not isinstance(exceptions, Mapping) or any(exceptions.values()):
        raise ActivationAnalysisError(
            "CANDIDATE_EXCEPTION_PRESENT",
            "activation capture contains a graph or probe exception",
        )
    probes = capture_plan.get("probe_mappings")
    integers = capture_plan.get("integer_mappings")
    query_order = capture_plan.get("query_order")
    if not isinstance(probes, list) or len(probes) != EXPECTED_FLOATING_PROBE_COUNT:
        raise ActivationAnalysisError(
            "ACTIVATION_AUTHORITY_INVALID",
            "floating probe mappings are incomplete",
        )
    if not isinstance(integers, list) or len(integers) != EXPECTED_INTEGER_PROBE_COUNT:
        raise ActivationAnalysisError(
            "ACTIVATION_AUTHORITY_INVALID",
            "integer probe mappings are incomplete",
        )
    if query_order != "query_id_utf8_byte_order":
        raise ActivationAnalysisError(
            "QUERY_ORDER_MISMATCH", "capture plan query-order rule is not authoritative"
        )
    probe_ids = [item.get("probe_id") for item in probes if isinstance(item, Mapping)]
    integer_ids = [item.get("probe_id") for item in integers if isinstance(item, Mapping)]
    if len(set(probe_ids)) != EXPECTED_FLOATING_PROBE_COUNT:
        raise ActivationAnalysisError(
            "ACTIVATION_AUTHORITY_INVALID",
            "floating probe identities are not unique",
        )
    if len(set(integer_ids)) != EXPECTED_INTEGER_PROBE_COUNT or not set(integer_ids).issubset(
        probe_ids
    ):
        raise ActivationAnalysisError(
            "ACTIVATION_AUTHORITY_INVALID",
            "integer probe identities are inconsistent",
        )


def verify_activation_analysis_input(
    bundle_path: str | Path, expected_manifest_sha256: str
) -> VerifiedActivationAnalysisInput:
    bundle = Path(bundle_path).resolve()
    if not bundle.is_file():
        raise ActivationAnalysisError(
            "ACTIVATION_AUTHORITY_MISSING",
            "activation replay bundle is missing",
        )
    root = bundle.parent
    try:
        replay_result = replay_activation_capture(bundle, expected_manifest_sha256)
    except Exception as exc:
        raise ActivationAnalysisError(
            "ACTIVATION_REPLAY_FAILED",
            f"source activation replay failed: {exc}",
        ) from exc
    required_replay = {
        "status": "COMPLETE",
        "activation_capture_status": "CAPTURED",
        "replay_verified": True,
        "status_match": True,
        "summary_match": True,
        "model_execution_used": False,
        "query_count": EXPECTED_QUERY_COUNT,
        "batch_count": EXPECTED_BATCH_COUNT,
        "floating_probe_count": EXPECTED_FLOATING_PROBE_COUNT,
        "integer_probe_count": EXPECTED_INTEGER_PROBE_COUNT,
    }
    if any(replay_result.get(key) != value for key, value in required_replay.items()):
        raise ActivationAnalysisError(
            "ACTIVATION_REPLAY_MISMATCH",
            "source activation replay is not authoritative",
        )
    replay_bundle = _load_json(bundle)
    if replay_bundle.get("replay_requires_model_execution") is not False:
        raise ActivationAnalysisError("MODEL_EXECUTION_REQUIRED", "source replay is not model-free")
    capture_plan = _load_json(_safe_artifact(root, replay_bundle.get("capture_plan_path")))
    batch_index = _load_json(_safe_artifact(root, replay_bundle.get("batch_index_path")))
    _require_exact_counts(capture_plan, batch_index)
    _require_batch_semantics(batch_index)
    _require_plan_semantics(capture_plan)
    return VerifiedActivationAnalysisInput(
        root=root,
        bundle_path=bundle,
        manifest_sha256=expected_manifest_sha256,
        capture_plan=capture_plan,
        batch_index=batch_index,
        replay_result=replay_result,
    )
