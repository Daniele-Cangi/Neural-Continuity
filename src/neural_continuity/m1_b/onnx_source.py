from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.evidence import sha256_file
from neural_continuity.m1_teacher_evidence import (
    _fail,
    _load_json,
    _require_mapping,
    _require_string,
    _safe_artifact_path,
    _verify_artifacts,
)


@dataclass(frozen=True)
class VerifiedOnnxSource:
    artifact_path: Path
    artifact_sha256: str
    transition_a_manifest_sha256: str
    execution_provider: str


def load_verified_onnx_source(
    transition_a_bundle: str | Path, contract: Mapping[str, Any]
) -> VerifiedOnnxSource:
    bundle_path = Path(transition_a_bundle).resolve()
    root = bundle_path.parent
    bundle = _load_json(bundle_path, "TRANSITION_A_BUNDLE_INVALID")
    if bundle.get("transition_id") != "A":
        raise _fail("TRANSITION_A_BUNDLE_INVALID", "source bundle is not Transition A")
    manifest_path = root / "evidence-manifest.json"
    manifest = _load_json(manifest_path, "TRANSITION_A_MANIFEST_INVALID")
    _verify_artifacts(root, manifest, "artifacts")
    transition_a = _require_mapping(
        _require_mapping(contract.get("preconditions"), "preconditions").get("transition_a"),
        "preconditions.transition_a",
    )
    expected_manifest = _require_string(
        transition_a.get("evidence_manifest_sha256"),
        "preconditions.transition_a.evidence_manifest_sha256",
    )
    if sha256_file(manifest_path) != expected_manifest:
        raise _fail(
            "TRANSITION_A_IDENTITY_MISMATCH", "Transition A evidence manifest SHA-256 differs"
        )
    decision = _load_json(root / "decision.json", "TRANSITION_A_DECISION_INVALID")
    if decision.get("transition_a_status") != "PASS":
        raise _fail("TRANSITION_A_NOT_PASS", "Transition A is not PASS")
    onnx_manifest = _load_json(root / "onnx-manifest.json", "TRANSITION_A_ONNX_INVALID")
    artifact_path = _safe_artifact_path(
        root, _require_string(onnx_manifest.get("artifact_path"), "onnx-manifest.artifact_path")
    )
    actual_hash = sha256_file(artifact_path)
    expected_hash = _require_string(
        transition_a.get("onnx_fp32_artifact_sha256"),
        "preconditions.transition_a.onnx_fp32_artifact_sha256",
    )
    if actual_hash != expected_hash or onnx_manifest.get("artifact_sha256") != expected_hash:
        raise _fail("TRANSITION_A_IDENTITY_MISMATCH", "Transition A ONNX artifact SHA-256 differs")
    provider = _require_string(
        onnx_manifest.get("requested_execution_provider"),
        "onnx-manifest.requested_execution_provider",
    )
    if provider != "CPUExecutionProvider":
        raise _fail("ONNX_PROVIDER_UNVERIFIED", "only CPUExecutionProvider is authorized")
    return VerifiedOnnxSource(artifact_path, actual_hash, expected_manifest, provider)


def open_onnx_session(source: VerifiedOnnxSource) -> Any:
    try:
        import onnxruntime
    except ModuleNotFoundError as exc:
        raise _fail("ONNX_DEPENDENCY_MISSING", f"missing dependency: {exc.name}") from exc
    if source.execution_provider not in onnxruntime.get_available_providers():
        raise _fail("ONNX_PROVIDER_MISSING", f"provider unavailable: {source.execution_provider}")
    try:
        session = onnxruntime.InferenceSession(
            str(source.artifact_path), providers=[source.execution_provider]
        )
    except Exception as exc:
        raise _fail(
            "ONNX_SESSION_FAILED", f"cannot load verified ONNX source: {exc}", "EXECUTION_ERROR"
        ) from exc
    if source.execution_provider not in session.get_providers():
        raise _fail("ONNX_PROVIDER_MISSING", "verified provider is not active")
    return session


def encode_onnx_source(
    teacher: Any, session: Any, texts: Sequence[str], batch_size: int, label: str
) -> np.ndarray:
    expected_inputs = {"input_ids", "attention_mask", "token_type_ids"}
    if {value.name for value in session.get_inputs()} != expected_inputs:
        raise _fail("ONNX_IO_INVALID", "ONNX source graph input names are not authoritative")
    batches: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        tokens = teacher.tokenize(list(texts[start : start + batch_size]))
        input_ids = tokens.get("input_ids")
        attention_mask = tokens.get("attention_mask")
        if input_ids is None or attention_mask is None:
            raise _fail("ONNX_TOKENIZATION_INVALID", f"tokenizer lacks inputs for {label}")
        token_type_ids = tokens.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = np.zeros_like(input_ids.detach().cpu().numpy())
        else:
            token_type_ids = token_type_ids.detach().cpu().numpy()
        try:
            values = session.run(
                ["embeddings"],
                {
                    "input_ids": input_ids.detach().cpu().numpy().astype(np.int64, copy=False),
                    "attention_mask": attention_mask.detach()
                    .cpu()
                    .numpy()
                    .astype(np.int64, copy=False),
                    "token_type_ids": np.asarray(token_type_ids, dtype=np.int64),
                },
            )[0]
        except Exception as exc:
            raise _fail(
                "ONNX_INFERENCE_FAILED", f"cannot encode {label}: {exc}", "EXECUTION_ERROR"
            ) from exc
        batches.append(np.asarray(values, dtype=np.float32))
    embeddings = np.ascontiguousarray(np.concatenate(batches, axis=0), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts) or embeddings.shape[1] == 0:
        raise _fail(
            "ONNX_OUTPUT_INVALID", f"invalid embedding shape for {label}", "EXECUTION_ERROR"
        )
    if not np.isfinite(embeddings).all():
        raise _fail(
            "ONNX_OUTPUT_INVALID", f"non-finite embedding values for {label}", "EXECUTION_ERROR"
        )
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise _fail(
            "ONNX_OUTPUT_INVALID", f"non-positive embedding norm for {label}", "EXECUTION_ERROR"
        )
    return np.ascontiguousarray(embeddings / norms[:, np.newaxis], dtype=np.float32)
