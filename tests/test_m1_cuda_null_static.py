"""Fail-closed tests for the model-free CUDA-null static package."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics import cuda_null_authority as authority_module
from neural_continuity.m1_diagnostics import (
    measurement_null_extension_evidence as extension_evidence,
)
from neural_continuity.m1_diagnostics.cuda_null_authority import (
    CORPUS_SHA256,
    CPU_EXTENSION_MANIFEST_SHA256,
    DATASET_MANIFEST_SHA256,
    HISTORICAL_CUDA_MANIFEST_SHA256,
    MATERIALIZATION_POLICY_SHA256,
    MEASUREMENT_QRELS_SHA256,
    MEASUREMENT_QUERIES_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    PARTITION_POLICY_SHA256,
    ROLE_ORDER,
    TEACHER_MANIFEST_SHA256,
    TRANSITION_A_MANIFEST_SHA256,
    TRANSITION_A_ONNX_SHA256,
    TRANSITION_B_CONTRACT_SHA256,
    CudaNullAuthorityBlocked,
    _snapshot_target_is_allowed,
)
from neural_continuity.m1_diagnostics.cuda_null_evidence import (
    replay_static_package,
    write_static_package,
)
from neural_continuity.m1_diagnostics.measurement_null_extension_authority import (
    MeasurementNullPlanError,
)


def _authority() -> dict[str, object]:
    digest = "a" * 64
    return {
        "kind": "m1_cuda_null_static_authority",
        "version": "1.0.0",
        "status": "STATIC_VERIFIED_EXECUTION_BLOCKED",
        "config_sha256": digest,
        "contract_sha256": TRANSITION_B_CONTRACT_SHA256,
        "dataset": {
            "materialization_manifest_sha256": DATASET_MANIFEST_SHA256,
            "partition_policy_sha256": PARTITION_POLICY_SHA256,
            "materialization_policy_sha256": MATERIALIZATION_POLICY_SHA256,
            "document_count": 5183,
            "measurement_query_count": 81,
            "measurement_qrel_count": 103,
            **{
                key: digest
                for key in (
                    "document_ids_sha256",
                    "query_ids_sha256",
                    "qrels_sha256",
                    "corpus_sha256",
                    "measurement_queries_sha256",
                    "measurement_qrels_sha256",
                )
            },
            "corpus_sha256": CORPUS_SHA256,
            "measurement_queries_sha256": MEASUREMENT_QUERIES_SHA256,
            "measurement_qrels_sha256": MEASUREMENT_QRELS_SHA256,
            "role_order": list(ROLE_ORDER),
        },
        "source": {
            "transition_a_manifest_sha256": TRANSITION_A_MANIFEST_SHA256,
            "onnx_sha256": TRANSITION_A_ONNX_SHA256,
            "teacher_manifest_sha256": TEACHER_MANIFEST_SHA256,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "embedding_dimension": 384,
            "normalization": "l2_unit_after_encode",
            "snapshot_files_sha256": digest,
        },
        "historical_cpu_extension_manifest_sha256": CPU_EXTENSION_MANIFEST_SHA256,
        "historical_cuda_preflight_manifest_sha256": HISTORICAL_CUDA_MANIFEST_SHA256,
        "historical_cuda_preflight_is_qualifying": False,
        "runtime_verified": False,
        "onnx_graph_loaded": False,
        "session_created": False,
        "activation_read": False,
        "execution_authorized": False,
    }


def test_static_package_replays_without_model(tmp_path: Path) -> None:
    run = tmp_path / "run"
    result = write_static_package(_authority(), run)
    replay = replay_static_package(run / "replay-bundle.json", result["manifest_sha256"])
    assert replay["integrity_replay_status"] == "PASS"
    assert replay["execution_authorized"] is False
    assert replay["scientific_outcome"] == "NOT_EVALUATED"


def test_static_replay_rejects_missing_declared_artifact(tmp_path: Path) -> None:
    run = tmp_path / "run"
    result = write_static_package(_authority(), run)
    (run / "decision.json").unlink()
    with pytest.raises(CudaNullAuthorityBlocked):
        replay_static_package(run / "replay-bundle.json", result["manifest_sha256"])


def test_static_replay_rejects_tampered_authority(tmp_path: Path) -> None:
    run = tmp_path / "run"
    result = write_static_package(_authority(), run)
    path = run / "static-authority.json"
    authority = json.loads(path.read_text(encoding="utf-8"))
    authority["execution_authorized"] = True
    path.write_text(json.dumps(authority), encoding="utf-8")
    with pytest.raises(CudaNullAuthorityBlocked):
        replay_static_package(run / "replay-bundle.json", result["manifest_sha256"])


def test_static_replay_requires_external_manifest_hash(tmp_path: Path) -> None:
    run = tmp_path / "run"
    write_static_package(_authority(), run)
    with pytest.raises(CudaNullAuthorityBlocked):
        replay_static_package(run / "replay-bundle.json", "b" * 64)


def test_static_replay_rejects_symlinked_package_parent(tmp_path: Path) -> None:
    run = tmp_path / "run"
    result = write_static_package(_authority(), run)
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(run, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")
    with pytest.raises(CudaNullAuthorityBlocked, match="symlink or junction"):
        replay_static_package(linked / "replay-bundle.json", result["manifest_sha256"])


def test_static_config_rejects_resealed_scope_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = authority_module.CUDA_NULL_CONFIG_PATH.read_text(encoding="utf-8")
    altered = original.replace("full_corpus_authorized: false", "full_corpus_authorized: true")
    assert altered != original
    config_path = tmp_path / "m1-cuda-null-v1.yaml"
    config_path.write_text(altered, encoding="utf-8")
    monkeypatch.setattr(authority_module, "CUDA_NULL_CONFIG_PATH", config_path)
    with pytest.raises(CudaNullAuthorityBlocked, match="not frozen"):
        authority_module._verify_config(config_path, sha256_file(config_path))


def test_historical_replay_error_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        authority_module,
        "_verify_config",
        lambda *_: authority_module.CUDA_NULL_CONFIG_SHA256,
    )
    monkeypatch.setattr(authority_module, "sha256_file", lambda *_: TRANSITION_B_CONTRACT_SHA256)
    monkeypatch.setattr(
        authority_module, "_read_json", lambda *_: {"contract_id": "m1-transition-b-v1"}
    )
    monkeypatch.setattr(
        authority_module,
        "_verify_dataset",
        lambda *_: {"materialization_manifest_sha256": DATASET_MANIFEST_SHA256},
    )
    monkeypatch.setattr(authority_module, "_verify_source", lambda *_: {})

    def missing_cpu_plan(*_: object) -> None:
        raise MeasurementNullPlanError("historical CPU package missing")

    monkeypatch.setattr(
        extension_evidence, "replay_measurement_null_extension_plan", missing_cpu_plan
    )
    with pytest.raises(CudaNullAuthorityBlocked, match="static authority could not be verified"):
        authority_module.build_static_authority(
            config_path=tmp_path / "m1-cuda-null-v1.yaml",
            external_config_sha256=authority_module.CUDA_NULL_CONFIG_SHA256,
            dataset_root=tmp_path,
            transition_a_bundle=tmp_path / "replay-bundle.json",
            teacher_snapshot_root=tmp_path,
            cpu_extension_bundle=tmp_path / "replay-bundle.json",
            historical_cuda_bundle=tmp_path / "replay-bundle.json",
        )


def test_static_package_cannot_claim_executable_state(tmp_path: Path) -> None:
    authority = _authority()
    authority["runtime_verified"] = True
    with pytest.raises(CudaNullAuthorityBlocked):
        write_static_package(authority, tmp_path / "run")


def test_snapshot_targets_are_confined_to_snapshot_or_model_blobs(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "models--sentence-transformers--all-MiniLM-L6-v2"
    snapshot = model_root / "snapshots" / "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    blob_root = model_root / "blobs"
    snapshot.mkdir(parents=True)
    blob_root.mkdir()
    assert _snapshot_target_is_allowed(snapshot, snapshot / "tokenizer.json")
    assert _snapshot_target_is_allowed(snapshot, blob_root / "sha256-blob")
    assert not _snapshot_target_is_allowed(snapshot, tmp_path / "other" / "tokenizer.json")
    assert not _snapshot_target_is_allowed(snapshot, model_root / "other" / "tokenizer.json")


def test_snapshot_rejects_wrong_revision_root(tmp_path: Path) -> None:
    model_root = tmp_path / "models--sentence-transformers--all-MiniLM-L6-v2"
    snapshot = model_root / "snapshots" / "other-revision"
    assert not _snapshot_target_is_allowed(snapshot, snapshot / "tokenizer.json")


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("root", "historical_cpu_extension_manifest_sha256"),
        ("dataset", "role_order"),
        ("dataset", "qrels_sha256"),
        ("source", "teacher_manifest_sha256"),
        ("source", "model_revision"),
    ],
)
def test_replay_rejects_missing_identity_even_if_hashes_are_resealed(
    tmp_path: Path, section: str, field: str
) -> None:
    run = tmp_path / "run"
    write_static_package(_authority(), run)
    authority_path = run / "static-authority.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    if section == "root":
        del authority[field]
    else:
        del authority[section][field]
    authority_path.write_bytes(canonical_json_bytes(authority) + b"\n")
    bundle_path = run / "replay-bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["authority_sha256"] = sha256_file(authority_path)
    bundle_path.write_bytes(canonical_json_bytes(bundle) + b"\n")
    manifest_path = run / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        path = run / artifact["path"]
        artifact["sha256"] = sha256_file(path)
        artifact["size_bytes"] = path.stat().st_size
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    with pytest.raises(CudaNullAuthorityBlocked):
        replay_static_package(bundle_path, sha256_file(manifest_path))
