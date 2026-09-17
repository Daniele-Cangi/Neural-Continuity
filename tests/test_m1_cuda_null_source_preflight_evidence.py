"""Model-free replay rejects missing or unprofiled source observations."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from neural_continuity.evidence import canonical_json_bytes
from neural_continuity.m1_diagnostics import cuda_null_source_preflight_evidence as evidence
from neural_continuity.m1_diagnostics.cuda_null_source_preflight_inputs import FROZEN_RUNS


def _capture(monkeypatch: object) -> dict[str, object]:
    runtime = {
        "status": "RUNTIME_IDENTITY_VERIFIED_EXECUTION_BLOCKED",
        "onnx_graph_loaded": False,
        "session_created": False,
        "execution_authorized": False,
    }
    runtime_hash = hashlib.sha256(canonical_json_bytes(runtime) + b"\n").hexdigest()
    monkeypatch.setattr(evidence, "RUNTIME_IDENTITY_SHA256", runtime_hash)
    authority = {
        "status": "SOURCE_ONLY_PREFLIGHT_AUTHORITY_VERIFIED",
        "authorization_spec_sha256": evidence.AUTHORIZATION_SPEC_SHA256,
        "parent_spec_sha256": evidence.PREFLIGHT_SPEC_SHA256,
        "readiness_record_sha256": evidence.READINESS_RECORD_SHA256,
        "runtime_identity_sha256": runtime_hash,
        "source_only": True,
        "document_count": 64,
        "query_count": 64,
        "qualifying_m1_evidence": False,
        "sentinel_allowed": False,
        "int8_allowed": False,
        "full_corpus_allowed": False,
        "holdout_allowed": False,
        "onnx_graph_loaded": False,
        "session_created": False,
        "technical_preflight_permission": "GRANTED_AFTER_REVIEW",
    }
    authority["record_sha256"] = hashlib.sha256(canonical_json_bytes(authority) + b"\n").hexdigest()
    documents = [f"d{index:05}" for index in range(5183)]
    queries = [f"q{index:02}" for index in range(81)]
    identity = {
        "kind": "m1_cuda_null_source_preflight_input_identity",
        "dataset_role": "measurement_null",
        "dataset_manifest_sha256": evidence.DATASET_MANIFEST_SHA256,
        "corpus_sha256": evidence.CORPUS_SHA256,
        "queries_sha256": evidence.QUERY_SHA256,
        "qrels_sha256": evidence.QRELS_SHA256,
        "source_onnx_sha256": evidence.SOURCE_ONNX_SHA256,
        "teacher_revision": evidence.TEACHER_REVISION,
        "normalization": "l2_unit_after_encode",
        "embedding_dimension": evidence.EMBEDDING_DIMENSION,
        "max_sequence_length": 256,
        "all_document_ids": documents,
        "all_query_ids": queries,
        "selected_document_ids": documents[:64],
        "selected_query_ids": queries[:64],
        "source_preflight_authority": authority,
    }
    observations = {}
    profiles = []
    for role, batch in FROZEN_RUNS:
        name = f"{role}_batch_{batch}"
        array = np.zeros((64, 384), dtype=np.float32)
        array[:, 0] = 1.0
        observations[name] = array
        profiles.append(
            {
                "role": role,
                "batch_size": batch,
                "observation_name": name,
                "observation_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                "item_count": 64,
                "session_seconds": 0.1,
                "encode_seconds": 1.0,
                "items_per_second": 64.0,
                "provider_event_counts": {"CUDAExecutionProvider": 1},
                "operator_event_counts": {"CUDAExecutionProvider": {"MatMul": 1}},
                "cpu_fallback_operator_types": [],
                "unclassified_cpu_events": 0,
                "undeclared_providers": [],
            }
        )
    return {
        "runtime_inventory": runtime,
        "input_identity": identity,
        "observations": observations,
        "provider_profiles": profiles,
    }


def test_corrupt_observation_archive_fails_closed(monkeypatch: object, tmp_path: Path) -> None:
    from zipfile import BadZipFile

    capture = _capture(monkeypatch)
    output, manifest_hash = evidence.write_source_preflight_package(capture, tmp_path / "run")
    for error_type in (BadZipFile, EOFError):

        def _broken_load(
            *_args: object,
            _error_type: type[Exception] = error_type,
            **_kwargs: object,
        ) -> None:
            raise _error_type("corrupt observation archive")

        monkeypatch.setattr(evidence.np, "load", _broken_load)
        replay = evidence.replay_source_preflight(output / "replay-bundle.json", manifest_hash)
        assert replay["replay_status"] == "BLOCKED"
        assert replay["technical_preflight_status"] == "BLOCKED"


def test_model_free_package_replays_and_tampering_blocks(
    monkeypatch: object, tmp_path: Path
) -> None:
    capture = _capture(monkeypatch)
    output, manifest_hash = evidence.write_source_preflight_package(capture, tmp_path / "run")
    replay = evidence.replay_source_preflight(output / "replay-bundle.json", manifest_hash)
    assert replay["replay_status"] == "PASS"
    assert replay["technical_preflight_status"] == "PASS"
    assert replay["model_loaded"] is False
    (output / "provider-profile.jsonl").write_text("", encoding="utf-8")
    blocked = evidence.replay_source_preflight(output / "replay-bundle.json", manifest_hash)
    assert blocked["replay_status"] == "BLOCKED"


def test_each_run_needs_cuda_activity(monkeypatch: object) -> None:
    capture = _capture(monkeypatch)
    capture["provider_profiles"][0]["provider_event_counts"] = {"CPUExecutionProvider": 1}
    capture["provider_profiles"][0]["operator_event_counts"] = {
        "CPUExecutionProvider": {"MatMul": 1}
    }
    try:
        evidence._decision(
            capture["runtime_inventory"],
            capture["input_identity"],
            capture["observations"],
            capture["provider_profiles"],
        )
    except ValueError as exc:
        assert "CUDA provider activity" in str(exc)
    else:
        raise AssertionError("CPU-only run was accepted")


@pytest.mark.parametrize(
    ("field", "altered"),
    [
        ("authorization_spec_sha256", "0" * 64),
        ("document_count", 65),
        ("query_count", 65),
        ("qualifying_m1_evidence", 0),
        ("sentinel_allowed", True),
        ("onnx_graph_loaded", True),
        ("session_created", True),
        ("unexpected_permission", True),
    ],
)
def test_resealed_authority_mutation_blocks(
    monkeypatch: object, field: str, altered: object
) -> None:
    capture = _capture(monkeypatch)
    record = capture["input_identity"]["source_preflight_authority"]
    record[field] = altered
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    record["record_sha256"] = hashlib.sha256(canonical_json_bytes(payload) + b"\n").hexdigest()
    with pytest.raises(ValueError, match="authority record"):
        evidence._decision(
            capture["runtime_inventory"],
            capture["input_identity"],
            capture["observations"],
            capture["provider_profiles"],
        )


def test_output_path_normalizes_before_repository_containment(tmp_path: Path) -> None:
    repository = Path(evidence.__file__).resolve().parents[3]
    disguised = (
        repository.parent
        / ".."
        / repository.parent.name
        / repository.name
        / f".preflight-containment-{tmp_path.name}"
    )
    with pytest.raises(ValueError, match="outside the repository"):
        evidence._external_output_directory(disguised)


def test_resealed_oversized_profile_number_fails_closed(
    monkeypatch: object, tmp_path: Path
) -> None:
    capture = _capture(monkeypatch)
    output, _ = evidence.write_source_preflight_package(capture, tmp_path / "run")
    profile_path = output / "provider-profile.jsonl"
    profiles = [
        evidence.json.loads(line) for line in profile_path.read_text(encoding="utf-8").splitlines()
    ]
    profiles[0]["encode_seconds"] = 10**1000
    profile_path.write_bytes(
        b"".join(canonical_json_bytes(profile) + b"\n" for profile in profiles)
    )
    manifest_path = output / "artifact-manifest.json"
    manifest = evidence._read_json(manifest_path)
    for entry in manifest["artifacts"]:
        if entry["path"] == "provider-profile.jsonl":
            entry["sha256"] = evidence.sha256_file(profile_path)
    evidence._write_json(manifest_path, manifest)
    replay = evidence.replay_source_preflight(
        output / "replay-bundle.json", evidence.sha256_file(manifest_path)
    )
    assert replay["replay_status"] == "BLOCKED"
    assert replay["technical_preflight_status"] == "BLOCKED"
