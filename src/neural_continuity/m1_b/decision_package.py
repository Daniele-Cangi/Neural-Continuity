from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_b.decision_policy import decide_transition_b
from neural_continuity.m1_b.pure_comparison import compare_paired_observations
from neural_continuity.m1_teacher_evidence import (
    TeacherEvidenceError,
    _artifact_entry,
    _fail,
    _load_json,
    _require_mapping,
    _require_string,
    _verify_artifacts,
    _write_bytes,
)


def _contracts(
    transition_b_path: str | Path, transition_a_path: str | Path
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    b_path = Path(transition_b_path)
    a_path = Path(transition_a_path)
    contract_b = _load_json(b_path, "TRANSITION_B_CONTRACT_INVALID")
    contract_a = _load_json(a_path, "TRANSITION_A_CONTRACT_INVALID")
    b_sha = sha256_file(b_path)
    a_sha = sha256_file(a_path)
    inheritance = _require_mapping(contract_b.get("tolerance_inheritance"), "tolerance_inheritance")
    if (
        contract_b.get("contract_id") != "m1-transition-b-v1"
        or contract_a.get("contract_id") != "m1-transition-a-v1"
        or inheritance.get("source_contract_sha256") != a_sha
    ):
        raise _fail(
            "TRANSITION_B_TOLERANCE_AUTHORITY_INVALID",
            "frozen tolerance authority differs",
        )
    return contract_b, contract_a, b_sha, a_sha


def _evidence_identity(bundle_path: str | Path) -> dict[str, str]:
    bundle = Path(bundle_path).resolve()
    manifest = bundle.parent / "evidence-manifest.json"
    if not bundle.is_file() or not manifest.is_file():
        raise _fail("MISSING_DECLARED_SOURCE_OBSERVATION", "paired evidence bundle is missing")
    return {
        "bundle_path": str(bundle),
        "bundle_sha256": sha256_file(bundle),
        "manifest_sha256": sha256_file(manifest),
    }


def create_transition_b_decision_package(
    source_bundle: str | Path,
    target_bundle: str | Path,
    transition_b_contract: str | Path,
    transition_a_contract: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    contract_b, contract_a, b_sha, a_sha = _contracts(transition_b_contract, transition_a_contract)
    comparison = compare_paired_observations(source_bundle, target_bundle)
    decision = decide_transition_b(comparison, contract_b, contract_a)
    source_identity = _evidence_identity(source_bundle)
    target_identity = _evidence_identity(target_bundle)
    output = Path(output_directory).resolve()
    if output.exists():
        raise _fail("OUTPUT_ALREADY_EXISTS", f"output already exists: {output}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        report_path = temporary / "comparison-report.json"
        decision_path = temporary / "decision.json"
        _write_bytes(report_path, canonical_json_bytes(comparison) + b"\n")
        _write_bytes(decision_path, canonical_json_bytes(decision) + b"\n")
        replay = {
            "replay_format_version": "1.0.0",
            "transition_id": "B",
            "contracts": {
                "transition_b": {
                    "path": str(Path(transition_b_contract).resolve()),
                    "sha256": b_sha,
                },
                "transition_a": {
                    "path": str(Path(transition_a_contract).resolve()),
                    "sha256": a_sha,
                },
            },
            "source_evidence": source_identity,
            "target_evidence": target_identity,
            "comparison_report_path": report_path.name,
            "decision_path": decision_path.name,
            "expected_status": decision["transition_b_status"],
            "replay_requires_model_execution": False,
        }
        bundle_path = temporary / "replay-bundle.json"
        _write_bytes(bundle_path, canonical_json_bytes(replay) + b"\n")
        artifacts = [
            _artifact_entry(temporary, path) for path in (report_path, decision_path, bundle_path)
        ]
        manifest = {
            "evidence_format_version": "1.0.0",
            "package_kind": "m1_transition_b_decision",
            "evidence_status": "DECIDED_PENDING_REPLAY",
            "transition_b_status": decision["transition_b_status"],
            "contract_id": contract_b["contract_id"],
            "contract_sha256": b_sha,
            "artifacts": sorted(artifacts, key=lambda entry: entry["path"]),
            "integrity": {
                "artifact_hash_algorithm": "SHA-256",
                "missing_evidence_behavior": "BLOCKED",
                "replay_without_model_execution_required": True,
            },
        }
        _write_bytes(temporary / "evidence-manifest.json", canonical_json_bytes(manifest) + b"\n")
        os.replace(temporary, output)
        return {
            "status": "PASS",
            "transition_b_status": decision["transition_b_status"],
            "output_directory": str(output),
            "evidence_manifest_sha256": sha256_file(output / "evidence-manifest.json"),
        }
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def replay_transition_b_decision(bundle_path: str | Path) -> dict[str, Any]:
    bundle = Path(bundle_path).resolve()
    root = bundle.parent
    replay = _load_json(bundle, "TRANSITION_B_REPLAY_INVALID")
    manifest = _load_json(root / "evidence-manifest.json", "TRANSITION_B_MANIFEST_INVALID")
    _verify_artifacts(root, manifest, "artifacts")
    contracts = _require_mapping(replay.get("contracts"), "contracts")
    b_ref = _require_mapping(contracts.get("transition_b"), "contracts.transition_b")
    a_ref = _require_mapping(contracts.get("transition_a"), "contracts.transition_a")
    contract_b, contract_a, b_sha, a_sha = _contracts(
        _require_string(b_ref.get("path"), "transition_b.path"),
        _require_string(a_ref.get("path"), "transition_a.path"),
    )
    if b_ref.get("sha256") != b_sha or a_ref.get("sha256") != a_sha:
        raise _fail("TRANSITION_B_TOLERANCE_AUTHORITY_INVALID", "replay contract hash differs")
    source = _require_mapping(replay.get("source_evidence"), "source_evidence")
    target = _require_mapping(replay.get("target_evidence"), "target_evidence")
    for identity in (source, target):
        evidence_bundle = Path(_require_string(identity.get("bundle_path"), "bundle_path"))
        if sha256_file(evidence_bundle) != identity.get("bundle_sha256"):
            raise _fail("EVIDENCE_ARTIFACT_HASH_MISMATCH", "paired replay bundle hash differs")
        if sha256_file(evidence_bundle.parent / "evidence-manifest.json") != identity.get(
            "manifest_sha256"
        ):
            raise _fail("EVIDENCE_ARTIFACT_HASH_MISMATCH", "paired manifest hash differs")
    comparison = compare_paired_observations(source["bundle_path"], target["bundle_path"])
    decision = decide_transition_b(comparison, contract_b, contract_a)
    recorded_report = _load_json(root / "comparison-report.json", "COMPARISON_REPORT_INVALID")
    recorded_decision = _load_json(root / "decision.json", "TRANSITION_B_DECISION_INVALID")
    status_match = decision["transition_b_status"] == replay.get("expected_status")
    replay_verified = comparison == recorded_report and decision == recorded_decision
    if not replay_verified or not status_match:
        raise _fail("TRANSITION_B_REPLAY_MISMATCH", "replayed comparison or decision differs")
    return {
        "status": "PASS",
        "replay_verified": replay_verified,
        "model_execution_used": False,
        "transition_b_status": decision["transition_b_status"],
        "status_match": status_match,
        "run_outcome_match": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create or replay the M1 Transition B decision.")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--source-bundle", required=True)
    create.add_argument("--target-bundle", required=True)
    create.add_argument("--transition-b-contract", required=True)
    create.add_argument("--transition-a-contract", required=True)
    create.add_argument("--output", required=True)
    replay = commands.add_parser("replay")
    replay.add_argument("--bundle", required=True)
    args = parser.parse_args(argv)
    try:
        result = (
            create_transition_b_decision_package(
                args.source_bundle,
                args.target_bundle,
                args.transition_b_contract,
                args.transition_a_contract,
                args.output,
            )
            if args.command == "create"
            else replay_transition_b_decision(args.bundle)
        )
    except TeacherEvidenceError as exc:
        print(json.dumps({"status": exc.status, "error": exc.__dict__}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
