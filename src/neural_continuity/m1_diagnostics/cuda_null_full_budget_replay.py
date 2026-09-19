"""Model-free structural replay for timing-only full-input preflight evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import uuid
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.cuda_null_full_budget_authority import (
    DATASET_MANIFEST_SHA256,
    GATE_MANIFEST_SHA256,
    SHA256,
    SOURCE_ONNX_SHA256,
    load_budget_spec,
)
from neural_continuity.m1_diagnostics.cuda_null_paths import has_linked_ancestor
from neural_continuity.m1_diagnostics.cuda_null_sentinel_execution_authority import (
    _load_spec,
)
from neural_continuity.m1_diagnostics.cuda_null_sentinel_postgate_package import (
    replay_sentinel_postgate,
)
from neural_continuity.m1_diagnostics.cuda_null_sentinel_readiness import EPOCH_LAYOUT
from neural_continuity.m1_diagnostics.cuda_null_source_preflight_evidence import (
    _external_output_directory,
)
from neural_continuity.m1_diagnostics.cuda_null_source_preflight_inputs import (
    CORPUS_SHA256,
    QRELS_SHA256,
    QUERY_SHA256,
    _load_jsonl,
    _verified_file,
)

FILES = ("budget-record.json", "replay-bundle.json", "artifact-manifest.json")
ROLES = ("documents", "measurement_null_queries")


def _read_json(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("budget artifact is not a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def validate_budget_record(record: dict[str, Any]) -> None:
    expected = {
        "kind": "m1_cuda_full_corpus_budget_observation",
        "version": "1.0.0",
        "status": "TECHNICAL_TIMING_CAPTURED_NOT_QUALIFYING",
        "sentinel_gate_manifest_sha256": GATE_MANIFEST_SHA256,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "source_onnx_sha256": SOURCE_ONNX_SHA256,
        "qrels_sha256": QRELS_SHA256,
        "ordered_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "projection_is_guaranteed_upper_bound": False,
        "observations_retained": False,
        "rankings_or_metrics_retained": False,
        "qualifying_detection_evidence": False,
        "bounded_full_input_timing_authorized": True,
        "full_corpus_qualification_started": False,
        "full_corpus_qualification_execution_authorized": False,
        "int8_executed": False,
        "holdout_accessed": False,
        "scientific_decision": "NOT_EVALUATED",
    }
    extra = {
        "budget_spec_sha256",
        "document_ids_sha256",
        "query_ids_sha256",
        "runtime_identity_sha256",
        "authority_elapsed_seconds",
        "execution_elapsed_seconds",
        "linear_120_epoch_projection_hours",
        "profiles",
    }
    if set(record) != set(expected) | extra or any(
        record.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("budget record scope differs")
    if any(
        not isinstance(record[key], str) or SHA256.fullmatch(record[key]) is None
        for key in (
            "budget_spec_sha256",
            "document_ids_sha256",
            "query_ids_sha256",
            "runtime_identity_sha256",
        )
    ):
        raise ValueError("budget identity hash is invalid")
    for key in (
        "authority_elapsed_seconds",
        "execution_elapsed_seconds",
        "linear_120_epoch_projection_hours",
    ):
        value = record[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"budget timing is invalid: {key}")
    projection = (
        120 * (record["authority_elapsed_seconds"] + record["execution_elapsed_seconds"]) / 3600
    )
    if abs(record["linear_120_epoch_projection_hours"] - projection) > 1e-9:
        raise ValueError("budget projection differs from recorded timings")
    profiles = record["profiles"]
    keys = [(label, role, size) for label, size in EPOCH_LAYOUT for role in ROLES]
    if not isinstance(profiles, list) or len(profiles) != len(keys):
        raise ValueError("budget provider profile set differs")
    for profile, (label, role, size) in zip(profiles, keys, strict=True):
        count = 5183 if role == "documents" else 81
        if not isinstance(profile, dict) or set(profile) != {
            "run_label",
            "role",
            "batch_size",
            "item_count",
            "elapsed_seconds",
            "profiled_item_count",
            "profile_scope",
            "full_input_profiled",
            "provider_event_counts",
            "operator_event_counts",
            "cpu_fallback_operator_types",
            "unclassified_cpu_events",
            "undeclared_providers",
        }:
            raise ValueError("budget profile schema differs")
        if (
            (profile["run_label"], profile["role"], profile["batch_size"], profile["item_count"])
            != (label, role, size, count)
            or profile["profiled_item_count"] != 64
            or profile["profile_scope"] != "canonical_first_64_ids_per_role"
            or profile["full_input_profiled"] is not False
            or type(profile["elapsed_seconds"]) not in (int, float)
            or not math.isfinite(profile["elapsed_seconds"])
            or profile["elapsed_seconds"] <= 0
            or profile["unclassified_cpu_events"] != 0
            or profile["undeclared_providers"] != []
        ):
            raise ValueError("budget profile identity or classification differs")
        counts = profile["provider_event_counts"]
        operators = profile["operator_event_counts"]
        if (
            not isinstance(counts, dict)
            or not isinstance(operators, dict)
            or set(counts) != set(operators)
            or not set(counts).issubset({"CUDAExecutionProvider", "CPUExecutionProvider"})
            or type(counts.get("CUDAExecutionProvider")) is not int
            or counts["CUDAExecutionProvider"] <= 0
        ):
            raise ValueError("budget profile lacks classified CUDA activity")
        for provider, total in counts.items():
            by_operator = operators[provider]
            if (
                type(total) is not int
                or total <= 0
                or not isinstance(by_operator, dict)
                or not by_operator
                or any(
                    not isinstance(name, str) or not name or type(events) is not int or events <= 0
                    for name, events in by_operator.items()
                )
                or sum(by_operator.values()) != total
            ):
                raise ValueError("budget provider/operator counts differ")
        if profile["cpu_fallback_operator_types"] != sorted(
            operators.get("CPUExecutionProvider", {})
        ):
            raise ValueError("budget CPU fallback declaration differs")


def replay_budget_package(bundle: Path, external_manifest_sha256: str) -> dict[str, Any]:
    try:
        bundle = Path(bundle).absolute()
        if (
            bundle.name != FILES[1]
            or has_linked_ancestor(bundle)
            or {p.name for p in bundle.parent.iterdir()} != set(FILES)
        ):
            raise ValueError("budget artifact set differs")
        manifest = bundle.parent / FILES[2]
        if sha256_file(manifest) != external_manifest_sha256:
            raise ValueError("external budget manifest hash differs")
        entries = _read_json(manifest)
        if (
            set(entries) != {"kind", "version", "artifacts"}
            or entries["kind"] != "m1_cuda_full_budget_manifest"
            or entries["version"] != "1.0.0"
            or not isinstance(entries["artifacts"], list)
            or len(entries["artifacts"]) != 2
        ):
            raise ValueError("budget manifest schema differs")
        for entry, name in zip(entries["artifacts"], FILES[:2], strict=True):
            path = bundle.parent / name
            if (
                not isinstance(entry, dict)
                or set(entry) != {"path", "sha256"}
                or entry["path"] != name
                or not path.is_file()
                or path.is_symlink()
                or sha256_file(path) != entry["sha256"]
            ):
                raise ValueError(f"budget artifact integrity differs: {name}")
        replay = _read_json(bundle)
        if (
            set(replay) != {"kind", "version", "budget_spec_sha256", "model_required_for_replay"}
            or replay["kind"] != "m1_cuda_full_budget_replay"
            or replay["version"] != "1.0.0"
            or replay["model_required_for_replay"] is not False
        ):
            raise ValueError("budget replay declaration differs")
        spec = load_budget_spec(replay["budget_spec_sha256"])
        gate = replay_sentinel_postgate(Path(spec["technical_gate_bundle"]), GATE_MANIFEST_SHA256)
        if gate.get("replay_status") != "PASS":
            raise ValueError("sentinel gate no longer replays")
        record = _read_json(bundle.parent / FILES[0])
        validate_budget_record(record)
        if record["budget_spec_sha256"] != replay["budget_spec_sha256"]:
            raise ValueError("budget authority hash differs")
        sentinel_spec = _load_spec(spec["sentinel_authority_sha256"])
        dataset = Path(sentinel_spec["paths"]["dataset_root"])
        documents = _load_jsonl(
            _verified_file(dataset / "corpus.jsonl", CORPUS_SHA256, "corpus"),
            "document_id",
            5183,
        )
        queries = _load_jsonl(
            _verified_file(
                dataset / "roles" / "measurement_null.queries.jsonl",
                QUERY_SHA256,
                "queries",
            ),
            "query_id",
            81,
        )
        _verified_file(dataset / "roles" / "measurement_null.qrels.tsv", QRELS_SHA256, "qrels")
        for field, rows in (("document_ids_sha256", documents), ("query_ids_sha256", queries)):
            digest = hashlib.sha256(
                canonical_json_bytes([identity for identity, _text in rows])
            ).hexdigest()
            if record[field] != digest:
                raise ValueError(f"budget input identity differs: {field}")
        return {
            "replay_status": "PASS",
            "status": record["status"],
            "artifact_manifest_sha256": external_manifest_sha256,
            "model_loaded": False,
            "timing_truth_reproduced": False,
            "full_corpus_qualification_execution_authorized": False,
        }
    except (OSError, ValueError, TypeError, KeyError, UnicodeError) as exc:
        return {"replay_status": "BLOCKED", "reason": str(exc), "model_loaded": False}


def write_budget_package(output: Path, record: dict[str, Any]) -> tuple[Path, str]:
    validate_budget_record(record)
    target = _external_output_directory(output)
    temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    _write_json(temporary / FILES[0], record)
    _write_json(
        temporary / FILES[1],
        {
            "kind": "m1_cuda_full_budget_replay",
            "version": "1.0.0",
            "budget_spec_sha256": record["budget_spec_sha256"],
            "model_required_for_replay": False,
        },
    )
    _write_json(
        temporary / FILES[2],
        {
            "kind": "m1_cuda_full_budget_manifest",
            "version": "1.0.0",
            "artifacts": [
                {"path": name, "sha256": sha256_file(temporary / name)} for name in FILES[:2]
            ],
        },
    )
    digest = sha256_file(temporary / FILES[2])
    if replay_budget_package(temporary / FILES[1], digest)["replay_status"] != "PASS":
        raise ValueError("new budget package failed model-free replay")
    temporary.replace(target)
    return target, digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("capture", "replay"))
    parser.add_argument("--spec-sha256")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.mode == "capture":
        if not args.spec_sha256 or args.bundle or args.manifest_sha256:
            parser.error("capture requires only external spec SHA-256")
        from neural_continuity.m1_diagnostics.cuda_null_full_budget_capture import (
            capture_full_budget,
        )

        result = capture_full_budget(args.spec_sha256)
    else:
        if not args.bundle or not args.manifest_sha256 or args.spec_sha256:
            parser.error("replay requires only bundle and external manifest SHA-256")
        result = replay_budget_package(args.bundle, args.manifest_sha256)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result.get("replay_status", "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
