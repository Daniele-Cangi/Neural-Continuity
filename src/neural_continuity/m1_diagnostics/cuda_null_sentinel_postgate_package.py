"""Tamper-evident package and model-free replay for the sentinel technical gate."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.cuda_null_sentinel_postgate import (
    SHA256,
    build_sentinel_postgate_report,
)
from neural_continuity.m1_diagnostics.cuda_null_source_preflight_evidence import (
    _external_output_directory,
)

FILES = ("technical-gate-report.json", "replay-bundle.json", "artifact-manifest.json")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)
    if not isinstance(value, dict):
        raise ValueError("gate artifact is not an object")
    return value


def replay_sentinel_postgate(bundle: Path, external_manifest_sha256: str) -> dict[str, Any]:
    """Verify package hashes, then recompute the exact gate without a model."""
    try:
        if SHA256.fullmatch(external_manifest_sha256) is None:
            raise ValueError("external gate manifest hash is invalid")
        bundle = Path(bundle).absolute()
        if bundle.name != "replay-bundle.json" or {p.name for p in bundle.parent.iterdir()} != set(
            FILES
        ):
            raise ValueError("gate artifact set differs")
        manifest = bundle.parent / "artifact-manifest.json"
        if sha256_file(manifest) != external_manifest_sha256:
            raise ValueError("external gate manifest hash mismatch")
        entries = _read_json(manifest)
        if (
            set(entries) != {"kind", "version", "artifacts"}
            or entries["kind"] != "m1_cuda_null_sentinel_postgate_manifest"
            or entries["version"] != "1.0.0"
            or not isinstance(entries["artifacts"], list)
            or len(entries["artifacts"]) != 2
        ):
            raise ValueError("gate manifest schema differs")
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
                raise ValueError(f"gate artifact integrity mismatch: {name}")
        record = _read_json(bundle)
        if (
            set(record)
            != {
                "kind",
                "version",
                "sentinel_root",
                "external_tip_sha256",
                "external_authority_sha256",
                "model_required_for_replay",
            }
            or record["kind"] != "m1_cuda_null_sentinel_postgate_replay"
            or record["version"] != "1.0.0"
            or record["model_required_for_replay"] is not False
        ):
            raise ValueError("gate replay declaration differs")
        recomputed = build_sentinel_postgate_report(
            root=Path(record["sentinel_root"]),
            external_tip_sha256=record["external_tip_sha256"],
            external_authority_sha256=record["external_authority_sha256"],
        )
        if _read_json(bundle.parent / "technical-gate-report.json") != recomputed:
            raise ValueError("gate report differs from model-free recomputation")
        return {
            "replay_status": "PASS",
            "status": recomputed["status"],
            "epoch_count": recomputed["epoch_count"],
            "artifact_manifest_sha256": external_manifest_sha256,
            "model_loaded": False,
            "full_corpus_execution_authorized": False,
        }
    except (OSError, ValueError, KeyError, TypeError, UnicodeError) as exc:
        return {"replay_status": "BLOCKED", "reason": str(exc), "model_loaded": False}


def write_sentinel_postgate(
    *, output: Path, root: Path, external_tip_sha256: str, external_authority_sha256: str
) -> tuple[Path, str]:
    report = build_sentinel_postgate_report(
        root=root,
        external_tip_sha256=external_tip_sha256,
        external_authority_sha256=external_authority_sha256,
    )
    target = _external_output_directory(output)
    temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    _write_json(temporary / FILES[0], report)
    _write_json(
        temporary / FILES[1],
        {
            "kind": "m1_cuda_null_sentinel_postgate_replay",
            "version": "1.0.0",
            "sentinel_root": str(root.absolute()),
            "external_tip_sha256": external_tip_sha256,
            "external_authority_sha256": external_authority_sha256,
            "model_required_for_replay": False,
        },
    )
    _write_json(
        temporary / FILES[2],
        {
            "kind": "m1_cuda_null_sentinel_postgate_manifest",
            "version": "1.0.0",
            "artifacts": [
                {"path": name, "sha256": sha256_file(temporary / name)} for name in FILES[:2]
            ],
        },
    )
    digest = sha256_file(temporary / FILES[2])
    replay = replay_sentinel_postgate(temporary / FILES[1], digest)
    if replay["replay_status"] != "PASS":
        raise ValueError("new sentinel gate package did not replay")
    temporary.replace(target)
    return target, digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("write", "replay"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--tip-sha256")
    parser.add_argument("--authority-sha256")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.mode == "write":
        if not all((args.output, args.root, args.tip_sha256, args.authority_sha256)):
            parser.error("write requires output, root, tip, and authority")
        output, digest = write_sentinel_postgate(
            output=args.output,
            root=args.root,
            external_tip_sha256=args.tip_sha256,
            external_authority_sha256=args.authority_sha256,
        )
        result = {"output": str(output), "artifact_manifest_sha256": digest}
    else:
        if not args.bundle or not args.manifest_sha256:
            parser.error("replay requires bundle and external manifest hash")
        result = replay_sentinel_postgate(args.bundle, args.manifest_sha256)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result.get("replay_status", "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
