"""Final model-free package for the qualifying CUDA full corpus."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from neural_continuity.evidence import canonical_json_bytes, sha256_file
from neural_continuity.m1_diagnostics.cuda_null_full_comparison_replay import (
    recompute_full_comparison,
)
from neural_continuity.m1_diagnostics.cuda_null_full_epoch_package import (
    replay_full_epoch_package,
)
from neural_continuity.m1_diagnostics.cuda_null_full_extrema import (
    aggregate_family_extrema,
    batch_epoch_unit,
    single_comparison_unit,
)
from neural_continuity.m1_diagnostics.cuda_null_paths import (
    has_linked_ancestor,
    snapshot_file_inventory,
)

_VERSION = "1.0.0"
_EPOCHS = 120
_FILES = {"corpus-summary.json", "replay-bundle.json", "artifact-manifest.json"}
_MAX_JSON_BYTES = 64 * 1024 * 1024


class FullCorpusPackageBlocked(ValueError):
    """The complete full-corpus package did not replay exactly."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise FullCorpusPackageBlocked(reason)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json(path: Path) -> Any:
    try:
        _require(path.stat().st_size <= _MAX_JSON_BYTES, f"{path.name} exceeds size limit")
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FullCorpusPackageBlocked(f"cannot decode {path.name}") from exc
    _require(raw == canonical_json_bytes(value) + b"\n", f"{path.name} is not canonical")
    return value


def _record(records: Any, run_label: str, role: str) -> dict[str, Any]:
    _require(isinstance(records, list), "epoch run records differ")
    matches = [
        item
        for item in records
        if isinstance(item, dict)
        and item.get("run_label") == run_label
        and item.get("role") == role
    ]
    _require(len(matches) == 1, "epoch batch-16 record is missing or duplicated")
    return matches[0]


def _recompute(
    root: Path,
    authority_sha256: str,
    epoch_manifests: list[dict[str, Any]],
) -> dict[str, Any]:
    _require(len(epoch_manifests) == _EPOCHS, "epoch manifest coverage incomplete")
    repeated_units: list[dict[str, Any]] = []
    batch_units: list[dict[str, Any]] = []
    restart_units: list[dict[str, Any]] = []
    prior: dict[str, Any] | None = None
    canonical_identity: tuple[Any, Any, Any] | None = None

    for epoch, declaration in enumerate(epoch_manifests, start=1):
        _require(
            isinstance(declaration, dict)
            and declaration.get("epoch") == epoch
            and _is_sha256(declaration.get("manifest_sha256")),
            "epoch manifest declaration differs",
        )
        directory = root / f"epoch-{epoch:04d}"
        replay = replay_full_epoch_package(
            directory / "replay-bundle.json",
            declaration["manifest_sha256"],
            authority_sha256,
        )
        _require(
            replay.get("replay_status") == "PASS" and replay.get("epoch_number") == epoch,
            f"epoch {epoch} replay blocked",
        )
        plan = _read_json(directory / "epoch-plan.json")
        comparisons = _read_json(directory / "within-epoch-comparisons.json")
        records = _read_json(directory / "run-records.json")
        rankings = _read_json(directory / "rankings.json")
        metrics = _read_json(directory / "metrics.json")
        _require(
            isinstance(plan, dict)
            and isinstance(comparisons, list)
            and len(comparisons) == 4
            and all(isinstance(item, dict) for item in comparisons)
            and isinstance(rankings, dict)
            and isinstance(metrics, dict),
            f"epoch {epoch} aggregate inputs differ",
        )
        identity = (plan.get("document_ids"), plan.get("query_ids"), plan.get("qrels"))
        if canonical_identity is None:
            canonical_identity = identity
        _require(identity == canonical_identity, "cross-epoch dataset identity differs")
        repeated_units.append(
            single_comparison_unit(
                "repeated_inference",
                epoch,
                comparisons[0]["comparison"],
            )
        )
        batch_units.append(
            batch_epoch_unit(
                epoch,
                [item["comparison"] for item in comparisons[1:]],
            )
        )
        current = {
            "directory": directory,
            "document_ids": plan["document_ids"],
            "query_ids": plan["query_ids"],
            "qrels": plan["qrels"],
            "document_record": _record(records, "batch_16_primary", "documents"),
            "query_record": _record(records, "batch_16_primary", "measurement_null_queries"),
            "rankings": rankings["batch_16_primary"],
            "metrics": metrics["batch_16_primary"],
        }
        if epoch % 2 == 0:
            if prior is None:
                raise FullCorpusPackageBlocked("restart pair predecessor missing")
            previous = prior
            comparison = recompute_full_comparison(
                previous["directory"],
                current["directory"],
                left_run_label="batch_16_primary",
                right_run_label="batch_16_primary",
                document_ids=current["document_ids"],
                query_ids=current["query_ids"],
                qrels=current["qrels"],
                left_document_record=previous["document_record"],
                left_query_record=previous["query_record"],
                right_document_record=current["document_record"],
                right_query_record=current["query_record"],
                left_rankings=previous["rankings"],
                left_metrics=previous["metrics"],
                right_rankings=current["rankings"],
                right_metrics=current["metrics"],
            )
            restart_units.append(
                single_comparison_unit(
                    "process_restart_variation",
                    epoch // 2,
                    comparison,
                )
            )
            prior = None
        else:
            prior = current

    _require(prior is None, "restart pair coverage incomplete")
    units = {
        "repeated_inference": repeated_units,
        "batch_size_variation": batch_units,
        "process_restart_variation": restart_units,
    }
    return {
        "units": units,
        "family_extrema": {
            family: aggregate_family_extrema(family, values) for family, values in units.items()
        },
    }


def replay_full_corpus_package(
    bundle: Path,
    external_manifest_sha256: str,
    authority_sha256: str,
    final_checkpoint_tip_sha256: str,
) -> dict[str, Any]:
    """Recompute all 300 units and three extrema families without a model."""
    try:
        _require(
            all(
                _is_sha256(value)
                for value in (
                    external_manifest_sha256,
                    authority_sha256,
                    final_checkpoint_tip_sha256,
                )
            ),
            "full-corpus external hash malformed",
        )
        bundle = Path(bundle).absolute()
        _require(
            bundle.name == "replay-bundle.json" and not has_linked_ancestor(bundle),
            "full-corpus replay path invalid",
        )
        package = bundle.parent
        root = package.parent
        _require(snapshot_file_inventory(package) == _FILES, "full-corpus inventory differs")
        manifest_path = package / "artifact-manifest.json"
        _require(
            sha256_file(manifest_path) == external_manifest_sha256,
            "full-corpus external manifest differs",
        )
        manifest = _read_json(manifest_path)
        names = sorted(_FILES - {"artifact-manifest.json"})
        _require(
            isinstance(manifest, dict)
            and manifest.get("kind") == "m1_cuda_null_full_corpus_manifest"
            and manifest.get("version") == _VERSION
            and manifest.get("artifacts")
            == [
                {
                    "path": name,
                    "size_bytes": (package / name).stat().st_size,
                    "sha256": sha256_file(package / name),
                }
                for name in names
            ],
            "full-corpus manifest schema differs",
        )
        declaration = _read_json(bundle)
        _require(
            declaration
            == {
                "kind": "m1_cuda_null_full_corpus_replay",
                "version": _VERSION,
                "authority_sha256": authority_sha256,
                "final_checkpoint_tip_sha256": final_checkpoint_tip_sha256,
                "model_required_for_replay": False,
            },
            "full-corpus replay declaration differs",
        )
        summary = _read_json(package / "corpus-summary.json")
        _require(
            isinstance(summary, dict)
            and summary.get("kind") == "m1_cuda_null_full_corpus_summary"
            and summary.get("version") == _VERSION
            and summary.get("status") == "CAPTURED_NOT_DECIDED"
            and summary.get("authority_sha256") == authority_sha256
            and summary.get("final_checkpoint_tip_sha256") == final_checkpoint_tip_sha256
            and summary.get("epoch_count") == _EPOCHS
            and summary.get("scientific_decision") == "NOT_EVALUATED",
            "full-corpus summary scope differs",
        )
        recomputed = _recompute(root, authority_sha256, summary["epoch_manifests"])
        _require(
            summary.get("units") == recomputed["units"]
            and summary.get("family_extrema") == recomputed["family_extrema"],
            "full-corpus recomputation differs",
        )
        return {
            "replay_status": "PASS",
            "status": "CAPTURED_NOT_DECIDED",
            "artifact_manifest_sha256": external_manifest_sha256,
            "epoch_count": _EPOCHS,
            "family_unit_counts": {
                family: len(units) for family, units in recomputed["units"].items()
            },
            "model_loaded": False,
            "scientific_decision": "NOT_EVALUATED",
        }
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, OverflowError) as exc:
        return {
            "replay_status": "BLOCKED",
            "model_loaded": False,
            "scientific_decision": "NOT_EVALUATED",
            "reason": str(exc),
        }


def finalize_full_corpus_package(
    root: Path,
    authority_sha256: str,
    final_checkpoint_tip_sha256: str,
    epoch_manifests: list[dict[str, Any]],
) -> tuple[Path, str]:
    """Build, replay, and atomically publish the complete corpus package."""
    root = Path(root).absolute()
    output = root / "full-corpus-package"
    _require(not has_linked_ancestor(root), "full-corpus root path contains a link")
    _require(not output.exists(), "full-corpus output exists")
    staging = root / f".full-corpus-package.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        recomputed = _recompute(root, authority_sha256, epoch_manifests)
        summary = {
            "kind": "m1_cuda_null_full_corpus_summary",
            "version": _VERSION,
            "status": "CAPTURED_NOT_DECIDED",
            "authority_sha256": authority_sha256,
            "final_checkpoint_tip_sha256": final_checkpoint_tip_sha256,
            "epoch_count": _EPOCHS,
            "epoch_manifests": epoch_manifests,
            **recomputed,
            "source_only": True,
            "candidate_or_int8_executed": False,
            "holdout_accessed": False,
            "model_required_for_replay": False,
            "scientific_decision": "NOT_EVALUATED",
        }
        (staging / "corpus-summary.json").write_bytes(canonical_json_bytes(summary) + b"\n")
        replay = {
            "kind": "m1_cuda_null_full_corpus_replay",
            "version": _VERSION,
            "authority_sha256": authority_sha256,
            "final_checkpoint_tip_sha256": final_checkpoint_tip_sha256,
            "model_required_for_replay": False,
        }
        (staging / "replay-bundle.json").write_bytes(canonical_json_bytes(replay) + b"\n")
        names = sorted(snapshot_file_inventory(staging))
        manifest = {
            "kind": "m1_cuda_null_full_corpus_manifest",
            "version": _VERSION,
            "artifacts": [
                {
                    "path": name,
                    "size_bytes": (staging / name).stat().st_size,
                    "sha256": sha256_file(staging / name),
                }
                for name in names
            ],
        }
        (staging / "artifact-manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
        digest = sha256_file(staging / "artifact-manifest.json")
        replayed = replay_full_corpus_package(
            staging / "replay-bundle.json",
            digest,
            authority_sha256,
            final_checkpoint_tip_sha256,
        )
        _require(replayed.get("replay_status") == "PASS", "new full-corpus package blocked")
        staging.replace(output)
        return output, digest
    except Exception:
        if staging.exists() and staging.parent == root:
            shutil.rmtree(staging)
        raise
