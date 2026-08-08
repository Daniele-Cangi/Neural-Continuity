from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.m1_b.paired_observation_compatibility import (
    verify_paired_observation_compatibility,
)
from neural_continuity.m1_onnx_transition import PairedRun, _comparison
from neural_continuity.m1_teacher_evidence import (
    TeacherEvidenceError,
    TeacherObservation,
    _fail,
    _load_json,
    _rank_and_measure,
    _safe_artifact_path,
)


def _observations(bundle_path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    bundle = Path(bundle_path).resolve()
    replay = _load_json(bundle, "COMPARISON_REPLAY_BUNDLE_INVALID")
    metadata = _load_json(
        _safe_artifact_path(bundle.parent, str(replay.get("metadata_path", ""))),
        "COMPARISON_METADATA_INVALID",
    )
    path = _safe_artifact_path(bundle.parent, str(replay.get("observation_path", "")))
    try:
        with np.load(path, allow_pickle=False) as archive:
            values = {key: archive[key] for key in archive.files}
    except Exception as exc:
        raise _fail("COMPARISON_OBSERVATION_INVALID", f"cannot load observations: {exc}") from exc
    return metadata, values


def _observation(
    values: dict[str, np.ndarray], metadata: dict[str, Any], index: int
) -> TeacherObservation:
    query_ids = values["query_ids"].astype(str).tolist()
    qrels = metadata.get("qrels")
    if not isinstance(qrels, dict) or set(qrels) != set(query_ids):
        raise _fail("COMPARISON_QRELS_INVALID", "qrels do not match canonical query IDs")
    return TeacherObservation(
        document_ids=values["document_ids"].astype(str).tolist(),
        document_embeddings=np.asarray(values["document_embeddings"][index], dtype=np.float32),
        query_ids=query_ids,
        query_embeddings=np.asarray(values["query_embeddings"][index], dtype=np.float32),
        query_roles=values["query_roles"].astype(str).tolist(),
        relevant_document_ids={query_id: list(qrels[query_id]) for query_id in query_ids},
    )


def compare_paired_observations(
    source_bundle: str | Path, target_bundle: str | Path, top_k: int = 10
) -> dict[str, Any]:
    compatibility = verify_paired_observation_compatibility(source_bundle, target_bundle)
    if compatibility["status"] != "PASS":
        raise _fail("PAIRED_OBSERVATION_INCOMPATIBLE", "compatibility gate did not pass")
    source_metadata, source = _observations(source_bundle)
    target_metadata, target = _observations(target_bundle)
    for key in ("document_embeddings", "query_embeddings", "run_ids", "batch_sizes"):
        if key not in source or key not in target or source[key].shape != target[key].shape:
            raise _fail("COMPARISON_OBSERVATION_INVALID", f"{key} is absent or incompatible")
    reports: list[dict[str, Any]] = []
    for index, run_id in enumerate(source["run_ids"].astype(str).tolist()):
        source_observation = _observation(source, source_metadata, index)
        target_observation = _observation(target, target_metadata, index)
        source_rankings, source_metrics = _rank_and_measure(source_observation, top_k)
        target_rankings, target_metrics = _rank_and_measure(target_observation, top_k)
        report = _comparison(
            PairedRun(
                run_id=run_id,
                batch_size=int(source["batch_sizes"][index]),
                source=source_observation,
                target=target_observation,
            ),
            source_rankings,
            target_rankings,
            source_metrics,
            target_metrics,
        )
        report["metrics"] = {"source": source_metrics, "target": target_metrics}
        reports.append(report)
    return {
        "status": "PASS",
        "comparison_state": "CAPTURED_NOT_DECIDED",
        "transition_b_decision": "NOT_EVALUATED",
        "dataset_id": source_metadata["dataset_id"],
        "top_k": top_k,
        "runs": reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare replay-verified M1 FP32 and INT8 observations."
    )
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--target-bundle", required=True)
    args = parser.parse_args(argv)
    try:
        result = compare_paired_observations(args.source_bundle, args.target_bundle)
    except TeacherEvidenceError as exc:
        print(json.dumps({"status": exc.status, "error": exc.__dict__}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
