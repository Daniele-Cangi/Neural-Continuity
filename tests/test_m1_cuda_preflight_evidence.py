import json
from pathlib import Path

from neural_continuity.m1_diagnostics.cuda_preflight_evidence import (
    replay_cuda_preflight,
    write_cuda_preflight_package,
)


def _authority() -> dict[str, object]:
    return {
        "authority_sha256": "authority",
        "provider_order": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "run_layout": [
            {"label": "batch_1_primary", "batch_size": 1},
            {"label": "batch_16_primary", "batch_size": 16},
            {"label": "batch_16_repeat", "batch_size": 16},
            {"label": "batch_64_primary", "batch_size": 64},
        ],
        "document_count": 256,
        "query_count": 81,
        "qualifying_m1_evidence": False,
        "full_corpus_authorized": False,
        "scientific_decision": "NOT_EVALUATED",
    }


def _runtime_result() -> dict[str, dict[str, object]]:
    layout = _authority()["run_layout"]
    runs = [dict(run) for run in layout]  # type: ignore[arg-type]
    profile = {
        "provider_event_counts": {
            "CUDAExecutionProvider": 10,
            "CPUExecutionProvider": 2,
        },
        "operator_event_counts": {
            "CUDAExecutionProvider": {"MatMul": 10},
            "CPUExecutionProvider": {"Shape": 2},
        },
        "cpu_fallback_operator_types": ["Shape"],
        "unclassified_cpu_events": 0,
        "undeclared_providers": [],
    }
    return {
        "runtime_inventory": {
            "declared_provider_order": [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        },
        "provider_assignment": {
            "source_fp32": profile,
            "candidate_int8_qdq": profile,
        },
        "benchmark": {
            "document_count": 256,
            "query_count": 81,
            "raw_embeddings_retained": False,
            "scientific_comparison_performed": False,
            "source_fp32": runs,
            "candidate_int8_qdq": runs,
        },
    }


def test_model_free_replay_matches_capture(tmp_path: Path) -> None:
    package, _manifest_hash = write_cuda_preflight_package(
        _authority(), _runtime_result(), tmp_path / "package"
    )
    result = replay_cuda_preflight(package / "replay-bundle.json")
    assert result["replay_status"] == "PASS"
    assert result["model_loaded"] is False
    assert result["onnx_graph_loaded"] is False


def test_replay_fails_closed_on_tampering(tmp_path: Path) -> None:
    package, _manifest_hash = write_cuda_preflight_package(
        _authority(), _runtime_result(), tmp_path / "package"
    )
    benchmark = package / "benchmark-summary.json"
    payload = json.loads(benchmark.read_text(encoding="utf-8"))
    payload["document_count"] = 255
    benchmark.write_text(json.dumps(payload), encoding="utf-8")
    result = replay_cuda_preflight(package / "replay-bundle.json")
    assert result["replay_status"] == "BLOCKED"
    assert result["artifact_integrity"] is False
