"""Bounded runtime probe for the declared M1 hybrid CUDA authority."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from neural_continuity.m1_b.onnx_source import encode_onnx_source
from neural_continuity.m1_diagnostics.cuda_preflight_authority import (
    CudaPreflightAuthority,
    CudaPreflightBlocked,
)
from neural_continuity.m1_teacher_evidence import _load_teacher


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise CudaPreflightBlocked(f"required distribution is missing: {name}") from exc


def _gpu_inventory() -> dict[str, str]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,compute_cap,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CudaPreflightBlocked(f"GPU inventory failed: {exc}") from exc

    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise CudaPreflightBlocked(
            f"exactly one GPU is required; nvidia-smi returned {len(rows)} rows"
        )
    values = [part.strip() for part in rows[0].split(",")]
    if len(values) != 5:
        raise CudaPreflightBlocked("unexpected nvidia-smi inventory shape")
    return {
        "name": values[0],
        "uuid": values[1],
        "driver_version": values[2],
        "compute_capability": values[3],
        "memory_total_mib": values[4],
    }


def capture_runtime_inventory(authority: CudaPreflightAuthority) -> dict[str, Any]:
    """Verify runtime identity before loading either ONNX graph."""

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise CudaPreflightBlocked("onnxruntime-gpu is unavailable") from exc

    try:
        ort.preload_dlls()
    except Exception as exc:
        raise CudaPreflightBlocked(f"CUDA DLL preload failed: {exc}") from exc

    gpu = _gpu_inventory()
    for key, expected in authority.expected_gpu.items():
        if gpu.get(key) != expected:
            raise CudaPreflightBlocked(
                f"GPU {key} mismatch: expected {expected!r}, observed {gpu.get(key)!r}"
            )

    ort_version = _distribution_version("onnxruntime-gpu")
    module_version = getattr(ort, "__version__", None)
    if (
        ort_version != authority.expected_onnxruntime_version
        or module_version != authority.expected_onnxruntime_version
    ):
        raise CudaPreflightBlocked(
            "onnxruntime-gpu version mismatch: "
            f"expected {authority.expected_onnxruntime_version}, "
            f"distribution {ort_version}, imported module {module_version}"
        )
    available = ort.get_available_providers()
    missing = [name for name in authority.provider_order if name not in available]
    if missing:
        raise CudaPreflightBlocked(f"declared providers are unavailable: {missing}")

    return {
        "version": 1,
        "kind": "m1_cuda_hybrid_runtime_inventory",
        "gpu": gpu,
        "onnxruntime_gpu_version": ort_version,
        "onnxruntime_device": ort.get_device(),
        "cuda_dlls_preloaded": True,
        "available_providers": available,
        "declared_provider_order": list(authority.provider_order),
        "package_versions": {
            name: _distribution_version(name)
            for name in (
                "numpy",
                "onnxruntime-gpu",
                "sentence-transformers",
                "tokenizers",
                "torch",
                "transformers",
            )
        },
    }


def _array_record(array: np.ndarray) -> dict[str, Any]:
    normalized = np.asarray(array, dtype="<f4", order="C")
    return {
        "shape": list(normalized.shape),
        "dtype": "float32",
        "sha256": hashlib.sha256(normalized.tobytes(order="C")).hexdigest(),
    }


def _run_model(
    teacher: Any,
    session: Any,
    document_texts: Sequence[str],
    query_texts: Sequence[str],
    run_layout: tuple[tuple[str, int], ...],
    model_label: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run_label, batch_size in run_layout:
        started = time.perf_counter()
        documents = encode_onnx_source(
            teacher,
            session,
            document_texts,
            batch_size=batch_size,
            label=f"{model_label}.{run_label}.documents",
        )
        queries = encode_onnx_source(
            teacher,
            session,
            query_texts,
            batch_size=batch_size,
            label=f"{model_label}.{run_label}.queries",
        )
        elapsed = time.perf_counter() - started
        records.append(
            {
                "label": run_label,
                "batch_size": batch_size,
                "elapsed_seconds": elapsed,
                "item_count": len(document_texts) + len(query_texts),
                "items_per_second": ((len(document_texts) + len(query_texts)) / elapsed),
                "documents": _array_record(documents),
                "queries": _array_record(queries),
            }
        )
    return records


def _profile_summary(profile_path: Path, declared: tuple[str, ...]) -> dict[str, Any]:
    try:
        events = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CudaPreflightBlocked(f"invalid ONNX Runtime profile: {profile_path}") from exc
    finally:
        profile_path.unlink(missing_ok=True)

    if not isinstance(events, list):
        raise CudaPreflightBlocked("ONNX Runtime profile root must be a list")

    provider_counts: dict[str, int] = {}
    operator_counts: dict[str, dict[str, int]] = {}
    unclassified_cpu_events = 0
    for event in events:
        if not isinstance(event, dict) or event.get("cat") != "Node":
            continue
        args = event.get("args")
        if not isinstance(args, dict):
            raise CudaPreflightBlocked("ONNX profile Node event lacks arguments")
        provider = args.get("provider")
        if not isinstance(provider, str) or not provider:
            raise CudaPreflightBlocked("ONNX profile Node event lacks a provider")
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
        op_name = args.get("op_name")
        if not isinstance(op_name, str) or not op_name:
            raise CudaPreflightBlocked("ONNX profile Node event lacks an operator type")
        by_operator = operator_counts.setdefault(provider, {})
        by_operator[op_name] = by_operator.get(op_name, 0) + 1

    undeclared = sorted(set(provider_counts) - set(declared))
    if undeclared:
        raise CudaPreflightBlocked(f"profile contains undeclared providers: {undeclared}")
    if provider_counts.get("CUDAExecutionProvider", 0) < 1:
        raise CudaPreflightBlocked("profile contains no CUDA execution event")
    if unclassified_cpu_events:
        raise CudaPreflightBlocked(
            f"{unclassified_cpu_events} CPU fallback events lack an operator type"
        )

    return {
        "provider_event_counts": dict(sorted(provider_counts.items())),
        "operator_event_counts": {
            provider: dict(sorted(counts.items()))
            for provider, counts in sorted(operator_counts.items())
        },
        "cpu_fallback_operator_types": sorted(operator_counts.get("CPUExecutionProvider", {})),
        "unclassified_cpu_events": unclassified_cpu_events,
        "undeclared_providers": undeclared,
    }


def _profiled_session(model_path: Path, profile_prefix: Path, providers: Sequence[str]) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(profile_prefix)
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=list(providers),
    )
    observed_providers = session.get_providers()
    if observed_providers != list(providers):
        profile_path = Path(session.end_profiling())
        profile_path.unlink(missing_ok=True)
        del session
        raise CudaPreflightBlocked(
            "session provider order differs from the declared hybrid authority: "
            f"{observed_providers}"
        )
    return session


def run_cuda_preflight(
    authority: CudaPreflightAuthority, working_directory: str | Path
) -> dict[str, Any]:
    """Run the bounded real-data preflight after complete static verification."""

    working = Path(working_directory).resolve()
    working.mkdir(parents=True, exist_ok=True)
    runtime_inventory = capture_runtime_inventory(authority)

    teacher, _teacher_identity = _load_teacher(authority.base.config)
    source_session = _profiled_session(
        authority.base.source.artifact_path,
        working / "source-profile",
        authority.provider_order,
    )
    candidate_session = _profiled_session(
        authority.candidate.artifact_path,
        working / "candidate-profile",
        authority.provider_order,
    )

    source_runs = _run_model(
        teacher,
        source_session,
        authority.base.selected_document_texts,
        authority.base.query_texts,
        authority.run_layout,
        "source_fp32",
    )
    candidate_runs = _run_model(
        teacher,
        candidate_session,
        authority.base.selected_document_texts,
        authority.base.query_texts,
        authority.run_layout,
        "candidate_int8_qdq",
    )
    source_profile = Path(source_session.end_profiling())
    candidate_profile = Path(candidate_session.end_profiling())

    provider_assignment = {
        "version": 1,
        "kind": "m1_cuda_hybrid_provider_assignment",
        "policy": {
            "provider_order": list(authority.provider_order),
            "classification_unit": "operator_type",
            "node_names_retained": False,
            "tensor_names_retained": False,
            "benchmark_specific_exceptions": False,
        },
        "source_fp32": _profile_summary(source_profile, authority.provider_order),
        "candidate_int8_qdq": _profile_summary(candidate_profile, authority.provider_order),
    }
    benchmark = {
        "version": 1,
        "kind": "m1_cuda_hybrid_bounded_benchmark",
        "document_count": len(authority.base.selected_document_ids),
        "query_count": len(authority.base.query_ids),
        "raw_embeddings_retained": False,
        "scientific_comparison_performed": False,
        "source_fp32": source_runs,
        "candidate_int8_qdq": candidate_runs,
    }
    return {
        "runtime_inventory": runtime_inventory,
        "provider_assignment": provider_assignment,
        "benchmark": benchmark,
    }
