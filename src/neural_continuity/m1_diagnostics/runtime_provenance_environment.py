from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from neural_continuity.evidence import sha256_file
from neural_continuity.m1_diagnostics.runtime_provenance_authority import RuntimeProvenanceError

DISTRIBUTIONS = (
    "numpy",
    "onnx",
    "onnxruntime",
    "sentence-transformers",
    "tokenizers",
    "torch",
    "transformers",
)
NUMERIC_ENVIRONMENT_KEYS = (
    "KMP_AFFINITY",
    "KMP_DUPLICATE_LIB_OK",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OMP_WAIT_POLICY",
    "OPENBLAS_NUM_THREADS",
    "ORT_NUM_THREADS",
)
HISTORICAL_RUNTIME_FIELDS = (
    "python_version",
    "numpy_version",
    "onnx_version",
    "onnxruntime_version",
    "onnxruntime_binary_sha256",
    "execution_provider",
    "provider_options",
    "numeric_environment",
    "cpu_identity",
    "torch_version",
    "sentence_transformers_version",
)


def _distribution_files(name: str) -> tuple[str, list[dict[str, Any]]]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeProvenanceError(
            "RUNTIME_PROVENANCE_DEPENDENCY_MISSING", f"required distribution is missing: {name}"
        ) from exc
    records: list[dict[str, Any]] = []
    for item in distribution.files or ():
        relative = item.as_posix()
        metadata_file = Path(relative).name in {"METADATA", "RECORD", "WHEEL"}
        native_file = name == "onnxruntime" and relative.lower().endswith((".dll", ".pyd"))
        if not metadata_file and not native_file:
            continue
        path = Path(str(distribution.locate_file(item))).resolve()
        if not path.is_file():
            raise RuntimeProvenanceError(
                "RUNTIME_PROVENANCE_DISTRIBUTION_FILE_MISSING",
                f"declared distribution file is missing: {name}:{relative}",
            )
        records.append(
            {"path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        )
    if not records:
        raise RuntimeProvenanceError(
            "RUNTIME_PROVENANCE_DISTRIBUTION_INVENTORY_EMPTY",
            f"no authority files selected for distribution: {name}",
        )
    return distribution.version, sorted(records, key=lambda record: str(record["path"]))


def capture_runtime_inventory() -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise RuntimeProvenanceError(
            "RUNTIME_PROVENANCE_DEPENDENCY_MISSING",
            "onnxruntime is required for provenance inventory",
        ) from exc
    distributions: dict[str, Any] = {}
    for name in DISTRIBUTIONS:
        version, files = _distribution_files(name)
        distributions[name] = {"version": version, "authority_files": files}
    executable = Path(sys.executable).resolve()
    return {
        "kind": "m1-runtime-provenance-inventory",
        "version": "1.0.0",
        "model_execution_used": False,
        "onnx_graph_loaded": False,
        "activation_read": False,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_name": executable.name,
            "executable_sha256": sha256_file(executable),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cpu_identity": {
            "processor_architecture": os.environ.get("PROCESSOR_ARCHITECTURE"),
            "processor_identifier": os.environ.get("PROCESSOR_IDENTIFIER"),
            "processor_level": os.environ.get("PROCESSOR_LEVEL"),
            "processor_revision": os.environ.get("PROCESSOR_REVISION"),
            "number_of_processors": os.environ.get("NUMBER_OF_PROCESSORS"),
        },
        "numeric_environment": {key: os.environ.get(key) for key in NUMERIC_ENVIRONMENT_KEYS},
        "onnxruntime": {
            "version": ort.__version__,
            "available_providers": list(ort.get_available_providers()),
            "build_info": ort.get_build_info(),
            "session_created": False,
        },
        "distributions": distributions,
    }


def _observed_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        keys = {str(key) for key in value}
        for child in value.values():
            keys.update(_observed_keys(child))
        return keys
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        sequence_keys: set[str] = set()
        for child in value:
            sequence_keys.update(_observed_keys(child))
        return sequence_keys
    return set()


def historical_runtime_coverage(baseline_manifests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observed: set[str] = set()
    for manifest in baseline_manifests:
        observed.update(_observed_keys(manifest))
    present = [field for field in HISTORICAL_RUNTIME_FIELDS if field in observed]
    missing = [field for field in HISTORICAL_RUNTIME_FIELDS if field not in observed]
    return {
        "required_fields": list(HISTORICAL_RUNTIME_FIELDS),
        "present_fields": present,
        "missing_fields": missing,
        "complete": not missing,
    }
