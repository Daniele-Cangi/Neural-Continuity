"""Exact CUDA runtime identity gate without ONNX graph or session creation."""

from __future__ import annotations

import importlib
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from neural_continuity.evidence import sha256_file
from neural_continuity.m1_diagnostics.cuda_null_authority import (
    CUDA_NULL_CONFIG_PATH,
    CUDA_NULL_CONFIG_SHA256,
)
from neural_continuity.m1_diagnostics.cuda_null_paths import has_linked_ancestor
from neural_continuity.m1_diagnostics.cuda_null_runtime_preimport import (
    import_verified_ort,
    verify_ort_package,
)

FROZEN_ENVIRONMENT = Path(r"D:\neural-continuity-runtime-cuda-v1")
NVIDIA_SMI_PATH = Path(r"C:\Windows\System32\nvidia-smi.exe")
NVIDIA_SMI_SHA256 = "957be91368f4d6f7bcad8723801c293d513af229f223f13215914244a0fec989"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DISTRIBUTIONS = {
    "onnxruntime_gpu_version": ("onnxruntime-gpu", "onnxruntime"),
    "numpy_version": ("numpy", "numpy"),
    "sentence_transformers_version": ("sentence-transformers", "sentence_transformers"),
    "tokenizers_version": ("tokenizers", "tokenizers"),
    "torch_version": ("torch", "torch"),
    "transformers_version": ("transformers", "transformers"),
    "nvidia_cuda_runtime_version": ("nvidia-cuda-runtime", None),
    "nvidia_cuda_nvrtc_version": ("nvidia-cuda-nvrtc", None),
    "nvidia_cublas_version": ("nvidia-cublas", None),
    "nvidia_cufft_version": ("nvidia-cufft", None),
    "nvidia_curand_version": ("nvidia-curand", None),
    "nvidia_nvjitlink_version": ("nvidia-nvjitlink", None),
    "nvidia_cudnn_cu13_version": ("nvidia-cudnn-cu13", None),
}
_CUDA_DLL_DISTRIBUTIONS = {
    "cudart64_13.dll": "nvidia-cuda-runtime",
    "cublas64_13.dll": "nvidia-cublas",
    "cublasLt64_13.dll": "nvidia-cublas",
    "cufft64_12.dll": "nvidia-cufft",
}
_CUDNN_DLL_DISTRIBUTIONS = {
    name: "nvidia-cudnn-cu13"
    for name in (
        "cudnn64_9.dll",
        "cudnn_adv64_9.dll",
        "cudnn_heuristic64_9.dll",
        "cudnn_engines_precompiled64_9.dll",
        "cudnn_graph64_9.dll",
        "cudnn_ops64_9.dll",
        "cudnn_engines_runtime_compiled64_9.dll",
        "cudnn_engines_tensor_ir64_9.dll",
    )
}
_INSTALLED_ONLY_DISTRIBUTIONS = {
    "nvrtc64_130_0.dll": "nvidia-cuda-nvrtc",
    "curand64_10.dll": "nvidia-curand",
    "nvJitLink_130_0.dll": "nvidia-nvjitlink",
}
_ORT_BINARIES = {
    "onnxruntime_providers_shared.dll",
    "onnxruntime_providers_cuda.dll",
    "onnxruntime_pybind11_state.pyd",
}
_DLL_GROUPS = {
    "loaded_cuda_dll_sha256": _CUDA_DLL_DISTRIBUTIONS,
    "loaded_cudnn_dll_sha256": _CUDNN_DLL_DISTRIBUTIONS,
    "installed_not_observed_loaded_dll_sha256": _INSTALLED_ONLY_DISTRIBUTIONS,
    "onnxruntime_binary_sha256": {name: "onnxruntime-gpu" for name in _ORT_BINARIES},
}
_PROVIDER_POLICY = {
    "ordered_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "require_cuda_activity_per_run": True,
    "classify_cpu_fallback_by_operator_type": True,
    "reject_undeclared_providers": True,
    "reject_unclassified_cpu_events": True,
    "node_name_exceptions": False,
    "tensor_name_exceptions": False,
    "index_exceptions": False,
    "benchmark_specific_exceptions": False,
}
_PIN_KEYS = {
    "gpu_name",
    "gpu_uuid",
    "compute_capability",
    "driver_version",
    "python_version",
    *_DISTRIBUTIONS,
    *_DLL_GROUPS,
}


class CudaNullRuntimeBlocked(RuntimeError):
    """A frozen runtime identity is missing, changed, or not observable."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise CudaNullRuntimeBlocked(reason)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise CudaNullRuntimeBlocked(f"{label}: expected object")
    return value


def _file_from_distribution(distribution_name: str, basename: str, root: Path) -> Path:
    distribution = metadata.distribution(distribution_name)
    matches = [
        Path(str(distribution.locate_file(item)))
        for item in distribution.files or ()
        if item.name.casefold() == basename.casefold()
    ]
    _require(len(matches) == 1, f"{distribution_name}: missing or duplicate {basename}")
    _require(not has_linked_ancestor(matches[0]), f"{basename}: linked installed path")
    path = matches[0].resolve()
    _require(
        path.is_file() and path.is_relative_to(root), f"{basename}: outside frozen environment"
    )
    return path


def _load_runtime_pins(config_path: Path, external_config_sha256: str) -> Mapping[str, Any]:
    _require(
        _SHA256.fullmatch(external_config_sha256) is not None, "external config SHA-256 invalid"
    )
    _require(config_path.resolve() == CUDA_NULL_CONFIG_PATH.resolve(), "config path mismatch")
    _require(
        external_config_sha256 == CUDA_NULL_CONFIG_SHA256,
        "external config hash is not frozen",
    )
    _require(sha256_file(config_path) == CUDA_NULL_CONFIG_SHA256, "config SHA-256 mismatch")
    config = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "CUDA null config")
    _require(config.get("status") == "DRAFT_NOT_EXECUTABLE", "unexpected config status")
    freeze = _mapping(config.get("freeze_gate"), "freeze gate")
    _require(freeze.get("technical_runtime_restored") is True, "runtime not restored")
    for key in (
        "independent_review_complete",
        "fail_closed_replay_verified",
        "fresh_source_only_preflight_verified",
    ):
        _require(freeze.get(key) is False, f"freeze gate {key} changed")
    _require(freeze.get("external_config_sha256") is None, "self-asserted config hash")
    _require(config.get("provider_policy") == _PROVIDER_POLICY, "provider policy mismatch")
    pins = _mapping(config.get("runtime_identity"), "runtime identity")
    _require(set(pins) == _PIN_KEYS, "runtime pin field set mismatch")
    for key in _PIN_KEYS - set(_DLL_GROUPS):
        _require(isinstance(pins[key], str) and bool(pins[key]), f"runtime pin {key} invalid")
    for group, names in _DLL_GROUPS.items():
        hashes = _mapping(pins[group], group)
        _require(set(hashes) == set(names), f"{group}: DLL set mismatch")
        for name, value in hashes.items():
            _require(
                isinstance(value, str) and _SHA256.fullmatch(value) is not None,
                f"{group}: invalid hash for {name}",
            )
    return pins


def _gpu_inventory() -> dict[str, str]:
    _require(
        NVIDIA_SMI_PATH.is_file() and not has_linked_ancestor(NVIDIA_SMI_PATH),
        "frozen NVIDIA utility missing or linked",
    )
    _require(sha256_file(NVIDIA_SMI_PATH) == NVIDIA_SMI_SHA256, "NVIDIA utility hash mismatch")
    result = subprocess.run(
        [
            str(NVIDIA_SMI_PATH),
            "--query-gpu=name,uuid,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows = [
        [part.strip() for part in line.split(",")]
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    _require(all(len(row) == 4 and all(row) for row in rows), "GPU inventory format mismatch")
    _require(len(rows) == 1, "expected exactly one frozen GPU")
    name, uuid, driver, capability = rows[0]
    return {
        "gpu_name": name,
        "gpu_uuid": uuid,
        "driver_version": driver,
        "compute_capability": capability,
    }


def _distribution_inventory(root: Path) -> dict[str, dict[str, str]]:
    observed: dict[str, dict[str, str]] = {}
    for field, (distribution_name, module_name) in _DISTRIBUTIONS.items():
        installed = metadata.version(distribution_name)
        item = {"distribution_version": installed}
        if module_name is not None:
            module = importlib.import_module(module_name)
            module_version = getattr(module, "__version__", None)
            module_file = getattr(module, "__file__", None)
            if not isinstance(module_version, str):
                raise CudaNullRuntimeBlocked(f"{module_name}: no imported version")
            if not isinstance(module_file, str):
                raise CudaNullRuntimeBlocked(f"{module_name}: no import path")
            resolved = Path(module_file).resolve()
            _require(
                resolved.is_file() and resolved.is_relative_to(root),
                f"{module_name}: outside frozen environment",
            )
            item["imported_version"] = module_version
            item["imported_path"] = str(resolved)
        observed[field] = item
    return observed


def _mapped_dll_paths(names: set[str]) -> dict[str, Path]:
    import psutil

    found: dict[str, set[Path]] = {name: set() for name in names}
    lookup = {name.casefold(): name for name in names}
    for mapping in psutil.Process().memory_maps():
        path = Path(mapping.path)
        name = lookup.get(path.name.casefold())
        if name is not None:
            found[name].add(path.resolve())
    for name, paths in found.items():
        _require(len(paths) <= 1, f"{name}: multiple mapped paths")
    return {name: next(iter(paths)) for name, paths in found.items() if paths}


def _installed_dlls(root: Path, pins: Mapping[str, Any]) -> dict[str, tuple[str, Path, str]]:
    installed: dict[str, tuple[str, Path, str]] = {}
    for group, names in _DLL_GROUPS.items():
        expected_hashes = _mapping(pins[group], group)
        for name, distribution_name in names.items():
            installed_path = _file_from_distribution(distribution_name, name, root)
            digest = sha256_file(installed_path)
            _require(digest == expected_hashes[name], f"{name}: installed hash mismatch")
            installed[name] = (distribution_name, installed_path, digest)
    return installed


def _dll_inventory(
    root: Path, installed: Mapping[str, tuple[str, Path, str]]
) -> dict[str, dict[str, dict[str, str | bool]]]:
    mapped = _mapped_dll_paths(set(installed))
    observed: dict[str, dict[str, dict[str, str | bool]]] = {}
    for group, names in _DLL_GROUPS.items():
        group_observed: dict[str, dict[str, str | bool]] = {}
        for name in names:
            distribution_name, installed_path, digest = installed[name]
            item: dict[str, str | bool] = {
                "distribution": distribution_name,
                "installed_path": str(installed_path),
                "sha256": digest,
            }
            mapped_path = mapped.get(name)
            if mapped_path is not None:
                _require(
                    mapped_path.is_relative_to(root), f"{name}: mapped outside frozen environment"
                )
                _require(
                    mapped_path == installed_path, f"{name}: mapped DLL differs from installed file"
                )
                item["mapped_path"] = str(mapped_path)
                item["mapped_sha256"] = sha256_file(mapped_path)
            item["observed_loaded"] = mapped_path is not None
            group_observed[name] = item
        observed[group] = group_observed
    return observed


def _compare_runtime(pins: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    gpu = _mapping(observed.get("gpu"), "observed GPU")
    for key in ("gpu_name", "gpu_uuid", "driver_version", "compute_capability"):
        _require(gpu.get(key) == pins[key], f"GPU {key} mismatch")
    _require(observed.get("python_version") == pins["python_version"], "Python version mismatch")
    software = _mapping(observed.get("software"), "software inventory")
    _require(set(software) == set(_DISTRIBUTIONS), "software inventory incomplete")
    for field, (_, module_name) in _DISTRIBUTIONS.items():
        item = _mapping(software[field], field)
        _require(item.get("distribution_version") == pins[field], f"{field}: distribution mismatch")
        if module_name is not None:
            _require(
                item.get("imported_version") == pins[field], f"{field}: imported version mismatch"
            )
    dlls = _mapping(observed.get("dlls"), "DLL inventory")
    _require(set(dlls) == set(_DLL_GROUPS), "DLL inventory group mismatch")
    for group, names in _DLL_GROUPS.items():
        items = _mapping(dlls[group], group)
        _require(set(items) == set(names), f"{group}: DLL inventory incomplete")
        expected_hashes = _mapping(pins[group], group)
        for name, distribution_name in names.items():
            item = _mapping(items[name], name)
            _require(
                item.get("distribution") == distribution_name, f"{name}: distribution mismatch"
            )
            _require(
                item.get("sha256") == expected_hashes[name], f"{name}: installed hash mismatch"
            )
            loaded = item.get("observed_loaded")
            _require(type(loaded) is bool, f"{name}: loaded state missing")
            if loaded:
                _require(
                    item.get("mapped_path") == item.get("installed_path"),
                    f"{name}: mapped path mismatch",
                )
                _require(
                    item.get("mapped_sha256") == expected_hashes[name],
                    f"{name}: mapped hash mismatch",
                )
            else:
                _require(
                    "mapped_path" not in item and "mapped_sha256" not in item,
                    f"{name}: unmapped DLL has mapping evidence",
                )
            if group in ("loaded_cuda_dll_sha256", "loaded_cudnn_dll_sha256"):
                _require(loaded is True, f"{name}: not observed loaded")
            elif group == "installed_not_observed_loaded_dll_sha256":
                _require(loaded is False, f"{name}: unexpectedly loaded")
            elif name == "onnxruntime_pybind11_state.pyd":
                _require(loaded is True, f"{name}: imported binary not mapped")
    providers = observed.get("available_providers")
    if not isinstance(providers, list):
        raise CudaNullRuntimeBlocked("available providers missing")
    # Available providers are capabilities, not the providers selected by a session.
    _require(
        {"CUDAExecutionProvider", "CPUExecutionProvider"}.issubset(set(providers)),
        "CUDA/CPU provider unavailable",
    )
    for key in ("onnx_graph_loaded", "session_created", "execution_authorized"):
        _require(observed.get(key) is False, f"runtime inventory cannot assert {key}")


def verify_runtime_identity(config_path: Path, external_config_sha256: str) -> dict[str, Any]:
    """Verify the frozen environment, without loading an ONNX graph."""
    try:
        pins = _load_runtime_pins(config_path, external_config_sha256)
        root = FROZEN_ENVIRONMENT.resolve()
        _require(Path(sys.prefix).resolve() == root, "wrong Python environment")
        verify_ort_package(
            root,
            str(pins["onnxruntime_gpu_version"]),
            _mapping(pins["onnxruntime_binary_sha256"], "ONNX Runtime binary pins"),
        )
        installed = _installed_dlls(root, pins)
        with import_verified_ort(installed) as ort:
            observed: dict[str, Any] = {
                "kind": "m1_cuda_null_runtime_identity",
                "version": "1.0.0",
                "status": "RUNTIME_IDENTITY_VERIFIED_EXECUTION_BLOCKED",
                "config_sha256": external_config_sha256,
                "environment_root": str(root),
                "python_version": platform.python_version(),
                "gpu": _gpu_inventory(),
                "software": _distribution_inventory(root),
                "dlls": _dll_inventory(root, installed),
                "available_providers": ort.get_available_providers(),
                "onnx_graph_loaded": False,
                "session_created": False,
                "execution_authorized": False,
            }
            _compare_runtime(pins, observed)
            return observed
    except CudaNullRuntimeBlocked:
        raise
    except Exception as exc:
        raise CudaNullRuntimeBlocked(f"runtime identity could not be verified: {exc}") from exc
