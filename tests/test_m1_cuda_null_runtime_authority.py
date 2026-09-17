"""Fail-closed comparison tests for the no-session runtime gate."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from neural_continuity.evidence import sha256_file
from neural_continuity.m1_diagnostics import cuda_null_runtime_authority as runtime_module
from neural_continuity.m1_diagnostics.cuda_null_runtime_authority import (
    _DISTRIBUTIONS,
    _DLL_GROUPS,
    CudaNullRuntimeBlocked,
    _compare_runtime,
)


def _matching_records() -> tuple[dict[str, object], dict[str, object]]:
    digest = "a" * 64
    pins: dict[str, object] = {
        "gpu_name": "GPU",
        "gpu_uuid": "GPU-test",
        "driver_version": "driver",
        "compute_capability": "7.5",
        "python_version": "3.12.10",
    }
    software = {}
    for field, (_, module_name) in _DISTRIBUTIONS.items():
        pins[field] = "version"
        item = {"distribution_version": "version"}
        if module_name is not None:
            item["imported_version"] = "version"
        software[field] = item
    dlls = {}
    for group, names in _DLL_GROUPS.items():
        pins[group] = {name: digest for name in names}
        dlls[group] = {}
        for name, distribution in names.items():
            is_loaded = group.startswith("loaded_") or name == "onnxruntime_pybind11_state.pyd"
            item = {
                "distribution": distribution,
                "installed_path": f"D:/env/{name}",
                "sha256": digest,
                "observed_loaded": is_loaded,
            }
            if is_loaded:
                item["mapped_path"] = f"D:/env/{name}"
                item["mapped_sha256"] = digest
            dlls[group][name] = item
    observed: dict[str, object] = {
        "gpu": {
            "gpu_name": "GPU",
            "gpu_uuid": "GPU-test",
            "driver_version": "driver",
            "compute_capability": "7.5",
        },
        "python_version": "3.12.10",
        "software": software,
        "dlls": dlls,
        "available_providers": [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
        "onnx_graph_loaded": False,
        "session_created": False,
        "execution_authorized": False,
    }
    return pins, observed


def test_matching_runtime_remains_non_executable() -> None:
    pins, observed = _matching_records()
    _compare_runtime(pins, observed)
    assert observed["session_created"] is False


@pytest.mark.parametrize(
    "change",
    [
        "wrong_gpu",
        "missing_loaded_dll",
        "shadowed_dll",
        "wrong_imported_ort",
        "no_cuda",
        "installed_only_loaded",
        "shadowed_ort_binary",
        "missing_ort_pybind",
        "wrong_ort_mapped_hash",
    ],
)
def test_runtime_mismatch_fails_closed(change: str) -> None:
    pins, observed = _matching_records()
    observed = deepcopy(observed)
    if change == "wrong_gpu":
        observed["gpu"]["gpu_uuid"] = "other"
    elif change == "missing_loaded_dll":
        observed["dlls"]["loaded_cuda_dll_sha256"]["cudart64_13.dll"]["observed_loaded"] = False
    elif change == "shadowed_dll":
        observed["dlls"]["loaded_cuda_dll_sha256"]["cudart64_13.dll"][
            "mapped_path"
        ] = "C:/Windows/cudart64_13.dll"
    elif change == "wrong_imported_ort":
        observed["software"]["onnxruntime_gpu_version"]["imported_version"] = "other"
    elif change == "installed_only_loaded":
        item = observed["dlls"]["installed_not_observed_loaded_dll_sha256"]["curand64_10.dll"]
        item["observed_loaded"] = True
        item["mapped_path"] = item["installed_path"]
        item["mapped_sha256"] = "a" * 64
    elif change == "shadowed_ort_binary":
        observed["dlls"]["onnxruntime_binary_sha256"]["onnxruntime_pybind11_state.pyd"][
            "mapped_path"
        ] = "C:/other/onnxruntime_pybind11_state.pyd"
    elif change == "missing_ort_pybind":
        item = observed["dlls"]["onnxruntime_binary_sha256"]["onnxruntime_pybind11_state.pyd"]
        item["observed_loaded"] = False
        del item["mapped_path"]
        del item["mapped_sha256"]
    elif change == "wrong_ort_mapped_hash":
        observed["dlls"]["onnxruntime_binary_sha256"]["onnxruntime_pybind11_state.pyd"][
            "mapped_sha256"
        ] = ("b" * 64)
    else:
        observed["available_providers"] = ["CPUExecutionProvider"]
    with pytest.raises(CudaNullRuntimeBlocked):
        _compare_runtime(pins, observed)


def test_runtime_config_rejects_resealed_identity_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = runtime_module.CUDA_NULL_CONFIG_PATH.read_text(encoding="utf-8")
    altered = original.replace('driver_version: "581.57"', 'driver_version: "other"')
    assert altered != original
    config_path = tmp_path / "m1-cuda-null-v1.yaml"
    config_path.write_text(altered, encoding="utf-8")
    monkeypatch.setattr(runtime_module, "CUDA_NULL_CONFIG_PATH", config_path)
    with pytest.raises(CudaNullRuntimeBlocked, match="not frozen"):
        runtime_module._load_runtime_pins(config_path, sha256_file(config_path))


def test_mapped_inventory_includes_optional_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapped_file = tmp_path / "curand64_10.dll"
    mapped_file.write_bytes(b"test")
    process = SimpleNamespace(memory_maps=lambda: [SimpleNamespace(path=str(mapped_file))])
    monkeypatch.setattr(psutil, "Process", lambda: process)
    assert runtime_module._mapped_dll_paths(
        {"curand64_10.dll", "onnxruntime_providers_cuda.dll"}
    ) == {"curand64_10.dll": mapped_file.resolve()}
