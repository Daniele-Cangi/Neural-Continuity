"""Fail-closed comparison tests for the no-session runtime gate."""

from __future__ import annotations

from copy import deepcopy

import pytest

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
            item = {
                "distribution": distribution,
                "installed_path": f"D:/env/{name}",
                "sha256": digest,
                "observed_loaded": group.startswith("loaded_"),
            }
            if group.startswith("loaded_"):
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
    ["wrong_gpu", "missing_loaded_dll", "shadowed_dll", "wrong_imported_ort", "no_cuda"],
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
    else:
        observed["available_providers"] = ["CPUExecutionProvider"]
    with pytest.raises(CudaNullRuntimeBlocked):
        _compare_runtime(pins, observed)
