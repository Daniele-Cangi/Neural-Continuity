"""Fail-closed tests for the CUDA-null ONNX Runtime pre-import boundary."""

from __future__ import annotations

import base64
import hashlib
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from neural_continuity.evidence import sha256_file
from neural_continuity.m1_diagnostics import cuda_null_paths as paths_module
from neural_continuity.m1_diagnostics import cuda_null_runtime_authority as runtime_module
from neural_continuity.m1_diagnostics import cuda_null_runtime_preimport as preimport


def _fake_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, external_script: bool = False
) -> tuple[Path, Path, dict[str, str]]:
    root = tmp_path / "venv"
    site = root / "Lib" / "site-packages"
    package = site / "onnxruntime"
    files = {
        "onnxruntime/__init__.py": b"safe init\n",
        "onnxruntime/capi/onnxruntime_pybind11_state.pyd": b"pyd",
        "onnxruntime/capi/onnxruntime_providers_shared.dll": b"shared",
        "onnxruntime/capi/onnxruntime_providers_cuda.dll": b"cuda",
    }
    records = []
    binary_pins = {}
    for name, content in files.items():
        path = site.joinpath(*name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        digest = hashlib.sha256(content).digest()
        encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
        records.append(f"{name},sha256={encoded},{len(content)}")
        if path.suffix in {".pyd", ".dll"}:
            binary_pins[path.name] = digest.hex()
    if external_script:
        script = root / "Scripts" / "onnxruntime_test.exe"
        script.parent.mkdir()
        script.write_bytes(b"signed test executable")
        digest = hashlib.sha256(script.read_bytes()).digest()
        encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
        records.append(
            f"../../Scripts/onnxruntime_test.exe,sha256={encoded},{script.stat().st_size}"
        )
    record = site / "onnxruntime_gpu-1.28.0.dist-info" / "RECORD"
    record.parent.mkdir()
    record.write_text("\n".join(records) + "\n", encoding="utf-8")
    monkeypatch.setattr(preimport, "ORT_RECORD_SHA256", sha256_file(record))
    monkeypatch.setattr(preimport.metadata, "version", lambda _: "1.28.0")
    monkeypatch.setattr(preimport, "_ort_already_loaded", lambda: False)
    monkeypatch.setattr(
        preimport.importlib.util,
        "find_spec",
        lambda _: SimpleNamespace(origin=str(package / "__init__.py")),
    )
    return root, package, binary_pins


def test_verified_package_never_imports_ort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, pins = _fake_package(tmp_path, monkeypatch)
    monkeypatch.setattr(
        preimport.importlib,
        "import_module",
        lambda _: pytest.fail("ONNX Runtime imported during file verification"),
    )
    preimport.verify_ort_package(root, "1.28.0", pins)


def test_signed_external_script_is_not_importable_package_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, pins = _fake_package(tmp_path, monkeypatch, external_script=True)
    preimport.verify_ort_package(root, "1.28.0", pins)


@pytest.mark.parametrize("change", ["source", "extra_module", "record", "shadow_origin", "reparse"])
def test_package_tampering_fails_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    root, package, pins = _fake_package(tmp_path, monkeypatch)
    if change == "source":
        (package / "__init__.py").write_bytes(b"evil init\n")
    elif change == "extra_module":
        (package / "capi" / "shadow.py").write_text("pass\n", encoding="utf-8")
    elif change == "record":
        record = root / "Lib" / "site-packages" / "onnxruntime_gpu-1.28.0.dist-info" / "RECORD"
        record.write_text(record.read_text(encoding="utf-8") + "unexpected\n", encoding="utf-8")
    elif change == "shadow_origin":
        monkeypatch.setattr(
            preimport.importlib.util,
            "find_spec",
            lambda _: SimpleNamespace(origin=str(tmp_path / "shadow" / "__init__.py")),
        )
    else:
        monkeypatch.setattr(paths_module, "_WINDOWS", True)
        monkeypatch.setattr(paths_module, "_windows_reparse", lambda path: path == package)
    monkeypatch.setattr(
        preimport.importlib,
        "import_module",
        lambda _: pytest.fail("ONNX Runtime imported before rejection"),
    )
    with pytest.raises(preimport.RuntimeImportBlocked):
        preimport.verify_ort_package(root, "1.28.0", pins)


def test_runtime_refuses_unverified_ort_before_dll_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_module, "FROZEN_ENVIRONMENT", Path(sys.prefix))
    monkeypatch.setattr(
        runtime_module,
        "_load_runtime_pins",
        lambda *_: {"onnxruntime_gpu_version": "1.28.0", "onnxruntime_binary_sha256": {}},
    )

    def reject_package(*_: object) -> None:
        raise preimport.RuntimeImportBlocked("unverified package")

    monkeypatch.setattr(runtime_module, "verify_ort_package", reject_package)
    monkeypatch.setattr(
        runtime_module,
        "_installed_dlls",
        lambda *_: pytest.fail("DLL inspection ran after package rejection"),
    )
    with pytest.raises(runtime_module.CudaNullRuntimeBlocked, match="unverified package"):
        runtime_module.verify_runtime_identity(Path("config.yaml"), "a" * 64)


@pytest.mark.skipif(os.name != "nt", reason="frozen CUDA runtime is Windows-only")
def test_controlled_import_preloads_absolute_dlls_before_ort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    directory = tmp_path / "dlls"
    directory.mkdir()
    installed = {
        name: ("distribution", directory / name, "a" * 64) for name in preimport._PRELOAD_ORDER
    }
    monkeypatch.setattr(preimport, "_ort_already_loaded", lambda: False)

    class ModuleHandle:
        argtypes: object = None
        restype: object = None

        def __call__(self, name: str) -> int:
            return 0

    def fake_dll(name: str, **_: object) -> object:
        events.append(name)
        return SimpleNamespace(GetModuleHandleW=ModuleHandle()) if name == "kernel32" else object()

    monkeypatch.setattr(preimport.ctypes, "WinDLL", fake_dll)
    monkeypatch.setattr(preimport.os, "add_dll_directory", lambda _: nullcontext())
    monkeypatch.setattr(
        preimport.importlib,
        "import_module",
        lambda name: events.append(name) or SimpleNamespace(),
    )
    with preimport.import_verified_ort(installed):
        assert events[-1] == "onnxruntime"
    assert events[1:-1] == [str(directory / name) for name in preimport._PRELOAD_ORDER]
