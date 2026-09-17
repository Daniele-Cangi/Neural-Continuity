"""Authenticate the frozen ONNX Runtime files before importing native code."""

from __future__ import annotations

import base64
import binascii
import csv
import ctypes
import hashlib
import importlib
import importlib.util
import io
import os
import sys
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from importlib import metadata
from pathlib import Path, PurePosixPath
from types import ModuleType

from neural_continuity.evidence import sha256_file
from neural_continuity.m1_diagnostics.cuda_null_paths import (
    has_linked_ancestor,
    path_is_link_or_reparse,
)

ORT_RECORD_SHA256 = "0f8ef4e6e6178a33aec2ca61f3dba4b19f84685809012848a2446c8eb0e95bb6"
_EXECUTABLE_SUFFIXES = frozenset({".py", ".pyd", ".dll"})
_PRELOAD_ORDER = (
    "onnxruntime_providers_shared.dll",
    "cudart64_13.dll",
    "cublasLt64_13.dll",
    "cublas64_13.dll",
    "cufft64_12.dll",
    "cudnn64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_engines_tensor_ir64_9.dll",
)


class RuntimeImportBlocked(RuntimeError):
    """The installed runtime cannot be trusted before native import."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeImportBlocked(reason)


def _ort_already_loaded() -> bool:
    return any(name == "onnxruntime" or name.startswith("onnxruntime.") for name in sys.modules)


def _record_hash(encoded: str) -> str:
    _require(encoded.startswith("sha256="), "ONNX Runtime RECORD hash missing")
    value = encoded.removeprefix("sha256=")
    try:
        digest = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (binascii.Error, ValueError) as exc:
        raise RuntimeImportBlocked("ONNX Runtime RECORD hash invalid") from exc
    _require(len(digest) == 32, "ONNX Runtime RECORD hash invalid")
    return digest.hex()


def verify_ort_package(root: Path, version: str, binary_pins: Mapping[str, str]) -> None:
    """Bind every importable ORT file to the pinned wheel RECORD, without import."""
    _require(version == "1.28.0", "ONNX Runtime version is not frozen")
    site = root / "Lib" / "site-packages"
    package = site / "onnxruntime"
    record = site / f"onnxruntime_gpu-{version}.dist-info" / "RECORD"
    _require(
        not has_linked_ancestor(package) and not has_linked_ancestor(record),
        "ONNX Runtime package contains a link or reparse point",
    )
    _require(package.is_dir() and record.is_file(), "ONNX Runtime package missing")
    _require(metadata.version("onnxruntime-gpu") == version, "ONNX Runtime distribution mismatch")
    record_bytes = record.read_bytes()
    _require(
        hashlib.sha256(record_bytes).hexdigest() == ORT_RECORD_SHA256,
        "ONNX Runtime wheel RECORD mismatch",
    )
    _require(not _ort_already_loaded(), "ONNX Runtime imported before provenance gate")
    spec = importlib.util.find_spec("onnxruntime")
    _require(
        spec is not None
        and isinstance(spec.origin, str)
        and Path(spec.origin).absolute() == (package / "__init__.py").absolute(),
        "ONNX Runtime import origin mismatch",
    )

    declared: set[str] = set()
    observed_binaries: set[str] = set()
    rows = csv.reader(io.StringIO(record_bytes.decode("utf-8")))
    for row in rows:
        _require(len(row) == 3, "ONNX Runtime RECORD row invalid")
        name, encoded_hash, size = row
        if not name.startswith("onnxruntime/"):
            continue
        relative = PurePosixPath(name)
        _require(
            bool(relative.parts)
            and not relative.is_absolute()
            and "\\" not in name
            and all(part not in (".", "..") for part in relative.parts),
            "ONNX Runtime RECORD path unsafe",
        )
        if relative.suffix.lower() not in _EXECUTABLE_SUFFIXES:
            continue
        _require(name not in declared, "ONNX Runtime RECORD duplicate executable")
        path = site.joinpath(*relative.parts)
        _require(not has_linked_ancestor(path), f"ONNX Runtime linked file: {name}")
        _require(path.is_file() and size.isdecimal(), f"ONNX Runtime file missing: {name}")
        _require(path.stat().st_size == int(size), f"ONNX Runtime file size mismatch: {name}")
        digest = sha256_file(path)
        _require(digest == _record_hash(encoded_hash), f"ONNX Runtime file hash mismatch: {name}")
        declared.add(name)
        if path.name in binary_pins:
            _require(path.name not in observed_binaries, "ONNX Runtime binary name duplicated")
            _require(digest == binary_pins[path.name], f"ONNX Runtime binary pin mismatch: {name}")
            observed_binaries.add(path.name)
    _require(observed_binaries == set(binary_pins), "ONNX Runtime binary set mismatch")

    actual: set[str] = set()
    pending = [package]
    while pending:
        for entry in pending.pop().iterdir():
            _require(not path_is_link_or_reparse(entry), "ONNX Runtime package contains a link")
            if entry.is_dir():
                pending.append(entry)
            elif entry.is_file():
                if entry.suffix.lower() in _EXECUTABLE_SUFFIXES:
                    actual.add(entry.relative_to(site).as_posix())
                elif entry.suffix.lower() == ".pyc" and "__pycache__" not in entry.parts:
                    raise RuntimeImportBlocked("ONNX Runtime contains sourceless bytecode")
            else:
                raise RuntimeImportBlocked("ONNX Runtime contains an unsupported entry")
    _require(actual == declared, "ONNX Runtime executable inventory mismatch")


@contextmanager
def import_verified_ort(
    installed: Mapping[str, tuple[str, Path, str]],
) -> Iterator[ModuleType]:
    """Import only after hashes pass, using validated absolute Windows DLL paths."""
    _require(os.name == "nt", "frozen ONNX Runtime requires Windows")
    _require(not _ort_already_loaded(), "ONNX Runtime imported before controlled load")
    _require(set(_PRELOAD_ORDER).issubset(installed), "validated DLL set incomplete")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_module_handle = kernel32.GetModuleHandleW
    get_module_handle.argtypes = [ctypes.c_wchar_p]
    get_module_handle.restype = ctypes.c_void_p
    for name in installed:
        _require(not get_module_handle(name), f"{name}: loaded before provenance gate")

    with ExitStack() as stack:
        directories = sorted({item[1].parent for item in installed.values()}, key=str)
        for directory in directories:
            _require(not has_linked_ancestor(directory), "linked DLL directory")
            stack.enter_context(os.add_dll_directory(str(directory)))
        cache = stack.enter_context(tempfile.TemporaryDirectory(prefix="nc-ort-import-"))
        previous_prefix = sys.pycache_prefix
        previous_no_bytecode = sys.dont_write_bytecode
        sys.pycache_prefix = cache
        sys.dont_write_bytecode = True
        try:
            loaded_handles = []
            for name in _PRELOAD_ORDER:
                loaded_handles.append(ctypes.WinDLL(str(installed[name][1])))
            module = importlib.import_module("onnxruntime")
            yield module
        finally:
            sys.pycache_prefix = previous_prefix
            sys.dont_write_bytecode = previous_no_bytecode
