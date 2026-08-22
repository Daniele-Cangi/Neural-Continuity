from __future__ import annotations

from types import ModuleType

from neural_continuity.m1_diagnostics import measurement_null_extension_evidence


def test_evidence_module_exposes_capture_and_replay() -> None:
    assert callable(measurement_null_extension_evidence.capture_measurement_null_extension_plan)
    assert callable(measurement_null_extension_evidence.replay_measurement_null_extension_plan)


def test_evidence_module_has_no_onnx_runtime_dependency() -> None:
    imported_modules = {
        value.__name__
        for value in vars(measurement_null_extension_evidence).values()
        if isinstance(value, ModuleType)
    }

    assert not any(name == "onnx" or name.startswith("onnx.") for name in imported_modules)
    assert not any(
        name == "onnxruntime" or name.startswith("onnxruntime.") for name in imported_modules
    )
