from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from neural_continuity.m1_b.onnx_fp32_observation import (
    capture_fp32_source_observation,
)
from neural_continuity.m1_b.onnx_int8_observation import (
    capture_int8_target_observation,
)
from neural_continuity.m1_diagnostics.stage0_authority import (
    Stage0Authority,
)


@contextmanager
def _working_directory(path: Path) -> Any:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def capture_stage0_observations(
    authority: Stage0Authority,
    build_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fp32_output = build_root / "fp32-control"
    int8_output = build_root / "int8-control"
    with _working_directory(authority.runtime_root):
        fp32 = capture_fp32_source_observation(
            authority.fp32_config_path,
            authority.dataset_root,
            authority.transition_a_bundle,
            fp32_output,
        )
        int8 = capture_int8_target_observation(
            authority.int8_config_path,
            authority.dataset_root,
            authority.candidate_root,
            int8_output,
        )
    return fp32, int8
