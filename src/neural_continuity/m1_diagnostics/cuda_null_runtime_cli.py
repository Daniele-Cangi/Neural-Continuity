"""Read-only runtime identity check for the CUDA-null preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_continuity.m1_diagnostics.cuda_null_runtime_authority import (
    CudaNullRuntimeBlocked,
    verify_runtime_identity,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify CUDA-null runtime without ONNX session")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    arguments = parser.parse_args()
    try:
        record = verify_runtime_identity(arguments.config, arguments.config_sha256)
    except CudaNullRuntimeBlocked as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
