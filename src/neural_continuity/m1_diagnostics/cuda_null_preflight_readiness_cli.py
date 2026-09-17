"""Non-executing CUDA-null source-preflight readiness entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_continuity.evidence import canonical_json_bytes
from neural_continuity.m1_diagnostics.cuda_null_preflight_readiness import (
    CudaNullPreflightBlocked,
    verify_preflight_readiness,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify CUDA-null preflight prerequisites without loading an ONNX graph"
    )
    parser.add_argument("--preflight-spec", type=Path, required=True)
    parser.add_argument("--preflight-spec-sha256", required=True)
    parser.add_argument("--static-bundle", type=Path, required=True)
    parser.add_argument("--static-manifest-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--transition-a-bundle", type=Path, required=True)
    parser.add_argument("--teacher-snapshot", type=Path, required=True)
    parser.add_argument("--cpu-extension-bundle", type=Path, required=True)
    parser.add_argument("--historical-cuda-bundle", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        record = verify_preflight_readiness(
            preflight_spec=arguments.preflight_spec,
            external_preflight_spec_sha256=arguments.preflight_spec_sha256,
            static_bundle=arguments.static_bundle,
            external_static_manifest_sha256=arguments.static_manifest_sha256,
            config_path=arguments.config,
            external_config_sha256=arguments.config_sha256,
            dataset_root=arguments.dataset,
            transition_a_bundle=arguments.transition_a_bundle,
            teacher_snapshot_root=arguments.teacher_snapshot,
            cpu_extension_bundle=arguments.cpu_extension_bundle,
            historical_cuda_bundle=arguments.historical_cuda_bundle,
        )
    except CudaNullPreflightBlocked as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(canonical_json_bytes(record).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
