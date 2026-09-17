"""CLI for non-executing CUDA sentinel readiness review."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from neural_continuity.m1_diagnostics.cuda_null_sentinel_readiness import (
    CudaNullSentinelReadinessBlocked,
    verify_cuda_sentinel_readiness,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify CUDA sentinel design without execution")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--source-preflight-bundle", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    arguments = parser.parse_args(argv)
    try:
        record = verify_cuda_sentinel_readiness(
            config_path=arguments.config,
            external_config_sha256=arguments.config_sha256,
            source_preflight_bundle=arguments.source_preflight_bundle,
            external_source_manifest_sha256=arguments.source_manifest_sha256,
        )
    except CudaNullSentinelReadinessBlocked as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
