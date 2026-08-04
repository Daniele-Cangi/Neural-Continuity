from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .observations import ModelObservation, observation_to_manifest


def canonical_json_bytes(payload: Mapping[str, Any] | list[Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def canonical_json(payload: Mapping[str, Any], path: Path) -> None:
    path.write_bytes(canonical_json_bytes(payload))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_environment_manifest() -> dict[str, Any]:
    deps = ["numpy", "pandas", "pyarrow", "psutil", "torch", "scipy", "PyYAML"]
    versions: dict[str, str] = {}
    for dep in deps:
        try:
            versions[dep] = importlib.metadata.version(dep)
        except importlib.metadata.PackageNotFoundError:
            versions[dep] = "not-installed"

    env: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os": platform.platform(),
        "processor": platform.processor(),
        "dependencies": versions,
    }
    try:
        import psutil

        env["cpu_count"] = psutil.cpu_count(logical=True)
        env["memory_total_bytes"] = psutil.virtual_memory().total
    except Exception:
        pass
    try:
        import torch

        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["cuda_device_count"] = torch.cuda.device_count()
            env["cuda_devices"] = [
                torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())
            ]
    except Exception:
        env["cuda_available"] = False
        env["cuda_device_count"] = 0
        env["cuda_devices"] = []
    return env


def get_git_commit_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
            text=True,
            cwd=Path.cwd(),
        ).strip()
    except Exception:
        return None


def write_artifacts(run_dir: Path, artifacts: Mapping[str, Any]) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    for name, payload in artifacts.items():
        path = run_dir / name
        if isinstance(payload, dict | list):
            path.write_bytes(canonical_json_bytes(payload))
        else:
            raise TypeError(f"unsupported artifact payload for {name}: {type(payload)!r}")
        manifest[name] = sha256_file(path)

    manifest_payload = {
        "artifacts": manifest,
        "run_id": run_dir.name,
        "run_manifest_type": "canonical_sha256",
        "git_commit": get_git_commit_sha(),
    }
    manifest_path = run_dir / "artifact-manifest.json"
    canonical_json(manifest_payload, manifest_path)
    return {
        "artifact-manifest-path": str(manifest_path),
        "artifact-manifest": manifest_payload,
    }


def save_replay_bundle(
    path: Path,
    observations: list[ModelObservation],
    *,
    dataset_identity: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    payload = {
        "format_version": "1.0.0",
        "dataset": dict(dataset_identity),
        "observations": [observation_to_manifest(observation) for observation in observations],
        "experiment": dict(config),
        "reproducibility": {
            "purpose": "Recompute all metric decisions from evidence artifacts",
            "seedable_fields": ["model_manifest", "query_embeddings", "system_metrics"],
        },
    }
    out = path / "replay-bundle.json"
    out.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return str(out)
