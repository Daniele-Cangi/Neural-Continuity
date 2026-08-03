from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Perturbation:
    perturbation_type: str

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def manifest(self) -> dict[str, Any]:
        return {"type": self.perturbation_type}


@dataclass(frozen=True)
class GaussianNoisePerturbation(Perturbation):
    strength: float
    seed: int

    def __post_init__(self) -> None:
        if self.strength < 0:
            raise ValueError("strength must be non-negative")

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        noise = rng.normal(loc=0.0, scale=self.strength, size=vectors.shape).astype(np.float32)
        return vectors + noise

    def manifest(self) -> dict[str, Any]:
        return {
            "type": self.perturbation_type,
            "strength": self.strength,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class DimensionMaskPerturbation(Perturbation):
    mask_fraction: float
    seed: int

    def __post_init__(self) -> None:
        if not 0 <= self.mask_fraction <= 1:
            raise ValueError("mask_fraction must be within [0,1]")

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        if self.mask_fraction == 0:
            return vectors

        dim = vectors.shape[1]
        mask = np.zeros(dim, dtype=np.float32)
        rng = np.random.default_rng(self.seed)
        mask_count = max(1, int(round(self.mask_fraction * dim))) if self.mask_fraction > 0 else 0
        masked = rng.choice(np.arange(dim), size=min(mask_count, dim), replace=False)
        mask[masked] = 1.0
        return vectors * (1.0 - mask)

    def manifest(self) -> dict[str, Any]:
        return {
            "type": self.perturbation_type,
            "mask_fraction": self.mask_fraction,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class RotationPerturbation(Perturbation):
    angle_scale: float
    seed: int

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        if self.angle_scale == 0:
            return vectors

        dim = vectors.shape[1]
        rng = np.random.default_rng(self.seed)
        # Deterministic, reproducible orthogonal perturbation:
        basis = rng.normal(size=(dim, dim))
        q, _ = np.linalg.qr(basis)
        return vectors @ (
            q * self.angle_scale + np.eye(dim, dtype=np.float32) * (1 - self.angle_scale)
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "type": self.perturbation_type,
            "angle_scale": self.angle_scale,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class OutputCorruptionPerturbation(Perturbation):
    corruption_fraction: float
    seed: int

    def __post_init__(self) -> None:
        if not 0 <= self.corruption_fraction <= 1:
            raise ValueError("corruption_fraction must be within [0,1]")

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        if self.corruption_fraction == 0:
            return vectors

        rng = np.random.default_rng(self.seed)
        corrupted = vectors.copy()
        dim = vectors.shape[1]
        count = (
            max(1, int(round(self.corruption_fraction * dim)))
            if self.corruption_fraction > 0
            else 0
        )
        for row in range(corrupted.shape[0]):
            idx = rng.choice(dim, size=min(count, dim), replace=False)
            corrupted[row, idx] = 0.0
        return corrupted

    def manifest(self) -> dict[str, Any]:
        return {
            "type": self.perturbation_type,
            "corruption_fraction": self.corruption_fraction,
            "seed": self.seed,
        }


def perturbation_from_config(cfg: dict[str, Any]) -> Perturbation:
    ptype = str(cfg.get("type", "gaussian_noise"))
    if ptype in {"gaussian_noise", "gaussian"}:
        return GaussianNoisePerturbation(
            perturbation_type="gaussian_noise",
            strength=float(cfg.get("strength", 0.5)),
            seed=int(cfg.get("seed", 0)),
        )
    if ptype in {"dimension_mask", "mask"}:
        return DimensionMaskPerturbation(
            perturbation_type="dimension_mask",
            mask_fraction=float(cfg.get("strength", 0.5)),
            seed=int(cfg.get("seed", 0)),
        )
    if ptype in {"rotation", "rotate"}:
        return RotationPerturbation(
            perturbation_type="rotation",
            angle_scale=float(cfg.get("strength", 0.5)),
            seed=int(cfg.get("seed", 0)),
        )
    if ptype in {"output_corruption", "corrupt"}:
        return OutputCorruptionPerturbation(
            perturbation_type="output_corruption",
            corruption_fraction=float(cfg.get("strength", 0.5)),
            seed=int(cfg.get("seed", 0)),
        )
    raise ValueError(f"unsupported perturbation type: {ptype}")
