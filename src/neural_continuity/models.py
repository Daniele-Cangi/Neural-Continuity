from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class EmbeddingModel(Protocol):
    model_id: str

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray: ...


@dataclass(frozen=True)
class ToyEmbeddingModel:
    dimension: int
    seed: int = 0
    model_id: str = "toy-deterministic"

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be > 0")

    def _vector_for_text(self, text: str) -> np.ndarray:
        base = f"{self.seed}::{text}".encode()
        digest = hashlib.sha256(base).hexdigest()
        seed = int(digest[:16], 16) % (2**63)
        rng = np.random.default_rng(seed)
        vector = rng.normal(0, 1, size=self.dimension).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            return vector
        return (vector / norm).astype(np.float32)

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        del batch_size
        return np.stack([self._vector_for_text(text) for text in texts]).astype(np.float32)


class SentenceTransformerModel:
    def __init__(self, model_id: str, device: str = "cpu", cache_only: bool = True):
        self.model_id = model_id
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise RuntimeError("sentence-transformers is not installed") from exc
        import torch

        self.model_id = model_id
        self.device = (
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.cache_only = cache_only
        kwargs: dict[str, Any] = {"device": self.device}
        try:
            self._model = SentenceTransformer(model_id, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"sentence-transformer model '{model_id}' not loadable: {exc}"
            ) from exc

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        return self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)


@dataclass(frozen=True)
class PerturbedModel:
    base_model: EmbeddingModel
    perturbation: Any
    seed: int
    perturbation_manifest: dict
    model_id: str

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        vectors = np.asarray(self.base_model.encode(texts, batch_size=batch_size), dtype=np.float32)
        return self.perturbation.apply(vectors)
