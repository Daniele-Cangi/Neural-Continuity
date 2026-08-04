from __future__ import annotations

import hashlib
import os
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
        self.device = device
        self.cache_only = cache_only
        self.model_path: str | None = None
        self.model_files: list[str] = []

        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise RuntimeError("sentence-transformers is not installed") from exc
        import builtins

        torch = builtins.__import__("torch")
        self.device = (
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        kwargs: dict[str, Any] = {"device": self.device}
        try:
            self._model = SentenceTransformer(model_id, local_files_only=cache_only, **kwargs)
        except Exception as exc:
            if cache_only:
                raise RuntimeError(
                    f"sentence-transformer model '{model_id}' not loadable from local cache: {exc}"
                ) from exc
            raise RuntimeError(
                f"sentence-transformer model '{model_id}' not loadable: {exc}"
            ) from exc

        self.model_path = self._infer_model_path()
        self.model_files = self._collect_model_files()

    def _infer_model_path(self) -> str | None:
        try:
            path = str(self._model._model_path)
            if path and os.path.isdir(path):
                return path
        except Exception:
            pass
        if os.path.isdir(self.model_id):
            return str(os.path.abspath(self.model_id))
        return None

    def _collect_model_files(self) -> list[str]:
        if not self.model_path:
            return []
        collected: list[str] = []
        for root, _, filenames in os.walk(self.model_path):
            for filename in sorted(filenames):
                lower = filename.lower()
                if lower.endswith((".bin", ".safetensors", ".json", ".txt", ".model", ".pt")):
                    collected.append(os.path.relpath(os.path.join(root, filename), self.model_path))
        return collected[:80]

    def manifest(self) -> dict[str, Any]:
        file_hashes: list[dict[str, str]] = []
        if self.model_path:
            for name in self.model_files:
                full_path = os.path.join(self.model_path, name)
                hasher = hashlib.sha256()
                with open(full_path, "rb") as handle:
                    for chunk in iter(lambda: handle.read(8192), b""):
                        hasher.update(chunk)
                file_hashes.append({"path": name, "sha256": hasher.hexdigest()})

        return {
            "path": self.model_path,
            "cache_only": self.cache_only,
            "files": file_hashes,
            "model_id": self.model_id,
            "device": self.device,
        }

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
