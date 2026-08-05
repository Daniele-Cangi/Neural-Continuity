from __future__ import annotations

import builtins
import hashlib
import importlib.metadata
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class EmbeddingModel(Protocol):
    model_id: str

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray: ...


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "not-installed"


def _package_available(name: str) -> bool:
    try:
        importlib.metadata.version(name)
        return True
    except Exception:
        return False


def _cache_miss_reason(exc: BaseException) -> bool:
    if isinstance(exc, FileNotFoundError):
        return True
    if getattr(exc, "errno", None) == 2:  # No such file
        return True
    message = str(exc).lower()
    indicators = (
        "local cache",
        "cache does not contain",
        "not in local cache",
        "cached files",
        "no sentence-transformers model found",
        "couldn't find it in the cached files",
        "no such file",
        "cannot find",
        "requested file",
    )
    return any(indicator in message for indicator in indicators)


def _safe_path(value: Any) -> Path | None:
    if not isinstance(value, str):
        return None
    candidate = Path(value).expanduser()
    if not candidate.exists():
        return None
    return candidate


def _snapshot_path_from_file_path(value: Any) -> str | None:
    path = _safe_path(value)
    if path is None:
        return None

    previous = path
    for parent in path.parents:
        if parent.name == "snapshots":
            if previous != path and previous.name:
                return str(previous)
            if parent.parent is not None:
                return str(parent.parent / previous.name)
        previous = parent
    return None


def _candidate_paths_from_object(obj: Any) -> list[str]:
    if obj is None:
        return []

    if isinstance(obj, str):
        return [obj]

    names: list[str] = []
    for attr in (
        "vocab_file",
        "vocab",
        "vocab_path",
        "tokenizer_file",
        "special_tokens_map_file",
        "merges_file",
        "added_tokens_file",
        "config_file",
    ):
        value = getattr(obj, attr, None)
        if value is not None:
            names.append(value)

    backend = getattr(obj, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "tokenizer"):
        names.extend(_candidate_paths_from_object(backend.tokenizer))

    return [str(v) for v in names if isinstance(v, str | Path)]


def _resolve_revision_from_modules(modules: list[Any]) -> str | None:
    for module in modules:
        for attr in ("config",):
            config = getattr(module, attr, None)
            if config is None:
                continue
            for key in ("_commit_hash", "_name_or_path"):
                value = getattr(config, key, None)
                if key == "_commit_hash" and _is_immutable_revision(value):
                    return value
        auto_model = getattr(module, "auto_model", None)
        if auto_model is not None:
            auto_config = getattr(auto_model, "config", None)
            if auto_config is None:
                continue
            value = getattr(auto_config, "_commit_hash", None)
            if _is_immutable_revision(value):
                return value
    return None


def _is_immutable_revision(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 40:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in value)


def _is_transient_file(name: str) -> bool:
    lowered = name.lower()
    transient_suffixes = {".lock", ".tmp", ".partial", ".part", ".tmp_download"}
    return lowered.startswith(".") or any(lowered.endswith(suffix) for suffix in transient_suffixes)


def _hash_file(path: str) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_manifest_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(k): _to_manifest_value(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list | tuple | set):
        return [_to_manifest_value(v) for v in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    try:
        return str(value)
    except Exception:
        return repr(value)


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
    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cpu",
        cache_only: bool = True,
        normalize_embeddings: bool = False,
        output_dtype: str = "float32",
        prompt_name: str | None = None,
        prompt: str | None = None,
        max_sequence_length: int | None = None,
        revision: str | None = None,
    ):
        self.model_id = model_id
        self.requested_device = str(device)
        if not isinstance(cache_only, bool):
            raise ValueError("cache_only must be a boolean")
        self.cache_only = cache_only
        if not isinstance(normalize_embeddings, bool):
            raise ValueError("normalize_embeddings must be a boolean")
        self.normalize_embeddings = normalize_embeddings
        self.prompt_name = prompt_name if prompt_name is not None else None
        self.prompt_text = prompt if prompt is not None else None
        if max_sequence_length is not None and (
            isinstance(max_sequence_length, bool)
            or not isinstance(max_sequence_length, int)
            or max_sequence_length <= 0
        ):
            raise ValueError("max_sequence_length must be a positive integer")
        self.requested_max_sequence_length = max_sequence_length
        if revision is not None and not _is_immutable_revision(revision):
            raise ValueError("revision must be an immutable 40-character commit SHA")
        self.requested_revision = revision
        if self.prompt_name is not None and self.prompt_text is not None:
            raise ValueError("prompt_name and prompt are mutually exclusive")

        normalized_dtype = str(output_dtype).lower()
        if normalized_dtype != "float32":
            raise ValueError("real teacher qualification requires output_dtype=float32")
        self.output_dtype = normalized_dtype

        if not _package_available("sentence-transformers"):
            raise RuntimeError("teacher_dependency_unavailable:sentence-transformers")

        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            dependency = str(getattr(exc, "name", None) or "unknown").split(".", 1)[0]
            raise RuntimeError(f"teacher_dependency_unavailable:{dependency}") from exc
        except Exception as exc:
            raise RuntimeError(f"teacher_runtime_import_error: {exc}") from exc
        try:
            torch = builtins.__import__("torch")
        except ModuleNotFoundError as exc:
            raise RuntimeError("teacher_dependency_unavailable:torch") from exc

        self.device = (
            self.requested_device
            if self.requested_device != "auto"
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

        try:
            load_options: dict[str, Any] = {
                "local_files_only": cache_only,
                "device": self.device,
            }
            if self.requested_revision is not None:
                load_options["revision"] = self.requested_revision
            self._model = SentenceTransformer(model_id, **load_options)
        except Exception as exc:
            if cache_only:
                if _cache_miss_reason(exc):
                    raise RuntimeError(f"teacher_not_available_from_local_cache: {exc}") from exc
                raise RuntimeError(f"teacher_runtime_import_error: {exc}") from exc
            raise RuntimeError(f"teacher_runtime_import_error: {exc}") from exc

        self.model_path: str | None = None
        self.model_path_reason: str | None = None
        self.model_path = self._infer_model_path()
        if self.model_path is None:
            raise RuntimeError("teacher_provenance_unavailable:model path could not be inferred")

        self.model_files = self._collect_model_files()
        if not self.model_files:
            raise RuntimeError("teacher_provenance_unavailable:model snapshot contains no files")
        self.model_implementation_class = (
            f"{type(self._model).__module__}.{type(self._model).__name__}"
        )

        self.embedding_dimension = self._infer_embedding_dimension()
        self.embedding_dimension_reason: str | None = None
        if self.embedding_dimension is None:
            self.embedding_dimension_reason = "embedding dimension not exposed by model instance"

        if self.requested_max_sequence_length is not None:
            try:
                self._model.max_seq_length = self.requested_max_sequence_length
            except Exception as exc:
                raise RuntimeError(
                    "teacher_preprocessing_configuration_error:max_sequence_length"
                ) from exc

        self.max_sequence_length = self._resolve_max_sequence_length()
        self.max_sequence_length_reason = (
            None if self.max_sequence_length is not None else "max sequence length not available"
        )
        if (
            self.requested_max_sequence_length is not None
            and self.max_sequence_length != self.requested_max_sequence_length
        ):
            raise RuntimeError(
                "teacher_preprocessing_configuration_error:max_sequence_length_not_applied"
            )

        (
            self.tokenizer_class,
            self.tokenizer_config,
            self.tokenizer_config_reason,
            self.tokenizer_config_sha256,
        ) = self._resolve_tokenizer()
        self.pipeline_modules = self._resolve_pipeline_modules()
        self.pooling_configuration, self.pooling_configuration_reason = self._resolve_pooling()
        self.pipeline_normalizes_embeddings = any(
            module.get("normalizes_embeddings") is True for module in self.pipeline_modules
        )
        self.effective_normalization = (
            self.normalize_embeddings or self.pipeline_normalizes_embeddings
        )
        self.model_revision, self.model_revision_reason = self._resolve_revision()
        if self.requested_revision is not None and self.model_revision != self.requested_revision:
            raise RuntimeError(
                "teacher_provenance_revision_mismatch:"
                f"requested={self.requested_revision},resolved={self.model_revision}"
            )

        self.evaluation_mode = False
        self.model_eval_reason: str | None = None
        try:
            self._model.eval()
            self.evaluation_mode = True
        except Exception as exc:
            raise RuntimeError("teacher_evaluation_mode_error") from exc
        else:
            self.model_eval_reason = None

    def _infer_model_path(self) -> str | None:
        model_path_attrs = ["_model_path", "cache_folder", "model_card_data"]
        for attr in model_path_attrs:
            try:
                value = getattr(self._model, attr)
            except Exception:
                continue
            if isinstance(value, str):
                if os.path.isdir(value):
                    return os.path.abspath(value)
                if os.path.isfile(value):
                    return os.path.dirname(value)

        tokenizer = getattr(self._model, "tokenizer", None)
        for path_value in _candidate_paths_from_object(tokenizer):
            model_path = _snapshot_path_from_file_path(path_value)
            if model_path is not None and os.path.isdir(model_path):
                return model_path

        for module in self._iter_sentence_modules():
            module_path = self._snapshot_path_from_module(module)
            if module_path is not None:
                return module_path

        if os.path.isdir(self.model_id):
            return os.path.abspath(self.model_id)
        return None

    def _collect_model_files(self) -> list[dict[str, Any]]:
        if not self.model_path:
            return []
        manifest: list[dict[str, Any]] = []
        for root, _, filenames in os.walk(self.model_path):
            filenames = sorted(filenames)
            for filename in filenames:
                if _is_transient_file(filename):
                    continue
                full_path = os.path.join(root, filename)
                if not os.path.isfile(full_path):
                    continue
                relative_path = os.path.relpath(full_path, self.model_path).replace(os.sep, "/")
                try:
                    file_hash, file_size = _hash_file(full_path)
                    manifest.append(
                        {
                            "relative_path": relative_path,
                            "sha256": file_hash,
                            "byte_size": file_size,
                        }
                    )
                except Exception as exc:
                    raise RuntimeError(f"teacher_provenance_hash_error:{relative_path}") from exc
        manifest.sort(key=lambda item: str(item.get("relative_path")))
        return manifest

    def _infer_embedding_dimension(self) -> int | None:
        dimension = getattr(self._model, "get_embedding_dimension", None)
        if callable(dimension):
            try:
                value = dimension()
                return _safe_int(value)
            except Exception:
                pass
        dimension = getattr(self._model, "get_sentence_embedding_dimension", None)
        if callable(dimension):
            try:
                value = dimension()
                return _safe_int(value)
            except Exception:
                pass
        for attr in ("embedding_dimension", "dimension"):
            candidate = getattr(self._model, attr, None)
            value = _safe_int(candidate)
            if value is not None and value > 0:
                return value
        for module in self._iter_sentence_modules():
            if hasattr(module, "get_sentence_embedding_dimension"):
                try:
                    value = module.get_sentence_embedding_dimension()
                    return _safe_int(value)
                except Exception:
                    continue
        return None

    def _iter_sentence_modules(self) -> list[Any]:
        return [module for _, module in self._iter_sentence_module_entries()]

    def _iter_sentence_module_entries(self) -> list[tuple[str, Any]]:
        modules = getattr(self._model, "_modules", None)
        if isinstance(modules, Mapping):
            return [(str(name), module) for name, module in modules.items()]
        return []

    def _snapshot_path_from_module(self, module: Any) -> str | None:
        for candidate in _candidate_paths_from_object(getattr(module, "auto_model", None)):
            model_path = _snapshot_path_from_file_path(candidate)
            if model_path and os.path.isdir(model_path):
                return model_path
        for candidate in _candidate_paths_from_object(getattr(module, "config", None)):
            model_path = _snapshot_path_from_file_path(candidate)
            if model_path and os.path.isdir(model_path):
                return model_path
        return None

    def _resolve_max_sequence_length(self) -> int | None:
        candidates = [getattr(self._model, "max_seq_length", None)]
        tokenizer = getattr(self._model, "tokenizer", None)
        if tokenizer is not None:
            candidates.extend(
                [
                    getattr(tokenizer, "model_max_length", None),
                    getattr(tokenizer, "max_len", None),
                    getattr(tokenizer, "max_seq_length", None),
                ]
            )
        if candidates:
            for value in candidates:
                safe = _safe_int(value)
                if safe is not None and safe > 0:
                    return safe
        return None

    def _resolve_tokenizer(
        self,
    ) -> tuple[str | None, dict[str, Any] | None, str | None, str | None]:
        tokenizer = getattr(self._model, "tokenizer", None)
        if tokenizer is None:
            return None, None, "tokenizer is unavailable", None
        tokenizer_class = f"{type(tokenizer).__module__}.{type(tokenizer).__name__}"
        backend = getattr(tokenizer, "backend_tokenizer", None)
        serializer = getattr(backend, "to_str", None)
        if callable(serializer):
            try:
                raw_config = json.loads(serializer())
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                return (
                    tokenizer_class,
                    None,
                    f"backend tokenizer serialization failed: {type(exc).__name__}",
                    None,
                )
        else:
            raw_config = (
                getattr(tokenizer, "init_kwargs", None)
                or getattr(tokenizer, "config", None)
                or {
                    "model_max_length": getattr(tokenizer, "model_max_length", None),
                }
            )
        manifest_config = _to_manifest_value(raw_config)
        if not isinstance(manifest_config, dict):
            manifest_config = {"value": manifest_config}
        canonical = json.dumps(
            manifest_config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return tokenizer_class, manifest_config, None, hashlib.sha256(canonical).hexdigest()

    @staticmethod
    def _stable_module_configuration(module: Any) -> dict[str, Any]:
        raw_config: Any = None
        get_config = getattr(module, "get_config_dict", None)
        if callable(get_config):
            try:
                raw_config = get_config()
            except Exception:
                raw_config = None
        if raw_config is None:
            config_keys = getattr(module, "config_keys", ())
            if isinstance(config_keys, Sequence) and not isinstance(config_keys, str | bytes):
                raw_config = {
                    str(key): getattr(module, str(key), None)
                    for key in config_keys
                    if isinstance(key, str)
                }
        converted = _to_manifest_value(raw_config if raw_config is not None else {})
        return converted if isinstance(converted, dict) else {"value": converted}

    def _resolve_pipeline_modules(self) -> list[dict[str, Any]]:
        manifest: list[dict[str, Any]] = []
        for name, module in self._iter_sentence_module_entries():
            module_class = f"{type(module).__module__}.{type(module).__name__}"
            class_name = type(module).__name__.lower()
            configuration = self._stable_module_configuration(module)
            is_pooling = class_name.endswith("pooling") or any(
                str(key).startswith("pooling_mode") for key in configuration
            )
            normalizes_embeddings = class_name.endswith("normalize") or bool(
                getattr(module, "normalizes_embeddings", False)
            )
            entry: dict[str, Any] = {
                "name": name,
                "class": module_class,
                "configuration": configuration,
                "normalizes_embeddings": normalizes_embeddings,
            }
            if is_pooling:
                entry["semantic_role"] = "pooling"
            elif normalizes_embeddings:
                entry["semantic_role"] = "embedding_normalization"
            manifest.append(entry)
        return manifest

    def _resolve_pooling(self) -> tuple[dict[str, Any] | None, str | None]:
        for module in self.pipeline_modules:
            if module.get("semantic_role") == "pooling":
                return {
                    "module_name": module["name"],
                    "module_class": module["class"],
                    "configuration": module["configuration"],
                }, None
        for attr in ("pooling", "pooling_layer", "pooler"):
            candidate = getattr(self._model, attr, None)
            if candidate is not None:
                return {
                    "module_name": attr,
                    "module_class": f"{type(candidate).__module__}.{type(candidate).__name__}",
                    "configuration": self._stable_module_configuration(candidate),
                }, None
        return None, "pooling module is unavailable"

    def _resolve_revision(self) -> tuple[str | None, str | None]:
        for attr in ("revision", "_revision", "_model_card"):
            value = getattr(self._model, attr, None)
            if _is_immutable_revision(value):
                return value, None
        for attr in ("_commit_hash",):
            value = getattr(getattr(self._model, "auto_model", None), attr, None)
            if _is_immutable_revision(value):
                return value, None

        modules = self._iter_sentence_modules()
        commit_hash = _resolve_revision_from_modules(modules)
        if commit_hash is not None:
            return commit_hash, None
        if self.model_path and os.path.isdir(self.model_path):
            revision_hint = Path(self.model_path).name
            if _is_immutable_revision(revision_hint):
                return revision_hint, "inferred from local snapshot directory name"

        return None, "immutable revision identity not exposed; content hashes are authoritative"

    def manifest(self) -> dict[str, Any]:
        file_hashes = list(self.model_files)
        package_versions = {
            "torch": _package_version("torch"),
            "sentence-transformers": _package_version("sentence-transformers"),
            "transformers": _package_version("transformers"),
            "huggingface-hub": _package_version("huggingface-hub"),
            "tokenizers": _package_version("tokenizers"),
            "safetensors": _package_version("safetensors"),
        }
        total_bytes = 0
        content_hasher = hashlib.sha256()
        for entry in file_hashes:
            bytes_value = entry.get("byte_size")
            if isinstance(bytes_value, int):
                total_bytes += bytes_value
            content_hasher.update(str(entry["relative_path"]).encode("utf-8"))
            content_hasher.update(b"\0")
            content_hasher.update(str(entry["sha256"]).encode("ascii"))
            content_hasher.update(b"\0")
            content_hasher.update(str(entry["byte_size"]).encode("ascii"))
            content_hasher.update(b"\n")
        return {
            "declared_model_id": self.model_id,
            "model_id": self.model_id,
            "path": self.model_path,
            "path_reason": self.model_path_reason,
            "resolved_snapshot_path": self.model_path,
            "cache_only": self.cache_only,
            "requested_device": self.requested_device,
            "resolved_device": self.device,
            "model_implementation_class": self.model_implementation_class,
            "embedding_dimension": self.embedding_dimension,
            "embedding_dimension_reason": self.embedding_dimension_reason,
            "max_sequence_length": self.max_sequence_length,
            "requested_max_sequence_length": self.requested_max_sequence_length,
            "max_sequence_length_reason": self.max_sequence_length_reason,
            "output_dtype": self.output_dtype,
            "evaluation_mode": self.evaluation_mode,
            "evaluation_mode_reason": self.model_eval_reason,
            "normalize_embeddings": self.normalize_embeddings,
            "encode_normalize_embeddings": self.normalize_embeddings,
            "pipeline_normalizes_embeddings": self.pipeline_normalizes_embeddings,
            "effective_normalization": self.effective_normalization,
            "tokenizer_class": self.tokenizer_class,
            "tokenizer_configuration": self.tokenizer_config,
            "tokenizer_configuration_reason": self.tokenizer_config_reason,
            "tokenizer_configuration_sha256": self.tokenizer_config_sha256,
            "pipeline_modules": self.pipeline_modules,
            "pooling_configuration": self.pooling_configuration,
            "pooling_configuration_reason": self.pooling_configuration_reason,
            "prompt_name": self.prompt_name,
            "prompt_text": self.prompt_text,
            "snapshot_revision": self.model_revision,
            "requested_revision": self.requested_revision,
            "snapshot_revision_reason": self.model_revision_reason,
            "package_versions": package_versions,
            "model_files": file_hashes,
            "model_file_count": len(file_hashes),
            "hashed_file_count": len(file_hashes),
            "hashed_file_bytes": total_bytes,
            "snapshot_content_sha256": content_hasher.hexdigest(),
        }

    def _validate_encoding(self, values: np.ndarray, expected_count: int) -> np.ndarray:
        if values.ndim != 2:
            raise RuntimeError("sentence-transformer model returned non-2D embeddings")
        if values.shape[0] != expected_count:
            raise RuntimeError(
                "sentence-transformer output row count mismatch "
                f"(expected {expected_count}, got {values.shape[0]})"
            )
        if not np.all(np.isfinite(values)):
            raise RuntimeError("sentence-transformer output contains NaN or infinity")
        if self.embedding_dimension is not None and values.shape[1] != self.embedding_dimension:
            raise RuntimeError(
                f"sentence-transformer embedding dimension changed across calls "
                f"(expected {self.embedding_dimension}, got {values.shape[1]})"
            )
        if values.dtype != np.float32:
            raise RuntimeError(
                f"sentence-transformer output dtype is {values.dtype}, expected float32"
            )
        self.embedding_dimension = values.shape[1]
        self.embedding_dimension_reason = None
        return values

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        values = self._model.encode(
            list(texts),
            batch_size=max(1, int(batch_size)),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            device=self.device,
            prompt_name=self.prompt_name,
            prompt=self.prompt_text,
        )
        if not isinstance(values, np.ndarray):
            values = np.asarray(values)
        values = np.asarray(values)
        values = self._validate_encoding(values, len(texts))
        return values.astype(np.float32)


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
