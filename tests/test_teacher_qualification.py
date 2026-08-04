from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

import neural_continuity.models as models
from neural_continuity.cli import run_m0, run_m0_replay
from neural_continuity.evidence import build_environment_manifest, canonical_json_bytes
from neural_continuity.models import SentenceTransformerModel


def _install_fake_torch(monkeypatch) -> None:
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def reset_peak_memory_stats(*_args) -> None:
            return None

        @staticmethod
        def max_memory_allocated(*_args) -> int:
            return 0

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = _FakeCuda
    fake_torch.float32 = np.float32
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def _make_fake_sentence_transformers_model(monkeypatch, *, model_path: Path, output_factory):
    init_calls: list[dict[str, object]] = []
    encode_calls: list[dict[str, object]] = []

    class FakeTokenizer:
        def __init__(self) -> None:
            self.model_max_length = 16
            self.config = {"name_or_path": "stub-tokenizer"}

    class FakeSentenceTransformer:
        def __init__(
            self,
            model_id: str,
            local_files_only: bool = True,
            device: str = "cpu",
            **kwargs,
        ):
            init_calls.append(
                {
                    "model_id": model_id,
                    "local_files_only": local_files_only,
                    "device": device,
                    "kwargs": dict(kwargs),
                }
            )
            self._model_path = str(model_path)
            self._eval_called = False
            self.model_id = model_id
            self.tokenizer = FakeTokenizer()
            self.auto_model = types.SimpleNamespace(
                config=types.SimpleNamespace(_commit_hash="a" * 40)
            )

        def eval(self):
            self._eval_called = True
            return self

        def get_sentence_embedding_dimension(self) -> int:
            return 4

        def encode(self, texts, **kwargs):
            encode_calls.append({"count": len(texts), "kwargs": dict(kwargs)})
            return output_factory(texts=list(texts), kwargs=kwargs)

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(models, "_package_available", lambda name: True)
    return init_calls, encode_calls, FakeSentenceTransformer


def _write_teacher_config(tmp_path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    fixture_src = repo_root / "tests" / "fixtures" / "local_retrieval_fixture.json"
    fixture_copy = tmp_path / "local_retrieval_fixture.json"
    fixture_copy.write_text(fixture_src.read_text(encoding="utf-8"), encoding="utf-8")

    config = {
        "experiment_name": "M1_REAL_TEACHER_QUALIFICATION",
        "contract": str(repo_root / "contracts" / "m0-measurement-integrity-v1.json"),
        "model": {
            "kind": "sentence-transformers",
            "model_id": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
            "cache_only": True,
            "normalize_embeddings": False,
        },
        "dataset": {"path": "local_retrieval_fixture.json"},
        "null": {
            "repeats": 2,
            "batch_sizes": [1],
            "bootstrap_samples": 25,
            "confidence_level": 0.99,
            "random_seed": 2026,
            "candidate_bootstrap_samples": 25,
            "candidate_confidence_level": 0.99,
        },
        "runtime": {"topology_k": 5},
        "controls": {
            "exact_repeat": {"enabled": True, "repeats": 1},
            "negative": {
                "enabled": True,
                "type": "gaussian_noise",
                "strength": 0.95,
                "seed": 777,
            },
            "boundary": {
                "enabled": True,
                "type": "gaussian_noise",
                "strength": 0.06,
                "seed": 901,
                "attempts": 4,
                "min_strength": 0.0,
                "max_strength": 0.2,
            },
        },
    }
    config_path = tmp_path / "m1-teacher-qualification.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _install_fake_sentence_transformers(monkeypatch, *, constructor_behaviour) -> None:
    class FakeSentenceTransformer:
        def __init__(
            self,
            model_id: str,
            local_files_only: bool = True,
            device: str = "cpu",
            **_kwargs,
        ):
            constructor_behaviour(model_id, local_files_only=local_files_only, device=device)

        def eval(self):
            return self

        def get_sentence_embedding_dimension(self) -> int:
            return 4

        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 4), dtype=np.float32)

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(models, "_package_available", lambda name: True)


def _classify_teacher_dependency_error(monkeypatch, reason: str, *, extra_setup=None) -> None:
    def _constructor(*_args, **_kwargs):
        raise RuntimeError(reason)

    if extra_setup is not None:
        extra_setup()
    import neural_continuity.cli as cli_module

    monkeypatch.setattr(cli_module, "SentenceTransformerModel", _constructor)


def _run_blocking_cli(
    monkeypatch, tmp_path: Path, *, reason: str
) -> tuple[int, dict[str, object], str]:
    config_path = _write_teacher_config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["model"]["allow_offline_skip"] = True
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def _raise_blocking_error(*_args, **_kwargs):
        raise RuntimeError(reason)

    import neural_continuity.cli as cli_module

    monkeypatch.setattr(cli_module, "SentenceTransformerModel", _raise_blocking_error)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exit_code = cli_module.main(
            ["m0-run", "--config", str(config_path), "--output", str(tmp_path / "runs")]
        )
    payload_text = output.getvalue().strip()
    assert payload_text, "expected blocked decision payload on stdout"
    return exit_code, json.loads(payload_text), payload_text


def test_sentence_transformer_dependency_classification_missing_sentence_transformers(
    monkeypatch,
) -> None:
    monkeypatch.setattr(models, "_package_available", lambda name: name != "sentence-transformers")
    with pytest.raises(RuntimeError, match=r"teacher_dependency_unavailable:sentence-transformers"):
        SentenceTransformerModel(model_id="sentence-transformers/all-MiniLM-L6-v2")


def test_sentence_transformer_dependency_classification_missing_torch(
    monkeypatch, tmp_path: Path
) -> None:
    def _raise_file(*_args, **_kwargs) -> None:
        raise RuntimeError("not reached")

    _install_fake_sentence_transformers(monkeypatch, constructor_behaviour=_raise_file)
    monkeypatch.setattr(models, "_package_available", lambda name: True)
    original_import = models.builtins.__import__

    def _import(name: str, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(models.builtins, "__import__", _import)
    with pytest.raises(RuntimeError, match=r"teacher_dependency_unavailable:torch"):
        SentenceTransformerModel(model_id="sentence-transformers/all-MiniLM-L6-v2")


def test_sentence_transformer_dependency_classification_missing_cache(
    monkeypatch, tmp_path: Path
) -> None:
    def _raise_cache_missing(*_args, **_kwargs) -> None:
        raise FileNotFoundError("Cannot find file in local cache")

    _install_fake_sentence_transformers(monkeypatch, constructor_behaviour=_raise_cache_missing)
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(models, "_package_available", lambda name: True)
    with pytest.raises(RuntimeError, match=r"teacher_not_available_from_local_cache"):
        SentenceTransformerModel(model_id="sentence-transformers/all-MiniLM-L6-v2")


def test_sentence_transformer_dependency_classification_cached_model_missing(
    monkeypatch, tmp_path: Path
) -> None:
    def _raise_cached_missing(*_args, **_kwargs) -> None:
        raise OSError(
            "No sentence-transformer model found with name sentence-transformers/all-MiniLM-L6-v2. "
            "Couldn't connect to 'https://huggingface.co' and couldn't find it in the cached files."
        )

    _install_fake_sentence_transformers(monkeypatch, constructor_behaviour=_raise_cached_missing)
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(models, "_package_available", lambda name: True)
    with pytest.raises(RuntimeError, match=r"teacher_not_available_from_local_cache"):
        SentenceTransformerModel(model_id="sentence-transformers/all-MiniLM-L6-v2")


def test_sentence_transformer_dependency_classification_runtime_import_error(
    monkeypatch, tmp_path: Path
) -> None:
    underlying = RuntimeError("boom during import")

    def _raise_runtime(*_args, **_kwargs):
        raise underlying

    _install_fake_sentence_transformers(monkeypatch, constructor_behaviour=_raise_runtime)
    _install_fake_torch(monkeypatch)
    monkeypatch.setattr(models, "_package_available", lambda name: True)
    with pytest.raises(RuntimeError, match=r"teacher_runtime_import_error") as exc:
        SentenceTransformerModel(model_id="sentence-transformers/all-MiniLM-L6-v2")
    assert exc.value.__cause__ is underlying


@pytest.mark.parametrize(
    ("reason", "execution_status", "exit_code"),
    [
        ("teacher_dependency_unavailable:sentence-transformers", "BLOCKED", 3),
        ("teacher_dependency_unavailable:torch", "BLOCKED", 3),
        ("teacher_not_available_from_local_cache: missing", "BLOCKED", 3),
        ("teacher_runtime_import_error: boom", "EXECUTION_ERROR", 2),
    ],
)
def test_teacher_failures_remain_outside_scientific_decisions(
    monkeypatch, tmp_path: Path, reason: str, execution_status: str, exit_code: int
) -> None:
    actual_exit_code, payload, _ = _run_blocking_cli(monkeypatch, tmp_path, reason=reason)
    assert actual_exit_code == exit_code
    assert payload["status"] == execution_status
    assert payload["execution_status"] == execution_status
    assert payload["reason"] == reason
    assert payload["real_teacher_executed"] is False
    assert payload["measurement_integrity_status"] is None
    assert payload["scientific_decision"] is None


def test_sentence_transformer_manifest_records_explicit_controls_and_cache(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.bin").write_text("x", encoding="utf-8")
    init_calls, _, _ = _make_fake_sentence_transformers_model(
        monkeypatch,
        model_path=snapshot,
        output_factory=lambda texts, kwargs: np.ones((len(texts), 4), dtype=np.float32),
    )
    _install_fake_torch(monkeypatch)

    model = SentenceTransformerModel(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        cache_only=True,
        normalize_embeddings=False,
        output_dtype="float32",
        prompt_name=None,
        prompt=None,
        max_sequence_length=42,
    )
    manifest = model.manifest()

    assert init_calls[-1]["local_files_only"] is True
    assert init_calls[-1]["device"] == "cpu"
    assert manifest["cache_only"] is True
    assert manifest["requested_device"] == "cpu"
    assert manifest["resolved_device"] == "cpu"
    assert manifest["normalize_embeddings"] is False
    assert manifest["output_dtype"] == "float32"
    assert manifest["max_sequence_length"] == 42
    assert manifest["evaluation_mode"] is True
    assert model._model.max_seq_length == 42


def test_sentence_transformer_manifest_has_full_inventory_and_deterministic_order(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for idx in range(90):
        (snapshot / f"file_{idx:03d}.txt").write_text(f"{idx}", encoding="utf-8")
    _install_fake_torch(monkeypatch)
    _make_fake_sentence_transformers_model(
        monkeypatch,
        model_path=snapshot,
        output_factory=lambda texts, kwargs: np.ones((len(texts), 4), dtype=np.float32),
    )

    one = SentenceTransformerModel(
        "sentence-transformers/all-MiniLM-L6-v2", device="cpu", cache_only=True
    ).manifest()
    two = SentenceTransformerModel(
        "sentence-transformers/all-MiniLM-L6-v2", device="cpu", cache_only=True
    ).manifest()

    file_order = [row["relative_path"] for row in one["model_files"]]
    assert len(file_order) >= 90
    assert file_order == sorted(file_order)
    assert one["model_files"] == two["model_files"]
    assert one["hashed_file_count"] == 90
    assert one["snapshot_content_sha256"] == two["snapshot_content_sha256"]


def test_sentence_transformer_rejects_non_fp32_output_contract(monkeypatch, tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.bin").write_text("x", encoding="utf-8")
    _install_fake_torch(monkeypatch)
    _make_fake_sentence_transformers_model(
        monkeypatch,
        model_path=snapshot,
        output_factory=lambda texts, kwargs: np.ones((len(texts), 4), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="requires output_dtype=float32"):
        SentenceTransformerModel("sentence-transformers/all-MiniLM-L6-v2", output_dtype="float64")


def test_sentence_transformer_fails_closed_on_snapshot_hash_error(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.bin").write_text("x", encoding="utf-8")
    _install_fake_torch(monkeypatch)
    _make_fake_sentence_transformers_model(
        monkeypatch,
        model_path=snapshot,
        output_factory=lambda texts, kwargs: np.ones((len(texts), 4), dtype=np.float32),
    )

    def _fail_hash(_path: str):
        raise OSError("unreadable")

    monkeypatch.setattr(models, "_hash_file", _fail_hash)
    with pytest.raises(RuntimeError, match="teacher_provenance_hash_error:model.bin"):
        SentenceTransformerModel("sentence-transformers/all-MiniLM-L6-v2")


def test_sentence_transformer_file_mutation_changes_manifest_hash(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    target = snapshot / "tokenizer.json"
    target.write_text("first", encoding="utf-8")
    _install_fake_torch(monkeypatch)
    _make_fake_sentence_transformers_model(
        monkeypatch,
        model_path=snapshot,
        output_factory=lambda texts, kwargs: np.ones((len(texts), 4), dtype=np.float32),
    )

    first = SentenceTransformerModel(
        "sentence-transformers/all-MiniLM-L6-v2", device="cpu", cache_only=True
    ).manifest()
    before = hashlib.sha256(canonical_json_bytes(first)).hexdigest()

    target.write_text("changed", encoding="utf-8")
    second = SentenceTransformerModel(
        "sentence-transformers/all-MiniLM-L6-v2", device="cpu", cache_only=True
    ).manifest()
    after = hashlib.sha256(canonical_json_bytes(second)).hexdigest()

    assert before != after


def test_sentence_transformer_encoding_validation_fails_for_invalid_outputs(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "one.bin").write_text("x", encoding="utf-8")
    _install_fake_torch(monkeypatch)

    def invalid_2d(texts, kwargs):
        del kwargs
        return np.ones((1, 4), dtype=np.float32)

    _make_fake_sentence_transformers_model(
        monkeypatch, model_path=snapshot, output_factory=invalid_2d
    )
    model = SentenceTransformerModel(
        "sentence-transformers/all-MiniLM-L6-v2", device="cpu", cache_only=True
    )
    with pytest.raises(RuntimeError, match="row count mismatch"):
        model.encode(["q1", "q2"], batch_size=2)

    def invalid_finite(texts, kwargs):
        del kwargs
        return np.array([[1.0, np.nan, 2.0, 3.0]], dtype=np.float32)

    _make_fake_sentence_transformers_model(
        monkeypatch, model_path=snapshot, output_factory=invalid_finite
    )
    model = SentenceTransformerModel(
        "sentence-transformers/all-MiniLM-L6-v2", device="cpu", cache_only=True
    )
    with pytest.raises(RuntimeError, match="contains NaN or infinity"):
        model.encode(["q1"], batch_size=1)


def test_sentence_transformer_encoding_validation_fails_on_dimension_churn(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "one.bin").write_text("x", encoding="utf-8")
    _install_fake_torch(monkeypatch)

    state = {"i": 0}

    def variable_dims(texts, kwargs):
        i = state["i"]
        state["i"] = i + 1
        if i == 0:
            return np.ones((len(texts), 4), dtype=np.float32)
        return np.ones((len(texts), 5), dtype=np.float32)

    _make_fake_sentence_transformers_model(
        monkeypatch, model_path=snapshot, output_factory=variable_dims
    )
    model = SentenceTransformerModel(
        "sentence-transformers/all-MiniLM-L6-v2", device="cpu", cache_only=True
    )

    _ = model.encode(["q1"], batch_size=1)
    with pytest.raises(RuntimeError, match="embedding dimension changed"):
        model.encode(["q2"], batch_size=1)


def test_environment_manifest_includes_teacher_dependencies() -> None:
    manifest = build_environment_manifest()
    for dep in [
        "sentence-transformers",
        "transformers",
        "huggingface-hub",
        "tokenizers",
        "safetensors",
    ]:
        assert dep in manifest["dependencies"]


def test_m0_replay_does_not_construct_sentence_transformer(monkeypatch, tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for idx in range(3):
        (snapshot / f"f{idx}.txt").write_text(f"{idx}", encoding="utf-8")

    _install_fake_torch(monkeypatch)
    _make_fake_sentence_transformers_model(
        monkeypatch,
        model_path=snapshot,
        output_factory=lambda texts, kwargs: np.full(
            (len(texts), 4), fill_value=0.5, dtype=np.float32
        ),
    )

    config_path = _write_teacher_config(tmp_path)
    run_summary = run_m0(config_path=config_path, output_root=tmp_path / "runs")
    replay_path = Path(run_summary["run_dir"]) / "replay-bundle.json"
    assert replay_path.exists()

    def _raise_constructor(*_args, **_kwargs):
        raise RuntimeError("should not be instantiated in replay path")

    import neural_continuity.cli as cli_module

    monkeypatch.setattr(cli_module, "SentenceTransformerModel", _raise_constructor)
    replay = run_m0_replay(replay_path)
    assert replay["status_match"] is True
    assert replay["control_outcome_match"] is True
    assert replay["measurement_integrity_status"] == run_summary["measurement_integrity_status"]
