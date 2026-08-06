from __future__ import annotations

import numpy as np
import pytest

from neural_continuity.m1_b.static_quantization import StaticCalibrationReader
from neural_continuity.m1_teacher_evidence import TeacherEvidenceError


class _Tokenizer:
    def tokenize(self, texts: list[str]) -> dict[str, np.ndarray]:
        return {
            "input_ids": np.asarray([[index + 1, 0] for index, _ in enumerate(texts)]),
            "attention_mask": np.asarray([[1, 0] for _ in texts]),
        }


class _IncompleteTokenizer:
    def tokenize(self, texts: list[str]) -> dict[str, np.ndarray]:
        return {"input_ids": np.asarray([[1] for _ in texts])}


def test_static_calibration_reader_emits_canonical_batches() -> None:
    reader = StaticCalibrationReader(_Tokenizer(), ["first", "second", "third"], 2)

    first = reader.get_next()
    second = reader.get_next()

    assert first is not None
    assert first["input_ids"].tolist() == [[1, 0], [2, 0]]
    assert first["token_type_ids"].tolist() == [[0, 0], [0, 0]]
    assert second is not None
    assert second["attention_mask"].tolist() == [[1, 0]]
    assert reader.get_next() is None


def test_static_calibration_reader_blocks_missing_attention_mask() -> None:
    reader = StaticCalibrationReader(_IncompleteTokenizer(), ["only"], 1)

    with pytest.raises(TeacherEvidenceError) as error:
        reader.get_next()

    assert error.value.code == "ONNX_TOKENIZATION_INVALID"
