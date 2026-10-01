from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_continuity.evidence import (
    sha256_file,
    sha256_lf_normalized_file,
)
from neural_continuity.m1_b.decision_package import _contracts
from neural_continuity.m1_teacher_evidence import TeacherEvidenceError

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_A = ROOT / "contracts" / "m1-transition-a-v1.json"
CONTRACT_B = ROOT / "contracts" / "m1-transition-b-v1.json"
EXPECTED_A_SHA256 = "772e0df5133de09f6108cb42144e9b2ee69e47c0694bdf5b60ca4d88c18ee5c4"


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_transition_b_inheritance_uses_lf_normalized_transition_a(
    tmp_path: Path, newline: bytes
) -> None:
    source_a = CONTRACT_A.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    source_b = CONTRACT_B.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    path_a = tmp_path / "transition-a.json"
    path_b = tmp_path / "transition-b.json"
    path_a.write_bytes(source_a.replace(b"\n", newline))
    path_b.write_bytes(source_b.replace(b"\n", newline))

    _, _, b_sha256, a_sha256 = _contracts(path_b, path_a)

    assert a_sha256 == EXPECTED_A_SHA256
    assert a_sha256 == sha256_lf_normalized_file(path_a)
    assert b_sha256 == sha256_file(path_b)


def test_transition_b_inheritance_rejects_non_newline_contract_changes(
    tmp_path: Path,
) -> None:
    source_a = CONTRACT_A.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    source_b = CONTRACT_B.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    path_a = tmp_path / "transition-a.json"
    path_b = tmp_path / "transition-b.json"
    path_a.write_bytes(source_a.replace(b"{", b'{"unrecognized_change":true,', 1))
    path_b.write_bytes(source_b)

    assert json.loads(path_a.read_text(encoding="utf-8"))["contract_id"] == "m1-transition-a-v1"
    assert sha256_lf_normalized_file(path_a) != EXPECTED_A_SHA256
    with pytest.raises(TeacherEvidenceError) as error:
        _contracts(path_b, path_a)

    assert error.value.status == "BLOCKED"
    assert error.value.code == "TRANSITION_B_TOLERANCE_AUTHORITY_INVALID"
