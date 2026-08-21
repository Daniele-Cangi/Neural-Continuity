"""Frozen authority verification that must precede every diagnostic graph load."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

AuthorityRole = Literal[
    "onnx_fp32_source",
    "onnx_int8_candidate",
    "calibration_manifest",
    "paired_fp32_evidence",
    "int8_target_evidence",
    "transition_b_decision",
    "transition_a_contract",
    "transition_b_v1_contract",
]

FROZEN_AUTHORITY_SHA256: Mapping[AuthorityRole, str] = {
    "onnx_fp32_source": "5c0d999bd6b5e64e36cad1f61a83ef8e7507d55be49086745780fabb7c648511",
    "onnx_int8_candidate": "8b28688438e249c42b523e276333a3a009ca30d0754a3ba6fcbb10d76de873e5",
    "calibration_manifest": "3ac7d68e01976ee444217cd80c5b4b7338f870d8c0ab5a350a960495baef0778",
    "paired_fp32_evidence": "cf03882df0913e84b456b61f02a1c00a14ec151cd0fd9cc07f7d0bf04745b4df",
    "int8_target_evidence": "4027c1edf9f24254e6174ca79bc722c98758c8f97f5ad175b380866f64063a80",
    "transition_b_decision": "eed7d7af553ae9aa77274104cc75f348de910df464d836272ab37e8760e78d4e",
    "transition_a_contract": "772e0df5133de09f6108cb42144e9b2ee69e47c0694bdf5b60ca4d88c18ee5c4",
    "transition_b_v1_contract": "ad8c04574b3121eb69028e89f98f81cd1a68c34f15ecc23f9dc85c66b45273b0",
}

_ROLE_ORDER: tuple[AuthorityRole, ...] = (
    "onnx_fp32_source",
    "onnx_int8_candidate",
    "calibration_manifest",
    "paired_fp32_evidence",
    "int8_target_evidence",
    "transition_b_decision",
    "transition_a_contract",
    "transition_b_v1_contract",
)
_VERIFICATION_SEAL = object()


class DiagnosticPreflightError(RuntimeError):
    """Structured fail-closed diagnostic failure."""

    def __init__(
        self,
        *,
        status: Literal["BLOCKED", "EXECUTION_ERROR"],
        code: str,
        message: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


@dataclass(frozen=True)
class FrozenAuthorityPaths:
    onnx_fp32_source: Path
    onnx_int8_candidate: Path
    calibration_manifest: Path
    paired_fp32_evidence: Path
    int8_target_evidence: Path
    transition_b_decision: Path
    transition_a_contract: Path
    transition_b_v1_contract: Path

    def path_for(self, role: AuthorityRole) -> Path:
        return cast(Path, getattr(self, role))


@dataclass(frozen=True)
class VerifiedAuthority:
    role: AuthorityRole
    path: Path
    sha256: str
    size_bytes: int
    verification_method: str = "byte_sha256"

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "verification_method": self.verification_method,
        }


@dataclass(frozen=True)
class VerifiedAuthoritySet:
    authorities: tuple[VerifiedAuthority, ...]
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._seal is not _VERIFICATION_SEAL:
            raise ValueError("VerifiedAuthoritySet can only be created by authority verification")

    def authority_for(self, role: AuthorityRole) -> VerifiedAuthority:
        for authority in self.authorities:
            if authority.role == role:
                return authority
        raise DiagnosticPreflightError(
            status="BLOCKED",
            code="FROZEN_AUTHORITY_NOT_VERIFIED",
            message=f"Frozen authority was not verified: {role}",
            details={"role": role},
        )

    def assert_complete(self) -> None:
        roles = tuple(authority.role for authority in self.authorities)
        if roles != _ROLE_ORDER or self._seal is not _VERIFICATION_SEAL:
            raise DiagnosticPreflightError(
                status="BLOCKED",
                code="FROZEN_AUTHORITY_SET_INCOMPLETE",
                message="The complete ordered frozen authority set has not passed verification",
                details={"verified_roles": list(roles)},
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "m1_transition_b_v2_diagnostic_authority",
            "status": "PASS",
            "authority_count": len(self.authorities),
            "all_authorities_verified": True,
            "authorities": [authority.to_dict() for authority in self.authorities],
            "model_execution_used": False,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_lf_normalized_file(path: Path) -> str:
    normalized = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def verify_frozen_authority_set(paths: FrozenAuthorityPaths) -> VerifiedAuthoritySet:
    """Verify every frozen authority without importing or loading ONNX."""

    verified: list[VerifiedAuthority] = []
    for role in _ROLE_ORDER:
        path = paths.path_for(role).resolve()
        if not path.is_file():
            raise DiagnosticPreflightError(
                status="BLOCKED",
                code="FROZEN_AUTHORITY_MISSING",
                message=f"Frozen authority is missing: {role}",
                details={"role": role, "path": str(path)},
            )
        is_contract = role in {"transition_a_contract", "transition_b_v1_contract"}
        actual_sha256 = _sha256_lf_normalized_file(path) if is_contract else _sha256_file(path)
        verification_method = "lf_normalized_sha256" if is_contract else "byte_sha256"
        expected_sha256 = FROZEN_AUTHORITY_SHA256[role]
        if actual_sha256 != expected_sha256:
            raise DiagnosticPreflightError(
                status="BLOCKED",
                code="FROZEN_AUTHORITY_HASH_MISMATCH",
                message=f"Frozen authority hash mismatch: {role}",
                details={
                    "role": role,
                    "path": str(path),
                    "expected_sha256": expected_sha256,
                    "actual_sha256": actual_sha256,
                },
            )
        verified.append(
            VerifiedAuthority(
                role=role,
                path=path,
                sha256=actual_sha256,
                size_bytes=path.stat().st_size,
                verification_method=verification_method,
            )
        )

    result = VerifiedAuthoritySet(authorities=tuple(verified), _seal=_VERIFICATION_SEAL)
    result.assert_complete()
    return result
