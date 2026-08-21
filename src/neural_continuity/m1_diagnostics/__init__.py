"""Fail-closed, model-free diagnostics for M1 Transition B."""

from neural_continuity.m1_diagnostics.authority import (
    DiagnosticPreflightError,
    FrozenAuthorityPaths,
    VerifiedAuthoritySet,
    verify_frozen_authority_set,
)

__all__ = [
    "DiagnosticPreflightError",
    "FrozenAuthorityPaths",
    "VerifiedAuthoritySet",
    "verify_frozen_authority_set",
]
