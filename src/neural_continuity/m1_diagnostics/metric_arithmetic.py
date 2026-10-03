"""Canonical binary64 arithmetic for M1-B v2 diagnostic metrics."""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable

from neural_continuity.m1_diagnostics.authority import DiagnosticPreflightError


def _blocked(code: str, message: str) -> DiagnosticPreflightError:
    return DiagnosticPreflightError(status="BLOCKED", code=code, message=message)


def pairwise_sum(values: Iterable[float]) -> float:
    """Sum binary64 values with the protocol's fixed adjacent-pair tree."""
    level = [float(value) for value in values]
    if any(not math.isfinite(value) for value in level):
        raise _blocked("METRIC_ARITHMETIC_NONFINITE", "Pairwise sum received a non-finite value")
    while len(level) > 1:
        next_level: list[float] = []
        for index in range(0, len(level) - 1, 2):
            value = level[index] + level[index + 1]
            if not math.isfinite(value):
                raise _blocked("METRIC_ARITHMETIC_NONFINITE", "Pairwise sum became non-finite")
            next_level.append(value)
        if len(level) % 2:
            next_level.append(level[-1])
        level = next_level
    return level[0] if level else 0.0


def pairwise_mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise _blocked("METRIC_EMPTY_VALID_SET", "Cannot average an empty valid element set")
    result = pairwise_sum(materialized) / float(len(materialized))
    if not math.isfinite(result):
        raise _blocked("METRIC_ARITHMETIC_NONFINITE", "Mean became non-finite")
    return result


def l2_norm(values: Iterable[float]) -> float:
    squares: list[float] = []
    for value in values:
        square = float(value) * float(value)
        if not math.isfinite(square):
            raise _blocked("METRIC_ARITHMETIC_NONFINITE", "Squared value became non-finite")
        squares.append(square)
    norm = math.sqrt(pairwise_sum(squares))
    if not math.isfinite(norm):
        raise _blocked("METRIC_ARITHMETIC_NONFINITE", "L2 norm became non-finite")
    return norm


def cosine_similarity(source: Iterable[float], target: Iterable[float]) -> float:
    left = [float(value) for value in source]
    right = [float(value) for value in target]
    if len(left) != len(right) or not left:
        raise _blocked("METRIC_COSINE_SHAPE_INVALID", "Cosine vectors must have equal nonzero size")
    left_zero = all(value == 0.0 for value in left)
    right_zero = all(value == 0.0 for value in right)
    if left_zero and right_zero:
        return 1.0
    if left_zero or right_zero:
        return 0.0
    dot = pairwise_sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = l2_norm(left)
    right_norm = l2_norm(right)
    denominator = left_norm * right_norm
    if not math.isfinite(denominator):
        raise _blocked("METRIC_ARITHMETIC_NONFINITE", "Cosine denominator became non-finite")
    if denominator == 0.0:
        raise _blocked("METRIC_COSINE_UNDERFLOW", "Nonzero cosine vector norm underflowed to zero")
    result = dot / denominator
    if not math.isfinite(result):
        raise _blocked("METRIC_ARITHMETIC_NONFINITE", "Cosine similarity became non-finite")
    return result


def binary64_hex(value: float) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise _blocked("METRIC_ARITHMETIC_NONFINITE", "Cannot serialize a non-finite metric")
    return numeric.hex().lower()


def binary64_bits(value: float | str) -> int:
    numeric = float.fromhex(value) if isinstance(value, str) else float(value)
    if not math.isfinite(numeric):
        raise _blocked("METRIC_ARITHMETIC_NONFINITE", "Cannot compare non-finite metric bits")
    return int.from_bytes(struct.pack("<d", numeric), byteorder="little", signed=False)
