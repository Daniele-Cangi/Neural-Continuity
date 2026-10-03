"""Frozen M1-B v2 H1-H4 diagnostic labels; not transition decisions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from neural_continuity.m1_diagnostics.authority import DiagnosticPreflightError
from neural_continuity.m1_diagnostics.metric_arithmetic import binary64_bits, binary64_hex

_HYPOTHESIS_FAMILIES = {
    "H1_NORMALIZATION_SENSITIVITY": "NORMALIZATION",
    "H2_ATTENTION_OR_MATMUL_SENSITIVITY": "ATTENTION_OR_MATMUL",
    "H4_OUTPUT_PATH_QUANTIZATION": "OUTPUT_PATH",
}


def _blocked(code: str, message: str) -> DiagnosticPreflightError:
    return DiagnosticPreflightError(status="BLOCKED", code=code, message=message)


def _family_status(
    boundaries: Sequence[str], families: Mapping[str, Sequence[str]], family: str
) -> str:
    matches = [family in families[probe_id] for probe_id in boundaries]
    if all(matches):
        return "SUPPORTED"
    if not any(matches):
        return "NOT_SUPPORTED"
    return "UNRESOLVED"


def evaluate_hypotheses_v2(
    *,
    localization: Mapping[str, Any],
    boundary_families: Mapping[str, Sequence[str]],
    saturation_boundary_ids: Sequence[str],
    saturation: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the preregistered H1-H4 rules to canonical replay outputs."""
    probe_scores = localization.get("probe_scores")
    dominant = localization.get("dominant_onset_boundary_probe_ids")
    maximum_growth_hex = localization.get("maximum_growth")
    if not isinstance(probe_scores, Mapping) or not isinstance(dominant, list):
        raise _blocked(
            "HYPOTHESIS_LOCALIZATION_MISSING", "Replay localization evidence is incomplete"
        )
    if set(boundary_families) != set(probe_scores):
        raise _blocked(
            "HYPOTHESIS_FAMILY_SET_MISMATCH",
            "Every planned probe requires structural family labels",
        )
    if not isinstance(maximum_growth_hex, str):
        raise _blocked(
            "HYPOTHESIS_LOCALIZATION_INVALID", "Maximum growth must be a hexadecimal string"
        )
    try:
        maximum_growth = float.fromhex(maximum_growth_hex)
    except (TypeError, ValueError) as exc:
        raise _blocked(
            "HYPOTHESIS_LOCALIZATION_INVALID", "Maximum growth is not hexadecimal binary64"
        ) from exc
    if not math.isfinite(maximum_growth) or binary64_hex(maximum_growth) != maximum_growth_hex:
        raise _blocked(
            "HYPOTHESIS_LOCALIZATION_INVALID", "Maximum growth is not finite canonical binary64"
        )
    if any(probe_id not in probe_scores for probe_id in dominant):
        raise _blocked(
            "HYPOTHESIS_LOCALIZATION_INVALID", "Dominant boundary references an unknown probe"
        )

    saturation_ids = tuple(sorted(saturation_boundary_ids))
    if not saturation_ids or len(set(saturation_ids)) != len(saturation_ids):
        raise _blocked(
            "HYPOTHESIS_SATURATION_BOUNDARIES_INVALID",
            "Quantized activation boundary identities must be unique",
        )
    if set(saturation) != set(saturation_ids) or any(
        value not in probe_scores for value in saturation_ids
    ):
        raise _blocked(
            "HYPOTHESIS_SATURATION_SET_MISMATCH",
            "Saturation evidence must cover every planned quantized boundary",
        )

    normalized_saturation: dict[str, dict[str, Any]] = {}
    for boundary_id in saturation_ids:
        record = saturation[boundary_id]
        measurable = record.get("measurable")
        if not isinstance(measurable, bool):
            raise _blocked(
                "HYPOTHESIS_SATURATION_INVALID", "Saturation measurability must be explicit"
            )
        if not measurable:
            if any(
                record.get(key) is not None
                for key in ("aggregate_count", "aggregate_total", "aggregate_fraction")
            ):
                raise _blocked(
                    "HYPOTHESIS_SATURATION_INVALID",
                    "Unmeasurable saturation cannot declare aggregate values",
                )
            normalized_saturation[boundary_id] = {
                "measurable": False,
                "aggregate_count": None,
                "aggregate_total": None,
                "aggregate_fraction": None,
            }
            continue
        count = record.get("aggregate_count")
        total = record.get("aggregate_total")
        fraction_hex = record.get("aggregate_fraction")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or isinstance(total, bool)
            or not isinstance(total, int)
            or total <= 0
            or count > total
        ):
            raise _blocked(
                "HYPOTHESIS_SATURATION_INVALID",
                "Saturation counts must satisfy 0 <= count <= total",
            )
        if not isinstance(fraction_hex, str):
            raise _blocked(
                "HYPOTHESIS_SATURATION_INVALID", "Saturation fraction must be a hexadecimal string"
            )
        try:
            fraction = float.fromhex(fraction_hex)
        except (TypeError, ValueError) as exc:
            raise _blocked(
                "HYPOTHESIS_SATURATION_INVALID", "Saturation fraction must be hexadecimal binary64"
            ) from exc
        if (
            not math.isfinite(fraction)
            or not 0.0 <= fraction <= 1.0
            or binary64_hex(fraction) != fraction_hex
            or binary64_hex(float(count) / float(total)) != fraction_hex
        ):
            raise _blocked(
                "HYPOTHESIS_SATURATION_INVALID", "Saturation count and fraction are inconsistent"
            )
        normalized_saturation[boundary_id] = {
            "measurable": True,
            "aggregate_count": count,
            "aggregate_total": total,
            "aggregate_fraction": fraction_hex,
        }

    if maximum_growth == 0.0:
        hypotheses = {
            name: "UNRESOLVED" for name in (*_HYPOTHESIS_FAMILIES, "H3_CALIBRATION_RANGE_MISMATCH")
        }
        maximum_saturation_ids: list[str] = []
    else:
        hypotheses = {
            name: _family_status(dominant, boundary_families, family)
            for name, family in _HYPOTHESIS_FAMILIES.items()
        }
        all_measurable = all(normalized_saturation[value]["measurable"] for value in saturation_ids)
        all_zero = all(
            normalized_saturation[value]["aggregate_count"] == 0 for value in saturation_ids
        )
        if all_measurable and all_zero:
            h3_status = "NOT_SUPPORTED"
            maximum_saturation_ids = list(saturation_ids)
        elif not all_measurable:
            h3_status = "UNRESOLVED"
            maximum_saturation_ids = []
        else:
            fractions = {
                value: float.fromhex(normalized_saturation[value]["aggregate_fraction"])
                for value in saturation_ids
            }
            maximum_fraction = max(fractions.values())
            maximum_bits = binary64_bits(maximum_fraction)
            maximum_saturation_ids = sorted(
                value
                for value, fraction in fractions.items()
                if fraction == maximum_fraction and binary64_bits(fraction) == maximum_bits
            )
            intersects = bool(set(maximum_saturation_ids).intersection(dominant))
            h3_status = "SUPPORTED" if maximum_fraction != 0.0 and intersects else "UNRESOLVED"
        hypotheses["H3_CALIBRATION_RANGE_MISMATCH"] = h3_status

    return {
        "protocol": "M1_TRANSITION_B_V2",
        "localization_status": localization.get("localization_status"),
        "dominant_onset_boundary_probe_ids": sorted(set(dominant)),
        "maximum_saturation_boundary_ids": maximum_saturation_ids,
        "saturation": normalized_saturation,
        "hypotheses": {name: {"status": status} for name, status in hypotheses.items()},
        "model_execution_used": False,
        "transition_decision_emitted": False,
    }
