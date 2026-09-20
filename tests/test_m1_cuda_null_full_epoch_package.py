from __future__ import annotations

import json

import pytest

from neural_continuity.evidence import canonical_json_bytes
from neural_continuity.m1_diagnostics.cuda_null_full_epoch_format import RUN_LAYOUT
from neural_continuity.m1_diagnostics.cuda_null_full_epoch_package import (
    FullEpochPackageBlocked,
    _validate_retrieval_run_maps,
)


def test_retrieval_maps_accept_canonical_json_key_order() -> None:
    labels = [label for label, _batch in RUN_LAYOUT]
    rankings = json.loads(canonical_json_bytes({label: [] for label in labels}))
    metrics = json.loads(canonical_json_bytes({label: {} for label in labels}))
    assert list(rankings) != labels
    _validate_retrieval_run_maps(rankings, metrics)


def test_retrieval_maps_reject_missing_run() -> None:
    labels = [label for label, _batch in RUN_LAYOUT]
    rankings = {label: [] for label in labels[:-1]}
    metrics = {label: {} for label in labels}
    with pytest.raises(FullEpochPackageBlocked, match="retrieval run set differs"):
        _validate_retrieval_run_maps(rankings, metrics)
