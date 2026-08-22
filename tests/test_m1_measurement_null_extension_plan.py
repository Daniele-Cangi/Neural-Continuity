from __future__ import annotations

import json
from pathlib import Path

from neural_continuity.m1_diagnostics.measurement_null_extension_plan import (
    DISJOINT_RESTART_PAIRS,
    PROCESS_EPOCHS,
    build_measurement_null_extension_plan,
)


class _AuthorityStub:
    provenance_bundle_path = Path("runtime-provenance/replay-bundle.json")
    provenance_manifest_sha256 = "a" * 64
    runtime_inventory: dict[str, object] = {}
    runtime_audit: dict[str, object] = {}
    provenance_replay: dict[str, object] = {}

    def __getattr__(self, name: str) -> object:
        if "path" in name or "bundle" in name:
            return self.provenance_bundle_path
        if "sha" in name:
            return self.provenance_manifest_sha256
        if "inventory" in name or "audit" in name or "replay" in name:
            return {}
        raise AttributeError(name)


def test_extension_plan_freezes_sampling_and_claim_limits() -> None:
    plan = build_measurement_null_extension_plan(_AuthorityStub())  # type: ignore[arg-type]
    encoded = json.dumps(plan, sort_keys=True)

    assert '"process_epoch_count": 120' in encoded
    assert PROCESS_EPOCHS == 120
    assert DISJOINT_RESTART_PAIRS == 60
    assert 2 * PROCESS_EPOCHS * 4 == 960
    assert '"ninety_fifth_percentile_claim_supported": true' in encoded
    assert '"prediction_interval_language_allowed": false' in encoded
    assert '"model_execution_used_for_preregistration": false' in encoded
    assert '"stage_1_execution_allowed": false' in encoded


def test_extension_plan_excludes_candidate_and_holdout_authority() -> None:
    plan = build_measurement_null_extension_plan(_AuthorityStub())  # type: ignore[arg-type]
    encoded = json.dumps(plan, sort_keys=True)

    assert '"measurement_null"' in encoded
    assert '"candidate_or_holdout_result_selected_design": false' in encoded
    assert '"holdout_query_access_allowed": false' in encoded
    assert '"early_stopping_allowed": false' in encoded
    assert '"adaptive_sample_size_allowed": false' in encoded
