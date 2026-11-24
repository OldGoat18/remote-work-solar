
from wlpr.io.bindings import asset as _asset_location
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / _asset_location('asset_0165')


def test_legacy_scope_builder_records_the_recovered_rule():
    source = (ROOT / "tools/build_legacy_study_scope.py").read_text(encoding="utf-8")
    assert "aggregate_use_observed" in source
    assert "dominant_pref_by_window_aggregate_use" in source
    assert "(34327, 34307, 20, 241)" in source


def test_materialized_legacy_scope_if_present():
    if not SCOPE.exists():
        return
    frame = pd.read_parquet(SCOPE)
    assert len(frame) == frame.mesh.nunique() == 34327
    assert (frame.aggregate_use_kwh > 0).sum() == 34307
    assert (frame.aggregate_use_kwh == 0).sum() == 20
    assert frame.pref.astype(str).isin(["11", "12", "13", "14"]).all()
