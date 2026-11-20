"""Phase 2 baseline-engine gates. Import/interface gates run always; data-run gates run only when the
frozen inputs are staged (skipif), so the suite stays green pre-staging and proves the chain post-staging."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_baseline_modules_import_and_expose_entrypoints():
    from wlpr.baseline import component_product, survey_anchor, ledger_p0
    assert callable(component_product.build_component_product)
    assert callable(component_product.run_gates)
    assert callable(survey_anchor.apply_survey_anchor)
    assert callable(ledger_p0.build_baseline_p0)
    # the load-fit WFH weight is industry wfh_eligible × wage_share (the documented census basis)
    src = (ROOT / "src/wlpr/baseline/component_product.py").read_text(encoding="utf-8")
    assert "wfh_eligible" in src and "wage_share" in src


def test_inputs_registered_for_baseline():
    from wlpr.io import paths
    idx = paths._inputs_index()
    for iid in ["closure_surface_share", "census_housing_T001194", "demographic_mesh_panel",
                "mesh_denominator_panel", "weather_amedas", "lv_load_hourly", "wfh_survey_pref_rates"]:
        assert iid in idx, iid


def test_ledger_keeps_Gpv_excludes_btm_userstate():
    """baseline_p0 keeps metered G_pv (for net-load) but must not pull in the BTM user-state twin."""
    src = (ROOT / "src/wlpr/baseline/ledger_p0.py").read_text(encoding="utf-8")
    assert "G_pv" in src
    for forbidden in ["eta", "kappa", "future_userstate", "theta_marg", "least("]:
        assert forbidden not in src, forbidden


_SCORECARD = ROOT / "outputs/scorecards/grid_component_scorecard.csv"


@pytest.mark.skipif(not _SCORECARD.exists(), reason="component product not built yet (run pipeline/02)")
def test_component_product_closes_to_load():
    """READ-ONLY check of the built product's scorecard — must NOT rebuild (a smoke rebuild would clobber the
    full production product). Asserts the load-fit closes to load exactly."""
    import pandas as pd
    sc = pd.read_csv(_SCORECARD).set_index("metric").value
    assert float(sc["agg_reldiff"]) < 1e-6
    assert float(sc["per_row_max_abs_load_minus_components"]) < 1e-6
    assert float(sc["null_component_rows"]) == 0 and float(sc["negative_component_rows"]) == 0
