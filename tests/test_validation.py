"""Validation-suite gates. Import/interface gates run always; data-run gates run only when the frozen inputs +
the derived component product are present (skipif), so the suite stays green pre-staging and proves the chain
post-staging. The headline assertion is system daily NMAE ~0.039 / rep-day (monthly-profile) ~0.023."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_validation_modules_import_and_expose_entrypoints():
    from wlpr.validation import (decompose, qa_dashboard, midday_diagnosis,
                                 midday_explainability, mobility_midday_test,
                                 component_aware_calibration, annual_generation)
    assert callable(decompose.run_decomposition) and callable(decompose.run_gates)
    assert callable(qa_dashboard.build_dashboard)
    assert callable(midday_diagnosis.diagnose_midday)
    assert callable(midday_explainability.run_explainability)
    assert callable(mobility_midday_test.run_mobility_test)
    assert callable(component_aware_calibration.run_calibration)
    assert callable(annual_generation.run_annual_generation)


def test_figure_modules_import_and_expose_entrypoints():
    from wlpr.figures import (main_validation, ablation, grid_validation,
                              annual, mobility_validation, manifest)
    assert callable(main_validation.build_main_figure)
    assert callable(ablation.build_ablation_figure)
    assert callable(grid_validation.build_grid_figure)
    assert callable(annual.build_annual_figure)
    assert callable(mobility_validation.build_mobility_figure)
    assert callable(manifest.write_figure_manifest)


def test_w1_conformal_fold_module_and_scorecards():
    """W1 (forward-driver conformal/fold uncertainty) is the producer of the two scorecards consumed by
    scenario/increment_bands. Import gate always; scorecard structure gate when built."""
    from wlpr.validation import w1_conformal_fold
    assert callable(w1_conformal_fold.build_w1_conformal_fold) and callable(w1_conformal_fold.run_gates)
    src = (ROOT / "src/wlpr/validation/w1_conformal_fold.py").read_text(encoding="utf-8")
    # clean-port contract: HDD/CDD bases come from the package config (model_config), dropping the excluded
    # forward-PV l1_weather_daily median derivation (numerically exact: source medians ARE 18.0/24.0)
    assert 'cfg["weather"]["hdd_base_c"]' in src and 'cfg["weather"]["cdd_base_c"]' in src
    cov = ROOT / "outputs/scorecards/w1_conformal_coverage.csv"
    if cov.exists():
        import pandas as pd
        c = pd.read_csv(cov)
        pooled = c[c.heldout == "ALL"]
        assert set(["pref", "q80_band", "coverage", "nmae"]).issubset(c.columns)
        assert 0.70 <= float(pooled.coverage.mean()) <= 0.92  # split-conformal 80% target band
        assert (pooled.q80_band > 0).all()


def test_product_read_routed_through_derived_not_outpath():
    """The component product is written by step 02 to data/derived/products/; every consumer must read it via
    derived_path (NOT out_path), or the figures/NMAE will read an empty path."""
    for rel in ["src/wlpr/validation/decompose.py",
                "src/wlpr/validation/component_aware_calibration.py",
                "src/wlpr/figures/main_validation.py"]:
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert 'derived_path("products", "grid_component_daily_product.parquet")' in src, rel
        assert 'out_path("products"' not in src, rel


def test_no_excluded_userstate_in_validation():
    """The validation cluster must not pull in the excluded PV/BTM user-state twin."""
    for rel in ["src/wlpr/validation/annual_generation.py",
                "src/wlpr/validation/decompose.py",
                "src/wlpr/figures/annual.py"]:
        src = (ROOT / rel).read_text(encoding="utf-8")
        for forbidden in ["theta_marg", "future_userstate", "eta_", "least(", "kappa"]:
            assert forbidden not in src, f"{rel}: {forbidden}"


def test_targets_and_mobility_input_registered():
    from wlpr.io import paths
    assert os.path.exists(os.path.join(paths.CONFIGS, "targets.yaml"))
    idx = paths._inputs_index()
    assert "mobility_presence_panel" in idx
    for iid in ["closure_surface_share", "census_housing_T001194", "weather_amedas",
                "lv_load_hourly", "mesh_denominator_panel", "demographic_mesh_panel"]:
        assert iid in idx, iid


def test_qa_targets_yaml_parses_and_has_aggregate_daily_nmae():
    from wlpr.io.paths import _load_yaml
    T = _load_yaml("targets.yaml")
    assert T["aggregate"]["daily_nmae"]["warn"] == 0.04
    assert T["aggregate"]["daily_nmae"].get("floor") is True


# data-gated end-to-end gates ---------------------------------------------------------------------
_FROZEN = (ROOT / "data/frozen/locked_baselines/_share_tmp.parquet").exists()
_PRODUCT = (ROOT / "data/derived/products/grid_component_daily_product.parquet").exists()


@pytest.mark.skipif(not (_FROZEN and _PRODUCT),
                    reason="frozen closure surface + derived component product not staged (run steps 02 + stage_raw)")
def test_decomposition_headline_nmae():
    from wlpr.validation.decompose import run_decomposition
    out = run_decomposition()
    sysrow = out["by_level"][out["by_level"].grain == "system"].iloc[0]
    repday = float(out["headline"].monthly_profile_nmae.iloc[0])
    assert sysrow.nmae <= 0.05, f"system daily NMAE {sysrow.nmae:.4f} (~0.039 expected)"
    assert repday <= 0.04, f"rep-day NMAE {repday:.4f} (~0.023 expected)"


@pytest.mark.skipif(not (_FROZEN and _PRODUCT), reason="frozen + product not staged")
def test_qa_dashboard_no_fail():
    from wlpr.validation.decompose import run_decomposition
    from wlpr.validation.qa_dashboard import build_dashboard
    run_decomposition()
    qa = build_dashboard()
    assert (qa.status == "FAIL").sum() == 0, qa[qa.status == "FAIL"][["category", "metric", "value"]].to_dict("records")
