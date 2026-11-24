"""wfh_state cluster gates — the cell-native a_dwfh WFH-increment engine.

Import/interface gates run always; data-run gates run only when the inputs are staged (skipif), so the suite
stays green pre-staging and proves the chain (closure / S1 anchor / G_IDENT cell≡factored) once data lands.
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_wfh_state_modules_import_and_expose_entrypoints():
    from wlpr.wfh_state import (state_layer, climate_response, component_loads,
                                scenario_downscaling, household_cells, materialize, emit_adwfh)
    assert callable(state_layer.mesh_state)
    assert callable(state_layer.occupancy_profiles)
    assert callable(state_layer.darkness_for_doy)
    assert callable(state_layer.workday_factor)
    assert callable(climate_response.g_hdd) and callable(climate_response.g_cdd)
    assert callable(climate_response.daily_tmean)
    assert callable(component_loads.subcomponents_date)
    assert callable(component_loads.subcomponents_date_cells)
    assert callable(component_loads.aggregate_cells_to_mesh)
    assert callable(component_loads.cell_identity_report)
    assert callable(scenario_downscaling.scenario_state)
    assert callable(scenario_downscaling.scenario_multipliers)
    assert callable(household_cells.build_cells) and callable(household_cells.ipf_2x2)
    assert callable(materialize.emit_main) and callable(materialize.emit_full_year)
    assert callable(materialize.anchor_G) and callable(materialize.mon_slot_from_factors)
    assert callable(emit_adwfh.materialize_a_dwfh)


def test_scenario_taxonomy_is_ten_and_anchor_constants_preserved():
    """The S1..S10 taxonomy + the load-bearing numerics (survey anchor, intensity) must be byte-preserved."""
    from wlpr.wfh_state import scenario_downscaling as SD
    from wlpr.wfh_state import emit_adwfh
    ids = [s[0] for s in SD.SCEN]
    assert ids == [f"S{i}" for i in range(1, 11)]
    assert SD.I_BASE == 0.349
    assert SD.I_FREQ == {"observed": 0.349, "2-3 days": 0.50, "3-4 days": 0.70, "high": 0.70}
    assert emit_adwfh.SURVEY_CENTRAL_S1_TOTAL_MWH == 12534.9
    assert emit_adwfh.OBS_MONTHS == [10, 11, 12, 1, 2]


def test_unit_powers_and_window_constants_preserved():
    from wlpr.wfh_state import component_loads as CL
    from wlpr.wfh_state import climate_response as CR
    assert (CL.DEVICE_KW, CL.LIGHT_KW, CL.LUNCH_KWH, CL.HEAT_KW, CL.COOL_KW, CL.WORK_H) == \
        (0.090, 0.050, 0.30, 0.260, 0.340, 8.0)
    assert CL.COMP == ["office_device", "lighting", "lunch", "heating", "cooling", "other"]
    assert (CR.HDD_FULL, CR.CDD_FULL, CR.CDD_P, CR.HUMIDEX_K) == (12.0, 8.0, 1.5, 3.0)


def test_emit_adwfh_excludes_btm_twin():
    """emit_adwfh ports ONLY materialize_a_dwfh — never the BTM/eta/least/kappa twin half."""
    src = (ROOT / "src/wlpr/wfh_state/emit_adwfh.py").read_text(encoding="utf-8")
    assert "materialize_a_dwfh" in src
    for forbidden in ["build_future_userstate_ledger", "row_sql", "eta_shrunk", "KAPPA_SELF",
                      "least(", "theta_marg", "owner_BTM"]:
        assert forbidden not in src, forbidden


def test_inputs_registered_for_wfh_state():
    from wlpr.io import paths
    idx = paths._inputs_index()
    for iid in ["timeuse_15min", "weather_amedas"]:
        assert iid in idx, iid


# --- data-gated run gates: skip until the timeuse frozen input + the derived census CSVs are staged ---
_TU = (ROOT / _asset_location('asset_0186')).exists()
_T93 = (ROOT / _asset_location('asset_0187')).exists()
_HOUSE = (ROOT / _asset_location('asset_0188')).exists()
_STAGED = _TU and _T93 and _HOUSE


@pytest.mark.skipif(not _STAGED, reason="timeuse frozen input + derived census CSVs not staged yet")
def test_scenario_state_S1_identity_and_conservation():
    """S1 returns N_persons / H_wfh unchanged (identity-exact); every scenario conserves the per-pref total."""
    import numpy as np
    from wlpr.wfh_state import state_layer as SL, scenario_downscaling as SD
    state = SL.mesh_state()
    cfg = SD.scenario_multipliers(); mm = dict(zip(cfg.scenario_id, cfg.wfh_energy_multiplier))
    ds1 = SD.scenario_state(state, "S1", mm)
    drift_np = float((ds1.set_index("mesh").N_persons - state.set_index("mesh").N_persons).abs().max())
    drift_hw = float((ds1.set_index("mesh").H_wfh - state.set_index("mesh").H_wfh).abs().max())
    assert drift_np < 1e-9 and drift_hw < 1e-9
    for sid in [s[0] for s in SD.SCEN]:
        ds = SD.scenario_state(state, sid, mm)
        tgt = (state.assign(Ng=state.workers * np.clip(state.pi_worker0 * mm[sid], 0, 1))
               .groupby("pref").Ng.sum())
        got = ds.groupby("pref").N_persons.sum()
        assert float((got - tgt).abs().max()) < 1e-6, sid


@pytest.mark.skipif(not _STAGED, reason="timeuse frozen input + derived census CSVs not staged yet")
def test_household_cells_reconcile_and_g_ident():
    """IPF cells reconcile to mesh_state (S1-safe) and the cell-native engine ≡ the mesh-aggregate engine."""
    from wlpr.wfh_state import component_loads as CL
    ok, tbl, _d = CL.cell_identity_report(month=1)
    assert ok, tbl.to_string(index=False)
    assert float(tbl.max_abs_delta_mesh_slot.max()) < 1e-9


@pytest.mark.skipif(not (_STAGED and (ROOT / _asset_location('asset_0189')).exists()),
                    reason="v2 a_dwfh not emitted yet (run pipeline/06)")
def test_emit_adwfh_window_and_anchor():
    """The canonical emit keeps window months only and reproduces the S1 survey-central anchor."""
    from wlpr.wfh_state import emit_adwfh
    out = emit_adwfh.materialize_a_dwfh(["S1"])
    assert set(int(x) for x in out.mon.unique()) <= set(emit_adwfh.OBS_MONTHS)
    s1tot = float(out[out.scenario_id == "S1"].dwfh.sum())
    assert abs(s1tot - emit_adwfh.SURVEY_CENTRAL_S1_TOTAL_MWH) < 1.0
    for k in [f"c_{c}" for c in ["office_device", "lighting", "lunch", "heating", "cooling", "other"]]:
        assert k in out.columns
