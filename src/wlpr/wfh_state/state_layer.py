"""Layer 1 — mesh x date x slot WFH human-state layer.

Ported verbatim (numerics unchanged) from the source
`configured asset`.

Static per-mesh census state (mesh_state, ported verbatim from the proven H1 mesh_params: non-circular
T001193 occupation + T001108 households + housing heat-loss -> p_home0/vacancy_margin/N_persons/H_wfh)
PLUS the date/intra-day drivers that turn that static state into a mesh x date x slot panel:
  - occupancy_profiles(): marginal commuter-away occupancy shape (date-independent; the §3 SHAPE driver)
  - darkness_for_doy():   civil-twilight darkness for a given day-of-year (lighting driver)
  - workday_factor():     calendar_registry workday/holiday gate (dWFH = 0 on non-workdays -> weekend placebo)

NO kWh-derived denominator; mobility/time-use are PRIORS not load. See WFH_STATE_LAYER_V2_SPEC.md.
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location
import os
import duckdb
import numpy as np
import pandas as pd

from wlpr.io.paths import input_path, derived_path

# Derived inputs are produced by the upstream census and housing builders.
T93 = derived_path(_asset_location('asset_0125'), "t001193_mesh_industry_occupation_selected.csv")
T08 = derived_path(_asset_location('asset_0125'), "t001108_mesh_employment_commute_selected.csv")
OCC = derived_path(_asset_location('asset_0125'), "wfh_s6_occupation_telework_rates.csv")
HOUSE = derived_path(_asset_location('asset_0107'), "housing_mesh_panel.parquet")
TU = input_path("timeuse_15min", must_exist=False)
CAL = derived_path(_asset_location('asset_0150'), "calendar_registry.parquet")

# occupation columns (t001193) -> wfh_s6 occupation codes  (verbatim from H1)
OCC_MAP = {"occupation_A_managerial": "01", "occupation_B_professional_technical": "02",
           "occupation_C_clerical": "03", "occupation_D_sales": "04", "occupation_E_service": "05",
           "occupation_F_security": "06", "occupation_G_agriculture_fishery": "07",
           "occupation_H_production_process": "08", "occupation_I_transport_machine": "09",
           "occupation_J_construction_mining": "10", "occupation_K_carrying_cleaning_packaging": "11"}

I0 = 0.349          # current intensity (telework day-share among teleworkers)
Q_NW, Q_WK = 0.85, 0.85   # daytime home-presence allowance (household-saturation, H0_SOURCE_AUDIT)

# WFH marginal-occupancy commuter-away window (verbatim from H1; the §3 SHAPE that fills the daytime valley)
LEAVE_START, LEAVE_END = 8.0, 13.0
RETURN_START, RETURN_END = 15.5, 18.5
LAT = 35.6          # Kanto latitude for daylight geometry


def mesh_state():
    """Per-mesh NON-CIRCULAR census state (ported from H1 mesh_params). Returns a DataFrame with
    pref, workers, teleworkers, H_worker, H_nonworker, H_total, p_home0, vacancy_margin, pi_worker0,
    workers_per_hh, heatloss, N_persons(S1), H_wfh(S1)."""
    con = duckdb.connect(); con.execute("SET threads=4")
    occ = pd.read_csv(OCC); occ["category_code"] = occ.category_code.astype(str).str.zfill(2)
    rate = dict(zip(occ.category_code, occ.telework_rate))
    t93 = pd.read_csv(T93); t93["mesh"] = t93.mesh.astype("int64").astype(str)
    occ_cols = [c for c in OCC_MAP if c in t93.columns]
    t93["workers"] = t93[occ_cols].sum(axis=1)
    t93["teleworkers"] = sum(t93[c] * rate.get(OCC_MAP[c], 0.0) for c in occ_cols)
    t93 = t93[["mesh", "workers", "teleworkers", "non_labor_force_total", "labor_force_total"]]
    t08 = pd.read_csv(T08); t08["mesh"] = t08.mesh_code.astype("int64").astype(str)
    t08 = t08[["mesh", "non_ag_worker_households", "non_worker_households", "resident_student_total"]]
    hp = con.execute(f"select cast(mesh as varchar) mesh, cast(pref as varchar) pref, heatloss "
                     f"from read_parquet('{HOUSE}')").df()
    con.close()
    m = hp.merge(t93, on="mesh", how="inner").merge(t08, on="mesh", how="inner")
    for c in ["workers", "teleworkers", "non_labor_force_total", "labor_force_total",
              "non_ag_worker_households", "non_worker_households", "resident_student_total", "heatloss"]:
        m[c] = pd.to_numeric(m[c], errors="coerce").fillna(0.0)
    m = m[(m.workers > 0) & (m.non_ag_worker_households > 0)].copy()
    m["H_worker"] = m.non_ag_worker_households
    m["H_nonworker"] = m.non_worker_households
    m["H_total"] = m.H_worker + m.H_nonworker
    H = np.clip(m.H_total.values, 1, None)
    adults = m.labor_force_total.values + m.non_labor_force_total.values
    persons_per_hh = adults / H
    workers_per_hh_res = m.workers.values / H
    students_per_hh = m.resident_student_total.values / H
    f_nw = np.clip((persons_per_hh - workers_per_hh_res - students_per_hh) /
                   np.clip(persons_per_hh, 1e-9, None), 0, 1)
    m["p_home0"] = np.clip((m.H_nonworker.values * Q_NW + m.H_worker.values * f_nw * Q_WK) / H, 0, 1)
    m["vacancy_margin"] = 1.0 - m["p_home0"]
    m["pi_worker0"] = np.clip(m.teleworkers / np.clip(m.workers, 1, None) * I0, 0, 1)
    m["workers_per_hh"] = m.workers / m.H_worker
    m["N_persons"] = m.workers * m.pi_worker0
    m["H_wfh"] = m.H_worker * (1.0 - np.power(1.0 - m.pi_worker0, m.workers_per_hh))
    return m.reset_index(drop=True)


def occupancy_profiles():
    """Marginal commuter-away occupancy shapes (date-independent; the §3 SHAPE driver). Ported verbatim
    from H1 occupancy_profiles. Returns (work48, lunch48, presence01)."""
    tu = pd.read_parquet(TU)
    wk = tu[tu.day_type == "weekday"]
    work = np.clip(wk[wk.activity == "work"].sort_values("slot").rate_pct.values, 0, None)
    hod = np.arange(96) / 4.0
    work_c = work * ((hod >= 6.5) & (hod <= 19.5)); work_c = work_c / work_c.sum()
    lw = (hod >= 11.0) & (hod <= 14.0)
    lunch_c = np.where(lw, 0.5 * (1 - np.cos(2 * np.pi * np.clip((hod - 11.0) / 3.0, 0, 1))), 0.0)
    lunch_c = lunch_c / lunch_c.sum()
    to48 = lambda v: v.reshape(48, 2).sum(axis=1)
    work48, lunch48 = to48(work_c), to48(lunch_c)
    kk = np.hanning(7); kk = kk / kk.sum()
    work48 = np.convolve(np.concatenate([work48[:3][::-1], work48, work48[-3:][::-1]]), kk, "same")[3:-3]
    work48 = work48 / work48.sum()
    hod48 = np.arange(48) / 2.0
    xr = np.clip((hod48 - LEAVE_START) / (LEAVE_END - LEAVE_START), 0.0, 1.0)
    rise = 0.5 * (1.0 - np.cos(np.pi * xr))
    xf = np.clip((hod48 - RETURN_START) / (RETURN_END - RETURN_START), 0.0, 1.0)
    fall = 0.5 * (1.0 + np.cos(np.pi * xf))
    conv = rise * fall
    marg48 = work48 * conv
    marg48 = marg48 / marg48.sum()
    presence01 = marg48 / marg48.max()
    return marg48, lunch48, presence01


def darkness_for_doy(doy):
    """Civil-twilight darkness (0 midday -> 1 night) for a given day-of-year (per-DATE generalisation of
    H1 dark48(month)). Smooth ~0.75h dawn/dusk so lighting ramps, not steps."""
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + int(doy)) / 365.0)
    w0 = np.arccos(np.clip(-np.tan(np.deg2rad(LAT)) * np.tan(decl), -1, 1))
    half = w0 * 12 / np.pi; noon = 12.0 - (139.7 - 135.0) / 15.0
    t = np.arange(48) / 2.0 + 0.25
    ta = noon - half + 0.5; td = noon + half - 0.5; tw = 0.75
    day = 0.5 * (1 + np.tanh((t - ta) / tw)) * 0.5 * (1 + np.tanh((td - t) / tw))
    return np.clip(1.0 - day, 0.0, 1.0)


def workday_factor(dates):
    """Registered study-date activity gate; no silent weekday-only fallback."""
    from wlpr.io.activity_calendar import workday_factors
    return workday_factors(dates)


def _dow_workday(ymd):
    """Fallback: Mon-Fri = workday, Sat/Sun = 0 (used only when the calendar registry is absent)."""
    ts = pd.Timestamp(int(str(ymd)[:4]), int(str(ymd)[4:6]), int(str(ymd)[6:8]))
    return 1.0 if ts.dayofweek < 5 else 0.0


def state_panel_explicit():
    """Materialise the EXPLICIT human-state probabilities (plan 15.3/15.14 state_mesh_date_slot), factored:
    per-mesh STATIC state + per-slot STATE SHAPES. Each component is driven by a named state, making the
    state -> component mapping auditable (not just a load curve).

      per-mesh static:  p_worker, p_eligible_wfh(=pi_worker0), p_home0, p_marginal_home(=vacancy_margin)
      per-slot shapes:  p_working_at_home(work48), p_marginal_occupancy(marg/presence01),
                        p_lunch_at_home(lunch48)   [p_lighting_need/p_hvac_control add darkness/weather at date grain]

    Returns (mesh_static_df, slot_shapes_df). The full mesh x date x slot panel is these x date weather/darkness
    (reconstructible on demand; we do NOT dump it — same factored discipline as the component layer)."""
    m = mesh_state()
    work48, lunch48, presence01 = occupancy_profiles()
    H = np.clip(m.H_total.values, 1, None)
    residents = m.labor_force_total.values + m.non_labor_force_total.values
    mesh_static = pd.DataFrame({
        "mesh": m.mesh.values, "pref": m.pref.values,
        "p_worker": np.clip(m.workers.values / np.clip(residents, 1, None), 0, 1),
        "p_eligible_wfh": m.pi_worker0.values,            # P(a worker WFHs on the rep-day) = participation x intensity
        "p_home0": m.p_home0.values,                       # P(>=1 member already home, counterfactual)
        "p_marginal_home": m.vacancy_margin.values,        # 1 - p_home0 (the WFH-marginal occupancy weight)
        "workers_per_hh": m.workers_per_hh.values,
    })
    slot = np.arange(48)
    slot_shapes = pd.DataFrame({
        "slot": slot,
        "p_working_at_home": work48,                       # office_device driver (per person)
        "p_marginal_occupancy": presence01,                # lighting/HVAC occupancy driver (per household)
        "p_lunch_at_home": lunch48,                        # lunch event driver
    })
    return mesh_static, slot_shapes
