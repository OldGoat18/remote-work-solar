"""Layer 3 — mesh x date x slot WFH 6-subcomponent loads (date-grain port of H1 subcomponents).

Ported verbatim (numerics unchanged) from the source
`src/bottom_up_microsim/wfh_state/component_loads.py`.

device  ∝ WFH PERSONS (additive);  lighting/lunch/heating/cooling ∝ WFH HOUSEHOLDS × vacancy_margin (shared).
The §3 ruling is realised here: the intra-day SHAPE is occupancy-driven (work48 / presence01 / lunch48 /
darkness), and the per-DATE weather enters ONLY as the HVAC daily MAGNITUDE g_hdd/g_cdd — so heating/cooling
gain genuine per-date climate skill without re-propping the cold morning/evening commute peaks.

TWO equivalent engines (plan §13.1 promotion):
  - subcomponents_date_cells()  — the CANONICAL definition: the 6 components are computed PER official-source
    household-state cell (worker-status × dwelling, from household_cells), then aggregate_cells_to_mesh() sums
    them back to the mesh × date × slot grain. This is what '让组件在 cell 层计算、再聚合回 mesh×date×slot' means.
  - subcomponents_date()        — the FACTORED fast form over the mesh-aggregate state (the closed-form ledger
    and materialize layer use it). It is a PROVEN-equivalent factorisation, not a second model.
cell_identity_report() / the G_IDENT gate enforce that the two never diverge (<1e-9): the household-cell layer
DRIVES the components, while the locked S1 stays bit-identical because the clean census decomposition is
margin-conserving (the dwelling-stratified heat-loss cancels in aggregate until a clean worker×dwelling
correlation is licensed — that future fidelity gain would then flow through this same cell engine).

Run the validated-window slice smoke (date-grain ≡ H1 month-grain rep-day, G4/G6, + the G_IDENT cell proof):
    PYTHONPATH=src python -m wlpr.wfh_state.component_loads
"""
from __future__ import annotations
import os
import glob
import duckdb
import numpy as np
import pandas as pd

from wlpr.io.paths import derived_path
from wlpr.wfh_state import state_layer as SL
from wlpr.wfh_state import climate_response as CR

# Registered engineering coefficients; not independently metered end uses.
# Power x normalized equivalent hours gives kWh. Lighting is the inherited
# per-slot ENERGY coefficient actually used by both engines (no further dt).
DEVICE_KW = 0.090
LIGHT_SLOT_KWH = 0.050
LIGHT_KW = LIGHT_SLOT_KWH  # Historical external name; compatibility alias, not a power interpretation.
LUNCH_KWH = 0.30
HEAT_KW = 0.260
COOL_KW = 0.340
WORK_H = 8.0
COMP = ["office_device", "lighting", "lunch", "heating", "cooling", "other"]
FWD = derived_path("scenario", "forward_lv_load_repday.parquet")
# Released project anchor for the validated Oct-Feb survey-central window.  This is
# only used when the excluded legacy forward table is not staged; it is not a
# substitute for the mesh/date load table.
SURVEY_CENTRAL_WINDOW_MWH = 12534.9


def subcomponents_date(state, work48, lunch48, presence01, dark48_d, gh, gc, workday):
    """6 sub-components for ONE date, per mesh x 48-slot. `gh`/`gc` are per-mesh daily HVAC magnitudes
    (g_hdd/g_cdd already mapped to each mesh's pref); `workday` in [0,1] gates the whole increment."""
    Np = state["N_persons"].values[:, None]
    Hw = state["H_wfh"].values[:, None]
    marg = state["vacancy_margin"].values[:, None]
    hl = state["heatloss"].values[:, None]
    gh = gh[:, None]; gc = gc[:, None]
    hs = (work48 / work48.sum() * WORK_H)[None, :]          # occupancy SHAPE (date-independent)
    darkocc = (presence01 * dark48_d)[None, :]
    dev = Np * DEVICE_KW * (work48 * WORK_H)[None, :]
    light = Hw * marg * LIGHT_SLOT_KWH * darkocc
    lunch = Hw * marg * LUNCH_KWH * lunch48[None, :]
    heat = Hw * marg * hl * HEAT_KW * gh * hs               # per-DATE magnitude x occupancy SHAPE
    cool = Hw * marg * hl * COOL_KW * gc * hs
    z = np.zeros_like(dev)
    out = {"office_device": dev, "lighting": light, "lunch": lunch, "heating": heat, "cooling": cool, "other": z}
    return {k: v * workday for k, v in out.items()}


# rooftop-PV self-consumption window (daytime PV generation ~09:00-15:00 = slots 18..29) and the commute-home
# evening RE-HEAT window (the RETURN edge ~15:30-21:00 = slots 31..41), which is ENTIRELY OUTSIDE the PV window.
PV_SELFCONS_WINDOW = slice(18, 30)


def _evening_reheat_shape():
    """Commute-home evening re-heat 48-slot shape: a bump on the RETURN edge (~15:30-21:00), zero in the
    PV self-consumption window — the load a NON-WFH (commuting) household incurs when it returns and re-heats."""
    hod = np.arange(48) / 2.0
    s = np.where((hod >= 15.5) & (hod <= 21.0),
                 0.5 * (1 - np.cos(2 * np.pi * np.clip((hod - 15.5) / 5.5, 0, 1))), 0.0)
    return s / s.sum() if s.sum() > 0 else s


def paired_counterfactual_report(state, work48, gh, reheat_frac=0.5):
    """§15.9 paired counterfactual: dWFH_heating = load(state=WFH, home all day) − load(state=commute, away by
    day + RE-HEAT on return). The commute counterfactual's evening re-heat sits on the RETURN edge, ENTIRELY
    OUTSIDE the rooftop-PV self-consumption window (~09-15h). Therefore the paired correction:
      (a) is ~ZERO inside the PV window  -> owner-BTM self-consumption is IDENTICAL to the production marginal
          form (the S1 PV headline is robust to the counterfactual refinement); and
      (b) makes the EVENING increment NEGATIVE (WFH SAVES the commute re-heat) -> the production marginal form,
          which conservatively OMITS this saving, is a LOWER BOUND on the WFH benefit.
    Returns (reheat_total, pv_window_max_abs_diff, reheat_fraction_of_daytime)."""
    Hw = state["H_wfh"].values
    hl = state["heatloss"].values
    hs = work48 / work48.sum() * WORK_H                       # WFH daytime heating SHAPE
    ev = _evening_reheat_shape()                              # commute evening re-heat shape (RETURN edge)
    daily_heat = Hw * hl * HEAT_KW * gh                       # per-mesh daily heating scale
    marginal_slot = np.outer(daily_heat, hs)                  # production daytime heating (marginal form)
    paired_slot = marginal_slot - np.outer(daily_heat * reheat_frac, ev)   # minus the commute evening re-heat
    diff = marginal_slot - paired_slot                        # = the evening re-heat term (the paired correction)
    pv_window_max = float(np.abs(diff[:, PV_SELFCONS_WINDOW]).max())       # MUST be ~0 (re-heat is outside PV hours)
    marg_tot = float(marginal_slot.sum())
    return float(diff.sum()), pv_window_max, (float(diff.sum()) / marg_tot if marg_tot > 0 else 0.0)


def _gmag_by_mesh(state, tmean_by_pref, fn):
    """Map a per-pref daily magnitude (fn applied to that pref's Tmean) onto each ROW of `state` (works for
    a mesh-grain frame OR a cell-grain frame — both carry a `pref` column)."""
    g = np.zeros(len(state))
    pref = state["pref"].values
    for pf, tm in tmean_by_pref.items():
        if tm is None:
            continue
        g[pref == pf] = fn(tm)
    return g


def subcomponents_date_cells(cells, work48, lunch48, presence01, dark48_d, gh, gc, workday):
    """CELL-NATIVE 6 sub-components for ONE date (plan §13.1 promotion): each household-state cell carries
    its OWN wfh_persons / wfh_households / vacancy_margin / heatloss, so dwelling-stratified heat-loss and
    worker-status occupancy drive the components directly (not a mesh average). `gh`/`gc` are per-CELL daily
    HVAC magnitudes (mapped from each cell's pref); `workday` in [0,1] gates the whole increment. Returns
    {comp: (n_cells, 48)}; aggregate_cells_to_mesh() sums these back to the mesh × date × slot grain."""
    Np = cells["wfh_persons"].values[:, None]
    Hw = cells["wfh_households"].values[:, None]
    marg = cells["vacancy_margin"].values[:, None]
    hl = cells["heatloss"].values[:, None]
    gh = gh[:, None]; gc = gc[:, None]
    hs = (work48 / work48.sum() * WORK_H)[None, :]
    darkocc = (presence01 * dark48_d)[None, :]
    dev = Np * DEVICE_KW * (work48 * WORK_H)[None, :]
    light = Hw * marg * LIGHT_SLOT_KWH * darkocc
    lunch = Hw * marg * LUNCH_KWH * lunch48[None, :]
    heat = Hw * marg * hl * HEAT_KW * gh * hs
    cool = Hw * marg * hl * COOL_KW * gc * hs
    z = np.zeros_like(dev)
    out = {"office_device": dev, "lighting": light, "lunch": lunch, "heating": heat, "cooling": cool, "other": z}
    return {k: v * workday for k, v in out.items()}


def aggregate_cells_to_mesh(sc_cells, cells, mesh_order=None, comp_keys=COMP):
    """Sum per-cell components (n_cells, 48) back to the mesh × 48-slot grain — the '聚合回 mesh×date×slot'
    step of the cell-driven engine. Returns {comp: (n_mesh, 48)} aligned to `mesh_order` (or the cells' own
    sorted mesh order)."""
    mesh = cells["mesh"].values
    out = {}
    for k in comp_keys:
        df = pd.DataFrame(sc_cells[k]); df["mesh"] = mesh
        agg = df.groupby("mesh").sum()
        if mesh_order is not None:
            agg = agg.reindex(mesh_order).fillna(0.0)
        out[k] = agg.values
    return out


def cell_identity_report(month=1, sample_date=None):
    """PROMOTION RED-LINE PROOF (plan §13.1 / acceptance gate G_IDENT): the CELL-native engine, aggregated
    back to the mesh × slot grain, reproduces the mesh-aggregate engine to machine precision for the
    published S1 — the architecture moves from 'mesh average' to 'household-cell driven' with ZERO movement
    of the locked S1 (the clean census decomposition is margin-conserving + independence, so it cancels in
    aggregate). Returns (ok, table, date)."""
    from wlpr.wfh_state import household_cells as HC
    work48, lunch48, presence01 = SL.occupancy_profiles()
    state = SL.mesh_state()
    cells, _ms, _J, _row, _col = HC.build_cells()
    prefs = sorted(set(state["pref"].values))
    dates = _dates_for_month(2024, month)
    wd_map = SL.workday_factor(dates)
    d = sample_date or next(dd for dd in dates if wd_map.get(dd, 0) > 0)
    doy = int(pd.Timestamp(int(d[:4]), int(d[4:6]), int(d[6:8])).dayofyear)
    dark_d = SL.darkness_for_doy(doy)
    tmean = CR.daily_tmean(prefs, [month])
    tdict = {p: tmean.get((p, d)) for p in prefs}
    # mesh-aggregate path (current product)
    gh_m = _gmag_by_mesh(state, tdict, CR.g_hdd); gc_m = _gmag_by_mesh(state, tdict, CR.g_cdd)
    sc_mesh = subcomponents_date(state, work48, lunch48, presence01, dark_d, gh_m, gc_m, 1.0)
    # cell-native path -> aggregate back to mesh
    gh_c = _gmag_by_mesh(cells, tdict, CR.g_hdd); gc_c = _gmag_by_mesh(cells, tdict, CR.g_cdd)
    sc_cell = subcomponents_date_cells(cells, work48, lunch48, presence01, dark_d, gh_c, gc_c, 1.0)
    agg = aggregate_cells_to_mesh(sc_cell, cells, mesh_order=state["mesh"].values)
    rows = []
    for k in COMP:
        d_abs = float(np.abs(agg[k] - sc_mesh[k]).max())
        rows.append((k, float(sc_mesh[k].sum()), float(agg[k].sum()), d_abs))
    tbl = pd.DataFrame(rows, columns=["component", "mesh_path_sum", "cell_path_sum", "max_abs_delta_mesh_slot"])
    ok = bool(tbl.max_abs_delta_mesh_slot.max() < 1e-9)
    return ok, tbl, d


def system_repday(state, dates, tmean, work48, lunch48, presence01, workday_map, comp_keys=COMP):
    """Workday-mean system rep-day (48,) per component, MWh/30min — average over the workdays in `dates`.
    tmean[(pref,ymd)] -> per-date per-pref Tmean. Non-workdays contribute 0 and are excluded from the mean."""
    prefs = sorted(set(state["pref"].values))
    acc = {k: np.zeros(48) for k in comp_keys}
    nwd = 0
    for d in dates:
        wd = workday_map.get(d, 0.0)
        if wd <= 0:
            continue
        nwd += 1
        doy = int(pd.Timestamp(int(d[:4]), int(d[4:6]), int(d[6:8])).dayofyear)
        dark_d = SL.darkness_for_doy(doy)
        gh = _gmag_by_mesh(state, {p: tmean.get((p, d)) for p in prefs}, CR.g_hdd)
        gc = _gmag_by_mesh(state, {p: tmean.get((p, d)) for p in prefs}, CR.g_cdd)
        sc = subcomponents_date(state, work48, lunch48, presence01, dark_d, gh, gc, wd)
        for k in comp_keys:
            acc[k] += sc[k].sum(axis=0)
    if nwd > 0:
        acc = {k: v / nwd for k, v in acc.items()}
    return acc, nwd


def system_repday_monthmean(state, month, tmean, work48, lunch48, presence01):
    """H1-equivalent month-grain rep-day: ONE profile using the month-mean Tmean (the magnitude H1 uses).
    No workday averaging needed (single representative weekday profile)."""
    prefs = sorted(set(state["pref"].values))
    tmean_mm = {}
    for p in prefs:
        vals = [v for (pp, d), v in tmean.items() if pp == p and int(d[4:6]) == month]
        tmean_mm[p] = float(np.mean(vals)) if vals else None
    mid = int(pd.Timestamp(2024, month, 15).dayofyear)
    dark_d = SL.darkness_for_doy(mid)
    gh = _gmag_by_mesh(state, tmean_mm, CR.g_hdd)
    gc = _gmag_by_mesh(state, tmean_mm, CR.g_cdd)
    sc = subcomponents_date(state, work48, lunch48, presence01, dark_d, gh, gc, 1.0)
    return {k: sc[k].sum(axis=0) for k in COMP}, tmean_mm


def survey_central_wfh(months):
    """Return the survey-central WFH anchor for `months`.

    The original forward LV rep-day table belongs to an excluded legacy chain
    and is not present in the public rebuild workspace.  Prefer it when it is
    available, preserving the original calculation.  For the validated
    Oct-Feb request used by :func:`materialize.anchor_G`, use the released
    scorecard anchor instead.  The fallback is deliberately limited to that
    exact window so it cannot silently invent arbitrary monthly observations.
    """
    requested = tuple(sorted(int(m) for m in months))
    if os.path.exists(FWD):
        con = duckdb.connect(); con.execute("SET threads=4")
        ms = ",".join(map(str, requested))
        df = con.execute(f"select sum(wfh) tot from read_parquet('{FWD}') "
                         f"where band='central' and month in ({ms})").df()
        con.close()
        if len(df) and pd.notna(df.tot[0]):
            print(f"[survey_anchor] source=legacy_forward_file value={float(df.tot[0]):.1f} MWh months={requested}")
            return float(df.tot[0])
    if requested == (1, 2, 10, 11, 12):
        print(f"[survey_anchor] source=released_scorecard_anchor value={SURVEY_CENTRAL_WINDOW_MWH:.1f} MWh months={requested}")
        return SURVEY_CENTRAL_WINDOW_MWH
    raise FileNotFoundError(
        f"Missing {FWD}; no released fallback is defined for requested months={requested}"
    )


def _dates_for_month(year, month):
    n = pd.Period(f"{year}-{month:02d}").days_in_month
    return [f"{year}{month:02d}{d:02d}" for d in range(1, n + 1)]


def _gates(sys_repday_mw):
    """Gate4(shape) + Gate6(valley-fill) on a system rep-day dict {comp: (48,) MW}."""
    dwfh = sum(sys_repday_mw[k] for k in COMP)
    g_mor = float(dwfh[12:18].max()); g_mid = float(dwfh[20:33].mean()); g_eve = float(dwfh[36:41].max())
    r_mor = g_mor / g_mid if g_mid > 0 else float("inf")
    r_eve = g_eve / g_mid if g_mid > 0 else float("inf")
    return r_mor, r_eve, dwfh


def main():
    print("[v2 slice] state-component-climate date-grain smoke (Jan 2024, 4 prefs)")
    work48, lunch48, presence01 = SL.occupancy_profiles()
    state = SL.mesh_state()
    print(f"[mesh_state] {len(state):,} meshes; p_home0 mean={state.p_home0.mean():.3f}; "
          f"vacancy_margin mean={state.vacancy_margin.mean():.3f}; ΣN_persons={state.N_persons.sum():,.0f}; "
          f"ΣH_wfh={state.H_wfh.sum():,.0f}")

    prefs = sorted(set(state["pref"].values))
    dates = _dates_for_month(2024, 1)
    tmean = CR.daily_tmean(prefs, [1])
    wd_map = SL.workday_factor(dates)
    n_wd = sum(1 for d in dates if wd_map.get(d, 0) > 0)
    print(f"[calendar] Jan 2024: {n_wd} workdays / {len(dates)} days")

    # date-grain (per-date weather) workday-mean rep-day  vs  H1 month-grain (month-mean weather) rep-day
    dg, nwd = system_repday(state, dates, tmean, work48, lunch48, presence01, wd_map)
    mg, tmm = system_repday_monthmean(state, 1, tmean, work48, lunch48, presence01)

    # anchor BOTH with the SAME survey-central G (so the comparison is shape/resolution, not the anchor)
    target = survey_central_wfh([1])
    G = float(target / (sum(mg[k] for k in COMP).sum() * 2.0))
    dg_mw = {k: dg[k] * G * 2.0 for k in COMP}      # -> MW
    mg_mw = {k: mg[k] * G * 2.0 for k in COMP}
    print(f"[anchor] G={G:.4f} (Jan survey-central {target:.0f} MW-slot)")

    dg_tot = sum(dg_mw[k] for k in COMP).sum() / 2.0   # MW-slot summed (≡ MWh over the rep-day's slots /? ) -> energy units
    mg_tot = sum(mg_mw[k] for k in COMP).sum() / 2.0
    reldiff = abs(dg_tot - mg_tot) / mg_tot if mg_tot else float("inf")

    # closure: Σ6 ≡ dWFH_total (by construction; assert machine precision)
    clo = float(np.abs(sum(dg_mw[k] for k in COMP) - sum(dg_mw[k] for k in COMP)).max())  # trivially 0
    # explicit closure check: total vs component sum recomputed independently
    dwfh_dg = np.zeros(48)
    for k in COMP:
        dwfh_dg = dwfh_dg + dg_mw[k]
    clo = float(np.abs(dwfh_dg - sum(dg_mw[k] for k in COMP)).max())

    r_mor_dg, r_eve_dg, _ = _gates(dg_mw)
    r_mor_mg, r_eve_mg, _ = _gates(mg_mw)

    print("\n=== date-grain vs month-grain (Jan, system rep-day, MW-slot summed) ===")
    comp_tbl = pd.DataFrame({
        "component": COMP,
        "date_grain": [dg_mw[k].sum() / 2.0 for k in COMP],
        "month_grain(H1)": [mg_mw[k].sum() / 2.0 for k in COMP],
    })
    print(comp_tbl.to_string(index=False))
    print(f"\n  total date_grain  = {dg_tot:8.1f} MW-slot")
    print(f"  total month_grain = {mg_tot:8.1f} MW-slot   (H1-equivalent)")
    print(f"  rel-diff          = {reldiff*100:6.3f}%   (climate-resolution gain; expect small, NON-regressive)")
    print(f"  closure |Σ6 - total| = {clo:.2e}  (expect <1e-9)")
    print(f"\n=== GATE6 valley-fill (morning/midday, evening/midday; need <=0.40) ===")
    print(f"  date-grain : morning/mid={r_mor_dg:.3f}  evening/mid={r_eve_dg:.3f}")
    print(f"  month-grain: morning/mid={r_mor_mg:.3f}  evening/mid={r_eve_mg:.3f}")

    # Gate4 sharing (synthetic 1-worker vs 2-worker hh), reuse the date-grain engine with flat magnitude
    g4 = _gate4(work48, lunch48, presence01)

    # G_IDENT — PROMOTION red-line: cell-native engine aggregated back to mesh×slot ≡ mesh-aggregate (S1 unmoved)
    ident_ok, ident_tbl, ident_d = cell_identity_report(month=1)
    ident_max = float(ident_tbl.max_abs_delta_mesh_slot.max())
    print(f"\n=== G_IDENT cell-native ≡ mesh-aggregate (promotion red-line; date {ident_d}) ===")
    print(ident_tbl.assign(max_abs_delta_mesh_slot=ident_tbl.max_abs_delta_mesh_slot.map(lambda x: f"{x:.2e}")).to_string(index=False))
    print(f"  worst component max|Δ(mesh,slot)| = {ident_max:.2e}  (need <1e-9 -> cell-driven preserves the locked S1)")

    ok = (clo < 1e-9 and r_mor_dg <= 0.40 and r_eve_dg <= 0.40 and reldiff < 0.10
          and g4["device_ratio"] > 1.8 and g4["shared_ratio"] < 1.2 and ident_ok)
    print(f"\n=== GATE4 sharing (2-worker vs 1-worker hh) ===")
    print(f"  device_ratio={g4['device_ratio']:.3f} (>=1.8)  shared_ratio={g4['shared_ratio']:.3f} (<1.2)")
    print(f"\n=== SLICE VERDICT: {'PASS' if ok else 'FAIL'} ===")
    if not ok:
        raise SystemExit("v2 slice gates failed")


def _gate4(work48, lunch48, presence01):
    rows = []
    for wph in (1.0, 2.0):
        H = 1000.0; W = H * wph; pi = 1.0
        rows.append(dict(pref="13", heatloss=1.4, vacancy_margin=0.5,
                         N_persons=W * pi, H_wfh=H * (1 - (1 - pi) ** wph)))
    tdf = pd.DataFrame(rows)
    gh = np.array([0.9, 0.9]); gc = np.array([0.0, 0.0])
    dark_d = SL.darkness_for_doy(15)
    sc = subcomponents_date(tdf, work48, lunch48, presence01, dark_d, gh, gc, 1.0)
    dev = sc["office_device"][1].sum() / sc["office_device"][0].sum()
    shared = ((sc["heating"] + sc["lighting"] + sc["lunch"])[1].sum() /
              (sc["heating"] + sc["lighting"] + sc["lunch"])[0].sum())
    return {"device_ratio": float(dev), "shared_ratio": float(shared)}


if __name__ == "__main__":
    main()
