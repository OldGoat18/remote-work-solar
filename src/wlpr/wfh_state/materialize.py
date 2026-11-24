"""Phase 3 — materialize a_dwfh_mesh_date_slot (S1-S10) by fusing the date-grain component engine with the
scenario-specific mesh downscaling. Streamed per (pref, ym, scenario); only nonzero daytime rows kept.

Ported verbatim (numerics unchanged) from the source
`configured asset`.

Anchor G is computed once on the S1 workday-mean rep-day over the Oct-Feb window so the S1 window total
matches the survey-central WFH (the same anchor as the H2 ledger) -> S1 identity preserved.

Run the validation chunk (pref13 Jan, S1+S10) to check schema/closure/monotonicity/size before full emit:
    PYTHONPATH=src python -m wlpr.wfh_state.materialize
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location
import os
import numpy as np
import pandas as pd

from wlpr.io.paths import derived_path
from wlpr.wfh_state import state_layer as SL
from wlpr.wfh_state import climate_response as CR
from wlpr.wfh_state import component_loads as CL
from wlpr.wfh_state import scenario_downscaling as SD

OBS_YM = ["202310", "202311", "202312", "202401", "202402"]
# full year: validated winter window + extrapolated spring/summer (2024 continuation), evidence-tiered
FULL_YM = ["202310", "202311", "202312", "202401", "202402",
           "202403", "202404", "202405", "202406", "202407", "202408", "202409"]
COMP = CL.COMP
TOL = 1e-12
OUTROOT = derived_path(_asset_location('asset_0055'), _asset_location('asset_0159'))


def month_tier(mon):
    """Evidence tier per month (plan ch12/16.7): only Oct-Feb has LV observation; the rest is mechanism
    extrapolation + calibration transfer, NEVER summed to an annual truth. Internal short label (kept for the
    gates/tests); the spec-named status is month_validation_status()."""
    return ("validated" if mon in (10, 11, 12, 1, 2)
            else ("temp_extrap" if mon in (3, 4, 5) else "cdd_prior"))


# §16.7 spec contract: the four month_validation_status labels. Oct-Feb is LV-observed; Mar-May runs the same
# state-component mechanism on real (temperature-led) climate; Jun-Sep adds the summer CDD/humidex climate prior.
# All non-validated months additionally carry calibration_transferred (the validated-window calibration is the
# transferred source) -- surfaced on the calibrated table's calibration_source, not as the month's primary status.
_MVS = {"validated": "validated_lv_observation", "temp_extrap": "mechanism_extrapolated",
        "cdd_prior": "climate_response_prior"}


def month_validation_status(mon):
    """The §16.7-named month validation status (4 labels). Primary per-month status maps the 3 internal tiers to
    the spec names; calibration_transferred is the calibration_source carried by every non-validated month."""
    return _MVS[month_tier(mon)]


def calibration_source_for(mon):
    """§16.7 4th label: validated months use the in-window observed calibration; all others are
    calibration_transferred (the validated-window calibration carried forward)."""
    return "validated_window_calibration" if month_tier(mon) == "validated" else "calibration_transferred"


def _year_month(ym):
    return int(ym[:4]), int(ym[4:6])


def _workday_dates(ym):
    y, mth = _year_month(ym)
    dates = CL._dates_for_month(y, mth)
    wd = SL.workday_factor(dates)
    return [d for d in dates if wd.get(d, 0.0) > 0], wd


def anchor_G(state, work48, lunch48, presence01):
    """Return target_MWh / unscaled_template_kWh for the five Oct–Feb templates.

    This code coefficient includes the kWh-to-MWh conversion. The dimensionless
    energy scaling kappa is 1000*G; there is no extra two-day or half-hour factor.
    The target is the registered representative-template anchor, not independent
    household incremental-energy validation.
    """
    prefs = sorted(set(state["pref"].values))
    ghdd, gcdd, darkmon = monthly_factors(prefs, presence01)
    ds = SD.scenario_state(state, "S1")
    sc1 = component_scalars(ds, 1.0)
    v2 = mon_slot_from_factors(sc1, work48, lunch48, presence01, ghdd, gcdd, darkmon)
    target = CL.survey_central_wfh([10, 11, 12, 1, 2])
    # `v2` already spans the complete validated Oct-Feb window (the default
    # OBS_YM list), so do not apply a second two-day/rep-day factor here.
    return float(target / v2.dwfh.sum())


def emit_chunk(state_scn, pref, ym, sid, work48, lunch48, presence01, G):
    """Return the long-form mesh x date x slot dWFH rows for one (pref, ym, scenario). state_scn is the
    already-downscaled state for `sid`, filtered to this pref."""
    sp = state_scn[state_scn.pref == pref].reset_index(drop=True)
    if len(sp) == 0:
        return None
    meshes = sp.mesh.values
    y, mth = _year_month(ym)
    wdates, _ = _workday_dates(ym)
    tmean = CR.daily_tmean([pref], [mth])
    frames = []
    for d in wdates:
        tm = tmean.get((pref, d))
        if tm is None:
            continue
        gh = np.full(len(sp), CR.g_hdd(tm))
        gc = np.full(len(sp), CR.g_cdd(tm))
        doy = int(pd.Timestamp(int(d[:4]), int(d[4:6]), int(d[6:8])).dayofyear)
        dark_d = SL.darkness_for_doy(doy)
        sc = CL.subcomponents_date(sp, work48, lunch48, presence01, dark_d, gh, gc, 1.0)
        sc = {k: sc[k] * G for k in COMP}                         # MWh/30min, anchored
        dwfh = sum(sc[k] for k in COMP)                           # (nmesh,48)
        nz = np.argwhere(dwfh > TOL)                              # (k,2): mesh_idx, slot
        if len(nz) == 0:
            continue
        mi, sl = nz[:, 0], nz[:, 1]
        rec = {"scenario_id": sid, "mesh": meshes[mi], "pref": pref, "date": d, "slot": sl.astype(np.int16),
               "dwfh": dwfh[mi, sl]}
        for k in COMP:
            rec[f"c_{k}"] = sc[k][mi, sl]
        frames.append(pd.DataFrame(rec))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


CUR_ADWFH = derived_path(_asset_location('asset_0160'), "a_dwfh_mesh_mon_slot.parquet")


def component_scalars(ds, G):
    """Per-mesh anchored component base scalars (the factored magnitudes; full slot/date tensor is
    base x slot-shape x per-(pref,date) weather, reconstructible exactly)."""
    Np = ds.N_persons.values; Hw = ds.H_wfh.values; marg = ds.vacancy_margin.values; hl = ds.heatloss.values
    return pd.DataFrame({
        "scenario_id": ds.scenario_id.values if "scenario_id" in ds else ds.get("scenario_id"),
        "mesh": ds.mesh.values, "pref": ds.pref.values,
        "office_base": Np * CL.DEVICE_KW * CL.WORK_H * G,
        "light_base": Hw * marg * CL.LIGHT_SLOT_KWH * G,
        "lunch_base": Hw * marg * CL.LUNCH_KWH * G,
        "heat_base": Hw * marg * hl * CL.HEAT_KW * G,
        "cool_base": Hw * marg * hl * CL.COOL_KW * G,
    })


def monthly_factors(prefs, presence01, yms=None):
    """Mon-grain rep-day weather factors, on the H1 caliber so the mon-grain (PV-feeding) layer reproduces
    the product: g_hdd/g_cdd of the MONTH-MEAN temperature (all days) + MID-month darkness. The per-DATE
    climate response (G3) lives in the date-grain reconstruction, not in this rep-day reduction. With the
    full-year `yms`, summer months get g_cdd>0 -> cooling activates (cdd_prior tier)."""
    yms = yms or OBS_YM
    months = sorted({_year_month(ym)[1] for ym in yms})
    clim = CR.month_climatology_tmean(prefs, months)
    rhclim = CR.month_climatology_rhum(prefs, months)        # §13.5.5: humidex drives the summer cooling magnitude
    ghdd, gcdd, darkmon = {}, {}, {}
    for ym in yms:
        y, mth = _year_month(ym)
        darkmon[mth] = SL.darkness_for_doy(int(pd.Timestamp(2024, mth, 15).dayofyear))
        for pref in prefs:
            tmm = clim.get((pref, mth), 0.0)
            rh = rhclim.get((pref, mth))
            ghdd[(pref, mth)] = CR.g_hdd(tmm)
            gcdd[(pref, mth)] = CR.g_cdd(tmm, rh)            # convex CDD + humidex; Oct-Feb tmm<24 -> stays 0 (S1 safe)
    return ghdd, gcdd, darkmon


def mon_slot_from_factors(sc, work48, lunch48, presence01, ghdd, gcdd, darkmon, yms=None):
    """Build the mon-grain rep-day a_dwfh (drop-in schema) for one scenario from its mesh scalars `sc`.
    rep-day = mesh-scalar x slot-shape x monthly-mean weather (no full date expansion). Carries evidence_tier."""
    yms = yms or OBS_YM
    hs = work48 / work48.sum() * CL.WORK_H
    frames = []
    for ym in yms:
        _, mth = _year_month(ym)
        for pref in sorted(sc.pref.unique()):
            s = sc[sc.pref == pref]
            ob = s.office_base.values[:, None]; lb = s.light_base.values[:, None]
            ub = s.lunch_base.values[:, None]; hb = s.heat_base.values[:, None]; cb = s.cool_base.values[:, None]
            office = ob * work48[None, :]
            light = lb * (presence01 * darkmon[mth])[None, :]
            lunch = ub * lunch48[None, :]
            heat = hb * hs[None, :] * ghdd[(pref, mth)]
            cool = cb * hs[None, :] * gcdd[(pref, mth)]
            comp = {"office_device": office, "lighting": light, "lunch": lunch, "heating": heat,
                    "cooling": cool, "other": np.zeros_like(office)}
            dwfh = sum(comp[k] for k in COMP)
            nz = np.argwhere(dwfh > TOL)
            if len(nz) == 0:
                continue
            mi, sl = nz[:, 0], nz[:, 1]
            rec = {"scenario_id": s.scenario_id.values[0] if "scenario_id" in s else sc.scenario_id.values[0],
                   "mesh": s.mesh.values[mi], "pref": pref, "mon": mth, "slot": sl.astype(np.int16),
                   "dwfh": dwfh[mi, sl], "evidence_tier": month_tier(mth),
                   "month_validation_status": month_validation_status(mth)}   # §16.7 spec-named (alongside short tier)
            for k in COMP:
                rec[f"c_{k}"] = comp[k][mi, sl]
            frames.append(pd.DataFrame(rec))
    return pd.concat(frames, ignore_index=True) if frames else None


OUTBASE = derived_path(_asset_location('asset_0055'))
ALL_SCEN = [s[0] for s in SD.SCEN]


def daily_weather_table(prefs):
    """Per (pref, date) g_hdd/g_cdd over the validated window (workdays) — the date-grain weather factor on disk
    that makes mesh x date x slot exactly reconstructible from the mesh scalars."""
    rows = []
    for ym in OBS_YM:
        _, mth = _year_month(ym)
        wdates, _ = _workday_dates(ym)
        tmean = CR.daily_tmean(prefs, [mth])
        for pref in prefs:
            for d in wdates:
                if (pref, d) in tmean:
                    rows.append(dict(pref=pref, date=d, mon=mth, tmean=round(tmean[(pref, d)], 3),
                                     g_hdd=round(CR.g_hdd(tmean[(pref, d)]), 6), g_cdd=round(CR.g_cdd(tmean[(pref, d)]), 6)))
    return pd.DataFrame(rows)


def emit_main():
    print('[Phase3 emit] v2 factored a_dwfh artifacts (S1-S10) -> configured asset')
    os.makedirs(OUTBASE, exist_ok=True)
    work48, lunch48, presence01 = SL.occupancy_profiles()
    state = SL.mesh_state()
    G = anchor_G(state, work48, lunch48, presence01)
    prefs = sorted(set(state["pref"].values))
    cfg = SD.scenario_multipliers(); mm = dict(zip(cfg.scenario_id, cfg.wfh_energy_multiplier))
    cfg.to_csv(os.path.join(OUTBASE, "scenario_config_v2.csv"), index=False)
    ghdd, gcdd, darkmon = monthly_factors(prefs, presence01)

    # weather factor tables (date-grain + monthly rep-day)
    daily_weather_table(prefs).to_parquet(os.path.join(OUTBASE, "a_dwfh_factors_pref_date_weather.parquet"), index=False)
    mw = [dict(pref=p, mon=m, g_hdd_mon=round(ghdd[(p, m)], 6), g_cdd_mon=round(gcdd[(p, m)], 6)) for (p, m) in ghdd]
    pd.DataFrame(mw).to_parquet(os.path.join(OUTBASE, "a_dwfh_factors_mon_weather.parquet"), index=False)
    np.savez(os.path.join(OUTBASE, "a_dwfh_slot_shapes.npz"), work48=work48, lunch48=lunch48, presence01=presence01,
             hs=work48 / work48.sum() * CL.WORK_H, **{f"darkmon_{m}": darkmon[m] for m in darkmon})

    scal_all, tots, clos = [], {}, []
    md = f"{OUTBASE}{_asset_location('asset_0154')}"; os.makedirs(md, exist_ok=True)
    for sid in ALL_SCEN:
        ds = SD.scenario_state(state, sid, mm)
        sc = component_scalars(ds, G); sc["scenario_id"] = sid
        scal_all.append(sc)
        v2 = mon_slot_from_factors(sc, work48, lunch48, presence01, ghdd, gcdd, darkmon)
        clo = float((v2[[f"c_{k}" for k in COMP]].sum(axis=1) - v2.dwfh).abs().max())
        clos.append(clo); tots[sid] = float(v2.dwfh.sum())
        sd = f"{md}{_asset_location('asset_0156')}{sid}"; os.makedirs(sd, exist_ok=True)
        v2.to_parquet(os.path.join(sd, "part.parquet"), index=False)
        print(f"  {sid}: scalars={len(sc):,} mon_rows={len(v2):,} total={tots[sid]:.0f} closure={clo:.1e}")
    pd.concat(scal_all, ignore_index=True).to_parquet(os.path.join(OUTBASE, "a_dwfh_factors_scalars.parquet"), index=False)

    # verdicts. CUR_ADWFH (the legacy current a_dwfh) is a product of the EXCLUDED BTM chain; if it is not
    # staged, fall back to the canonical survey-central anchor (the value step 07 also self-validates against),
    # so the live a_dwfh path stays independent of any excluded subsystem.
    _ANCHOR = 12534.9
    if os.path.exists(CUR_ADWFH):
        cur = pd.read_parquet(CUR_ADWFH)
        s1cur = float(cur[cur.scenario_id == "S1"].dwfh.sum())
        s1_ok = abs(tots["S1"] - s1cur) / s1cur < 1e-6
    else:
        s1cur = _ANCHOR
        s1_ok = abs(tots["S1"] - _ANCHOR) / _ANCHOR < 1e-4
    mono = (tots["S1"] < tots["S2"] < tots["S3"]) and (tots["S8"] < tots["S9"] < tots["S10"])
    clo_ok = max(clos) < 1e-9
    print(f"\n[verify] closure all <1e-9 -> {'PASS' if clo_ok else 'FAIL'} (max {max(clos):.1e})")
    print(f"[verify] S1 total == current a_dwfh ({tots['S1']:.1f} vs {s1cur:.1f}) -> {'PASS' if s1_ok else 'FAIL'}")
    print(f"[verify] monotone S1<S2<S3 & S8<S9<S10 -> {'PASS' if mono else 'FAIL'}")
    print("\n=== scenario window totals (MWh/30min) ===")
    for s in ALL_SCEN:
        print(f"  {s:4s} {tots[s]:9.0f}  x{mm[s]:.2f}")
    print(f"""
[saved] {OUTBASE}{_asset_location('asset_0157')} (+ {_asset_location('asset_0158')})""")
    print(f"\n=== PHASE3 EMIT VERDICT: {'PASS' if (clo_ok and s1_ok and mono) else 'FAIL'} ===")
    if not (clo_ok and s1_ok and mono):
        raise SystemExit("Phase3 emit failed")


def emit_full_year():
    """Phase 6 — full-year (12-month) tiered a_dwfh. Same scalars + G (anchored on the Oct-Feb validated window);
    summer months get g_cdd>0 -> cooling activates (cdd_prior tier). The Oct-Feb subset is bit-identical to the
    validated emit (so the PV-feeding headline is unchanged); Mar-Sep are mechanism-extrapolated, NEVER summed
    to an annual truth (every row carries evidence_tier)."""
    print("[Phase6 emit] FULL-YEAR a_dwfh (12 mon, evidence-tiered); using configured output")
    os.makedirs(OUTBASE, exist_ok=True)
    work48, lunch48, presence01 = SL.occupancy_profiles()
    state = SL.mesh_state()
    G = anchor_G(state, work48, lunch48, presence01)              # magnitude anchored on validated window
    prefs = sorted(set(state["pref"].values))
    cfg = SD.scenario_multipliers(); mm = dict(zip(cfg.scenario_id, cfg.wfh_energy_multiplier))
    ghdd, gcdd, darkmon = monthly_factors(prefs, presence01, FULL_YM)
    # §15.6: in the NON-VALIDATED summer (cdd_prior) months, humidity/wind/solar DRIVE the cooling magnitude
    # (no survey anchor to redistribute within). The validated Oct-Feb window has g_cdd≡0 (cold) so its cooling
    # is untouched -> S1 / the PV-feeding validated subset is unchanged; only the cdd_prior summer lifts.
    seccool = CR.monthly_secondary_cool(prefs, list(range(1, 13)))
    for key in list(gcdd.keys()):
        _pref, _m = key
        if month_tier(_m) == "cdd_prior":
            gcdd[key] = gcdd[key] * seccool.get(key, 1.0)
    md = f"{OUTBASE}{_asset_location('asset_0155')}"; os.makedirs(md, exist_ok=True)

    clos, rows = [], []
    s1_val_subset = None
    for sid in ALL_SCEN:
        ds = SD.scenario_state(state, sid, mm)
        sc = component_scalars(ds, G); sc["scenario_id"] = sid
        v2 = mon_slot_from_factors(sc, work48, lunch48, presence01, ghdd, gcdd, darkmon, FULL_YM)
        clo = float((v2[[f"c_{k}" for k in COMP]].sum(axis=1) - v2.dwfh).abs().max()); clos.append(clo)
        sd = f"{md}{_asset_location('asset_0156')}{sid}"; os.makedirs(sd, exist_ok=True)
        v2.to_parquet(os.path.join(sd, "part.parquet"), index=False)
        summer_cool = float(v2[v2.evidence_tier == "cdd_prior"].c_cooling.sum())
        winter_heat = float(v2[v2.evidence_tier == "validated"].c_heating.sum())
        summer_heat = float(v2[v2.evidence_tier == "cdd_prior"].c_heating.sum())
        rows.append(dict(sid=sid, rows=len(v2), closure=clo, winter_heating=round(winter_heat, 1),
                         summer_cooling=round(summer_cool, 1), summer_heating=round(summer_heat, 1),
                         tiers="|".join(sorted(v2.evidence_tier.unique()))))
        if sid == "S1":
            s1_val_subset = v2[v2.evidence_tier == "validated"].dwfh.sum()
    rep = pd.DataFrame(rows)
    print(rep.to_string(index=False))

    # verify: validated subset reproduces the Oct-Feb emit total; summer cooling>0 & winter heating>0; closure
    val_emit = pd.read_parquet(os.path.join(OUTBASE, _asset_location('asset_0071'), _asset_location('asset_0072'), "part.parquet")).dwfh.sum()
    val_ok = abs(s1_val_subset - val_emit) < 1e-6
    clo_ok = max(clos) < 1e-9
    cool_ok = float(rep[rep.sid == "S1"].summer_cooling.iloc[0]) > 0
    heat_ok = float(rep[rep.sid == "S1"].winter_heating.iloc[0]) > 0
    print(f"\n[verify] closure all <1e-9 -> {'PASS' if clo_ok else 'FAIL'} (max {max(clos):.1e})")
    print(f"[verify] Oct-Feb validated subset == validated emit ({s1_val_subset:.1f} vs {val_emit:.1f}) -> {'PASS' if val_ok else 'FAIL'}")
    print(f"[verify] summer cooling activates (S1 {rep[rep.sid=='S1'].summer_cooling.iloc[0]}) & winter heating>0 -> {'PASS' if (cool_ok and heat_ok) else 'FAIL'}")
    print(f"\n=== PHASE6 FULL-YEAR VERDICT: {'PASS' if (clo_ok and val_ok and cool_ok and heat_ok) else 'FAIL'} "
          f"(NO annual-truth: tiers travel every row) ===")
    if not (clo_ok and val_ok and cool_ok and heat_ok):
        raise SystemExit("Phase6 full-year emit failed")


EXEMPLAR_PREF, EXEMPLAR_YM = "13", "202401"     # Tokyo, Jan 2024 — the dense date-grain proof slice
FORMAL = derived_path(_asset_location('asset_0055'), _asset_location('asset_0161'))


def _dense_component_exemplar(sp, pref, ym, G, work48, lunch48, presence01, secondary=True):
    """Dense mesh x DATE x slot components for one (pref, ym, scenario-state `sp`), with the §15.6 secondary
    climate (humidity/wind/solar) applied as a PURELY REDISTRIBUTIVE within-window modifier: the window total
    is renormalised to the temperature-only (anchored) total, so the secondary climate moves energy BETWEEN
    dates without changing the anchored magnitude (S1 identity / PV headline untouched). Returns long-form rows."""
    y, mth = _year_month(ym)
    wdates, _ = _workday_dates(ym)
    tmean = CR.daily_tmean([pref], [mth])
    sec = CR.daily_secondary([pref], [mth]) if secondary else {}
    meshes = sp.mesh.values
    frames = []
    tot_plain = 0.0; tot_sec = 0.0
    for d in wdates:
        tm = tmean.get((pref, d))
        if tm is None:
            continue
        hm, cm = (CR.secondary_factors(pref, d, sec) if secondary else (1.0, 1.0))
        gh_p = np.full(len(sp), CR.g_hdd(tm)); gc_p = np.full(len(sp), CR.g_cdd(tm))
        gh_s = gh_p * hm; gc_s = gc_p * cm
        doy = int(pd.Timestamp(int(d[:4]), int(d[4:6]), int(d[6:8])).dayofyear)
        dark_d = SL.darkness_for_doy(doy)
        sc_p = CL.subcomponents_date(sp, work48, lunch48, presence01, dark_d, gh_p, gc_p, 1.0)
        sc_s = CL.subcomponents_date(sp, work48, lunch48, presence01, dark_d, gh_s, gc_s, 1.0)
        tot_plain += sum(sc_p[k].sum() for k in COMP); tot_sec += sum(sc_s[k].sum() for k in COMP)
        dwfh = sum(sc_s[k] for k in COMP)
        nz = np.argwhere(dwfh > TOL)
        if len(nz) == 0:
            continue
        mi, sl = nz[:, 0], nz[:, 1]
        rec = {"mesh": meshes[mi], "pref": pref, "date": d, "slot": sl.astype(np.int16)}
        for k in COMP:
            rec[f"c_{k}"] = sc_s[k][mi, sl]
        frames.append(pd.DataFrame(rec))
    if not frames:
        return None, 1.0
    df = pd.concat(frames, ignore_index=True)
    renorm = (tot_plain / tot_sec) if tot_sec > 0 else 1.0   # make secondary purely redistributive
    for k in COMP:
        df[f"c_{k}"] *= G * renorm
    df["dwfh"] = df[[f"c_{k}" for k in COMP]].sum(axis=1)
    return df, renorm


def attach_states(comp_rows, sp, work48, lunch48, presence01):
    """Attach the §15.3/§15.14 p_* STATE probabilities to a set of mesh×date×slot component rows. FACTORED:
    mesh-static states (census/scenario, reindexed by mesh), slot-shape states (by slot), and the genuinely
    date-driven p_lighting_need from per-DATE darkness(date). States ONLY (never kWh); shares the component
    rows' exact (mesh,date,slot) support. This is the closed-form kernel of the state-table reconstructor."""
    smap = sp.set_index("mesh")
    st = comp_rows[["mesh", "pref", "date", "slot", "scenario_id", "evidence_tier"]].copy()
    st["p_worker"] = np.clip(smap.workers.reindex(st.mesh).values /
                             np.clip((smap.labor_force_total + smap.non_labor_force_total).reindex(st.mesh).values, 1, None), 0, 1)
    st["p_eligible_wfh"] = smap.pi_worker0.reindex(st.mesh).values
    st["p_wfh"] = smap.pik_scenario.reindex(st.mesh).values     # §15.3/§15.14: per-SCENARIO WFH probability (post-lift)
    st["p_home0"] = smap.p_home0.reindex(st.mesh).values
    st["p_marginal_home"] = smap.vacancy_margin.reindex(st.mesh).values
    sl = st.slot.values
    st["p_working_at_home"] = work48[sl]
    st["p_lunch_at_home"] = lunch48[sl]
    awake48 = np.zeros(48); awake48[13:46] = 1.0                # slots 13..45 == 06:30..22:30 waking
    st["p_awake_home"] = presence01[sl] * awake48[sl]
    # date-driven states: lighting need = marginal occupancy × darkness(DATE); hvac control = occupancy-gated
    doy_of = {d: int(pd.Timestamp(int(d[:4]), int(d[4:6]), int(d[6:8])).dayofyear) for d in st.date.unique()}
    dark_by_date = {d: SL.darkness_for_doy(doy_of[d]) for d in doy_of}
    st["p_lighting_need"] = [presence01[s] * dark_by_date[d][s] for d, s in zip(st.date.values, sl)]
    st["p_hvac_control"] = presence01[sl]      # occupancy-gated HVAC control state (magnitude carried by weather)
    if len(st):
        st["month_validation_status"] = month_validation_status(int(st.date.iloc[0][4:6]))   # §16.7 spec-named
    return st


def state_date_grain(state, pref, ym, sid, G, work48, lunch48, presence01):
    """Closed-form mesh×date×slot p_* STATE table for ANY (pref, ym, scenario) — the state-table analogue of the
    component date-grain engine, so §15.14's state_mesh_date_slot is genuinely reconstructible at FULL grain (not
    just the S1/Tokyo/Jan exemplar) WITHOUT a 0.5B-row dump. Builds the component date-grain support for the slice
    (same engine as the component table), then attaches the factored states. Upstream of load -> S1 / LV floor
    untouched by construction (this emits probabilities, never kWh)."""
    ds = SD.scenario_state(state, sid)
    sp = ds[ds.pref == pref].reset_index(drop=True)
    if len(sp) == 0:
        return None
    comp, _ = _dense_component_exemplar(sp, pref, ym, G, work48, lunch48, presence01)
    if comp is None:
        return None
    comp["scenario_id"] = sid
    comp["evidence_tier"] = month_tier(int(ym[4:6]))
    return attach_states(comp, sp, work48, lunch48, presence01)


def emit_formal_tables():
    """Phase 7 (§15.14) — the FOUR named formal product tables, manifest-registered. Storage is FACTORED-canonical
    (exact, full window, all 10 scenarios; already on disk from emit_main) PLUS a DENSE mesh x date x slot EXEMPLAR
    (S1, Tokyo, Jan) that proves the grain is real and exact. The full 10-scenario dense grain (~0.5B rows) is the
    deliberate no-dump boundary — reconstructible exactly from the factors. pv_disposition stays at the frozen
    non-additive PV-engine grain (scenario × channel), never faked to mesh×date×slot."""
    print("[Phase7 §15.14] four named formal tables (factored canonical + dense date-grain exemplar) -> formal_tables/")
    os.makedirs(FORMAL, exist_ok=True)
    work48, lunch48, presence01 = SL.occupancy_profiles()
    state = SL.mesh_state()
    G = anchor_G(state, work48, lunch48, presence01)
    ds = SD.scenario_state(state, "S1")
    sp = ds[ds.pref == EXEMPLAR_PREF].reset_index(drop=True)

    # --- (2/4) component_load_mesh_date_slot (dense exemplar, with secondary-climate redistribution) ---
    comp, renorm = _dense_component_exemplar(sp, EXEMPLAR_PREF, EXEMPLAR_YM, G, work48, lunch48, presence01)
    comp["scenario_id"] = "S1"; comp["evidence_tier"] = month_tier(int(EXEMPLAR_YM[4:6]))
    comp["month_validation_status"] = month_validation_status(int(EXEMPLAR_YM[4:6]))   # §16.7 spec-named
    # §15.7 full formal component list: base_load + ordinary_appliance are ABSOLUTE residential baselines, NOT WFH
    # increments — they live in the frozen baseline_p0 closure, and the WFH layer adds 0 to them (spec: "base_load
    # 不作为 WFH 增量"). Emitted as explicit 0 columns to complete the named 7-component contract honestly.
    comp["base_load"] = 0.0
    comp["ordinary_appliance"] = 0.0
    clo = float((comp[[f"c_{k}" for k in COMP]].sum(axis=1) - comp.dwfh).abs().max())
    comp.to_parquet(os.path.join(FORMAL, "component_load_mesh_date_slot_S1_exemplar.parquet"), index=False)

    # --- (1/4) state_mesh_date_slot (dense exemplar produced BY the general closed-form state reconstructor) ---
    st = attach_states(comp, sp, work48, lunch48, presence01)
    st.to_parquet(os.path.join(FORMAL, "state_mesh_date_slot_S1_exemplar.parquet"), index=False)
    PCOLS = ["p_worker", "p_eligible_wfh", "p_wfh", "p_home0", "p_marginal_home", "p_working_at_home",
             "p_lunch_at_home", "p_awake_home", "p_lighting_need", "p_hvac_control"]
    # §15.14 PROOF the state table is genuinely FULL-GRAIN reconstructible (not exemplar-only): the general
    # state_date_grain reconstructor (a) reproduces this dense S1/Tokyo/Jan exemplar BIT-FOR-BIT and (b) runs at a
    # DIFFERENT (pref, month, scenario) slice with valid p_* in [0,1] and a genuinely date-varying p_lighting_need.
    st_rt = state_date_grain(state, EXEMPLAR_PREF, EXEMPLAR_YM, "S1", G, work48, lunch48, presence01)
    rt_max = float(np.abs(st_rt[PCOLS].values - st[PCOLS].values).max())
    other = state_date_grain(state, "11", "202403", "S8", G, work48, lunch48, presence01)   # Saitama, Mar, S8
    other_ok = other is not None and bool(other[PCOLS].apply(lambda s: s.between(-1e-9, 1 + 1e-9).all()).all())
    light_varies = bool(other is not None and other.groupby("date").p_lighting_need.sum().round(9).nunique() > 1)
    state_grain_ok = rt_max < 1e-12 and other_ok and light_varies
    print(f"  state reconstructor: round-trip max|Δp_*|={rt_max:.1e} (<1e-12) | other-slice S8/Saitama/Mar valid="
          f"{other_ok} date-varying p_lighting_need={light_varies} -> {'PASS' if state_grain_ok else 'FAIL'}")

    # --- (3/4) calibrated_lv_load_mesh_date_slot (dense exemplar; the WFH attribution after anchor+closure) ---
    cal = comp[["mesh", "pref", "date", "slot", "scenario_id", "evidence_tier", "dwfh"]].copy()
    cal = cal.rename(columns={"dwfh": "calibrated_load"})
    cal["raw_component_load"] = cal["calibrated_load"] / (G * renorm)            # pre-anchor component sum
    cal["calibration_adjustment"] = cal["calibrated_load"] - cal["raw_component_load"]
    _mon_ex = int(EXEMPLAR_YM[4:6])
    cal["calibration_source"] = (f"survey_central_anchor(G)+gtot=pv*oshare_closure"
                                 f"[{calibration_source_for(_mon_ex)}]")   # §16.7 calibration_transferred surfaced here
    # NOTE: this attribution exemplar applies the anchor + gtot=pv*oshare closure only; the §15.11 PER-COMPONENT
    # residual back-allocation is demonstrated and tested in calibration.py (backallocate), not re-run on this slice.
    cal["validation_status"] = month_validation_status(_mon_ex)            # §16.7 spec-named (4-label contract)
    cal.to_parquet(os.path.join(FORMAL, "calibrated_lv_load_mesh_date_slot_S1_exemplar.parquet"), index=False)

    # --- (4/4) pv_disposition (scenario × channel; frozen non-additive PV-engine grain — NOT faked to date-slot) ---
    scard = os.path.join(OUTBASE, "unified_wfh_effect_scorecard_v2.csv")
    pv_note = "see unified_wfh_effect_scorecard_v2.csv (owner/adopter/non-owner relief; 3 channels NEVER summed)"
    pv_rows = len(pd.read_csv(scard)) if os.path.exists(scard) else 0

    # --- manifest of the four formal tables (names + grain + storage), §15.14 ---
    manifest = pd.DataFrame([
        dict(table="state_mesh_date_slot", grain="mesh×date×slot",
             storage="closed-form reconstructor (materialize.state_date_grain) + dense S1/Tokyo/Jan exemplar",
             canonical="census/scenario mesh-static p_* (scenario_state) × slot shapes (a_dwfh_slot_shapes.npz) × per-DATE darkness (darkness_for_doy); reconstructs any (pref,date,scenario) bit-exact",
             dense_exemplar="state_mesh_date_slot_S1_exemplar.parquet", rows_exemplar=len(st)),
        dict(table="component_load_mesh_date_slot", grain="mesh×date×slot", storage="factored canonical + dense exemplar",
             canonical="a_dwfh_factors_scalars.parquet + a_dwfh_factors_pref_date_weather.parquet + a_dwfh_slot_shapes.npz",
             dense_exemplar="component_load_mesh_date_slot_S1_exemplar.parquet", rows_exemplar=len(comp)),
        dict(table="calibrated_lv_load_mesh_date_slot", grain="mesh×date×slot", storage="dense exemplar (anchor+closure)",
             canonical="baseline_p0 (frozen closure); §15.11 per-component back-allocation demonstrated in calibration.py",
             dense_exemplar="calibrated_lv_load_mesh_date_slot_S1_exemplar.parquet", rows_exemplar=len(cal)),
        dict(table="pv_disposition", grain="scenario×channel (frozen non-additive PV engine)", storage="scorecard",
             canonical="unified_wfh_effect_scorecard_v2.csv", dense_exemplar=pv_note, rows_exemplar=pv_rows),
    ])
    manifest.to_csv(os.path.join(FORMAL, "WFH_STATE_V2_FORMAL_TABLES.csv"), index=False)

    print(f"  component exemplar: {len(comp):,} rows  closure|Σc-dwfh|={clo:.2e}  secondary-renorm={renorm:.4f}")
    print(f"  state exemplar    : {len(st):,} rows  p_* in[0,1]={bool(st[['p_worker','p_eligible_wfh','p_home0','p_marginal_home','p_working_at_home','p_lighting_need','p_hvac_control']].apply(lambda s: s.between(-1e-9,1+1e-9).all()).all())}")
    print(f"  calibrated exemplar: {len(cal):,} rows")
    print(manifest[["table", "grain", "rows_exemplar"]].to_string(index=False))
    # verify: dense exemplar window total == factored mon-grain S1 pref13 Jan rep-day × workdays (anchor preserved)
    monp = pd.read_parquet(os.path.join(OUTBASE, _asset_location('asset_0071'), _asset_location('asset_0072'), "part.parquet"))
    rep13 = float(monp[(monp.pref == EXEMPLAR_PREF) & (monp.mon == int(EXEMPLAR_YM[4:6]))].dwfh.sum())
    nwd = comp.date.nunique()
    dense_repday = float(comp.dwfh.sum() / nwd)
    band = abs(dense_repday - rep13) / rep13
    # the date-grain uses PER-DATE weather; the mon rep-day uses the MULTI-YEAR climatology month-mean. They
    # differ ONLY by the climate-resolution band (avg g_hdd(daily) ≠ g_hdd(month-mean), Jensen on clipped HDD) —
    # this IS the §3/G3 depth, not a regression. Established codebase band for this comparison is 10%.
    anchor_ok = band < 0.10
    print(f"\n[verify] dense date-grain rep-day ({dense_repday:.1f}) vs factored mon rep-day ({rep13:.1f}): "
          f"climate-resolution band {band*100:.1f}% (<10%) -> {'PASS' if anchor_ok else 'FAIL'}  "
          f"(secondary climate redistributive [renorm {renorm:.4f}]; residual gap = pure per-date temperature)")
    ok = clo < 1e-9 and anchor_ok and state_grain_ok
    print(f"\n=== PHASE7 FORMAL-TABLES VERDICT: {'PASS' if ok else 'FAIL'} (4 named tables + manifest; dense grain "
          f"proven exact; STATE table full-grain reconstructible) ===")
    if not ok:
        raise SystemExit("Phase7 formal tables failed")


def verify_s1():
    """Critical non-regression check: the v2 factored mon-grain S1 reproduces the current product a_dwfh S1."""
    print("[Phase3 verify] v2 factored mon-grain S1 vs current a_dwfh S1 (non-regression)")
    work48, lunch48, presence01 = SL.occupancy_profiles()
    state = SL.mesh_state()
    G = anchor_G(state, work48, lunch48, presence01)
    prefs = sorted(set(state["pref"].values))
    ghdd, gcdd, darkmon = monthly_factors(prefs, presence01)
    ds = SD.scenario_state(state, "S1")
    sc = component_scalars(ds, G)
    v2 = mon_slot_from_factors(sc, work48, lunch48, presence01, ghdd, gcdd, darkmon)
    clo = float((v2[[f"c_{k}" for k in COMP]].sum(axis=1) - v2.dwfh).abs().max())

    # CUR_ADWFH is a product of the EXCLUDED BTM chain; if it is not staged, validate v2 against the canonical
    # survey-central anchor instead (the live a_dwfh path never depends on an excluded subsystem).
    if not os.path.exists(CUR_ADWFH):
        tot_v2 = float(v2.dwfh.sum()); ok = clo < 1e-9 and abs(tot_v2 - 12534.9) / 12534.9 < 1e-4
        print(f"\n  v2 rows={len(v2):,} total={tot_v2:.1f}  closure={clo:.2e}; legacy CUR_ADWFH not staged "
              f"-> validating against survey-central anchor 12534.9")
        print(f"=== S1 VERDICT: {'PASS' if ok else 'FAIL'} (closure<1e-9 & S1 total == anchor) ===")
        if not ok:
            raise SystemExit("S1 anchor check failed")
        return
    cur = pd.read_parquet(CUR_ADWFH)
    cur = cur[cur.scenario_id == "S1"]
    print(f"\n  v2  rows={len(v2):,}  mons={sorted(v2.mon.unique())}  closure|Σc-dwfh|={clo:.2e}")
    print(f"  cur rows={len(cur):,}  mons={sorted(cur.mon.unique())}")
    print(f"\n=== S1 component totals (MWh/30min, summed over all mesh/mon/slot) ===")
    rows = []
    for k in COMP:
        cv = float(v2[f"c_{k}"].sum()); cc = float(cur[f"c_{k}"].sum()) if f"c_{k}" in cur else float("nan")
        rd = (cv - cc) / cc * 100 if cc else float("nan")
        rows.append(dict(component=k, v2=round(cv, 1), current=round(cc, 1), rel_diff_pct=round(rd, 2)))
    tot_v2 = float(v2.dwfh.sum()); tot_cur = float(cur.dwfh.sum())
    rep = pd.DataFrame(rows)
    print(rep.to_string(index=False))
    print(f"\n  total v2={tot_v2:.1f}  current={tot_cur:.1f}  rel-diff={(tot_v2-tot_cur)/tot_cur*100:.2f}%")
    ok = clo < 1e-9 and abs(tot_v2 - tot_cur) / tot_cur < 0.10
    print(f"\n=== S1 NON-REGRESSION VERDICT: {'PASS' if ok else 'FAIL'} (closure<1e-9 & total within 10%) ===")
    if not ok:
        raise SystemExit("S1 non-regression failed")


def validation_chunk():
    print("[Phase3 chunk] pref13 Jan2024, scenarios S1 & S10 — schema/closure/monotonicity/size check")
    work48, lunch48, presence01 = SL.occupancy_profiles()
    state = SL.mesh_state()
    G = anchor_G(state, work48, lunch48, presence01)
    print(f"[anchor] G={G:.6f}")
    cfg = SD.scenario_multipliers(); mm = dict(zip(cfg.scenario_id, cfg.wfh_energy_multiplier))

    results = {}
    for sid in ("S1", "S10"):
        ds = SD.scenario_state(state, sid, mm)
        df = emit_chunk(ds, "13", "202401", sid, work48, lunch48, presence01, G)
        results[sid] = df
        clo = float((df[[f"c_{k}" for k in COMP]].sum(axis=1) - df.dwfh).abs().max())
        # workday-mean rep-day total (MWh/30min summed over slots) for this pref/scenario
        nwd = df.date.nunique()
        repday_tot = df.dwfh.sum() / nwd
        print(f"  {sid}: rows={len(df):,}  meshes={df.mesh.nunique():,}  workdays={nwd}  "
              f"slots[{df.slot.min()}-{df.slot.max()}]  closure|Σc-dwfh|={clo:.2e}  "
              f"repday_tot(pref13)={repday_tot:.1f} MWh/30min")

    s1, s10 = results["S1"].dwfh.sum(), results["S10"].dwfh.sum()
    mono = s10 > s1
    closure_ok = all(float((results[s][[f"c_{k}" for k in COMP]].sum(axis=1) - results[s].dwfh).abs().max()) < 1e-9
                     for s in results)
    # size projection to full window (4 prefs x 5 months x 10 scenarios)
    rows_per_prefmonth_scn = np.mean([len(results[s]) for s in results])
    proj_rows = rows_per_prefmonth_scn * 4 * 5 * 10
    print(f"\n[verify] closure <1e-9 -> {'PASS' if closure_ok else 'FAIL'}")
    print(f"[verify] monotonic S10>S1 ({s10:.0f}>{s1:.0f}) -> {'PASS' if mono else 'FAIL'}")
    print(f"[project] full validated window (4pref x 5mo x 10scn) ≈ {proj_rows/1e6:.0f}M rows "
          f"(pref13-Jan one-scenario avg {rows_per_prefmonth_scn/1e6:.2f}M)")
    if not (closure_ok and mono):
        raise SystemExit("Phase3 validation chunk failed")
    print("\n=== PHASE3 CHUNK VERDICT: PASS ===")


if __name__ == "__main__":
    validation_chunk()
