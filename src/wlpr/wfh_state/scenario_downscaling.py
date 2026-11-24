"""Scenario-specific mesh downscaling (plan ch5 / 15.4; gap G1).

Ported verbatim (numerics unchanged) from the source
`configured asset`.

Replaces the H1 single GLOBAL multiplier `scenario_mesh(m, mult)` (pi_worker0 x mult, identical every mesh)
with a mesh-specific lift:  lift_s(g) = score_s(g) / score_base(g),  score_s(g) = Σ_cat workers_{g,cat}·rate_{s,cat}
  - industry scenarios (S4/S5)   -> T001193 INDUSTRY columns x industry telework rates (sectoral diffusion, current freq)
  - sectoral institutionalization (S6/S7) = the INDUSTRY x FREQUENCY combo: same ICT / knowledge-service mesh
    allocation as S4/S5 but at 3-4 day frequency -> concentrated industry geography at HIGH intensity (the realistic
    future). Enterprise size was RETIRED as a lever (2026-06-23): establishment-size has no residence-mesh geography
    -- it is workplace-indexed and spatially spanned by the occupation/industry kernels -- so it could only ever
    reproduce S1 geography. The real 就業構造基本調査 size table is KEPT as a frozen source, just unused here.
  - occupation scenarios (S8-10) -> T001193 OCCUPATION columns x occupation telework rates (broad, high intensity)
  - frequency scenarios (S2/S3)  -> lift = 1 (frequency changes HOW OFTEN, not WHO/WHERE)

CONSERVATION (the no-regression guarantee): after applying lift+intensity, N_persons is RE-NORMALISED per
prefecture to the SAME pref total the global multiplier produces. So the cross-pref survey anchor and the
pref-level WFH energy are preserved exactly -> S1 reproduces the locked identity (owner 6.888/future 12.003/
relief 6.161 GWh; the old 6.086/10.309/5.036 triple was retired Jun-16 pre-trapezoid) and the closed LV total
is untouched. Only the WITHIN-pref spatial pattern of the increment concentrates into the scenario-relevant
meshes (the plan's goal).
"""
from __future__ import annotations

from wlpr.io.bindings import asset as _asset_location
import os
import numpy as np
import pandas as pd

from wlpr.io.paths import derived_path

T93 = derived_path(_asset_location('asset_0125'), "t001193_mesh_industry_occupation_selected.csv")
RATE = {"ind": derived_path(_asset_location('asset_0125'), "wfh_s5_industry_telework_rates.csv"),
        "occ": derived_path(_asset_location('asset_0125'), "wfh_s6_occupation_telework_rates.csv")}
# enterprise-size rates (wfh_enterprise_size_telework_rates.csv) remain a frozen source but are no longer a
# scenario lever -- size has no residence-mesh geography (retired 2026-06-23; see module docstring).

I_BASE = 0.349
I_FREQ = {"observed": 0.349, "2-3 days": 0.50, "3-4 days": 0.70, "high": 0.70}

# 10-scenario taxonomy (ported verbatim from build_wfh_attribution_scenario_ledger SCEN — canonical home is now src)
IND = {"ICT": "07", "PROF": "12", "FIN": "10", "UTIL": "06"}
OCCC = {"PROF": "02", "CLERK": "03", "MGMT": "01", "SALES": "04"}
SCEN = [
    ("S1", "当前调查锚定", "频度", "observed", "observed", "observed", None, {}),
    ("S2", "标准混合办公(2-3日)", "频度", "frequency", "2-3 days", "survey-anchored", None, {}),
    ("S3", "高频混合办公(3-4日)", "频度", "frequency", "3-4 days", "survey-anchored", None, {}),
    ("S4", "数字原生行业高采用", "行业", "industry", "observed", "survey-anchored", "ind", {IND["ICT"]: 0.90, IND["UTIL"]: 0.70}),
    ("S5", "知识密集服务业扩散", "行业", "industry", "observed", "survey-anchored", "ind", {IND["PROF"]: 0.80, IND["FIN"]: 0.80}),
    ("S6", "数字原生行业制度化(3-4日)", "组合", "industry", "3-4 days", "survey-anchored", "ind", {IND["ICT"]: 0.90, IND["UTIL"]: 0.70}),
    ("S7", "知识服务业制度化(3-4日)", "组合", "industry", "3-4 days", "survey-anchored", "ind", {IND["PROF"]: 0.80, IND["FIN"]: 0.80}),
    ("S8", "可远程职业充分采用", "结构", "occupation", "2-3 days", "upper-bound", "occ", {OCCC["PROF"]: 0.70, OCCC["CLERK"]: 0.70, OCCC["MGMT"]: 0.70, OCCC["SALES"]: 0.70}),
    ("S9", "政策强化高采用", "结构", "policy_bundle", "3-4 days", "policy-counterfactual", "occ", {OCCC["PROF"]: 0.75, OCCC["CLERK"]: 0.75, OCCC["MGMT"]: 0.75, OCCC["SALES"]: 0.75}),
    ("S10", "结构普及上限", "结构", "policy_bundle", "high", "upper-bound", "occ", {OCCC["PROF"]: 0.85, OCCC["CLERK"]: 0.85, OCCC["MGMT"]: 0.85, OCCC["SALES"]: 0.85}),
]
# spatial-downscaling evidence label per lever (transparent in every emitted row)
LEVER_SPATIAL = {"observed": "occupation_mesh_T001193", "frequency": "current_spatial_intensity_only",
                 "industry": "industry_mesh_T001193", "occupation": "occupation_mesh_T001193",
                 "policy_bundle": "occupation_mesh_T001193"}


def _col_code(col):
    """industry_G_information... / occupation_C_clerical -> 2-digit category code from the A..Z letter."""
    letter = col.split("_")[1]
    return f"{ord(letter) - 64:02d}"


def category_workers(table):
    """mesh x category-code worker counts from T001193 (industry_* or occupation_* columns)."""
    prefix = "industry_" if table == "ind" else "occupation_"
    df = pd.read_csv(T93)
    df["mesh"] = df.mesh.astype("int64").astype(str)
    cols = [c for c in df.columns if c.startswith(prefix)]
    code_of = {c: _col_code(c) for c in cols}
    out = df[["mesh"]].copy()
    for c in cols:
        out[code_of[c]] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return out, sorted(set(code_of.values()))


def base_rates(table):
    r = pd.read_csv(RATE[table])
    r["category_code"] = r.category_code.astype(str).str.zfill(2)
    return dict(zip(r.category_code, r.telework_rate.astype(float))), dict(zip(r.category_code, r.total_workers.astype(float)))


MOBW = derived_path(_asset_location('asset_0054'), "wfh_spatial_weight.parquet")


def mobility_weight(col="current_presence"):
    """Mesh-static mobility likelihood normalized to mean-1 WITHIN pref (a SOFT spatial prior, plan 15.5).
    RED LINE: mobility is presence/occupancy, NEVER kWh or WFH-truth; it only nudges WHERE the scenario
    INCREMENT lands (guardrail against placing WFH at max PV headroom), conserving the pref total. Missing -> 1.
    `col`: current_presence (current WFH spatial) or scenario_potential (future conversion). Returns {mesh: w} or None."""
    if not os.path.exists(MOBW):
        return None
    w = pd.read_parquet(MOBW)
    if col not in w.columns:
        return None
    w["mesh"] = w.mesh.astype("int64").astype(str) if pd.api.types.is_numeric_dtype(w.mesh) else w.mesh.astype(str)
    w["pref"] = w.pref.astype(str)
    w["mob"] = pd.to_numeric(w[col], errors="coerce").fillna(0.0).clip(lower=0.0)
    w["mob"] = w.groupby("pref").mob.transform(lambda s: s / s[s > 0].mean() if (s > 0).any() else 1.0)
    w.loc[~np.isfinite(w.mob), "mob"] = 1.0
    return dict(zip(w.mesh, w.mob))


def _recompute_P(rates, workers, raises):
    codes = list(rates)
    r = np.array([max(rates[c], raises.get(c, 0.0)) for c in codes])
    w = np.array([workers[c] for c in codes])
    return float((w * r).sum() / w.sum())


def scenario_multipliers():
    """Aggregate participation_ratio + global multiplier per scenario (the CONSERVED pref-total target)."""
    cache = {t: base_rates(t) for t in RATE}
    Pbase = {t: _recompute_P(*cache[t], {}) for t in RATE}
    rows = []
    for (sid, name, grp, lever, freq, tier, table, raises) in SCEN:
        p_ratio = 1.0 if table is None else _recompute_P(*cache[table], raises) / Pbase[table]
        mult = p_ratio * (I_FREQ[freq] / I_BASE)
        rows.append(dict(scenario_id=sid, scenario_name=name, group=grp, primary_lever=lever,
                         frequency_assumption=freq, evidence_tier=tier, spatial_basis=LEVER_SPATIAL[lever],
                         participation_ratio=round(p_ratio, 4), intensity=I_FREQ[freq],
                         # §15.4 spec-named output fields (alongside the internal intensity/spatial_basis):
                         frequency_intensity=I_FREQ[freq], mesh_weight_source=LEVER_SPATIAL[lever],
                         wfh_energy_multiplier=round(mult, 4)))
    return pd.DataFrame(rows)


def mesh_lift(table, raises):
    """lift_s(g) = score_s(g)/score_base(g) per mesh; mean ~ aggregate participation_ratio but spatially varying.
    DIAGNOSTIC ONLY (kept for the corr(lift, ICT) provenance check): a ratio near 1 for narrow/broad levers, so it
    does NOT drive the allocation -- the genuine bottom-up spatial signal is in _marginal_pool() below."""
    W, codes = category_workers(table)
    rates, _ = base_rates(table)
    rb = np.array([rates.get(c, 0.0) for c in codes])
    rs = np.array([max(rates.get(c, 0.0), raises.get(c, 0.0)) for c in codes])
    M = W[codes].to_numpy()
    score_b = M @ rb
    score_s = M @ rs
    lift = np.divide(score_s, score_b, out=np.ones_like(score_s), where=score_b > 0)
    return pd.DataFrame({"mesh": W.mesh.values, "lift": lift})


def _marginal_pool(m, table, raises, n_s1):
    """The per-mesh MARGINAL adopter pool a scenario's WFH increment is allocated by (genuine bottom-up):
       pool_s(g) = Σ_cat workers_{g,cat} · max(rate_{s,cat} − rate_base,cat, 0)
    so the S4 (ICT) increment concentrates where ICT workers actually are, the S8 (broad-occupation) increment
    spreads over office workers -> S4 ≠ S8 spatially (capped by the real occupation co-location ~0.96). Frequency
    levers carry NO mesh occupation signal -> the increment follows the existing S1 WFH (more of the same people,
    same places). Per-pref fallback to S1 where a pref has no targeted-category workers."""
    n_s1 = np.clip(np.asarray(n_s1, float), 0, None)
    if table not in ("ind", "occ"):
        return n_s1
    W, codes = category_workers(table)
    W["mesh"] = W.mesh.astype(str)
    rates, _ = base_rates(table)
    raise_vec = np.clip(np.array([max(rates.get(c, 0.0), raises.get(c, 0.0)) - rates.get(c, 0.0) for c in codes]),
                        0, None)
    M = W.set_index("mesh").reindex(m.mesh.astype(str))[codes].fillna(0.0).to_numpy()
    pool = M @ raise_vec
    pref = m.pref.astype(str).values
    poolp = pd.Series(pool).groupby(pref).transform("sum").values   # fall back to S1 in prefs with no such workers
    return np.where(poolp > 0, pool, n_s1)


def scenario_state(state, sid, mult_map=None, use_mobility=False):
    """Return a copy of `state` with N_persons / H_wfh re-derived under scenario `sid` via mesh-specific
    employment lift + per-pref conservation. S1 returns state unchanged (identity-exact). With use_mobility,
    a SOFT mobility likelihood additionally nudges WHERE the scenario INCREMENT (above S1) lands, conserving
    the pref total exactly (so S1 stays identity-exact and the LV floor / PV identity are preserved)."""
    sdef = {s[0]: s for s in SCEN}[sid]
    _, name, grp, lever, freq, tier, table, raises = sdef
    if mult_map is None:
        cfg = scenario_multipliers(); mult_map = dict(zip(cfg.scenario_id, cfg.wfh_energy_multiplier))
    mult = mult_map[sid]
    intensity_ratio = I_FREQ[freq] / I_BASE

    m = state.copy()
    if table in ("ind", "occ"):
        lf = mesh_lift(table, raises)
        m = m.merge(lf, on="mesh", how="left")
        m["lift"] = m["lift"].fillna(1.0)
    else:
        m["lift"] = 1.0   # frequency: no within-pref spatial move (same people, more days)

    pi0 = m.pi_worker0.values
    workers = m.workers.values
    pref = m.pref.astype(str).values
    N_s1 = state["N_persons"].values                       # the S1 base (validated anchor; left-merge order kept)
    # survey-anchored per-pref WFH-person TOTAL (the conserved target, exactly the global multiplier's pref sum)
    N_glob = workers * np.clip(pi0 * mult, 0, 1)
    tgt = pd.Series(N_glob).groupby(pref).sum()
    cur = pd.Series(N_s1).groupby(pref).sum()
    dT = (tgt - cur).to_dict()                              # per-pref increment of WFH persons to place
    # BOTTOM-UP allocation: place that increment across meshes by the scenario's MARGINAL occupation/industry
    # adopter pool (not a flat scalar) -> S4 lands where ICT workers are, S8 over office workers, etc. S1: dT=0,
    # so N_persons == N_s1 EXACTLY (identity preserved). Per-pref sum stays == the survey-anchored target.
    pool = _marginal_pool(m, table, raises, N_s1)
    poolp = pd.Series(pool).groupby(pref).transform("sum").values
    share = np.divide(pool, poolp, out=np.zeros_like(pool), where=poolp > 0)
    dvec = np.array([dT.get(p, 0.0) for p in pref])
    m["N_persons"] = np.clip(N_s1 + dvec * share, 0, workers)
    m["mobility_applied"] = False

    # SOFT mobility nudge on the INCREMENT only (keeps S1 exact + pref-total conserved)
    if use_mobility and sid != "S1":
        mob = mobility_weight("scenario_potential" if lever in ("occupation", "policy_bundle") else "current_presence")
        if mob is not None:
            N_s1 = state["N_persons"].values                       # S1 base (aligned: left-merge preserved order)
            delta = m["N_persons"].values - N_s1
            mobf = m.mesh.map(mob).fillna(1.0).values
            dm = delta * mobf
            dfp = pd.DataFrame({"pref": m.pref.values, "d": delta, "dm": dm})
            sd = dfp.groupby("pref").d.sum(); sm = dfp.groupby("pref").dm.sum()
            sc_map = (sd / sm.replace(0, np.nan)).fillna(0.0).to_dict()
            dm = dm * m.pref.map(sc_map).fillna(0.0).values
            m["N_persons"] = np.clip(N_s1 + dm, 0, workers)
            m["mobility_applied"] = True

    pik_s = np.divide(m.N_persons.values, workers, out=np.zeros_like(workers, dtype=float), where=workers > 0)
    pik_s = np.clip(pik_s, 0, 1)
    m["H_wfh"] = m.H_worker.values * (1.0 - np.power(1.0 - pik_s, m.workers_per_hh.values))
    m["pik_scenario"] = pik_s
    m["scenario_id"] = sid
    m["spatial_basis"] = LEVER_SPATIAL[lever]
    m["mesh_weight_source"] = LEVER_SPATIAL[lever]   # §15.4 spec-named alias
    return m


def main():
    from wlpr.wfh_state.state_layer import mesh_state
    cfg = scenario_multipliers()
    mult_map = dict(zip(cfg.scenario_id, cfg.wfh_energy_multiplier))
    print("=== scenario multipliers (aggregate, the conserved pref-target) ===")
    print(cfg[["scenario_id", "primary_lever", "spatial_basis", "participation_ratio", "wfh_energy_multiplier"]].to_string(index=False))

    state = mesh_state()
    base_NP = state.groupby("pref").N_persons.sum()
    base_HW = state.H_wfh.sum()

    # industry-G (ICT) share per mesh, for the concentration check
    W, codes = category_workers("ind")
    W["emp"] = W[codes].sum(axis=1)
    W["ict_share"] = np.divide(W["07"].values, W["emp"].values, out=np.zeros(len(W)), where=W["emp"].values > 0)
    ict = dict(zip(W.mesh, W.ict_share))

    print("\n=== per-scenario verification ===")
    rows = []
    for sid, *_ in SCEN:
        ds = scenario_state(state, sid, mult_map)
        # conservation: per-pref ΣN_persons == global-multiplier pref target
        tgt = (state.assign(Ng=state.workers * np.clip(state.pi_worker0 * mult_map[sid], 0, 1))
               .groupby("pref").Ng.sum())
        got = ds.groupby("pref").N_persons.sum()
        cons = float((got - tgt).abs().max())
        # S1 identity: N_persons & H_wfh unchanged
        s1_np = float((ds.set_index("mesh").N_persons - state.set_index("mesh").N_persons).abs().max()) if sid == "S1" else np.nan
        s1_hw = float((ds.set_index("mesh").H_wfh - state.set_index("mesh").H_wfh).abs().max()) if sid == "S1" else np.nan
        # concentration: corr(lift, ict_share) for industry scenarios
        ds["ict"] = ds.mesh.map(ict)
        corr = (float(np.corrcoef(ds["lift"], ds["ict"].fillna(0))[0, 1])
                if "lift" in ds and ds["lift"].std() > 0 else np.nan)
        rows.append(dict(sid=sid, mult=mult_map[sid], lift_mean=round(float(ds["lift"].mean()), 4),
                         lift_max=round(float(ds["lift"].max()), 3), conserve_resid=cons,
                         corr_lift_ICT=round(corr, 3) if corr == corr else None,
                         s1_NP_drift=s1_np, s1_HW_drift=s1_hw, ΣH_wfh=round(float(ds.H_wfh.sum()), 0)))
    rep = pd.DataFrame(rows)
    print(rep.to_string(index=False))

    cons_ok = float(rep.conserve_resid.max()) < 1e-6
    s1ok = (abs(rep[rep.sid == "S1"].s1_NP_drift.iloc[0]) < 1e-9 and abs(rep[rep.sid == "S1"].s1_HW_drift.iloc[0]) < 1e-9)
    s4corr = rep[rep.sid == "S4"].corr_lift_ICT.iloc[0]
    conc_ok = (s4corr is not None) and (s4corr > 0.2)
    print(f"\n[verify] per-pref conservation max resid = {rep.conserve_resid.max():.2e} (<1e-6) -> {'PASS' if cons_ok else 'FAIL'}")
    print(f"[verify] S1 identity drift (N_persons & H_wfh = 0)               -> {'PASS' if s1ok else 'FAIL'}")
    print(f"[verify] S4 concentration corr(lift, ICT_share)={s4corr}        -> {'PASS' if conc_ok else 'FAIL'}")

    # G5 mobility soft-weight: S1 identity preserved + pref conserved + increment nudged toward the scenario's
    # OWN mobility likelihood (future/occupation scenarios -> scenario_potential outflow meshes, per plan 15.5).
    mob_pot = mobility_weight("scenario_potential")
    mob_ok = True
    if mob_pot is not None:
        ds1m = scenario_state(state, "S1", mult_map, use_mobility=True)
        s1m_drift = float((ds1m.set_index("mesh").N_persons - state.set_index("mesh").N_persons).abs().max())
        ds8 = scenario_state(state, "S8", mult_map)
        ds8m = scenario_state(state, "S8", mult_map, use_mobility=True)
        cons8 = float((ds8m.groupby("pref").N_persons.sum() - ds8.groupby("pref").N_persons.sum()).abs().max())
        base = state.set_index("mesh").N_persons.reindex(ds8.mesh.values).values
        shift = (ds8m.N_persons.values - base) - (ds8.N_persons.values - base)
        mobv = np.array([mob_pot.get(x, 1.0) for x in ds8.mesh.values])
        corr = float(np.corrcoef(shift, mobv)[0, 1]) if shift.std() > 0 else 0.0
        mob_ok = s1m_drift < 1e-9 and cons8 < 1e-6 and corr > 0
        print(f"[verify] G5 mobility: S1 drift={s1m_drift:.1e} | S8 pref-conserve={cons8:.1e} | "
              f"corr(S8 increment shift, scenario_potential)={corr:.3f} -> {'PASS' if mob_ok else 'FAIL'}")
    else:
        print("[verify] G5 mobility: wfh_spatial_weight.parquet absent -> nudge unavailable (skip)")

    # GENUINE BOTTOM-UP DIFFERENTIATION: the S4 (ICT-industry) and S8 (broad-occupation) WFH INCREMENTS must land
    # in DIFFERENT meshes (their increment maps are no longer a scalar copy of S1). Capped by the real occupation
    # co-location (~0.96), so we require a clear-but-modest separation, not zero correlation.
    s1np = state.set_index("mesh").N_persons
    def _inc_share(sid):
        d = (scenario_state(state, sid, mult_map).set_index("mesh").N_persons - s1np).clip(lower=0)
        tot = d.sum()
        return (d / tot).reindex(state.mesh.values).fillna(0.0).values if tot > 0 else np.zeros(len(state))
    i4, i8 = _inc_share("S4"), _inc_share("S8")
    msk = (i4 + i8) > 0
    diff_corr = float(np.corrcoef(i4[msk], i8[msk])[0, 1])
    dissim = float(0.5 * np.abs(i4 - i8).sum())
    diff_ok = (diff_corr < 0.99) and (dissim > 0.05)
    print(f"[verify] GENUINE differentiation: S4(ICT) vs S8(office) increment corr={diff_corr:.3f} "
          f"dissim={dissim:.3f} (was corr~1.000) -> {'PASS' if diff_ok else 'FAIL'}")

    # ENTERPRISE RETIRED as a lever + the sectoral-institutionalization combo S6 concentrates like S4 (ICT industry
    # geography carried to 3-4 day frequency): no scenario may carry the enterprise lever, and S6 must concentrate.
    no_ent = all(s[3] != "enterprise" for s in SCEN) and ("ent" not in RATE)
    s6corr = rep[rep.sid == "S6"].corr_lift_ICT.iloc[0]
    s6_ok = (s6corr is not None) and (s6corr > 0.2)
    print(f"[verify] enterprise lever retired (no SCEN uses it)              -> {'PASS' if no_ent else 'FAIL'}")
    print(f"[verify] S6 sectoral-combo concentration corr(lift,ICT)={s6corr}  -> {'PASS' if s6_ok else 'FAIL'}")

    allok = cons_ok and s1ok and conc_ok and mob_ok and diff_ok and no_ent and s6_ok
    print(f"\n=== DOWNSCALING VERDICT: {'PASS' if allok else 'FAIL'} ===")
    if not allok:
        raise SystemExit("scenario downscaling gates failed")


if __name__ == "__main__":
    main()
